"""Parse Kalshi screenshots into structured markets via Claude vision.

The user uploads a photo of the Kalshi app — typically the crypto event list or
a single market's detail page. We send the image to Claude (vision-capable
model) with a strict JSON schema describing what to extract: title, horizon,
symbol, strike, side labels, and the implied probability + payout multiplier
per side. The result is scored by the existing decision engine.

Uses prompt caching on the system prompt so repeated uploads in a session
only pay for the image tokens, not the schema-explaining preamble.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any

try:
    import anthropic
except ImportError:  # pragma: no cover - surfaced at runtime
    anthropic = None  # type: ignore[assignment]


# The vision model we use. Sonnet is the right tier here: the task is
# perception + structured extraction, not multi-step reasoning.
VISION_MODEL = "claude-sonnet-4-6"


SYSTEM_PROMPT = """You extract structured market data from screenshots of the Kalshi mobile app's crypto markets.

For each market visible in the screenshot, extract:

- symbol: the underlying crypto (BTC, ETH, SOL, XRP). Use the icon + title.
- horizon: "15min" (e.g. "ETH 15 min · $2,261.91 target"), "hourly" (e.g. "Bitcoin price today at 11am EDT"), "daily" (e.g. "Bitcoin price today at 5pm EDT"), "weekly" (e.g. "Bitcoin price on Friday at 5pm EDT"), or "longer" for end-of-year etc.
- title: the full market title as shown.
- target_price: the strike or target price the market resolves against (for 15-min Up/Down markets, the single target like 2261.91; for range markets, leave null — use sides instead).
- sides: a list of the visible sides. Each side has:
    - label: exactly as displayed (e.g. "Up", "Down", "$79,750 or above", "$80,000 or above", "$79,500 to $79,749.99")
    - prob_pct: implied probability percentage shown on the right (e.g. 42, 58, 89, 12). Integer 0-100.
    - payout: the multiplier shown next to the probability (e.g. 2.01, 1.76). Float, or null if not visible.
    - strike: numeric strike for range markets (e.g. 79750 for "$79,750 or above"). Null for Up/Down.
    - strike_type: "greater" for "X or above", "less" for "below X", "between" for "X to Y", "up" for Up, "down" for Down.
- volume_usd: the dollar volume shown (e.g. "$7,419 vol" → 7419). Integer, null if not visible.
- time_remaining_seconds: countdown until close if shown (e.g. "8:35" near a 15-min market → 515; "53:44" → 3224; "6:53:49" → 24829). Integer, null if not visible.

Return ONLY a JSON object matching the requested schema. Do not include any prose, explanations, or markdown fences.

If the screenshot contains multiple distinct markets (e.g. the Crypto tab list), return all of them. If the screenshot shows a single market detail page, return one entry. If nothing crypto-related is visible, return an empty markets list."""


PARSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "markets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "enum": ["BTC", "ETH", "SOL", "XRP"]},
                    "horizon": {
                        "type": "string",
                        "enum": ["15min", "hourly", "daily", "weekly", "longer"],
                    },
                    "title": {"type": "string"},
                    "target_price": {"type": ["number", "null"]},
                    "sides": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "prob_pct": {"type": "integer", "minimum": 0, "maximum": 100},
                                "payout": {"type": ["number", "null"]},
                                "strike": {"type": ["number", "null"]},
                                "strike_type": {
                                    "type": "string",
                                    "enum": ["greater", "less", "between", "up", "down"],
                                },
                            },
                            "required": ["label", "prob_pct", "strike_type"],
                            "additionalProperties": False,
                        },
                    },
                    "volume_usd": {"type": ["integer", "null"]},
                    "time_remaining_seconds": {"type": ["integer", "null"]},
                },
                "required": ["symbol", "horizon", "title", "sides"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["markets"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ParsedSide:
    label: str
    prob_pct: int
    payout: float | None
    strike: float | None
    strike_type: str


@dataclass(frozen=True)
class ParsedMarket:
    """A market extracted from a screenshot.

    Mirrors the JSON schema but is dataclass-typed for the decision pipeline.
    """

    symbol: str
    horizon: str
    title: str
    sides: list[ParsedSide]
    target_price: float | None = None
    volume_usd: int | None = None
    time_remaining_seconds: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def parse_screenshot(
    image_bytes: bytes,
    media_type: str = "image/png",
    *,
    model: str = VISION_MODEL,
    client: Any | None = None,
) -> list[ParsedMarket]:
    """Send `image_bytes` to Claude vision and return parsed markets.

    Raises:
        RuntimeError: if the anthropic SDK is not installed or no API key set.
        ValueError: if the response cannot be parsed as the expected schema.
    """
    if anthropic is None:
        raise RuntimeError(
            "anthropic SDK not installed. Add `anthropic` to dependencies."
        )
    if client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to .env to enable photo parsing."
            )
        client = anthropic.Anthropic()

    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    # System prompt is cached — the schema-explaining preamble doesn't change
    # across uploads, so repeated parses pay only for the (small) image and
    # the (smaller) JSON response.
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        output_config={"format": {"type": "json_schema", "schema": PARSE_SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract every crypto market visible. Return JSON only.",
                    },
                ],
            }
        ],
    )

    text = _first_text(response)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude returned non-JSON: {text[:200]!r}") from e

    return _payload_to_markets(payload)


def _first_text(response: Any) -> str:
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", "") or ""
    return ""


def _payload_to_markets(payload: dict[str, Any]) -> list[ParsedMarket]:
    raw_markets = payload.get("markets", []) or []
    out: list[ParsedMarket] = []
    for raw in raw_markets:
        sides_raw = raw.get("sides", []) or []
        sides = [
            ParsedSide(
                label=str(s.get("label", "")),
                prob_pct=int(s.get("prob_pct", 0)),
                payout=(float(s["payout"]) if s.get("payout") is not None else None),
                strike=(float(s["strike"]) if s.get("strike") is not None else None),
                strike_type=str(s.get("strike_type", "")),
            )
            for s in sides_raw
        ]
        target = raw.get("target_price")
        out.append(
            ParsedMarket(
                symbol=str(raw.get("symbol", "")).upper(),
                horizon=str(raw.get("horizon", "")),
                title=str(raw.get("title", "")),
                sides=sides,
                target_price=float(target) if target is not None else None,
                volume_usd=raw.get("volume_usd"),
                time_remaining_seconds=raw.get("time_remaining_seconds"),
                raw=raw,
            )
        )
    return out
