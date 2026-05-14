"""Read a chart screenshot with Claude vision.

The user drops a TradingView (or any) screenshot into the sidebar chat
widget. We forward the image bytes to `claude-haiku-4-5` with a strict
JSON schema and pull back:

    - ticker            (best guess from chart label / context)
    - timeframe         ("1m", "5m", "15m", "1h", "4h", "1d", ...)
    - setup             (one-line description of what's happening)
    - key_levels        (support / resistance numerics seen on the chart)
    - suspected_pattern (e.g. "ascending triangle", "double bottom")
    - bias              ("bullish" | "bearish" | "neutral")

If the API key isn't set, the SDK isn't installed, or the call fails for
any reason, we return a `ChartRead` with `source="heuristic"` and
populated only with whatever the caller passed (typically nothing — the
UI then asks the user to fill in the ticker by hand).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

DEFAULT_MODEL = "claude-haiku-4-5"
_CACHE_TTL_SECONDS = 600.0
_MAX_TOKENS = 600
_CACHE: dict[str, tuple[float, "ChartRead"]] = {}


@dataclass
class ChartRead:
    ticker: str = ""
    timeframe: str = ""
    setup: str = ""
    key_levels: list[float] = field(default_factory=list)
    suspected_pattern: str = ""
    bias: str = "neutral"          # "bullish" | "bearish" | "neutral"
    source: str = "ai"             # "ai" | "heuristic"
    model: str = ""
    elapsed_ms: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_chart(
    image_bytes: bytes,
    *,
    media_type: str = "image/png",
    enable_ai: bool = True,
    model: Optional[str] = None,
    cache: bool = True,
) -> ChartRead:
    """Extract structured chart attributes from `image_bytes` via Claude vision.

    Never raises — failures degrade to a heuristic-empty `ChartRead` with
    `error` populated.
    """
    if not image_bytes:
        return ChartRead(source="heuristic", error="no image bytes provided")

    key = _cache_key(image_bytes, model or "")
    if cache:
        hit = _CACHE.get(key)
        if hit and (time.time() - hit[0]) < _CACHE_TTL_SECONDS:
            return hit[1]

    started = time.time()
    if not enable_ai:
        return _heuristic_read(error="AI disabled by caller")

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
        "AI_INTEGRATIONS_ANTHROPIC_API_KEY"
    )
    if not api_key:
        return _heuristic_read(error="ANTHROPIC_API_KEY not set")

    try:
        import anthropic
    except ImportError:
        return _heuristic_read(error="anthropic SDK not installed")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model or os.environ.get("ANTHROPIC_VISION_MODEL") or DEFAULT_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Read this chart and reply with the JSON object described "
                                "in the system prompt. No prose."
                            ),
                        },
                    ],
                }
            ],
        )
    except Exception as e:  # noqa: BLE001
        return _heuristic_read(error=f"vision API failed: {type(e).__name__}: {str(e)[:120]}")

    text = "".join(
        getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text"
    ).strip()
    parsed, err = _parse_payload(text)
    if parsed is None:
        return _heuristic_read(error=err or "vision reply unparseable")

    result = ChartRead(
        ticker=parsed.get("ticker", "")[:20],
        timeframe=parsed.get("timeframe", "")[:10],
        setup=parsed.get("setup", "")[:240],
        key_levels=parsed.get("key_levels", [])[:6],
        suspected_pattern=parsed.get("suspected_pattern", "")[:120],
        bias=parsed.get("bias", "neutral"),
        source="ai",
        model=model or DEFAULT_MODEL,
        elapsed_ms=int((time.time() - started) * 1000),
    )
    if cache:
        _CACHE[key] = (time.time(), result)
    return result


def clear_cache() -> None:
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You read trading-chart screenshots. Extract the following structured info as STRICT JSON. No prose outside the object.

Schema:
{
  "ticker": "<symbol shown on the chart, e.g. BTCUSD or SPY. If unsure, leave empty.>",
  "timeframe": "<1m|5m|15m|30m|1h|4h|1d|1w  — guess from the chart's bar density / label>",
  "setup": "<one sentence describing the structure: trend / chop / breakout / pullback / range>",
  "key_levels": [<numeric support/resistance prices visible on the chart, max 6, ordered low→high>],
  "suspected_pattern": "<e.g. 'ascending triangle', 'double bottom', 'bull flag', 'failed breakout', or '' if none>",
  "bias": "bullish|bearish|neutral"
}

Rules:
- Be conservative. If you cannot tell the ticker or timeframe, leave the field empty rather than inventing.
- Only include numeric levels that are clearly drawn / labelled on the chart. No guessed values.
- "bias" reflects what the price action implies, not your prediction."""


def _parse_payload(text: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.rstrip("`").strip()
    if not s.startswith("{"):
        lo, hi = s.find("{"), s.rfind("}")
        if lo == -1 or hi == -1 or hi <= lo:
            return None, "no JSON object in reply"
        s = s[lo : hi + 1]
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        return None, f"JSON parse failed: {e.msg}"

    bias = str(data.get("bias", "neutral")).lower()
    if bias not in {"bullish", "bearish", "neutral"}:
        bias = "neutral"

    levels: list[float] = []
    raw_levels = data.get("key_levels") or []
    for x in raw_levels:
        try:
            levels.append(float(x))
        except (TypeError, ValueError):
            continue

    return {
        "ticker": str(data.get("ticker", "")).strip().upper(),
        "timeframe": str(data.get("timeframe", "")).strip().lower(),
        "setup": str(data.get("setup", "")).strip(),
        "key_levels": sorted(levels),
        "suspected_pattern": str(data.get("suspected_pattern", "")).strip(),
        "bias": bias,
    }, None


def _heuristic_read(*, error: str) -> ChartRead:
    return ChartRead(source="heuristic", error=error, bias="neutral")


def _cache_key(image_bytes: bytes, model: str) -> str:
    h = hashlib.sha1()
    h.update(model.encode())
    h.update(b"\x00")
    h.update(image_bytes)
    return h.hexdigest()


__all__ = ["ChartRead", "read_chart", "clear_cache"]
