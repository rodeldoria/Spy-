"""Perplexity news/event lookup for signal confirmation.

The dashboard already pieces together a technical view. When the user sees a
BUY pattern fire we also want to know "what's the news telling us right now?"
so the trade is not taken blind into an earnings release, an ETF outflow, an
exploit, or an FOMC day.

This module is a thin wrapper around the Perplexity chat completions API
(`/chat/completions`, `sonar-small-online`-style models). The API key is read
from `PERPLEXITY_API_KEY` at call time. When the key is missing we return a
graceful "unconfigured" `NewsBrief` instead of raising — the UI keeps working
and just shows a hint that adding the key unlocks confirmation.

Responses are cached on disk for a short TTL so the dashboard's 30s refresh
loop does not hammer the API.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request

PERPLEXITY_ENDPOINT = "https://api.perplexity.ai/chat/completions"
DEFAULT_MODEL = "sonar"
CACHE_DIR = Path(os.path.expanduser("~/.monte/cache/perplexity"))
CACHE_TTL_SECONDS = 60 * 15  # 15 minutes


@dataclass
class NewsBrief:
    symbol: str
    configured: bool
    sentiment: str  # "bullish" | "bearish" | "neutral" | "unknown"
    summary: str
    headlines: list[str] = field(default_factory=list)
    catalysts: list[str] = field(default_factory=list)
    fetched_at: float = 0.0
    error: str = ""

    def aligns_with(self, action: str) -> str:
        """Return 'confirms' / 'conflicts' / 'neutral' vs a technical action."""
        action = (action or "").upper()
        if self.sentiment == "unknown":
            return "neutral"
        bullish_actions = {"BUY", "STRONG_BUY"}
        bearish_actions = {"SELL", "STRONG_SELL"}
        if self.sentiment == "bullish" and action in bullish_actions:
            return "confirms"
        if self.sentiment == "bearish" and action in bearish_actions:
            return "confirms"
        if self.sentiment == "bullish" and action in bearish_actions:
            return "conflicts"
        if self.sentiment == "bearish" and action in bullish_actions:
            return "conflicts"
        return "neutral"


def _cache_path(symbol: str, action: str) -> Path:
    key = hashlib.sha1(f"{symbol}|{action}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{key}.json"


def _load_cache(symbol: str, action: str) -> NewsBrief | None:
    path = _cache_path(symbol, action)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - float(payload.get("fetched_at", 0)) > CACHE_TTL_SECONDS:
        return None
    return NewsBrief(**payload)


def _save_cache(brief: NewsBrief) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(brief.symbol, brief.sentiment).write_text(
            json.dumps(brief.__dict__)
        )
    except OSError:
        pass


def _prompt(symbol: str, action: str) -> str:
    return (
        f"You are a markets news analyst. Summarise the most relevant news and "
        f"upcoming catalysts for {symbol} from the last 48 hours that a trader "
        f"considering a {action} should know about. "
        "Reply ONLY as compact JSON with these keys:\n"
        '{"sentiment":"bullish|bearish|neutral",'
        '"summary":"<=2 sentence plain English",'
        '"headlines":["..."],'
        '"catalysts":["upcoming events to watch"]}\n'
        "Do not include any prose outside the JSON object."
    )


def _parse_response(symbol: str, raw: str) -> NewsBrief:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        data: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        # Fall back: best-effort if the model wrapped JSON in commentary.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return NewsBrief(
                symbol=symbol,
                configured=True,
                sentiment="unknown",
                summary=text[:280] or "no usable response",
                fetched_at=time.time(),
            )
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return NewsBrief(
                symbol=symbol,
                configured=True,
                sentiment="unknown",
                summary="could not parse model output",
                fetched_at=time.time(),
            )

    sentiment = str(data.get("sentiment", "neutral")).lower()
    if sentiment not in {"bullish", "bearish", "neutral"}:
        sentiment = "neutral"
    return NewsBrief(
        symbol=symbol,
        configured=True,
        sentiment=sentiment,
        summary=str(data.get("summary", ""))[:600],
        headlines=[str(x) for x in (data.get("headlines") or [])][:5],
        catalysts=[str(x) for x in (data.get("catalysts") or [])][:5],
        fetched_at=time.time(),
    )


def fetch_news(symbol: str, action: str = "BUY", *, model: str | None = None) -> NewsBrief:
    """Return a `NewsBrief` for `symbol`. Caches results for 15 minutes.

    Never raises — failures (missing key, network error, parse error) become a
    `NewsBrief` with `error` populated so the UI can show the reason.
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY", "").strip()
    if not api_key:
        return NewsBrief(
            symbol=symbol,
            configured=False,
            sentiment="unknown",
            summary="PERPLEXITY_API_KEY not set — add it to .env to unlock news confirmation.",
            fetched_at=time.time(),
        )

    cached = _load_cache(symbol, action)
    if cached is not None:
        return cached

    body = json.dumps(
        {
            "model": model or DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": "You are concise. Reply with JSON only."},
                {"role": "user", "content": _prompt(symbol, action)},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
        }
    ).encode()

    req = request.Request(
        PERPLEXITY_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode())
    except error.HTTPError as e:
        return NewsBrief(
            symbol=symbol,
            configured=True,
            sentiment="unknown",
            summary=f"Perplexity API error: HTTP {e.code}",
            fetched_at=time.time(),
            error=str(e),
        )
    except Exception as e:
        return NewsBrief(
            symbol=symbol,
            configured=True,
            sentiment="unknown",
            summary=f"Perplexity request failed: {e}",
            fetched_at=time.time(),
            error=str(e),
        )

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return NewsBrief(
            symbol=symbol,
            configured=True,
            sentiment="unknown",
            summary="Perplexity returned an empty response.",
            fetched_at=time.time(),
        )

    brief = _parse_response(symbol, content)
    _save_cache(brief)
    return brief


__all__ = ["NewsBrief", "fetch_news"]
