"""Influencer-pulse fetcher.

Big-account commentary moves crypto markets. Elon's BTC tweet, Saylor's
buy announcement, Cathie Wood's price target, Arthur Hayes' macro post —
each can spike or dump prices within minutes. This module asks Perplexity
to scan the last 24 hours of major-account posts on a given asset and
return a structured `InfluencerPulse` with:

  - overall_sentiment: "bullish" | "bearish" | "mixed" | "quiet"
  - net_bias: -1.0..+1.0 score
  - voices: list of Voice records (handle, sentiment, paraphrased quote,
    estimated market impact "low|med|high")
  - summary: 1-2 sentence plain-English

Cached on disk for 30 minutes so we don't hammer the API on every
Streamlit refresh.

Tracked accounts (curated, not exhaustive): Elon Musk, Michael Saylor,
Arthur Hayes, Cathie Wood, Anthony Pompliano, Vitalik Buterin,
Changpeng Zhao, Brian Armstrong, Raoul Pal, PlanB, Willy Woo.

Honest about limits: Perplexity is a search-and-summarise tool, not a
direct Twitter feed. We're getting the *reported* commentary, not
real-time tweets. For sub-minute reaction this isn't enough — but for
"what's the loud-voice narrative right now?" it works well.
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
CACHE_DIR = Path(os.path.expanduser("~/.monte/cache/influencers"))
CACHE_TTL_SECONDS = 60 * 30  # 30 minutes

INFLUENCERS = [
    "Elon Musk (@elonmusk)",
    "Michael Saylor (@saylor)",
    "Arthur Hayes (@CryptoHayes)",
    "Cathie Wood (@CathieDWood)",
    "Anthony Pompliano (@APompliano)",
    "Vitalik Buterin (@VitalikButerin)",
    "Changpeng Zhao (@cz_binance)",
    "Brian Armstrong (@brian_armstrong)",
    "Raoul Pal (@RaoulGMI)",
    "PlanB (@100trillionUSD)",
    "Willy Woo (@woonomic)",
]


@dataclass
class Voice:
    handle: str
    sentiment: str   # "bullish" | "bearish" | "neutral"
    quote: str       # paraphrased ≤200 chars
    impact: str      # "low" | "med" | "high"


@dataclass
class InfluencerPulse:
    symbol: str
    configured: bool
    overall_sentiment: str   # bullish | bearish | mixed | quiet
    net_bias: float          # -1..+1
    summary: str
    voices: list[Voice] = field(default_factory=list)
    fetched_at: float = 0.0
    error: str = ""

    @property
    def emoji(self) -> str:
        return {
            "bullish": "🟢",
            "bearish": "🔴",
            "mixed": "🟡",
            "quiet": "⚪",
        }.get(self.overall_sentiment, "⚪")


def _cache_path(symbol: str) -> Path:
    key = hashlib.sha1(f"infl|{symbol}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{key}.json"


QUIET_TTL_SECONDS = 60 * 5  # short cache for quiet/error so we still throttle


def _load_cache(symbol: str) -> InfluencerPulse | None:
    p = _cache_path(symbol)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        ttl = CACHE_TTL_SECONDS
        if data.get("error") or data.get("overall_sentiment") == "quiet":
            ttl = QUIET_TTL_SECONDS
        if time.time() - float(data.get("fetched_at", 0)) > ttl:
            return None
        raw_voices = data.pop("voices", []) or []
        voices = []
        for v in raw_voices:
            if not isinstance(v, dict):
                continue
            try:
                voices.append(Voice(
                    handle=str(v.get("handle", "")),
                    sentiment=str(v.get("sentiment", "neutral")),
                    quote=str(v.get("quote", "")),
                    impact=str(v.get("impact", "low")),
                ))
            except (TypeError, ValueError):
                continue
        pulse = InfluencerPulse(
            symbol=str(data.get("symbol", symbol)),
            configured=bool(data.get("configured", True)),
            overall_sentiment=str(data.get("overall_sentiment", "quiet")),
            net_bias=float(data.get("net_bias", 0.0) or 0.0),
            summary=str(data.get("summary", "")),
            voices=voices,
            fetched_at=float(data.get("fetched_at", 0.0) or 0.0),
            error=str(data.get("error", "")),
        )
        return pulse
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def _save_cache(pulse: InfluencerPulse) -> None:
    # Cache *everything* (including quiet/error) so we don't hammer the API
    # on every Streamlit auto-refresh during a quiet news window.
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        d = pulse.__dict__.copy()
        d["voices"] = [v.__dict__ for v in pulse.voices]
        _cache_path(pulse.symbol).write_text(json.dumps(d))
    except OSError:
        pass


def _prompt(symbol: str) -> str:
    handles = ", ".join(INFLUENCERS)
    return (
        f"Scan the last 24 hours of public commentary from these crypto "
        f"market-moving accounts about {symbol}: {handles}. "
        "For each account that posted something material, extract a single "
        "paraphrased quote (under 200 chars), classify the sentiment "
        "(bullish/bearish/neutral) and the estimated market impact "
        "(low/med/high — 'high' means a post likely to move price >2% on its own). "
        "Then assess the overall narrative.\n\n"
        "Reply ONLY as compact JSON with this shape:\n"
        '{"overall_sentiment":"bullish|bearish|mixed|quiet",'
        '"net_bias":-1.0..1.0,'
        '"summary":"<=2 sentence plain English narrative",'
        '"voices":[{"handle":"@x","sentiment":"bullish|bearish|neutral",'
        '"quote":"...","impact":"low|med|high"}]}\n'
        "If no relevant posts in 24h, return overall_sentiment='quiet' with "
        "an empty voices list. No prose outside the JSON object."
    )


def _parse(symbol: str, raw: str) -> InfluencerPulse:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        data: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e == -1:
            return InfluencerPulse(
                symbol=symbol, configured=True, overall_sentiment="quiet",
                net_bias=0.0, summary="no usable response",
                fetched_at=time.time(),
            )
        try:
            data = json.loads(text[s:e+1])
        except json.JSONDecodeError:
            return InfluencerPulse(
                symbol=symbol, configured=True, overall_sentiment="quiet",
                net_bias=0.0, summary="parse failure",
                fetched_at=time.time(),
            )

    sent = str(data.get("overall_sentiment", "quiet")).lower()
    if sent not in {"bullish", "bearish", "mixed", "quiet"}:
        sent = "mixed"
    try:
        bias = float(data.get("net_bias", 0.0))
        bias = max(-1.0, min(1.0, bias))
    except (TypeError, ValueError):
        bias = 0.0

    voices = []
    for v in (data.get("voices") or [])[:10]:
        if not isinstance(v, dict):
            continue
        v_sent = str(v.get("sentiment", "neutral")).lower()
        if v_sent not in {"bullish", "bearish", "neutral"}:
            v_sent = "neutral"
        v_imp = str(v.get("impact", "low")).lower()
        if v_imp not in {"low", "med", "high"}:
            v_imp = "low"
        voices.append(Voice(
            handle=str(v.get("handle", ""))[:60],
            sentiment=v_sent,
            quote=str(v.get("quote", ""))[:240],
            impact=v_imp,
        ))

    return InfluencerPulse(
        symbol=symbol,
        configured=True,
        overall_sentiment=sent,
        net_bias=bias,
        summary=str(data.get("summary", ""))[:500],
        voices=voices,
        fetched_at=time.time(),
    )


def fetch_influencer_pulse(symbol: str, *, model: str | None = None) -> InfluencerPulse:
    """Return an `InfluencerPulse` for `symbol`. 30-minute cache.

    Never raises — failures become a pulse with `error` populated.
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY", "").strip()
    if not api_key:
        return InfluencerPulse(
            symbol=symbol, configured=False, overall_sentiment="quiet",
            net_bias=0.0,
            summary="PERPLEXITY_API_KEY not set — add it to unlock influencer pulse.",
            fetched_at=time.time(),
        )

    cached = _load_cache(symbol)
    if cached is not None:
        return cached

    body = json.dumps({
        "model": model or DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "You are concise. Reply with JSON only."},
            {"role": "user", "content": _prompt(symbol)},
        ],
        "temperature": 0.2,
        "max_tokens": 700,
    }).encode()

    req = request.Request(
        PERPLEXITY_ENDPOINT, data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode())
    except error.HTTPError as e:
        # Perplexity returns HTTP 401 for both bad keys AND exhausted
        # quota — read the body so we can give an honest reason.
        try:
            err_body = e.read().decode("utf-8", errors="ignore")
            err_json = json.loads(err_body)
            err_type = (err_json.get("error") or {}).get("type", "")
            err_msg = (err_json.get("error") or {}).get("message", "")
        except Exception:
            err_type, err_msg = "", ""
        if err_type == "insufficient_quota" or "quota" in err_msg.lower():
            human = (
                "Perplexity account is out of quota — top up at "
                "perplexity.ai/settings/api or swap in a funded key."
            )
        elif e.code == 401:
            human = "Perplexity API key rejected (401). Check the key value."
        else:
            human = f"Perplexity HTTP {e.code} {err_msg or ''}".strip()
        return InfluencerPulse(
            symbol=symbol, configured=True, overall_sentiment="quiet",
            net_bias=0.0, summary=human,
            fetched_at=time.time(), error=human,
        )
    except Exception as e:
        return InfluencerPulse(
            symbol=symbol, configured=True, overall_sentiment="quiet",
            net_bias=0.0, summary=f"Request failed: {e}",
            fetched_at=time.time(), error=str(e),
        )

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return InfluencerPulse(
            symbol=symbol, configured=True, overall_sentiment="quiet",
            net_bias=0.0, summary="empty Perplexity response",
            fetched_at=time.time(),
        )

    pulse = _parse(symbol, content)
    _save_cache(pulse)
    return pulse


__all__ = ["InfluencerPulse", "Voice", "fetch_influencer_pulse", "INFLUENCERS"]
