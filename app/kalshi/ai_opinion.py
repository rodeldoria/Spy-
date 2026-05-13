"""Claude second opinion for Kalshi order recommendations.

Optional advisory layer that takes a `Decision` + `CouncilResult` and asks
Claude to AGREE / DISAGREE / UNSURE in one sentence. Designed to fail
soft: if the SDK isn't installed, the env var isn't set, or the API call
errors, we return a `skipped` verdict with a human-readable reason so the
UI can show why instead of pretending everything is fine.

Caching: each market ticker × spot-price-bucket combination is cached in
session state for ~60s so we don't burn tokens on auto-refresh ticks.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Literal

from app.kalshi.council import CouncilResult
from app.kalshi.decisions import Decision

Verdict = Literal["AGREE", "DISAGREE", "UNSURE", "SKIPPED"]

# Claude calls are cached this long per market/spot bucket. Auto-refresh
# defaults are 5–30s on the page, so 60s gives us ≥1 free cache hit per
# market between cycles without hiding genuinely fresh re-evaluations.
CACHE_TTL_SECONDS = 60

# Spot rounded to this many decimal places forms part of the cache key.
# 0 = whole-dollar buckets, which is enough resolution for "did the price
# move enough to merit a fresh second opinion?"
SPOT_BUCKET_DECIMALS = 0

# Token budget is small on purpose — we want a one-sentence verdict, not
# an essay. Anything more is over-spend for a recurring per-market call.
MAX_TOKENS = 200

# Model: cheapest current Claude that's still strong enough to evaluate
# a structured numeric argument. Override via env if you want Opus.
# Note: prompt caching is intentionally not used — Haiku 4.5's minimum
# cacheable prefix is 4096 tokens and our system+user prompts come in
# around 500 tokens, so a cache_control breakpoint would silently no-op.
DEFAULT_MODEL = "claude-haiku-4-5"


@dataclass(frozen=True)
class Opinion:
    """Result of asking Claude whether the order recommendation is sound."""

    verdict: Verdict
    reasoning: str        # one-sentence summary, or skip reason
    model: str            # which model answered, or "" if skipped
    elapsed_ms: int       # round-trip time, 0 if skipped


def _cache_key(decision: Decision) -> str:
    bucket = round(decision.spot_price, SPOT_BUCKET_DECIMALS)
    return f"{decision.market_ticker}|{decision.direction}|{bucket}"


def _read_cache(cache: dict, key: str) -> Opinion | None:
    entry = cache.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > CACHE_TTL_SECONDS:
        return None
    return entry["opinion"]


def _write_cache(cache: dict, key: str, opinion: Opinion) -> None:
    cache[key] = {"ts": time.time(), "opinion": opinion}


def get_opinion(
    decision: Decision,
    council: CouncilResult,
    *,
    cache: dict | None = None,
    model: str | None = None,
) -> Opinion:
    """Ask Claude whether the recommendation is sound. Always returns an Opinion.

    `cache` should be a dict (typically `st.session_state["kalshi_ai_cache"]`)
    so repeated reruns on the same market don't burn API tokens. Pass None
    to skip caching.

    The verdict is one of:
      - AGREE     — Claude agrees with the recommendation
      - DISAGREE  — Claude disagrees (UI should surface a caution badge)
      - UNSURE    — Claude can't tell; UI shows the badge but doesn't block
      - SKIPPED   — call wasn't made (no key, no SDK, API error)
    """
    if cache is not None:
        cached = _read_cache(cache, _cache_key(decision))
        if cached is not None:
            return cached

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
        "AI_INTEGRATIONS_ANTHROPIC_API_KEY"
    )
    if not api_key:
        return Opinion(
            verdict="SKIPPED",
            reasoning="ANTHROPIC_API_KEY not set",
            model="",
            elapsed_ms=0,
        )

    try:
        import anthropic
    except ImportError:
        return Opinion(
            verdict="SKIPPED",
            reasoning="anthropic SDK not installed (pip install anthropic)",
            model="",
            elapsed_ms=0,
        )

    client = anthropic.Anthropic(api_key=api_key)
    model_id = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL

    start = time.time()
    try:
        msg = client.messages.create(
            model=model_id,
            max_tokens=MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _user_prompt(decision, council)}],
        )
    except Exception as e:  # anthropic.APIError, network, etc.
        return Opinion(
            verdict="SKIPPED",
            reasoning=f"Claude API error: {type(e).__name__}: {e}",
            model=model_id,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    elapsed_ms = int((time.time() - start) * 1000)
    text = "".join(
        block.text for block in msg.content if getattr(block, "type", None) == "text"
    ).strip()

    verdict, reasoning = _parse_response(text)
    opinion = Opinion(
        verdict=verdict,
        reasoning=reasoning,
        model=model_id,
        elapsed_ms=elapsed_ms,
    )
    if cache is not None:
        _write_cache(cache, _cache_key(decision), opinion)
    return opinion


_SYSTEM_PROMPT = """You are a risk-conscious second-opinion on a binary prediction-market trade.

The user has been recommended a side (YES or NO) on a Kalshi market by a
driftless GBM vol model. Your job is to sanity-check that recommendation
in one sentence.

Reply ONLY with a single JSON object, no markdown, no preamble:
{"verdict": "AGREE" | "DISAGREE" | "UNSURE", "reasoning": "<one sentence>"}

Guidelines:
- AGREE: the math looks coherent and you'd take the bet at this size.
- DISAGREE: there's a specific reason the recommendation is dangerous
  (e.g. structural mispricing of tails, very wide spread relative to
  edge, news-driven move not in the vol model).
- UNSURE: insufficient context to take a side. This is fine; don't
  pretend confidence you don't have.

Be brief. One sentence. No emojis."""


def _user_prompt(decision: Decision, council: CouncilResult) -> str:
    chosen = decision.chosen
    assert chosen is not None  # caller filters out PASS
    council_lines = "\n".join(
        f"  - {c.name}: {'PASS' if c.passed else 'FAIL'} ({c.detail})"
        for c in council.checks
    )
    return f"""Market: {decision.title}
Bet: {decision.bet_summary}
Closes in: {decision.horizon_seconds / 60:.1f} minutes
Spot price: ${decision.spot_price:,.2f} ({decision.spot_source})
Realised vol (σ/min): {decision.sigma_per_min * 100:.4f}%

Recommendation: BUY {decision.direction} @ {chosen.ask_cents}¢
  - Model probability: {chosen.model_prob * 100:.1f}%
  - Book implied probability: {chosen.implied_prob * 100:.1f}%
  - Edge: {chosen.edge * 100:+.1f}pp
  - EV per $1 staked: {chosen.ev_per_dollar * 100:+.1f}¢
  - Kelly fraction: {chosen.kelly_fraction * 100:.1f}%
  - Conviction: {decision.confidence_pct:.0f}%

Council gate ({council.passed}/{council.total} pass):
{council_lines}

Decision reasoning: {decision.reasoning}
Warnings: {'; '.join(decision.warnings) if decision.warnings else 'none'}

Do you AGREE, DISAGREE, or are you UNSURE about this trade?"""


def _parse_response(text: str) -> tuple[Verdict, str]:
    """Parse Claude's JSON reply.

    Falls back to keyword sniffing if the JSON contract is broken — Claude
    very occasionally wraps the response in markdown despite the system
    prompt, and the user is better served by a degraded parse than a SKIP.
    """
    try:
        # Strip markdown fences if present.
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("```", 2)[1]
            if stripped.startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.rstrip("`").strip()
        data = json.loads(stripped)
        verdict = str(data.get("verdict", "")).upper()
        reasoning = str(data.get("reasoning", "")).strip()
        if verdict in ("AGREE", "DISAGREE", "UNSURE") and reasoning:
            return verdict, reasoning  # type: ignore[return-value]
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    upper = text.upper()
    if "DISAGREE" in upper:
        return "DISAGREE", text.strip()[:240] or "Claude returned unstructured DISAGREE"
    if "AGREE" in upper:
        return "AGREE", text.strip()[:240] or "Claude returned unstructured AGREE"
    return "UNSURE", text.strip()[:240] or "Claude reply could not be parsed"
