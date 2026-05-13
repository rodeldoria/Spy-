"""Investment-decision premortem — "this trade already lost. Why?".

Wharton/Cornell's "prospective hindsight" trick: when you ask "what could go
wrong?" people give cautious, generic answers. When you ASSUME the plan has
already failed and ask "why?", the brain produces specific, creative, honest
explanations. Klein's premortem.

This module runs that drill against a short-horizon investment idea (a trade,
a council verdict, a paper-portfolio entry, a swing thesis) and returns:

  - 3 ranked failure modes — each with the chain of events, the hidden
    assumption it rests on, and an early-warning signal to watch for.
  - The single biggest hidden assumption the whole plan depends on.
  - A revised version of the plan with the weak spots patched.
  - A 5-item pre-launch checklist to clear before you place the order.

Designed for short windows: minutes to weeks, not multi-year strategy.
The horizon argument tunes the failure-mode lens (intraday: liquidity /
news shock / fat-finger; swing: regime change / vol expansion; position:
drawdown / margin; long: thesis decay).

Network use:
- If ANTHROPIC_API_KEY is set and the SDK is installed, asks Claude Haiku
  for the structured premortem.
- Otherwise falls back to a deterministic heuristic that still produces
  useful generic failure modes tuned to the horizon. Either way the
  output schema is identical so the UI doesn't branch.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional


Horizon = Literal["intraday", "swing", "position", "long"]

HORIZON_LABELS: dict[Horizon, str] = {
    "intraday": "Intraday (minutes–hours)",
    "swing": "Swing (1–10 days)",
    "position": "Position (weeks–months)",
    "long": "Long-term thesis (quarters+)",
}

Severity = Literal["low", "medium", "high", "critical"]

_SEV_RANK: dict[Severity, int] = {"low": 1, "medium": 2, "high": 3, "critical": 4}

DEFAULT_MODEL = "claude-haiku-4-5"
_CACHE_TTL_SECONDS = 600.0
_MAX_TOKENS = 900
_PLAN_TRIM = 4000   # hard cap on input length we send to the model


@dataclass
class FailureMode:
    name: str
    likelihood: Severity
    danger: Severity
    chain: str
    hidden_assumption: str
    early_warning: str

    @property
    def risk_score(self) -> int:
        """0..16 — combined likelihood × danger ranking."""
        return _SEV_RANK[self.likelihood] * _SEV_RANK[self.danger]


@dataclass
class PremortemResult:
    failure_modes: list[FailureMode]
    biggest_hidden_assumption: str
    revised_plan: list[str]
    prelaunch_checklist: list[str]
    horizon: Horizon
    source: Literal["ai", "heuristic"]
    model: str = ""
    elapsed_ms: int = 0
    ai_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["failure_modes"] = [asdict(f) for f in self.failure_modes]
        return d

    @property
    def top_failure_mode(self) -> Optional[FailureMode]:
        if not self.failure_modes:
            return None
        return max(self.failure_modes, key=lambda f: f.risk_score)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_CACHE: dict[str, tuple[float, PremortemResult]] = {}


def premortem(
    *,
    title: str,
    plan: str,
    horizon: Horizon = "swing",
    context: Optional[dict[str, Any]] = None,
    enable_ai: bool = True,
    model: Optional[str] = None,
    cache: bool = True,
) -> PremortemResult:
    """Run a premortem on a short-horizon investment idea.

    `title` is one line ("Long BTC on 4h breakout"); `plan` is the body of
    the idea — entries, stops, targets, why you think the edge exists, etc.
    `context` is an optional dict of structured numbers (RSI, ATR, edge_pp,
    book conditions, news state) that gets serialised into the prompt.
    """
    plan = plan.strip()
    title = title.strip()
    if not plan:
        return _empty_result(horizon, reason="no plan provided")

    key = _cache_key(title, plan, horizon, context, enable_ai)
    if cache:
        hit = _read_cache(key)
        if hit is not None:
            return hit

    started = time.time()
    result: Optional[PremortemResult] = None
    ai_error: Optional[str] = None

    if enable_ai:
        result, ai_error = _ai_premortem(
            title=title,
            plan=plan,
            horizon=horizon,
            context=context or {},
            model=model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL,
        )

    if result is None:
        result = _heuristic_premortem(title=title, plan=plan, horizon=horizon)
        result.ai_error = ai_error

    result.elapsed_ms = int((time.time() - started) * 1000)
    if cache:
        _CACHE[key] = (time.time(), result)
    return result


def clear_cache() -> None:
    """Drop the in-process premortem cache. Used by tests."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# AI path
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a skeptical hedge-fund risk officer running a Klein-style premortem on a SHORT-HORIZON investment decision. The trade has ALREADY failed — your job is to explain why, in concrete trade-level terms (not vague platitudes).

Output strict JSON. No prose outside it. Schema:

{
  "failure_modes": [
    {
      "name": "<3-6 word label>",
      "likelihood": "low|medium|high|critical",
      "danger": "low|medium|high|critical",
      "chain": "<the specific chain of events: what happens first, then what, then the loss. <= 2 sentences>",
      "hidden_assumption": "<the assumption the trade silently depends on for THIS failure mode>",
      "early_warning": "<one observable signal (price, vol, news, order-book) that would warn you BEFORE the loss>"
    }
    // exactly 3 modes, ranked from most-dangerous to least
  ],
  "biggest_hidden_assumption": "<the single assumption the WHOLE plan quietly rests on — the thing so obvious you forgot it was an assumption>",
  "revised_plan": [
    "<bullet 1: concrete change that patches a weak spot>",
    "<bullet 2: ...>",
    "<bullet 3: ...>",
    "<bullet 4: ...>"
  ],
  "prelaunch_checklist": [
    "<short imperative — pass/fail before you click the order button>",
    "...",
    "<5 items total>"
  ]
}

Rules:
- Be SPECIFIC to the trade described. Generic answers like "the market could move against you" are useless.
- Each failure mode must reference real mechanics (slippage on a thin book, gap through stop on news, vol expansion, regime flip, correlation break, dealer hedging flow, funding flip, etc.).
- For intraday/swing horizons, weight liquidity, slippage, news catalysts, vol regime, and stop placement heavily.
- For position/long horizons, weight thesis decay, drawdown sequencing, position-size error, correlation break, and macro regime.
- The pre-launch checklist must be ACTIONABLE in <5 minutes ("verify ATR-based stop ≥ X", not "make sure the trade is good").
- Do NOT pad. Five checklist items, three failure modes, four revised-plan bullets. Hard limits.
"""


def _user_prompt(title: str, plan: str, horizon: Horizon, context: dict[str, Any]) -> str:
    plan_trim = plan if len(plan) <= _PLAN_TRIM else plan[:_PLAN_TRIM] + " […truncated]"
    ctx_block = json.dumps(context, indent=2, default=str) if context else "{}"
    return (
        f"HORIZON: {horizon} ({HORIZON_LABELS[horizon]})\n"
        f"TITLE: {title or '(no title)'}\n\n"
        f"PLAN / ANALYSIS:\n{plan_trim}\n\n"
        f"STRUCTURED CONTEXT (optional, may be empty):\n{ctx_block}\n\n"
        "This trade has already lost money. Run the premortem and return the JSON."
    )


def _ai_premortem(
    *,
    title: str,
    plan: str,
    horizon: Horizon,
    context: dict[str, Any],
    model: str,
) -> tuple[Optional[PremortemResult], Optional[str]]:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
        "AI_INTEGRATIONS_ANTHROPIC_API_KEY"
    )
    if not api_key:
        return None, "ANTHROPIC_API_KEY not set"
    try:
        import anthropic
    except ImportError:
        return None, "anthropic SDK not installed"

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": _user_prompt(title, plan, horizon, context)}
            ],
        )
    except Exception as e:
        return None, f"Claude API error: {type(e).__name__}: {str(e)[:160]}"

    text = "".join(
        getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text"
    ).strip()

    parsed, err = _parse_ai_payload(text)
    if parsed is None:
        return None, err

    result = PremortemResult(
        failure_modes=parsed["failure_modes"],
        biggest_hidden_assumption=parsed["biggest_hidden_assumption"],
        revised_plan=parsed["revised_plan"],
        prelaunch_checklist=parsed["prelaunch_checklist"],
        horizon=horizon,
        source="ai",
        model=model,
    )
    return result, None


def _parse_ai_payload(text: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Tolerant parse: strip code fences, locate the outermost JSON object,
    coerce severities to the allowed enum, clip list lengths."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.rstrip("`").strip()
    # Some models still wrap with prose — grab the first {...} block.
    if not s.startswith("{"):
        lo, hi = s.find("{"), s.rfind("}")
        if lo == -1 or hi == -1 or hi <= lo:
            return None, "AI reply contained no JSON object"
        s = s[lo : hi + 1]

    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        return None, f"AI JSON parse failed: {e.msg}"

    try:
        raw_modes = data.get("failure_modes") or []
        modes: list[FailureMode] = []
        for m in raw_modes[:3]:
            modes.append(
                FailureMode(
                    name=str(m.get("name", ""))[:80] or "Unnamed failure",
                    likelihood=_coerce_severity(m.get("likelihood")),
                    danger=_coerce_severity(m.get("danger")),
                    chain=str(m.get("chain", "")).strip()[:600],
                    hidden_assumption=str(m.get("hidden_assumption", "")).strip()[:400],
                    early_warning=str(m.get("early_warning", "")).strip()[:300],
                )
            )
        if not modes:
            return None, "AI reply had no failure modes"

        return {
            "failure_modes": modes,
            "biggest_hidden_assumption": str(
                data.get("biggest_hidden_assumption", "")
            ).strip()[:400] or "Not identified.",
            "revised_plan": [
                str(x).strip()[:300]
                for x in (data.get("revised_plan") or [])
                if str(x).strip()
            ][:4],
            "prelaunch_checklist": [
                str(x).strip()[:200]
                for x in (data.get("prelaunch_checklist") or [])
                if str(x).strip()
            ][:5],
        }, None
    except Exception as e:
        return None, f"AI reply shape unexpected: {type(e).__name__}"


def _coerce_severity(value: Any) -> Severity:
    s = str(value or "").strip().lower()
    if s in _SEV_RANK:
        return s  # type: ignore[return-value]
    # Common aliases the model sometimes returns.
    if s in {"med", "mid", "moderate"}:
        return "medium"
    if s in {"very high", "extreme", "severe"}:
        return "critical"
    return "medium"


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

# Each horizon gets a short, opinionated list of canonical investor failure
# modes. We're not pretending these match the specific plan — they cover the
# ground every trade in that window has to clear. The UI flags it as
# `source="heuristic"` so the user knows they didn't get a tailored premortem.
_HEURISTIC_LIBRARY: dict[Horizon, list[FailureMode]] = {
    "intraday": [
        FailureMode(
            name="Gap through stop on news",
            likelihood="medium",
            danger="critical",
            chain="A scheduled or unscheduled headline drops while you're in the position. "
            "The book gaps past your stop before any resting order fills, "
            "realising a loss multiples of what your stop priced.",
            hidden_assumption="That intraday liquidity stays linear and your stop will fill near its price.",
            early_warning="Watch the economic calendar and unusual options skew the hour before entry.",
        ),
        FailureMode(
            name="Slippage on thin book",
            likelihood="high",
            danger="medium",
            chain="Mid-day liquidity drains, your size is large relative to top-of-book, "
            "and the fill walks the book — eating most of the modelled edge.",
            hidden_assumption="That your size doesn't move the market.",
            early_warning="Top-of-book depth falling below 5× your intended size.",
        ),
        FailureMode(
            name="Vol regime mis-read",
            likelihood="medium",
            danger="high",
            chain="The pattern you traded was calibrated on a low-vol session. Realised "
            "vol expands, your target is no longer special, and stops trigger on noise.",
            hidden_assumption="That today's σ ≈ recent average σ.",
            early_warning="ATR / realised σ rising >50% above the 20-bar median before entry.",
        ),
    ],
    "swing": [
        FailureMode(
            name="Regime flip mid-hold",
            likelihood="medium",
            danger="high",
            chain="The trend or mean-reversion regime that powers the setup flips while "
            "you're holding. The signal that got you in now points the other way.",
            hidden_assumption="That the regime persists for the duration of your hold.",
            early_warning="ADX / regime classifier dropping back into chop territory.",
        ),
        FailureMode(
            name="Correlation break to benchmark",
            likelihood="medium",
            danger="medium",
            chain="The asset decouples from its usual driver (BTC, SPY, dollar) mid-trade, "
            "invalidating the macro side of the thesis even though the chart still looks fine.",
            hidden_assumption="That the historical correlation holds.",
            early_warning="30-day rolling correlation drifting outside its 6-month band.",
        ),
        FailureMode(
            name="Position size too big for the stop",
            likelihood="high",
            danger="high",
            chain="Stop is technically correct but, sized to your account, a normal "
            "drawdown blows past your daily/weekly loss budget. Forced to exit early on a winner.",
            hidden_assumption="That your stop survives normal noise at this size.",
            early_warning="Daily P&L volatility > 2× the size you can emotionally absorb.",
        ),
    ],
    "position": [
        FailureMode(
            name="Thesis decay without exit rule",
            likelihood="medium",
            danger="critical",
            chain="The catalyst that justified the position resolves or fades, but no "
            "explicit invalidation rule exists, so the position drifts into a sunk-cost hold.",
            hidden_assumption="That you'll notice the thesis is dead in real time.",
            early_warning="The original three reasons you bought no longer being mentioned in your own notes.",
        ),
        FailureMode(
            name="Drawdown sequence ruins compounding",
            likelihood="medium",
            danger="high",
            chain="Even with a positive expected return, a streak of normal-sized losses "
            "early in the hold drops the bankroll low enough that the recovery math no longer works.",
            hidden_assumption="That sequence of returns doesn't matter — only the average.",
            early_warning="Drawdown crossing the 1× ATR weekly band you implicitly priced.",
        ),
        FailureMode(
            name="Correlation cluster at the worst time",
            likelihood="medium",
            danger="high",
            chain="Several 'independent' positions all carry the same hidden factor "
            "(USD strength, rates, risk-off). One macro event compounds the loss across the book.",
            hidden_assumption="That your positions are diversified by ticker.",
            early_warning="Sharp same-direction moves across nominally unrelated names.",
        ),
    ],
    "long": [
        FailureMode(
            name="Slow thesis erosion",
            likelihood="high",
            danger="high",
            chain="The qualitative reasons (adoption, regulation, narrative) quietly weaken "
            "over months. No single event triggers an exit, but the edge is gone.",
            hidden_assumption="That the original thesis still describes today's world.",
            early_warning="Quarterly re-read of the thesis no longer convincing you to re-enter from scratch.",
        ),
        FailureMode(
            name="Regulatory / structural shock",
            likelihood="low",
            danger="critical",
            chain="A policy or structural change (rate path, ETF flow halt, exchange "
            "blow-up) repricing the entire asset class overnight.",
            hidden_assumption="That regulatory baseline holds for the hold duration.",
            early_warning="Watchlist of policy / committee dates curated and reviewed weekly.",
        ),
        FailureMode(
            name="Opportunity cost outpaces gains",
            likelihood="medium",
            danger="medium",
            chain="Trade returns 8%/yr while a passive benchmark returns 15%/yr. "
            "You're 'right' but lose relative to doing nothing.",
            hidden_assumption="That this position is the best use of the capital.",
            early_warning="Rolling 90-day return < benchmark for two consecutive quarters.",
        ),
    ],
}


def _heuristic_premortem(*, title: str, plan: str, horizon: Horizon) -> PremortemResult:
    modes = list(_HEURISTIC_LIBRARY[horizon])
    revised = _heuristic_revised_plan(horizon)
    checklist = _heuristic_checklist(horizon)
    biggest = (
        "The plan implicitly assumes the regime, liquidity, and correlation structure "
        "that exist right now will persist for your entire hold. Spot-check each one before sizing."
    )
    return PremortemResult(
        failure_modes=modes,
        biggest_hidden_assumption=biggest,
        revised_plan=revised,
        prelaunch_checklist=checklist,
        horizon=horizon,
        source="heuristic",
        model="",
    )


def _heuristic_revised_plan(horizon: Horizon) -> list[str]:
    common = [
        "Write the invalidation in one line BEFORE entry — what specifically would force an exit.",
        "Halve size if any one of the three failure modes is in 'early-warning' state at entry.",
    ]
    by_horizon: dict[Horizon, list[str]] = {
        "intraday": [
            "Skip if a tier-1 macro print or earnings hits inside your expected hold.",
            "Size so a 2×ATR adverse move = ≤ 0.5% of bankroll.",
        ],
        "swing": [
            "Set a time-stop: if the setup hasn't moved 1×ATR in your favour within N bars, flatten.",
            "Re-check the regime classifier at every daily close.",
        ],
        "position": [
            "Pre-commit to a re-evaluation date — review thesis as if entering fresh.",
            "Cap the cluster: total notional in correlated names ≤ 1.5× a single-name limit.",
        ],
        "long": [
            "Schedule quarterly thesis review with explicit rewrite — if you can't, you've exited.",
            "Benchmark vs. passive every 90 days; if losing for 2 quarters, halve.",
        ],
    }
    return by_horizon[horizon] + common


def _heuristic_checklist(horizon: Horizon) -> list[str]:
    base = [
        "State the invalidation in one sentence. If you can't, don't enter.",
        "Confirm stop distance is ≥ 1×ATR on the operative timeframe.",
        "Confirm size puts max loss ≤ daily / weekly risk budget.",
    ]
    extra: dict[Horizon, list[str]] = {
        "intraday": [
            "Check the next 4h economic calendar — no tier-1 prints inside the hold.",
            "Top-of-book depth ≥ 5× intended size on both sides.",
        ],
        "swing": [
            "Regime classifier (trend/chop) agrees with the trade direction.",
            "Correlation with primary driver (BTC / SPY / DXY) inside its 6-mo band.",
        ],
        "position": [
            "Position does not raise total cluster notional above the cap.",
            "Re-evaluation date is on the calendar with a reminder.",
        ],
        "long": [
            "Thesis written down in <100 words; you'd re-enter from scratch today.",
            "Position size assumes a 30%+ drawdown is survivable.",
        ],
    }
    return (extra[horizon] + base)[:5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_result(horizon: Horizon, *, reason: str) -> PremortemResult:
    return PremortemResult(
        failure_modes=[],
        biggest_hidden_assumption=f"({reason})",
        revised_plan=[],
        prelaunch_checklist=[],
        horizon=horizon,
        source="heuristic",
        ai_error=reason,
    )


def _cache_key(
    title: str,
    plan: str,
    horizon: Horizon,
    context: Optional[dict[str, Any]],
    enable_ai: bool,
) -> str:
    h = hashlib.sha1()
    h.update(title.encode("utf-8"))
    h.update(b"\x00")
    h.update(plan.encode("utf-8"))
    h.update(b"\x00")
    h.update(horizon.encode("utf-8"))
    h.update(b"\x00")
    h.update(json.dumps(context or {}, sort_keys=True, default=str).encode("utf-8"))
    h.update(b"\x00")
    h.update(b"ai" if enable_ai else b"no-ai")
    return h.hexdigest()


def _read_cache(key: str) -> Optional[PremortemResult]:
    entry = _CACHE.get(key)
    if not entry:
        return None
    ts, result = entry
    if (time.time() - ts) > _CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return result


__all__ = [
    "premortem",
    "clear_cache",
    "PremortemResult",
    "FailureMode",
    "Horizon",
    "HORIZON_LABELS",
]
