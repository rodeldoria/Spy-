"""AI-powered decision council — the "should I pull the trigger?" engine.

We score every Kalshi opportunity through eight battle-tested decision
frameworks, then optionally ask Claude to sanity-check the call. The
output is a single 0–100 trigger score, a green/yellow/red verdict,
and a 3-step playbook ("size at X%, enter now, exit at Y").

Frameworks scored (each up to 12.5 points = 100 max):

  1. EDGE        — Thorp/Kelly: model_prob - book_prob ≥ 4pp.
  2. KELLY       — Kelly criterion suggests sizing ≥ 2% of bankroll.
  3. EV          — Expected value ≥ 2¢ per $1 (covers fees + slippage).
  4. CONVICTION  — Model confidence ≥ 65% in chosen direction.
  5. LIQUIDITY   — Book deep enough that we can actually fill at the ask.
  6. CALIBRATION — Recent kalshi_calibration shift small (model has been
                   accurate within ±5pp lately) — Tetlock's superforecaster
                   "track-record" principle.
  7. TRIANGULATION — ≥ 3 independent signals agree (Druckenmiller's
                     "multiple confirmation" rule). Pulled from the
                     triangulation engine for crypto markets.
  8. PRE-MORTEM  — No active warnings on the decision (no "unrecognised
                   market shape", no "thin book", no "stale spot").
                   Inverted Kahneman pre-mortem: count and penalise risk
                   flags.

The mechanical score is independent of any LLM. If `enable_ai=True` and
the Anthropic key is present, we ALSO ask Claude for a 0–100 trigger
score and a one-paragraph "would you take this trade?" explanation,
then average it 50/50 with the mechanical score for the final verdict.

Verdict bands:
  ≥ 80 → 🟢  PULL THE TRIGGER
  ≥ 60 → 🟡  ALMOST THERE — wait for one more confirmation
  ≥ 40 → 🟠  WATCH ONLY
  <  40 → 🔴  STAND DOWN
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Checkpoint:
    name: str          # short label for the UI
    framework: str     # well-known framework / source
    passed: bool
    score: float       # 0..12.5
    detail: str        # plain-English explanation


@dataclass
class CouncilVerdict:
    trigger_score: float          # 0..100 final (mechanical or blended, post-learning)
    mechanical_score: float       # 0..100 from the 8 checkpoints
    ai_score: Optional[float]     # 0..100 from Claude, or None
    verdict_emoji: str            # 🟢 / 🟡 / 🟠 / 🔴
    verdict_label: str            # "PULL THE TRIGGER" / etc.
    headline: str                 # one-line action ("BUY YES @ 32¢, size 4% bankroll, hold to close")
    checkpoints: list[Checkpoint] = field(default_factory=list)
    ai_summary: Optional[str] = None
    ai_error: Optional[str] = None
    playbook: list[str] = field(default_factory=list)   # 3 steps
    # Pattern-tracker learning
    signature: str = ""           # e.g. '11010110' — which checkpoints passed
    learning_multiplier: float = 1.0   # signature's historical confidence multiplier
    learning_label: str = ""      # e.g. "learned (12 settled, 75% hit, +18.4% ROI)"
    pre_learning_score: float = 0.0   # raw score before applying multiplier
    # Plain-English resolution condition — what the chosen side actually pays
    # out on. Surface on the UI so the user can sanity-check the ticket
    # label (YES/NO) against the underlying directional bet (above/below).
    resolution_summary: str = ""      # "BTC closes above $80,000 at 12:00 UTC"
    resolution_relation: str = ""     # "above" / "below" / "between" / "outside" / ""
    resolution_threshold: str = ""    # "$80,000" / "$79,500–$79,750"
    resolution_symbol: str = ""       # "BTC" / "ETH" / "SOL"

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checkpoints if c.passed)


# ---------------------------------------------------------------------------
# Mechanical scorecard
# ---------------------------------------------------------------------------

def _check_edge(edge: float) -> Checkpoint:
    edge_pp = edge * 100
    passed = edge_pp >= 4.0
    score = max(0.0, min(12.5, edge_pp / 8.0 * 12.5))   # full credit at +8pp
    return Checkpoint(
        "Edge", "Thorp · model vs book",
        passed, score,
        f"{edge_pp:+.1f}pp model-vs-book edge "
        f"({'PASS' if passed else 'fail'} threshold +4.0pp).",
    )


def _check_kelly(kf: float) -> Checkpoint:
    kpct = kf * 100
    passed = kpct >= 2.0
    score = max(0.0, min(12.5, kpct / 10.0 * 12.5))
    return Checkpoint(
        "Kelly sizing", "Kelly criterion",
        passed, score,
        f"Kelly suggests {kpct:.1f}% bankroll sizing "
        f"({'PASS' if passed else 'fail'} threshold 2.0%).",
    )


def _check_ev(ev_per_dollar: float) -> Checkpoint:
    ev_c = ev_per_dollar * 100
    passed = ev_c >= 2.0
    score = max(0.0, min(12.5, ev_c / 12.0 * 12.5))
    return Checkpoint(
        "Expected value", "EV per $1",
        passed, score,
        f"EV {ev_c:+.1f}¢ per $1 staked "
        f"({'PASS' if passed else 'fail'} threshold +2.0¢).",
    )


def _check_conviction(conf_pct: float) -> Checkpoint:
    passed = conf_pct >= 65.0
    score = max(0.0, min(12.5, conf_pct / 100.0 * 12.5))
    return Checkpoint(
        "Conviction", "Model confidence",
        passed, score,
        f"Model {conf_pct:.0f}% confident in chosen direction "
        f"({'PASS' if passed else 'fail'} threshold 65%).",
    )


def _check_liquidity(payout: float, ask_cents: int) -> Checkpoint:
    """Without an explicit depth field, use ask price as a proxy:
    very-low-priced contracts (≤3¢) tend to be illiquid, mid-range is fine."""
    passed = 5 <= ask_cents <= 95
    score = 12.5 if passed else max(0.0, 12.5 - abs(ask_cents - 50) / 8.0)
    return Checkpoint(
        "Liquidity", "Tradeable price band",
        passed, score,
        f"Ask {ask_cents}¢ is "
        f"{'inside' if passed else 'outside'} the 5¢–95¢ tradeable band.",
    )


def _check_calibration(cal_shift_pp: Optional[float]) -> Checkpoint:
    """How much did the calibrator nudge the raw model probability? Small
    shifts = the model has been accurate lately; big shifts = it's been
    wrong and the calibrator is correcting it."""
    if cal_shift_pp is None:
        return Checkpoint(
            "Calibration", "Tetlock track record",
            False, 4.0,
            "No calibration history yet — no track record to lean on.",
        )
    abs_shift = abs(cal_shift_pp)
    passed = abs_shift <= 5.0
    score = max(0.0, 12.5 - abs_shift)
    return Checkpoint(
        "Calibration", "Tetlock track record",
        passed, score,
        f"Calibrator shift {cal_shift_pp:+.1f}pp "
        f"({'PASS' if passed else 'fail'} threshold ±5.0pp). "
        f"{'Model has been accurate.' if passed else 'Model has been off lately — discount its claim.'}",
    )


def _check_triangulation(tri_n_agree: Optional[int],
                         tri_n_total: Optional[int],
                         tri_direction_match: Optional[bool]) -> Checkpoint:
    """Druckenmiller: don't act on one indicator. Count independent agree-ers."""
    if tri_n_agree is None or tri_n_total is None:
        return Checkpoint(
            "Triangulation", "Druckenmiller multi-confirm",
            False, 4.0,
            "Triangulation engine couldn't run for this market type.",
        )
    if tri_direction_match is False:
        return Checkpoint(
            "Triangulation", "Druckenmiller multi-confirm",
            False, 0.0,
            f"External signals DISAGREE with the trade direction "
            f"({tri_n_agree}/{tri_n_total} agreeing).",
        )
    passed = tri_n_agree >= 3
    score = max(0.0, min(12.5, tri_n_agree / 4.0 * 12.5))
    return Checkpoint(
        "Triangulation", "Druckenmiller multi-confirm",
        passed, score,
        f"{tri_n_agree}/{tri_n_total} independent signals agree "
        f"({'PASS' if passed else 'fail'} threshold ≥3).",
    )


def _check_pre_mortem(warnings: list[str]) -> Checkpoint:
    """Kahneman pre-mortem: imagine this trade has failed — what red flags
    were already visible? Each existing warning costs us points."""
    n = len(warnings or [])
    passed = n == 0
    score = max(0.0, 12.5 - n * 4.0)
    detail = (
        "No risk flags raised — clean setup."
        if passed
        else f"{n} risk flag(s): " + "; ".join((warnings or [])[:3])
    )
    return Checkpoint(
        "Pre-mortem", "Kahneman risk audit",
        passed, score, detail,
    )


# ---------------------------------------------------------------------------
# Optional AI second opinion
# ---------------------------------------------------------------------------

def _ai_verdict(payload: dict[str, Any]) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """Ask Claude for a 0-100 trigger score + a 2-sentence verdict.
    Returns (score, summary, error). All None if the SDK is unavailable
    or the API key isn't set."""
    if not (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY")
    ):
        return None, None, "ANTHROPIC_API_KEY not set."
    try:
        import anthropic
    except ImportError:
        return None, None, "anthropic SDK not installed."

    prompt = (
        "You are a hedge-fund risk officer reviewing a single Kalshi "
        "prediction-market trade. Look at the data below and decide: "
        "would you pull the trigger? Be sceptical of weak edges, thin "
        "books, and conflicting signals.\n\n"
        f"TRADE DATA:\n{json.dumps(payload, indent=2)}\n\n"
        "Respond with EXACTLY this JSON (no prose outside it):\n"
        '{\n  "trigger_score": <0-100 int>,\n'
        '  "verdict": "<PULL_TRIGGER | ALMOST | WATCH | STAND_DOWN>",\n'
        '  "summary": "<≤2 sentences explaining your call in plain English>"\n}'
    )

    try:
        api_key = (
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY")
        )
        base_url = (
            os.environ.get("ANTHROPIC_BASE_URL")
            or os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL")
        )
        client = (
            anthropic.Anthropic(api_key=api_key, base_url=base_url)
            if base_url
            else anthropic.Anthropic(api_key=api_key)
        )
        msg = client.messages.create(
            model="claude-haiku-4-5",   # cheap + fast for per-trade calls
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
        # Strip optional code-fence
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
        score = float(data.get("trigger_score", 0))
        summary = str(data.get("summary", ""))[:400]
        return max(0.0, min(100.0, score)), summary, None
    except Exception as e:
        return None, None, f"AI call failed: {str(e)[:120]}"


# ---------------------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------------------

# Process-level cache so we don't burn Anthropic quota when Streamlit
# re-renders the page on autorefresh.
_AI_CACHE: dict[str, tuple[float, tuple]] = {}
_AI_TTL = 600.0   # 10 minutes


def evaluate(
    *,
    direction: str,                 # "YES" / "NO" / "PASS"
    edge: float,                    # signed, decimal (0.05 = 5pp)
    ev_per_dollar: float,           # decimal (0.04 = 4¢)
    kelly_fraction: float,          # 0..1
    confidence_pct: float,          # 0..100
    payout: float,                  # multiplier
    ask_cents: int,
    warnings: Optional[list[str]] = None,
    cal_shift_pp: Optional[float] = None,
    tri_n_agree: Optional[int] = None,
    tri_n_total: Optional[int] = None,
    tri_direction_match: Optional[bool] = None,
    bet_summary: str = "",
    market_ticker: str = "",
    enable_ai: bool = False,
    bankroll: float = 1000.0,
    resolution_summary: str = "",
    resolution_relation: str = "",
    resolution_threshold: str = "",
    resolution_symbol: str = "",
) -> CouncilVerdict:
    """Run the council scorecard for a single Kalshi opportunity."""

    # If the underlying engine itself said PASS, short-circuit to STAND_DOWN
    # rather than scoring an invalid play.
    if direction == "PASS":
        return CouncilVerdict(
            trigger_score=0.0,
            mechanical_score=0.0,
            ai_score=None,
            verdict_emoji="🔴",
            verdict_label="STAND DOWN",
            headline="No edge — model agrees with the book. Don't trade.",
            checkpoints=[],
            playbook=[
                "Skip this market.",
                "Wait for the spread to widen or the model to disagree.",
                "Conserve bankroll for high-edge setups.",
            ],
            resolution_summary=resolution_summary,
            resolution_relation=resolution_relation,
            resolution_threshold=resolution_threshold,
            resolution_symbol=resolution_symbol,
        )

    checks = [
        _check_edge(edge),
        _check_kelly(kelly_fraction),
        _check_ev(ev_per_dollar),
        _check_conviction(confidence_pct),
        _check_liquidity(payout, ask_cents),
        _check_calibration(cal_shift_pp),
        _check_triangulation(tri_n_agree, tri_n_total, tri_direction_match),
        _check_pre_mortem(warnings or []),
    ]
    mech = sum(c.score for c in checks)   # 0..100

    ai_score: Optional[float] = None
    ai_summary: Optional[str] = None
    ai_error: Optional[str] = None
    if enable_ai and mech >= 40.0:        # don't waste AI on obvious losers
        # Cache key includes everything that meaningfully changes the AI's
        # answer, so a stale verdict isn't reused after the trade context
        # shifts (warnings appear, edge drifts, triangulation flips, etc).
        cache_key = (
            f"{market_ticker}:{direction}:{ask_cents}:"
            f"e{round(edge, 3)}:k{round(kelly_fraction, 3)}:"
            f"v{round(ev_per_dollar, 3)}:c{round(confidence_pct, 0)}:"
            f"cal{round(cal_shift_pp, 1) if cal_shift_pp is not None else 'NA'}:"
            f"tri{tri_n_agree}/{tri_n_total}:"
            f"w{len(warnings or [])}"
        )
        cached = _AI_CACHE.get(cache_key)
        if cached and (time.time() - cached[0]) < _AI_TTL:
            ai_score, ai_summary, ai_error = cached[1]
        else:
            payload = {
                "market": bet_summary or market_ticker,
                "side": direction,
                "ask_cents": ask_cents,
                "payout_multiplier": round(payout, 2),
                "edge_pp": round(edge * 100, 1),
                "ev_cents_per_dollar": round(ev_per_dollar * 100, 1),
                "kelly_pct": round(kelly_fraction * 100, 1),
                "confidence_pct": round(confidence_pct, 0),
                "calibration_shift_pp": round(cal_shift_pp, 1) if cal_shift_pp is not None else None,
                "triangulation_agree": tri_n_agree,
                "triangulation_total": tri_n_total,
                "warnings": warnings or [],
            }
            ai_score, ai_summary, ai_error = _ai_verdict(payload)
            _AI_CACHE[cache_key] = (time.time(), (ai_score, ai_summary, ai_error))

    if ai_score is not None:
        raw_final = (mech + ai_score) / 2.0
    else:
        raw_final = mech

    # ----- Pattern-tracker learning bump ---------------------------------
    # Look up this checkpoint signature's historical hit rate and apply a
    # confidence multiplier (0.6 .. 1.4). After ≥5 settled outcomes we
    # actually move the score; before that the multiplier is 1.0.
    signature = ""
    learning_mult = 1.0
    learning_label = "raw (no history yet)"
    try:
        from monte.learning import pattern_tracker as ptrack
        signature = ptrack.signature_from_checkpoints(checks)
        learning_mult, learning_label = ptrack.confidence_multiplier(signature)
    except Exception:
        pass
    pre_learning = raw_final
    final = max(0.0, min(100.0, raw_final * learning_mult))

    if final >= 80:
        emoji, label = "🟢", "PULL THE TRIGGER"
    elif final >= 60:
        emoji, label = "🟡", "ALMOST THERE"
    elif final >= 40:
        emoji, label = "🟠", "WATCH ONLY"
    else:
        emoji, label = "🔴", "STAND DOWN"

    # Compact directional clause so every line that mentions "BUY YES/NO"
    # also reminds the user *what* that ticket actually resolves on.
    # Examples:
    #   "BUY YES (BTC above $80,000)"
    #   "BUY NO (ETH below $2,250)"
    # Falls back to "BUY YES" when the market shape didn't parse.
    if resolution_relation in ("above", "below") and resolution_threshold:
        sym = resolution_symbol or "price"
        dir_clause = f"BUY {direction} ({sym} {resolution_relation} {resolution_threshold})"
    elif resolution_relation in ("between", "outside") and resolution_threshold:
        sym = resolution_symbol or "price"
        dir_clause = f"BUY {direction} ({sym} {resolution_relation} {resolution_threshold})"
    else:
        dir_clause = f"BUY {direction}"

    # 3-step playbook
    size_dollars = max(0.0, kelly_fraction * bankroll)
    if final >= 80:
        playbook = [
            f"1️⃣  Size: \\${size_dollars:.0f} (Kelly {kelly_fraction*100:.1f}% of \\${bankroll:.0f} bankroll).",
            f"2️⃣  Enter: {dir_clause} at {ask_cents}¢ now — wait no longer than 5 min.",
            f"3️⃣  Hold to close. Don't add if it moves against you; close out only if a new warning fires.",
        ]
    elif final >= 60:
        playbook = [
            f"1️⃣  Half-size: \\${size_dollars * 0.5:.0f} (half-Kelly because confidence is mid).",
            "2️⃣  Wait 1 refresh cycle for one more signal to confirm before entering.",
            f"3️⃣  If confirmation arrives, {dir_clause} at ≤ {ask_cents + 1}¢ and hold to close.",
        ]
    elif final >= 40:
        playbook = [
            "1️⃣  Don't bet yet — put it on watchlist.",
            "2️⃣  Re-check next refresh; act only if score crosses 60 with no new warnings.",
            "3️⃣  If close window drops under 30 min and score hasn't improved, drop it.",
        ]
    else:
        playbook = [
            "1️⃣  Skip — too many cautions.",
            "2️⃣  Don't revisit this market for at least 1 hour.",
            "3️⃣  Look for setups with edge ≥ 4pp AND ≥ 3 agreeing signals.",
        ]

    headline = (
        f"{emoji} {label} — {dir_clause} @ {ask_cents}¢ "
        f"(score {final:.0f}/100"
        + (f", AI {ai_score:.0f}" if ai_score is not None else "")
        + ")"
    )

    return CouncilVerdict(
        trigger_score=final,
        mechanical_score=mech,
        ai_score=ai_score,
        verdict_emoji=emoji,
        verdict_label=label,
        headline=headline,
        checkpoints=checks,
        ai_summary=ai_summary,
        ai_error=ai_error,
        playbook=playbook,
        signature=signature,
        learning_multiplier=learning_mult,
        learning_label=learning_label,
        pre_learning_score=pre_learning,
        resolution_summary=resolution_summary,
        resolution_relation=resolution_relation,
        resolution_threshold=resolution_threshold,
        resolution_symbol=resolution_symbol,
    )


__all__ = ["evaluate", "CouncilVerdict", "Checkpoint"]
