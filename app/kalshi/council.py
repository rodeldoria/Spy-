"""Decision Council — minimum-viable gate in front of Kalshi order buttons.

The page already places a buy/sell recommendation on each market via
`score_market`. The Council is a second pass on top of that recommendation
that asks: "given everything we know about this Decision, is it actually
safe to surface a one-click order link?"

It does NOT recompute probabilities or run any new analysis. It re-reads
the data already on a `Decision` and checks five independent gates:

  1. EDGE        — model_prob − implied_prob ≥ user's min_edge threshold
  2. EV          — expected return per $1 staked ≥ user's min_ev threshold
  3. KELLY       — Kelly fraction ≥ 1% (below that, model can't tell sides
                   apart and the bet is statistical noise)
  4. CONVICTION  — confidence_pct ≥ 50%
  5. LIQUIDITY   — no liquidity warnings (wide spread, zero volume, last-
                   minute close, ask outside the 5-95¢ band)

A button is "armed" only when ≥ 4 of 5 gates pass. The 5/5 case is shown
as a strong green pill; 4/5 yellow ("acceptable, one caveat"); below that
the button is suppressed entirely.

LIQUIDITY band rationale: an actionable ask outside [5¢, 95¢] is almost
always a "1¢ free money" mirage — either zero-volume dust on a stale book
or a tail strike no real money is sitting in. The decision engine's GBM
model defaults to 50% on unrecognised market shapes, so a 1¢ ask combined
with that default produces a phantom +49pp edge. The liquidity gate is
what catches those.

This module deliberately reuses *only* fields already on the Decision —
no extra HTTP calls, no model re-runs. The Claude second-opinion (separate
module `ai_opinion.py`) is layered on top of this gate, not inside it.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.kalshi.decisions import Decision, SideAssessment


# The "armed" threshold — how many of the 5 gates must pass before we
# surface a one-click order button. 4/5 lets one yellow caveat through;
# 5/5 is the "all green" badge.
ARMED_THRESHOLD = 4

# Liquidity band for the recommended side's ask. Outside this range, the
# implied probability is too compressed against the wall to be reliable —
# 1¢ asks are stale-book artefacts, 99¢ asks are tail strikes with no flow.
LIQUIDITY_BAND_MIN_CENTS = 5
LIQUIDITY_BAND_MAX_CENTS = 95

# Below 1% Kelly the bet is below the model's own noise floor.
MIN_KELLY = 0.01

# Below this conviction the edge could just be model error.
MIN_CONVICTION_PCT = 50.0


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class CouncilResult:
    """Five-gate verdict on a single Decision.

    `armed` is the boolean the UI checks before showing the order button.
    `passed` / `total` are surfaced as an "N/5" pill so the user can see
    which gates failed and why.
    """

    decision_ticker: str
    side: str                       # "YES" or "NO" — the side being assessed
    checks: list[CheckResult]
    armed: bool

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def score_label(self) -> str:
        return f"{self.passed}/{self.total}"

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


def evaluate(
    decision: Decision,
    *,
    min_edge: float,
    min_ev: float,
) -> CouncilResult | None:
    """Run the 5-gate council on `decision`.

    Returns None for PASS-direction decisions — there's no side to arm a
    button for. For YES / NO directions, returns a CouncilResult with one
    CheckResult per gate.

    `min_edge` and `min_ev` should match the user's sidebar thresholds so
    the council label stays consistent with the rest of the page.
    """
    chosen = decision.chosen
    if chosen is None or decision.direction == "PASS":
        return None

    checks = [
        _check_edge(chosen, min_edge),
        _check_ev(chosen, min_ev),
        _check_kelly(chosen),
        _check_conviction(decision),
        _check_liquidity(decision, chosen),
    ]
    armed = sum(1 for c in checks if c.passed) >= ARMED_THRESHOLD
    return CouncilResult(
        decision_ticker=decision.market_ticker,
        side=decision.direction,
        checks=checks,
        armed=armed,
    )


def _check_edge(side: SideAssessment, min_edge: float) -> CheckResult:
    passed = side.edge >= min_edge
    detail = (
        f"edge {side.edge * 100:+.1f}pp "
        f"({'≥' if passed else '<'} {min_edge * 100:.0f}pp threshold)"
    )
    return CheckResult(name="EDGE", passed=passed, detail=detail)


def _check_ev(side: SideAssessment, min_ev: float) -> CheckResult:
    passed = side.ev_per_dollar >= min_ev
    detail = (
        f"EV {side.ev_per_dollar * 100:+.1f}¢/$1 "
        f"({'≥' if passed else '<'} {min_ev * 100:.0f}¢ threshold)"
    )
    return CheckResult(name="EV", passed=passed, detail=detail)


def _check_kelly(side: SideAssessment) -> CheckResult:
    passed = side.kelly_fraction >= MIN_KELLY
    detail = (
        f"Kelly {side.kelly_fraction * 100:.1f}% "
        f"({'≥' if passed else '<'} {MIN_KELLY * 100:.0f}% noise floor)"
    )
    return CheckResult(name="KELLY", passed=passed, detail=detail)


def _check_conviction(decision: Decision) -> CheckResult:
    passed = decision.confidence_pct >= MIN_CONVICTION_PCT
    detail = (
        f"conviction {decision.confidence_pct:.0f}% "
        f"({'≥' if passed else '<'} {MIN_CONVICTION_PCT:.0f}%)"
    )
    return CheckResult(name="CONVICTION", passed=passed, detail=detail)


def _check_liquidity(decision: Decision, chosen: SideAssessment) -> CheckResult:
    """Liquidity band + warnings check.

    Fails if (a) the recommended side's ask sits outside the 5–95¢ band
    (stale-book / tail-strike artefact), or (b) `score_market` already
    appended a liquidity-flavoured warning (wide spread / zero volume /
    sub-minute close / unrecognised shape).
    """
    ask = chosen.ask_cents
    in_band = LIQUIDITY_BAND_MIN_CENTS <= ask <= LIQUIDITY_BAND_MAX_CENTS
    blocking_warning_keywords = ("wide spread", "zero volume", "less than a minute", "unrecognised")
    has_warning = any(
        any(kw in w.lower() for kw in blocking_warning_keywords)
        for w in decision.warnings
    )
    passed = in_band and not has_warning

    if not in_band:
        reason = f"ask {ask}¢ outside {LIQUIDITY_BAND_MIN_CENTS}–{LIQUIDITY_BAND_MAX_CENTS}¢ liquidity band"
    elif has_warning:
        reason = "decision flagged a liquidity/shape warning"
    else:
        reason = f"ask {ask}¢ inside liquidity band, no warnings"
    return CheckResult(name="LIQUIDITY", passed=passed, detail=reason)
