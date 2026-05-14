"""Confirmation gate for the event-aware chat widget.

Combines five independent axes and emits a single GateVerdict:

  1. NEWS / SENTIMENT     — Perplexity brief alignment with idea direction.
  2. REGIME               — HMM bull/bear probability, Hurst label,
                             Wyckoff phase compatibility.
  3. MACRO                — Dalio quadrant equity/crypto bias.
  4. MICROSTRUCTURE       — VWAP extension, CVD divergence, imbalance, RV regime.
  5. CATALYSTS            — Tier-1 calendar events / token unlocks inside the
                             hold window are HARD blockers (verdict = STAND_DOWN).

Each axis casts a vote ∈ {-1, 0, +1} with a 0..1 strength. The vote is
multiplied by the configured weight from `settings.gate_weights`. The
weighted sum is passed through a calibrated logistic to produce P(hit) —
"the chance the trade reaches its stated target before its stop, given
this multi-axis context".

Calibration: v1 uses a fixed-shape logistic centred at the user's
configured `confluence_min`. v2 should refit on `~/.monte/alerts.jsonl`
hit-rate to make P(hit) actually predictive — this v1 is honest enough
to gate but not honest enough to bet on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from monte.config import settings
from monte.intel.event_aggregator import EventBundle, IdeaContext
from monte.microstructure import MicrostructureReport
from monte.regime import RegimeReport


@dataclass
class AxisVote:
    name: str
    direction: int                 # -1 (bear), 0 (neutral), +1 (bull)
    strength: float                # 0..1 — how confident this axis is
    detail: str                    # plain-English one-liner


@dataclass
class GateVerdict:
    action: str                    # "GO" | "CAUTION" | "STAND_DOWN"
    p_hit: float                   # 0..1
    confluence_count: int          # axes voting in idea's direction
    axes: list[AxisVote] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    score: float = 0.0             # signed weighted sum (idea-aligned positive)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "p_hit": self.p_hit,
            "confluence_count": self.confluence_count,
            "blockers": list(self.blockers),
            "score": self.score,
            "note": self.note,
            "axes": [
                {"name": a.name, "direction": a.direction, "strength": a.strength, "detail": a.detail}
                for a in self.axes
            ],
        }


def score(
    *,
    idea: IdeaContext,
    bundle: EventBundle,
    regime: Optional[RegimeReport] = None,
    microstructure: Optional[MicrostructureReport] = None,
    premortem_critical_acknowledged: bool = True,
) -> GateVerdict:
    """Compute the gate verdict for `idea` given all collected evidence."""
    direction_sign = +1 if idea.direction == "long" else -1
    weights = {
        "news": 0.5,
        "regime": float(settings.gate_weights.get("regime", 0.7)),
        "macro": float(settings.gate_weights.get("macro", 0.5)),
        "microstructure": float(settings.gate_weights.get("microstructure", 0.4)),
        "catalysts": 0.6,
    }

    axes: list[AxisVote] = [
        _news_axis(bundle),
        _regime_axis(regime, idea),
        _macro_axis(regime, idea),
        _microstructure_axis(microstructure, idea),
        _catalysts_axis(bundle, idea),
    ]

    weighted_sum = 0.0
    weight_sum = 0.0
    confluence = 0
    for a in axes:
        w = weights.get(a.name.lower(), 0.5)
        contribution = a.direction * a.strength * w
        # Vote in idea's direction means contribution * direction_sign > 0.
        if a.direction != 0 and a.direction == direction_sign:
            confluence += 1
        weighted_sum += contribution * direction_sign
        weight_sum += abs(a.strength * w) if a.direction != 0 else 0.0

    blockers: list[str] = []

    tier_1 = bundle.tier_1_in_window()
    if tier_1 and idea.horizon_hours <= 168:
        names = ", ".join(e.name for e in tier_1[:3])
        blockers.append(f"Tier-1 catalyst inside hold window: {names}")

    unlock_h = bundle.unlock_in_window()
    if unlock_h is not None and idea.direction == "long" and idea.is_crypto:
        blockers.append(
            f"Token unlock in {unlock_h:.0f}h ({bundle.onchain.next_unlock_label}, "
            f"{(bundle.onchain.next_unlock_pct_supply or 0):.2f}% supply)"
        )

    if not premortem_critical_acknowledged:
        blockers.append("Critical premortem failure mode not acknowledged")

    confluence_min = int(getattr(settings, "confluence_min", 3))
    p_hit = _logistic(weighted_sum, weight_sum, confluence, confluence_min)

    if blockers:
        action = "STAND_DOWN"
    elif confluence >= confluence_min and weighted_sum > 0 and p_hit >= 0.55:
        action = "GO"
    elif weighted_sum < 0 and confluence < confluence_min:
        action = "STAND_DOWN"
    else:
        action = "CAUTION"

    note = _verdict_note(action, idea, confluence, confluence_min, blockers)

    return GateVerdict(
        action=action,
        p_hit=p_hit,
        confluence_count=confluence,
        axes=axes,
        blockers=blockers,
        score=weighted_sum,
        note=note,
    )


# ---------------------------------------------------------------------------
# Per-axis voters
# ---------------------------------------------------------------------------

def _news_axis(bundle: EventBundle) -> AxisVote:
    n = bundle.news
    if n is None or not n.configured:
        return AxisVote("news", 0, 0.0, "News not configured.")
    if n.error:
        return AxisVote("news", 0, 0.0, f"News error: {n.error[:80]}")
    if n.sentiment == "bullish":
        return AxisVote("news", +1, 0.7, n.summary[:160] or "Bullish news flow.")
    if n.sentiment == "bearish":
        return AxisVote("news", -1, 0.7, n.summary[:160] or "Bearish news flow.")
    return AxisVote("news", 0, 0.3, n.summary[:160] or "Neutral news.")


def _regime_axis(regime: Optional[RegimeReport], idea: IdeaContext) -> AxisVote:
    if regime is None:
        return AxisVote("regime", 0, 0.0, "Regime data unavailable.")
    bias = regime.directional_bias()
    if bias == "bull":
        direction, strength = +1, 0.7
    elif bias == "bear":
        direction, strength = -1, 0.7
    else:
        direction, strength = 0, 0.3
    parts: list[str] = []
    if regime.hmm:
        parts.append(f"HMM {regime.hmm.label} ({regime.hmm.bull_prob:.0%} bull)")
    if regime.hurst is not None:
        parts.append(regime.hurst_label)
    if regime.wyckoff:
        parts.append(f"Wyckoff: {regime.wyckoff.phase}")
    detail = "; ".join(parts) or "regime: indeterminate"
    return AxisVote("regime", direction, strength, detail[:200])


def _macro_axis(regime: Optional[RegimeReport], idea: IdeaContext) -> AxisVote:
    if regime is None or regime.macro is None:
        return AxisVote("macro", 0, 0.0, "Macro quadrant unavailable.")
    bias_field = "crypto_bias" if idea.is_crypto else "equity_bias"
    bias = getattr(regime.macro, bias_field, "neutral")
    direction = {"bull": +1, "bear": -1}.get(bias, 0)
    strength = 0.6 if direction != 0 else 0.2
    detail = f"{regime.macro.quadrant} (G{regime.macro.growth_label} / I{regime.macro.inflation_label}). " \
             f"{regime.macro.note}"
    return AxisVote("macro", direction, strength, detail[:240])


def _microstructure_axis(report: Optional[MicrostructureReport], idea: IdeaContext) -> AxisVote:
    if report is None:
        return AxisVote("microstructure", 0, 0.0, "Microstructure unavailable.")
    direction = 0
    strength = 0.3
    parts: list[str] = []

    if report.vwap_band_sigma is not None:
        if abs(report.vwap_band_sigma) >= 2.0:
            # Extension penalty: pushes against idea direction.
            extension_dir = -1 if report.vwap_band_sigma > 0 else +1
            direction += extension_dir
            strength = max(strength, 0.5)
            parts.append(f"{report.vwap_band_sigma:+.1f}σ from VWAP — extended")
        else:
            parts.append(report.vwap_relation())

    if report.cvd_divergence != 0:
        direction += report.cvd_divergence  # +1 bullish, -1 bearish
        strength = max(strength, 0.55)
        sign = "bullish" if report.cvd_divergence > 0 else "bearish"
        parts.append(f"CVD divergence {sign}")

    if report.imbalance_score is not None and abs(report.imbalance_score) >= 0.5:
        sign = +1 if report.imbalance_score > 0 else -1
        direction += sign
        strength = max(strength, 0.45)
        parts.append(f"order-flow imbalance {report.imbalance_score:+.2f}")

    if report.rv_zscore is not None and report.rv_zscore >= 1.5:
        # Hot vol regime is a small penalty against any directional bet.
        direction -= 1 if direction == 0 else 0
        strength = max(strength, 0.4)
        parts.append(f"realized-vol z={report.rv_zscore:+.1f} (hot)")

    direction = max(-1, min(1, direction))
    return AxisVote("microstructure", direction, strength, "; ".join(parts) or "no microstructure edge")


def _catalysts_axis(bundle: EventBundle, idea: IdeaContext) -> AxisVote:
    parts: list[str] = []
    direction = 0
    strength = 0.3

    tier1 = bundle.tier_1_in_window()
    if tier1:
        names = ", ".join(e.name for e in tier1[:3])
        # Tier-1 inside a long-bias hold counts as a vote against entering.
        direction = -1
        strength = 0.8
        parts.append(f"tier-1 catalyst in window: {names}")

    if bundle.onchain and bundle.onchain.etf_net_flow_5d_musd is not None:
        flow = bundle.onchain.etf_net_flow_5d_musd
        if abs(flow) >= 200:
            sign = +1 if flow > 0 else -1
            direction = direction or sign
            strength = max(strength, 0.5)
            parts.append(f"ETF 5d net ${flow:+,.0f}M")

    if bundle.onchain and bundle.onchain.funding_rate_z_30d is not None:
        z = bundle.onchain.funding_rate_z_30d
        if abs(z) >= 1.5:
            # Extreme positive funding = crowded longs → mild bearish vote.
            sign = -1 if z > 0 else +1
            direction = direction or sign
            strength = max(strength, 0.45)
            parts.append(f"funding z={z:+.1f} (crowded {'longs' if z > 0 else 'shorts'})")

    if not parts:
        return AxisVote("catalysts", 0, 0.2, "no notable catalyst pressure")
    return AxisVote("catalysts", direction, strength, "; ".join(parts)[:240])


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _logistic(weighted_sum: float, weight_sum: float, confluence: int, confluence_min: int) -> float:
    """Map (signed weighted sum, confluence) to a P(hit) ∈ [0.05, 0.95].

    Centred so weighted_sum=0 → 0.5, +1 → ~0.73, -1 → ~0.27. Confluence
    above the user's threshold lifts the floor a touch; below it caps the
    ceiling so a single dominant axis can't pretend to be conviction.
    """
    base = 1.0 / (1.0 + math.exp(-1.4 * weighted_sum))
    if confluence >= confluence_min:
        floor = 0.45
        ceiling = 0.92
    else:
        floor = 0.08
        ceiling = 0.65
    return float(min(ceiling, max(floor, base)))


def _verdict_note(
    action: str,
    idea: IdeaContext,
    confluence: int,
    confluence_min: int,
    blockers: list[str],
) -> str:
    if blockers:
        return "Blocked: " + "; ".join(blockers[:2])
    direction = idea.direction.upper()
    if action == "GO":
        return (
            f"{confluence} of 5 axes vote with the {direction} thesis "
            f"(threshold {confluence_min}). Risk-defined entry warranted."
        )
    if action == "STAND_DOWN":
        return (
            f"Only {confluence} axes vote with the {direction} thesis; weight "
            f"of evidence skews against. Skip or wait."
        )
    return (
        f"Mixed picture: {confluence}/5 axes agree (need {confluence_min}). "
        f"Watch for one more confirmation before sizing."
    )


__all__ = ["AxisVote", "GateVerdict", "score"]
