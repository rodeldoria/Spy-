"""Map a signal + timeframe + regime onto a trade horizon.

The watchlist surfaces a clear answer to "if I take this trade, am I scalping
it the same session, swinging it for a few days, or holding for weeks?"
The classifier is intentionally simple — three buckets with deterministic
rules — so the recommendation is always traceable to the inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from monte.indicators.regime import RegimeLabel


class Horizon(str, Enum):
    DAY_TRADE = "DAY_TRADE"
    SWING = "SWING"
    LONG_HOLD = "LONG_HOLD"


HORIZON_LABEL = {
    Horizon.DAY_TRADE: "Day Trade",
    Horizon.SWING:     "Swing (2-10 days)",
    Horizon.LONG_HOLD: "Long-Term Hold",
}


HORIZON_HOLD_HINT = {
    Horizon.DAY_TRADE: "exit before session close · trail tight",
    Horizon.SWING:     "hold 2-10 days · trail stop with ATR",
    Horizon.LONG_HOLD: "hold weeks-months · scale in on dips",
}


@dataclass
class HorizonCall:
    horizon: Horizon
    rationale: str


# Intraday vs swing vs long-term defaults derived from the chart timeframe.
_TF_BASE = {
    "1m":  Horizon.DAY_TRADE,
    "5m":  Horizon.DAY_TRADE,
    "15m": Horizon.DAY_TRADE,
    "30m": Horizon.SWING,
    "1h":  Horizon.SWING,
    "4h":  Horizon.SWING,
    "1d":  Horizon.LONG_HOLD,
    "1wk": Horizon.LONG_HOLD,
}


def classify_horizon(
    timeframe: str,
    regime: RegimeLabel,
    adx: float,
    score: float,
) -> HorizonCall:
    """Pick a trade horizon from chart timeframe, regime and signal strength."""
    base = _TF_BASE.get(timeframe, Horizon.SWING)
    reasons = [f"{timeframe} chart → {HORIZON_LABEL[base]}"]

    # A strong trend on a swing timeframe is the textbook long-hold setup.
    if base is Horizon.SWING and regime is RegimeLabel.TRENDING_UP and adx >= 25 and score > 0:
        base = Horizon.LONG_HOLD
        reasons.append(f"strong uptrend (ADX {adx:.0f}) → upgrade to long hold")

    # A weak/ranging regime on a daily chart is a swing, not a multi-week hold.
    if base is Horizon.LONG_HOLD and (regime is RegimeLabel.RANGING or adx < 18):
        base = Horizon.SWING
        reasons.append(f"daily ranging (ADX {adx:.0f}) → downgrade to swing")

    # Volatile regime on intraday: scalp it, don't hold through the chop.
    if regime is RegimeLabel.VOLATILE and base is not Horizon.LONG_HOLD:
        base = Horizon.DAY_TRADE
        reasons.append("volatile regime → day-trade only")

    return HorizonCall(horizon=base, rationale=" · ".join(reasons))


__all__ = ["Horizon", "HorizonCall", "classify_horizon", "HORIZON_LABEL", "HORIZON_HOLD_HINT"]
