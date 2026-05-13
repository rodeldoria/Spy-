"""Monte Edge — opinion layer on top of `dip_pump.detect()`.

Five rules:
  1. Macro filter — only long when SPY > 200-day SMA; only short when below.
  2. Confluence >= 3 of 5 contributors must agree with the signal direction.
  3. Confidence-scaled risk per trade — bigger size on higher conviction.
  4. ATR-based stop, 1.5x ATR; existing engine already provides this.
  5. Drawdown brake — halve risk past 5% DD, halt new entries past 10% DD.

These rules turn a "BUY 52%" into a clear ACT_NOW / WATCH / STAND_DOWN tier
with a one-sentence "why this works" the user can read on a lock screen.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

import pandas as pd

from monte.signals.dip_pump import Alert, detect


# ---------- Canned reasoning library ----------

REASONING_LIBRARY: dict[str, str] = {
    "MACRO_LONG": (
        "SPY is above its 200-day average → broad risk-on regime; long setups "
        "historically carry a positive drift edge in this state."
    ),
    "MACRO_SHORT": (
        "SPY is below its 200-day average → defensive regime; longs face a "
        "negative drift headwind, puts/shorts have the edge."
    ),
    "MACRO_UNKNOWN": (
        "Macro filter not available (SPY 200-SMA unknown); treating signal "
        "as informational only."
    ),
    "RSI_REVERT_IN_TREND": (
        "RSI < 30 inside a confirmed uptrend → pullback-in-trend; this is the "
        "highest hit-rate setup in swing trading."
    ),
    "BB_BREAKOUT": (
        "Close above the upper Bollinger band with ADX > 25 → trend-continuation "
        "breakout; momentum tends to persist for several bars."
    ),
    "MACD_CROSS_EARLY_TREND": (
        "MACD crosses up with RSI 50-65 → early-stage trend transition; best "
        "R:R window before momentum becomes obvious to everyone."
    ),
    "STRONG_DOWNTREND": (
        "ADX > 25 with price below SMA20 and SMA50 → confirmed downtrend; "
        "rallies are sells, not buy-the-dip."
    ),
    "OPTION_DEBIT": (
        "Buying an ATM call/put 30-45 DTE captures the strongest gamma-to-theta "
        "zone; defined risk = premium paid."
    ),
    "CONFLUENCE_HIGH": (
        "4+ of 5 indicators agree with the trade direction → high-conviction "
        "setup; concentrate risk here."
    ),
    "MACRO_MISMATCH": (
        "Signal direction disagrees with macro filter → standing down even "
        "though local indicators line up. Wait for macro alignment."
    ),
    "DRAWDOWN_BRAKE": (
        "Equity is in drawdown → risk per trade is halved until equity makes "
        "a new high. Capital preservation > prediction."
    ),
}


# ---------- Types ----------

class EdgeTier(str, Enum):
    ACT_NOW = "ACT_NOW"
    WATCH = "WATCH"
    STAND_DOWN = "STAND_DOWN"


TIER_LABEL = {
    EdgeTier.ACT_NOW: "ACT NOW",
    EdgeTier.WATCH: "Watch",
    EdgeTier.STAND_DOWN: "Stand down",
}


@dataclass
class EdgeSignal:
    """Decorated `Alert` with macro, confluence, tier and reasoning."""
    symbol: str
    timeframe: str
    action: str
    tier: EdgeTier
    confidence: float
    score: float
    confluence: int                 # how many of the 5 contributors agree
    macro_aligned: bool | None      # None if macro unavailable
    spot: float
    entry: float
    stop: float
    target: float
    rr: float
    horizon: str
    regime: str
    risk_per_share: float
    contributions: list[dict[str, Any]] = field(default_factory=list)
    indicator_snapshot: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    reasoning_codes: list[str] = field(default_factory=list)
    macro_note: str = ""
    horizon_rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tier"] = self.tier.value
        return d


# ---------- Helpers ----------

CONFLUENCE_MIN = 3
ACT_NOW_CONFLUENCE = 4
ACT_NOW_CONFIDENCE = 75.0
WATCH_CONFIDENCE = 60.0


def _spy_above_200ma(spy_daily: pd.DataFrame | None) -> bool | None:
    """Return True if SPY closes above its 200-day SMA, False below, None if unknown."""
    if spy_daily is None or spy_daily.empty or "Close" not in spy_daily.columns:
        return None
    closes = spy_daily["Close"].dropna()
    if len(closes) < 50:
        return None
    sma = closes.rolling(min(200, len(closes))).mean()
    last_close = float(closes.iloc[-1])
    last_sma = float(sma.iloc[-1])
    if last_sma <= 0:
        return None
    return last_close > last_sma


def _count_confluence(contributions: list[dict[str, Any]], score: float) -> int:
    """How many contributors point in the same direction as the composite score?"""
    if not contributions or score == 0:
        return 0
    direction = 1 if score > 0 else -1
    return sum(
        1 for c in contributions
        if (c.get("score", 0) > 0.05 and direction > 0)
        or (c.get("score", 0) < -0.05 and direction < 0)
    )


def _build_reasoning(
    alert: Alert,
    confluence: int,
    macro_aligned: bool | None,
    drawdown_brake: bool,
) -> tuple[str, list[str]]:
    """Pick 1–3 reasoning codes and stitch them into a one-line story."""
    snap = alert.indicator_snapshot or {}
    rsi_v = snap.get("rsi", 50.0)
    adx_v = snap.get("adx", 0.0)
    bb_v = snap.get("bb_pctb", 0.5)
    score = alert.score
    long_side = score > 0
    codes: list[str] = []

    if macro_aligned is True and long_side:
        codes.append("MACRO_LONG")
    elif macro_aligned is False and not long_side:
        codes.append("MACRO_SHORT")
    elif macro_aligned is False and long_side:
        codes.append("MACRO_MISMATCH")
    elif macro_aligned is True and not long_side:
        codes.append("MACRO_MISMATCH")
    else:
        codes.append("MACRO_UNKNOWN")

    if long_side:
        if rsi_v < 35 and adx_v >= 18:
            codes.append("RSI_REVERT_IN_TREND")
        elif bb_v >= 0.85 and adx_v >= 25:
            codes.append("BB_BREAKOUT")
        elif 50 <= rsi_v <= 65 and snap.get("macd_hist", 0.0) > 0:
            codes.append("MACD_CROSS_EARLY_TREND")
    else:
        if adx_v >= 25:
            codes.append("STRONG_DOWNTREND")

    if confluence >= ACT_NOW_CONFLUENCE:
        codes.append("CONFLUENCE_HIGH")
    if drawdown_brake:
        codes.append("DRAWDOWN_BRAKE")

    seen: set[str] = set()
    text_bits: list[str] = []
    for c in codes:
        if c in seen:
            continue
        seen.add(c)
        copy = REASONING_LIBRARY.get(c)
        if copy:
            text_bits.append(copy)
    return " ".join(text_bits[:3]), codes


def tier_from_signal(
    alert: Alert,
    confluence: int,
    macro_aligned: bool | None,
    drawdown_halt: bool,
) -> EdgeTier:
    """Classify an alert into ACT_NOW / WATCH / STAND_DOWN."""
    action = alert.action.value
    if action == "HOLD":
        return EdgeTier.STAND_DOWN
    if drawdown_halt:
        return EdgeTier.STAND_DOWN
    # Macro mismatch demotes to STAND_DOWN unless confluence is overwhelming.
    if macro_aligned is False and confluence < 5:
        return EdgeTier.STAND_DOWN
    if confluence < CONFLUENCE_MIN:
        return EdgeTier.STAND_DOWN
    if (
        alert.confidence >= ACT_NOW_CONFIDENCE
        and confluence >= ACT_NOW_CONFLUENCE
        and (macro_aligned is not False)
    ):
        return EdgeTier.ACT_NOW
    if alert.confidence >= WATCH_CONFIDENCE:
        return EdgeTier.WATCH
    return EdgeTier.STAND_DOWN


# ---------- Risk sizing ----------

def confidence_scaled_risk_pct(
    confidence: float,
    *,
    drawdown_pct: float = 0.0,
    base_floor: float = 0.005,
    base_max: float = 0.015,
) -> float:
    """0.5%–1.5% per trade scaled by confidence; halved during drawdown."""
    c = max(50.0, min(100.0, float(confidence)))
    raw = base_floor + (base_max - base_floor) * (c - 50.0) / 50.0
    if drawdown_pct <= -0.05:
        raw *= 0.5
    if drawdown_pct <= -0.10:
        raw = 0.0
    return max(0.0, min(base_max, raw))


# ---------- Public entry point ----------

def evaluate(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    *,
    spy_daily: pd.DataFrame | None = None,
    drawdown_pct: float = 0.0,
) -> EdgeSignal:
    """Run the triangulation engine, layer Monte Edge rules on top, return EdgeSignal."""
    alert = detect(df, symbol=symbol, timeframe=timeframe)
    spy_long_bias = _spy_above_200ma(spy_daily) if spy_daily is not None else None
    long_side = alert.score > 0

    if spy_long_bias is None:
        macro_aligned: bool | None = None
    else:
        macro_aligned = (long_side and spy_long_bias) or (
            (not long_side) and (not spy_long_bias)
        )
    macro_note = ""
    if spy_long_bias is True:
        macro_note = "SPY > 200-SMA · risk-on"
    elif spy_long_bias is False:
        macro_note = "SPY < 200-SMA · risk-off"
    else:
        macro_note = "macro filter unavailable"

    confluence = _count_confluence(alert.contributions, alert.score)
    drawdown_halt = drawdown_pct <= -0.10
    tier = tier_from_signal(alert, confluence, macro_aligned, drawdown_halt)
    reasoning, codes = _build_reasoning(
        alert, confluence, macro_aligned, drawdown_pct <= -0.05
    )

    risk_per_share = abs(alert.entry - alert.stop)

    return EdgeSignal(
        symbol=symbol,
        timeframe=timeframe,
        action=alert.action.value,
        tier=tier,
        confidence=float(alert.confidence),
        score=float(alert.score),
        confluence=int(confluence),
        macro_aligned=macro_aligned,
        spot=float(alert.entry),
        entry=float(alert.entry),
        stop=float(alert.stop),
        target=float(alert.target),
        rr=float(alert.rr),
        horizon=alert.horizon.value,
        regime=alert.regime,
        risk_per_share=float(risk_per_share),
        contributions=list(alert.contributions),
        indicator_snapshot=dict(alert.indicator_snapshot),
        reasoning=reasoning,
        reasoning_codes=codes,
        macro_note=macro_note,
        horizon_rationale=alert.horizon_rationale,
    )


__all__ = [
    "EdgeSignal",
    "EdgeTier",
    "TIER_LABEL",
    "REASONING_LIBRARY",
    "evaluate",
    "tier_from_signal",
    "confidence_scaled_risk_pct",
    "CONFLUENCE_MIN",
    "ACT_NOW_CONFLUENCE",
    "ACT_NOW_CONFIDENCE",
    "WATCH_CONFIDENCE",
]
