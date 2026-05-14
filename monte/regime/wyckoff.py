"""Wyckoff-style phase classifier: accumulation / markup / distribution / markdown.

Classification rules (price + vol only, no order flow needed):

  - **markup**         — price > 200MA AND 50d slope > 0  (uptrend)
  - **markdown**       — price < 200MA AND 50d slope < 0  (downtrend)
  - **accumulation**   — price < 200MA AND 50d slope ≥ 0 AND vol percentile in 25–75 band
  - **distribution**   — price > 200MA AND 50d slope ≤ 0 AND vol percentile in 25–75 band

This is a deliberate simplification of the Wyckoff method — the user's
discretionary calls (springs, upthrusts, signs of weakness) require
interactive chart analysis. What matters for the gate is whether the
*macro structure* of the chart is constructive or destructive for the
trade direction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class WyckoffPhase:
    phase: str           # "accumulation" | "markup" | "distribution" | "markdown" | "unclear"
    confidence: float    # 0..1
    note: str
    price_vs_200ma_pct: float
    slope_50_pct: float


def wyckoff_phase(df: pd.DataFrame) -> Optional[WyckoffPhase]:
    """Classify the chart phase from a daily OHLCV frame."""
    if df is None or df.empty:
        return None
    closes = df["Close"].astype(float).dropna() if "Close" in df.columns else None
    if closes is None and "close" in df.columns:
        closes = df["close"].astype(float).dropna()
    if closes is None or len(closes) < 60:
        return None

    spot = float(closes.iloc[-1])
    ma_200 = float(closes.rolling(200, min_periods=50).mean().iloc[-1])
    if not ma_200 or ma_200 <= 0:
        return None
    above_pct = (spot - ma_200) / ma_200 * 100.0

    slope_50 = (
        (spot - float(closes.iloc[-51])) / float(closes.iloc[-51]) * 100.0
        if len(closes) >= 51
        else 0.0
    )

    vol = None
    for col in ("Volume", "volume"):
        if col in df.columns:
            vol = df[col].astype(float).dropna()
            break

    vol_percentile = 0.5
    if vol is not None and len(vol) >= 60:
        recent_vol = float(vol.iloc[-1])
        sample = vol.tail(120)
        vol_percentile = float((sample <= recent_vol).mean())

    if above_pct > 1.0 and slope_50 > 1.0:
        phase = "markup"
        conf = min(1.0, abs(slope_50) / 10.0 + 0.5)
        note = f"Price {above_pct:+.1f}% above 200MA, 50d trend {slope_50:+.1f}% — uptrend in force."
    elif above_pct < -1.0 and slope_50 < -1.0:
        phase = "markdown"
        conf = min(1.0, abs(slope_50) / 10.0 + 0.5)
        note = f"Price {above_pct:+.1f}% below 200MA, 50d trend {slope_50:+.1f}% — downtrend in force."
    elif above_pct < 0 and slope_50 >= -1.0 and 0.25 <= vol_percentile <= 0.85:
        phase = "accumulation"
        conf = 0.5 + min(0.4, vol_percentile - 0.25)
        note = (
            f"Price {above_pct:+.1f}% below 200MA but slope flattening ({slope_50:+.1f}% over 50d); "
            "volume sitting mid-band — classic accumulation footprint."
        )
    elif above_pct > 0 and slope_50 <= 1.0 and 0.25 <= vol_percentile <= 0.85:
        phase = "distribution"
        conf = 0.5 + min(0.4, vol_percentile - 0.25)
        note = (
            f"Price {above_pct:+.1f}% above 200MA but slope flattening ({slope_50:+.1f}% over 50d); "
            "supply emerging — distribution risk."
        )
    else:
        phase = "unclear"
        conf = 0.2
        note = "Phase unclear — chart between regimes."

    return WyckoffPhase(
        phase=phase,
        confidence=conf,
        note=note,
        price_vs_200ma_pct=above_pct,
        slope_50_pct=slope_50,
    )


__all__ = ["WyckoffPhase", "wyckoff_phase"]
