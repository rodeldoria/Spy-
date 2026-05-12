"""Stub regime classifier."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class RegimeLabel(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"


@dataclass
class RegimeResult:
    regime: RegimeLabel
    adx: float


def classify_regime(df: pd.DataFrame, period: int = 14) -> RegimeResult:
    """Stub regime classifier using a simple ADX approximation."""
    try:
        close = df["Close"]
        high = df.get("High", close)
        low = df.get("Low", close)

        tr = pd.concat(
            [
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        plus_dm = (high - high.shift()).clip(lower=0)
        minus_dm = (low.shift() - low).clip(lower=0)

        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, float("nan"))
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, float("nan"))

        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))).fillna(0)
        adx = float(dx.ewm(span=period, adjust=False).mean().iloc[-1])

        last_plus = float(plus_di.iloc[-1]) if not plus_di.empty else 0.0
        last_minus = float(minus_di.iloc[-1]) if not minus_di.empty else 0.0

        if adx > 25:
            label = RegimeLabel.TRENDING_UP if last_plus > last_minus else RegimeLabel.TRENDING_DOWN
        elif adx > 15:
            label = RegimeLabel.RANGING
        else:
            label = RegimeLabel.VOLATILE
    except Exception:
        adx = 0.0
        label = RegimeLabel.RANGING

    return RegimeResult(regime=label, adx=adx)
