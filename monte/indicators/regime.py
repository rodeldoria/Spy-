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


def _series(x, fallback) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0] if x.shape[1] else fallback
    if x is None:
        return fallback
    return pd.to_numeric(pd.Series(x), errors="coerce")


def classify_regime(df: pd.DataFrame, period: int = 14) -> RegimeResult:
    """Regime classifier using a simple ADX approximation."""
    try:
        if df is None or len(df) == 0:
            return RegimeResult(regime=RegimeLabel.RANGING, adx=0.0)
        close = _series(df["Close"], pd.Series(dtype=float))
        high = _series(df["High"] if "High" in df.columns else close, close)
        low = _series(df["Low"] if "Low" in df.columns else close, close)

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
