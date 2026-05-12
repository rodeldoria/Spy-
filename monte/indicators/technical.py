"""Stub technical indicators."""
from __future__ import annotations

import pandas as pd


def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    result = 100 - (100 / (1 + rs))
    result = result.fillna(50.0)
    return result


def bollinger(closes: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = closes.rolling(period).mean()
    sigma = closes.rolling(period).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    bb_pctb = (closes - lower) / (upper - lower).replace(0, float("nan"))
    bb_pctb = bb_pctb.fillna(0.5)
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "bb_pctb": bb_pctb})


def macd(
    closes: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})
