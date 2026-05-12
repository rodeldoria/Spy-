"""Technical indicators (RSI / Bollinger / MACD).

All functions accept a 1-D close series. They tolerate accidental DataFrame
inputs (e.g. from a multi-level column slice) by squeezing to a Series, and
they return frames with a real index even when the input is too short to
produce any non-NaN values — that prevents pandas from raising
``ValueError: If using all scalar values, you must pass an index``.
"""
from __future__ import annotations

import pandas as pd


def _as_series(closes) -> pd.Series:
    """Coerce input to a 1-D float Series with a usable index."""
    if isinstance(closes, pd.DataFrame):
        if closes.shape[1] == 1:
            closes = closes.iloc[:, 0]
        else:
            # pick the first numeric column as a best-effort fallback
            num = closes.select_dtypes("number")
            closes = num.iloc[:, 0] if num.shape[1] else closes.iloc[:, 0]
    if not isinstance(closes, pd.Series):
        closes = pd.Series(closes)
    return pd.to_numeric(closes, errors="coerce").astype(float)


def rsi(closes, period: int = 14) -> pd.Series:
    closes = _as_series(closes)
    if closes.empty:
        return pd.Series(dtype=float, name="rsi")
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    result = 100 - (100 / (1 + rs))
    return result.fillna(50.0)


def bollinger(closes, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    closes = _as_series(closes)
    if closes.empty:
        return pd.DataFrame(
            {"mid": [], "upper": [], "lower": [], "bb_pctb": []},
            dtype=float,
        )
    mid = closes.rolling(period, min_periods=1).mean()
    sigma = closes.rolling(period, min_periods=1).std().fillna(0.0)
    upper = mid + std * sigma
    lower = mid - std * sigma
    width = (upper - lower).replace(0, float("nan"))
    bb_pctb = ((closes - lower) / width).fillna(0.5)
    out = pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "bb_pctb": bb_pctb},
        index=closes.index,
    )
    return out


def macd(
    closes,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    closes = _as_series(closes)
    if closes.empty:
        return pd.DataFrame({"macd": [], "signal": [], "hist": []}, dtype=float)
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": hist},
        index=closes.index,
    )


__all__ = ["rsi", "bollinger", "macd"]
