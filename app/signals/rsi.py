"""RSI signal — pure function over a close series.

Uses Wilder's classic smoothing (SMA seed for the first `period` diffs,
then recursive `(prev * (period - 1) + current) / period`).  No I/O,
no caching, no Streamlit.  Returns `monte.strategies.signals.Signal`.
"""

from __future__ import annotations

import pandas as pd

from monte.strategies.signals import Signal


def rsi(
    closes: pd.Series,
    *,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    strong_oversold: float = 20.0,
    strong_overbought: float = 80.0,
    timeframe: str = "",
) -> Signal:
    """Score the last bar against Wilder-smoothed RSI thresholds.

    Score rules:
      RSI <= strong_oversold:    +1.0
      RSI <= oversold:           +0.4
      strong_overbought <= RSI:  -1.0
      overbought      <= RSI:    -0.4
      otherwise:                  0.0

    Raises ValueError when `period < 2` or `len(closes) < period + 1`.
    A truly flat series (no up *or* down moves) returns RSI = 50.0.
    """
    if period < 2:
        raise ValueError(f"period ({period}) must be >= 2")
    if len(closes) < period + 1:
        raise ValueError(
            f"need at least period+1={period + 1} closes, got {len(closes)}"
        )
    if not (
        strong_oversold < oversold < overbought < strong_overbought
    ):
        raise ValueError(
            "thresholds must satisfy strong_oversold < oversold < "
            f"overbought < strong_overbought, got "
            f"({strong_oversold}, {oversold}, {overbought}, {strong_overbought})"
        )

    diffs = closes.diff().iloc[1:]
    gains = diffs.where(diffs > 0, 0.0).astype(float)
    losses = (-diffs).where(diffs < 0, 0.0).astype(float)

    avg_gain = float(gains.iloc[:period].mean())
    avg_loss = float(losses.iloc[:period].mean())
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + float(gains.iloc[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses.iloc[i])) / period

    if avg_loss == 0.0:
        rsi_val = 100.0 if avg_gain > 0.0 else 50.0
    else:
        rs = avg_gain / avg_loss
        rsi_val = 100.0 - 100.0 / (1.0 + rs)

    if rsi_val <= strong_oversold:
        score = 1.0
        rationale = f"RSI={rsi_val:.1f} <= {strong_oversold} (strong oversold)"
    elif rsi_val <= oversold:
        score = 0.4
        rationale = f"RSI={rsi_val:.1f} <= {oversold} (oversold)"
    elif rsi_val >= strong_overbought:
        score = -1.0
        rationale = f"RSI={rsi_val:.1f} >= {strong_overbought} (strong overbought)"
    elif rsi_val >= overbought:
        score = -0.4
        rationale = f"RSI={rsi_val:.1f} >= {overbought} (overbought)"
    else:
        score = 0.0
        rationale = f"RSI={rsi_val:.1f} (neutral)"

    return Signal(
        name="RSI",
        score=score,
        rationale=rationale,
        timeframe=timeframe,
        meta={
            "period": period,
            "rsi": rsi_val,
            "avg_gain": avg_gain,
            "avg_loss": avg_loss,
        },
    )
