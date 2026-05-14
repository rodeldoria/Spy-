"""Cumulative Volume Delta proxy from OHLCV.

Real CVD needs trade-level prints with aggressor side. As an OHLCV-only
proxy we use the **tick rule**: bar volume is treated as buy volume when
close > prior close, sell volume when close < prior close, and split
50/50 on equality. The cumulative sum is the CVD curve. Despite being a
proxy, the divergence signal (price up + CVD down → bearish, price down
+ CVD up → bullish) tracks the L2 version closely on liquid pairs.

Returns (cvd_array, divergence_sign) where divergence_sign is +1 (bullish
divergence vs price), -1 (bearish), or 0 (none).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def cvd_series(df: pd.DataFrame, *, lookback_bars: int = 20) -> tuple[Optional[np.ndarray], int]:
    if df is None or df.empty:
        return None, 0

    close = df.get("Close", df.get("close"))
    vol = df.get("Volume", df.get("volume"))
    if close is None or vol is None:
        return None, 0
    closes = close.astype(float).to_numpy()
    volumes = vol.astype(float).clip(lower=0).to_numpy()
    if len(closes) < 2:
        return None, 0

    diffs = np.diff(closes, prepend=closes[0])
    direction = np.sign(diffs)
    direction[direction == 0] = 0.0   # tie → no contribution
    signed_vol = direction * volumes
    cvd = np.cumsum(signed_vol)

    if len(cvd) < lookback_bars + 2:
        return cvd, 0

    price_slope = float(closes[-1] - closes[-lookback_bars])
    cvd_slope = float(cvd[-1] - cvd[-lookback_bars])
    if price_slope > 0 and cvd_slope < 0:
        return cvd, -1
    if price_slope < 0 and cvd_slope > 0:
        return cvd, +1
    return cvd, 0


__all__ = ["cvd_series"]
