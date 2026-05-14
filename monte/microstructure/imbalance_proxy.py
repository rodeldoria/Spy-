"""Order-book imbalance proxy from OHLCV.

Without L2 data we approximate "where the aggressors are pressing" as a
volume-weighted bar-direction average. Returns a signed score in [-1, 1]:

  +1 = recent bars heavily up-volume vs trailing baseline
  -1 = recent bars heavily down-volume vs trailing baseline
   0 = balanced or no signal

This is intentionally a simple proxy. The interface matches what a real
L2 imbalance feed would emit, so swapping in `coinglass`/`databento`/etc
is a one-line change.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def imbalance_score(df: pd.DataFrame, *, recent: int = 5, baseline: int = 60) -> Optional[float]:
    if df is None or df.empty:
        return None
    close = df.get("Close", df.get("close"))
    vol = df.get("Volume", df.get("volume"))
    if close is None or vol is None:
        return None
    closes = close.astype(float).to_numpy()
    volumes = vol.astype(float).clip(lower=0).to_numpy()
    n = len(closes)
    if n < baseline + recent + 1:
        return None

    diffs = np.diff(closes, prepend=closes[0])
    direction = np.sign(diffs)
    signed_vol = direction * volumes

    recent_vol = signed_vol[-recent:].sum()
    base_abs = np.abs(signed_vol[-(baseline + recent) : -recent]).mean() * recent
    if base_abs <= 0:
        return 0.0
    score = recent_vol / base_abs
    return float(np.clip(score, -1.0, 1.0))


__all__ = ["imbalance_score"]
