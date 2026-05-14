"""Realized-vol regime check on a 1m/5m frame.

Returns the z-score of the most recent 60-bar realized σ vs the trailing
20-session distribution. Positive z = vol regime is hot (penalize new
entries that don't account for it); negative z = vol regime is calm.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def realized_vol_zscore(
    df: pd.DataFrame,
    *,
    bars_per_session: int = 390,
    sessions: int = 20,
    window: int = 60,
) -> Optional[float]:
    if df is None or df.empty:
        return None
    close = df.get("Close", df.get("close"))
    if close is None:
        return None
    closes = close.astype(float).dropna()
    if len(closes) < bars_per_session * 2:
        return None
    log_rets = np.log(closes).diff().dropna()
    if len(log_rets) < window * sessions:
        return None

    rolling_sigma = log_rets.rolling(window).std()
    sample = rolling_sigma.dropna().iloc[-bars_per_session * sessions :]
    if sample.empty:
        return None
    mu = float(sample.mean())
    sd = float(sample.std()) or 1e-9
    return float((float(rolling_sigma.iloc[-1]) - mu) / sd)


__all__ = ["realized_vol_zscore"]
