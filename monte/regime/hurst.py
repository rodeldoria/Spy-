"""R/S Hurst exponent estimator.

H ≈ 0.5 → random walk; H > 0.55 → trending; H < 0.45 → mean-reverting.
Implementation is the textbook rescaled-range method. Pure numpy.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def hurst_exponent(prices: np.ndarray) -> Optional[float]:
    """Estimate the Hurst exponent of `prices`. Returns None if too short."""
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 64:
        return None

    log_returns = np.diff(np.log(np.maximum(arr, 1e-12)))
    n = len(log_returns)
    if n < 64:
        return None

    # Use a small-but-spread set of lags so the regression is stable.
    max_lag = max(20, n // 4)
    lags = sorted({int(round(lag)) for lag in np.geomspace(8, max_lag, num=8)})
    rs_values: list[float] = []
    log_lags: list[float] = []
    for lag in lags:
        if lag >= n:
            continue
        rs = _rs_for_lag(log_returns, lag)
        if rs is None or rs <= 0:
            continue
        rs_values.append(math.log(rs))
        log_lags.append(math.log(lag))

    if len(rs_values) < 4:
        return None

    slope, _ = np.polyfit(log_lags, rs_values, 1)
    return float(np.clip(slope, 0.0, 1.0))


def _rs_for_lag(returns: np.ndarray, lag: int) -> Optional[float]:
    n_chunks = len(returns) // lag
    if n_chunks < 2:
        return None
    rs_per_chunk: list[float] = []
    for i in range(n_chunks):
        chunk = returns[i * lag : (i + 1) * lag]
        mean = chunk.mean()
        deviations = np.cumsum(chunk - mean)
        r = float(deviations.max() - deviations.min())
        s = float(chunk.std(ddof=0))
        if s <= 0:
            continue
        rs_per_chunk.append(r / s)
    if not rs_per_chunk:
        return None
    return float(np.mean(rs_per_chunk))


__all__ = ["hurst_exponent"]
