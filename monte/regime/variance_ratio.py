"""Lo-MacKinlay variance-ratio test with heteroskedasticity correction.

VR(q) = Var(q-period returns) / (q · Var(1-period returns)).

Under the random-walk null VR(q) → 1. VR > 1 → positive serial correlation
(trending), VR < 1 → negative serial correlation (mean-reverting). The
heteroskedasticity-consistent z-statistic is normally distributed under
the null, giving us a clean p-value.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def variance_ratio_test(returns: np.ndarray, *, q: int = 4) -> Optional[tuple[float, float]]:
    """Return (VR(q), two-sided p-value) or None if input too short."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < q * 4:
        return None

    mu = r.mean()
    # Unbiased 1-period variance estimator (matches Lo-MacKinlay's σ_a²).
    var1 = float(((r - mu) ** 2).sum() / (n - 1))
    if var1 <= 0:
        return None

    # q-period overlapping return variance (Lo-MacKinlay's σ_b²(q)). The
    # m-correction already absorbs the q-period normalization, so VR is
    # var_q / var_1, not var_q / (q * var_1).
    m = q * (n - q + 1) * (1.0 - q / n)
    cum = np.cumsum(np.concatenate([[0.0], r]))
    q_returns = cum[q:] - cum[:-q] - q * mu
    varq = float((q_returns**2).sum() / m)

    vr = varq / var1

    # Heteroskedasticity-consistent variance of (VR - 1) (theta in their paper).
    theta = 0.0
    centered_sq = (r - mu) ** 2
    for k in range(1, q):
        delta_num = float((centered_sq[k:] * centered_sq[:-k]).sum())
        delta_den = float(centered_sq.sum() ** 2)
        if delta_den <= 0:
            return None
        delta_k = (n * delta_num) / delta_den
        weight = (2.0 * (q - k) / q) ** 2
        theta += weight * delta_k

    if theta <= 0:
        return None
    z = (vr - 1.0) * math.sqrt(n / theta)
    p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return float(vr), float(p_value)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


__all__ = ["variance_ratio_test"]
