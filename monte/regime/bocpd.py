"""Bayesian Online Change-Point Detection (Adams & MacKay 2007).

Implements the constant-hazard variant inline (numpy only — no third-party
dep). Returns the posterior P(run length = 0 | data_{1..t}) at the latest
point, which the gate reads as "probability the regime has just shifted".

This is a simplified, robust implementation: a Gaussian observation model
with conjugate Normal-Inverse-Gamma updates and a constant hazard
1 / lambda. Fast enough to run on every gate evaluation against ~180-bar
windows.
"""
from __future__ import annotations

import math

import numpy as np


def bocpd_posterior(
    returns: np.ndarray,
    *,
    hazard_lambda: float = 250.0,
    mu0: float = 0.0,
    kappa0: float = 0.1,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> float:
    """Return P(change-point at the latest observation).

    `hazard_lambda` is the prior expected run length; ~250 ≈ "regime lasts
    about a year of daily returns".
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return 0.0

    # Run-length distribution; entry r = P(run length = r at time t).
    R = np.zeros(n + 1)
    R[0] = 1.0

    # Parameter posteriors per run length.
    mu = np.full(n + 1, mu0)
    kappa = np.full(n + 1, kappa0)
    alpha = np.full(n + 1, alpha0)
    beta = np.full(n + 1, beta0)

    H = 1.0 / hazard_lambda

    for t in range(n):
        xt = x[t]
        # Predictive: Student-t with 2*alpha dof.
        dof = 2.0 * alpha[: t + 1]
        scale = beta[: t + 1] * (kappa[: t + 1] + 1.0) / (alpha[: t + 1] * kappa[: t + 1])
        scale = np.maximum(scale, 1e-12)
        pred = _student_t_pdf(xt, df=dof, loc=mu[: t + 1], scale=np.sqrt(scale))
        # Growth probabilities (no change point).
        growth = R[: t + 1] * pred * (1.0 - H)
        # Change-point probability (sum across all current run lengths).
        cp_mass = float(np.sum(R[: t + 1] * pred * H))

        # New run-length distribution.
        new_R = np.zeros(n + 1)
        new_R[0] = cp_mass
        new_R[1 : t + 2] = growth
        z = float(new_R.sum())
        if z <= 0 or not math.isfinite(z):
            # Numerical underflow — restart cleanly.
            new_R = np.zeros(n + 1)
            new_R[0] = 1.0
            z = 1.0
        R = new_R / z

        # Update sufficient stats per run length.
        new_mu = np.copy(mu)
        new_kappa = np.copy(kappa)
        new_alpha = np.copy(alpha)
        new_beta = np.copy(beta)
        new_mu[1 : t + 2] = (kappa[: t + 1] * mu[: t + 1] + xt) / (kappa[: t + 1] + 1.0)
        new_kappa[1 : t + 2] = kappa[: t + 1] + 1.0
        new_alpha[1 : t + 2] = alpha[: t + 1] + 0.5
        new_beta[1 : t + 2] = beta[: t + 1] + (
            kappa[: t + 1] * (xt - mu[: t + 1]) ** 2 / (2.0 * (kappa[: t + 1] + 1.0))
        )
        # Reset slot 0 to the prior so the next iteration's "cp" branch stays clean.
        new_mu[0] = mu0
        new_kappa[0] = kappa0
        new_alpha[0] = alpha0
        new_beta[0] = beta0

        mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta

    return float(R[0])


def _student_t_pdf(x: float, *, df: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (x - loc) / scale
    log_norm = (
        np.log(np.maximum(scale, 1e-12))
        - 0.5 * np.log(df * math.pi)
        + np.array([math.lgamma((d + 1) / 2.0) - math.lgamma(d / 2.0) for d in df])
    )
    log_pdf = log_norm - ((df + 1) / 2.0) * np.log1p(z * z / df)
    return np.exp(log_pdf)


__all__ = ["bocpd_posterior"]
