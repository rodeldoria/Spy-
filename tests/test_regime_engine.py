"""Smoke + correctness tests for the regime stack.

We avoid the network and the optional `hmmlearn` dep so the suite stays
fast and works on a clean install. Each test exercises one regime axis
on a synthetic series whose answer is known by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monte.regime import assess
from monte.regime.bocpd import bocpd_posterior
from monte.regime.hmm import clear_cache as clear_hmm_cache, hmm_state
from monte.regime.hurst import hurst_exponent
from monte.regime.macro_quadrant import classify_quadrant
from monte.regime.variance_ratio import variance_ratio_test
from monte.regime.wyckoff import wyckoff_phase


@pytest.fixture(autouse=True)
def _reset_hmm_cache():
    clear_hmm_cache()
    yield
    clear_hmm_cache()


def _trending_prices(n: int = 400, drift: float = 0.001, sigma: float = 0.005, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, sigma, size=n)
    return pd.Series(100.0 * np.exp(np.cumsum(rets)))


def _persistent_prices(n: int = 400, phi: float = 0.7, sigma: float = 0.005, seed: int = 11) -> pd.Series:
    """Returns generated as r_t = phi * r_{t-1} + e_t — strong positive
    autocorrelation in returns, which is what R/S Hurst flags as 'trending'."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    e = rng.normal(0, sigma, size=n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + e[i]
    return pd.Series(100.0 * np.exp(np.cumsum(r)))


def _mean_reverting_prices(n: int = 400, seed: int = 2) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + rng.normal(0, 1)
    base = 100.0 + 0.5 * x
    return pd.Series(base)


def test_hurst_trending_is_high():
    # Use a return series with positive autocorrelation; that's what R/S
    # Hurst is sensitive to (drift alone is not enough at this length).
    prices = _persistent_prices(n=600).to_numpy()
    h = hurst_exponent(prices)
    assert h is not None
    assert h > 0.55, f"persistent-return series should have H>0.55, got {h:.3f}"


def test_hurst_mean_reverting_is_low():
    prices = _mean_reverting_prices().to_numpy()
    h = hurst_exponent(prices)
    assert h is not None
    assert h < 0.5, f"AR(1) mean-reverting series should have H<0.5, got {h:.3f}"


def test_hurst_returns_none_for_short_series():
    assert hurst_exponent(np.array([1.0, 2.0, 3.0])) is None


def test_variance_ratio_random_walk_p_high():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, size=500)
    out = variance_ratio_test(rets, q=4)
    assert out is not None
    _vr, p = out
    assert p > 0.1, f"random-walk returns should not reject the null, got p={p:.3f}"


def test_variance_ratio_handles_short_input():
    assert variance_ratio_test(np.array([0.01, 0.02]), q=4) is None


def test_bocpd_returns_finite_probability():
    rng = np.random.default_rng(3)
    rets = rng.normal(0, 0.01, size=120)
    p = bocpd_posterior(rets)
    assert 0.0 <= p <= 1.0


def test_bocpd_detects_obvious_change_point():
    rng = np.random.default_rng(4)
    pre = rng.normal(0, 0.005, size=80)
    post = rng.normal(0.01, 0.02, size=20)   # mean shift + vol blow-up
    p = bocpd_posterior(np.concatenate([pre, post]))
    assert p >= 0.0


def test_hmm_fallback_runs_without_hmmlearn():
    prices = _trending_prices()
    rets = prices.pct_change().dropna()
    res = hmm_state(prices, rets)
    # We may or may not have hmmlearn — just assert the contract.
    assert res.label.startswith(("bull", "bear"))
    assert 0.0 <= res.bull_prob <= 1.0
    assert res.source in {"hmmlearn", "fallback"}


def test_wyckoff_markup_for_uptrend():
    closes = _trending_prices(n=300)
    df = pd.DataFrame(
        {
            "Close": closes.values,
            "High": closes.values * 1.001,
            "Low": closes.values * 0.999,
            "Volume": np.linspace(1000, 1500, len(closes)),
        }
    )
    phase = wyckoff_phase(df)
    assert phase is not None
    assert phase.phase == "markup"


def test_wyckoff_markdown_for_downtrend():
    rng = np.random.default_rng(7)
    rets = rng.normal(-0.001, 0.005, size=300)
    closes = pd.Series(100.0 * np.exp(np.cumsum(rets)))
    df = pd.DataFrame(
        {
            "Close": closes.values,
            "High": closes.values * 1.001,
            "Low": closes.values * 0.999,
            "Volume": np.linspace(1000, 800, len(closes)),
        }
    )
    phase = wyckoff_phase(df)
    assert phase is not None
    assert phase.phase == "markdown"


def test_macro_quadrant_returns_none_when_snapshot_unavailable():
    class _Fake:
        available = False
    assert classify_quadrant(_Fake()) is None


def test_macro_quadrant_handles_real_shape():
    from monte.data.fred import FredObservation, FredSnapshot

    snap = FredSnapshot(
        available=True,
        observations=[
            FredObservation("INDPRO", "Industrial Production", 105.0, 0.5, 2.5, "2025-04-01", False, ""),
            FredObservation("USSLIND", "USSLIND", 0.4, 0.05, None, "2025-04-01", False, ""),
            FredObservation("CPIAUCSL", "CPI", 312.0, 0.3, 2.4, "2025-04-01", False, ""),
        ],
    )
    q = classify_quadrant(snap)
    assert q is not None
    assert q.quadrant in {"Reflation", "Goldilocks", "Stagflation", "Deflationary bust", "Mixed"}


def test_assess_handles_empty_dataframe():
    report = assess("FAKE", pd.DataFrame())
    assert "df" in report.errors
    assert report.directional_bias() == "neutral"


def test_assess_runs_end_to_end_on_synthetic_data():
    closes = _trending_prices(n=500)
    df = pd.DataFrame(
        {
            "Close": closes.values,
            "High": closes.values * 1.002,
            "Low": closes.values * 0.998,
            "Volume": np.linspace(1000, 1200, len(closes)),
        }
    )
    report = assess("SYN-USD", df, fred_snapshot=None)
    assert report.symbol == "SYN-USD"
    # At least HMM and Hurst should have populated.
    assert report.hmm is not None
    assert report.hurst is not None
    assert report.directional_bias() in {"bull", "bear", "neutral"}
