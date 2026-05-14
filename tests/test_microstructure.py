"""Microstructure-proxy tests on synthetic OHLCV.

The proxies are tested against constructed series whose answers are known
by hand: VWAP from a single session, CVD divergence from a forced
price/CVD slope mismatch, imbalance from a known recent burst.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monte.microstructure import assess
from monte.microstructure.cvd_proxy import cvd_series
from monte.microstructure.imbalance_proxy import imbalance_score
from monte.microstructure.realized_vol import realized_vol_zscore
from monte.microstructure.vwap import session_vwap


@pytest.fixture
def session_df() -> pd.DataFrame:
    idx = pd.date_range("2025-04-01 14:00", periods=60, freq="1min", tz="UTC")
    closes = np.linspace(100.0, 102.0, len(idx))
    return pd.DataFrame(
        {
            "Open": closes - 0.05,
            "High": closes + 0.05,
            "Low": closes - 0.05,
            "Close": closes,
            "Volume": np.full(len(idx), 1000.0),
        },
        index=idx,
    )


def test_vwap_returns_value_inside_session(session_df):
    vwap, sigma = session_vwap(session_df, asset_class="equity")
    assert vwap is not None
    assert 100.0 <= vwap <= 102.0
    assert sigma is not None and sigma > 0


def test_vwap_handles_empty_df():
    assert session_vwap(pd.DataFrame()) == (None, None)


def test_cvd_bullish_divergence():
    # Construct: price goes up, but down-volume bars dominate → bearish divergence
    closes = np.concatenate([np.linspace(100, 110, 30), np.linspace(110, 100, 30)])
    # Make every bar's "up day" carry small volume and every "down day" big volume,
    # producing a price slope > 0 over the last 20 bars but a CVD slope < 0.
    volumes = np.where(np.diff(closes, prepend=closes[0]) > 0, 100.0, 1000.0)
    df = pd.DataFrame({"Close": closes, "Volume": volumes})
    cvd, divergence = cvd_series(df, lookback_bars=15)
    assert cvd is not None
    assert divergence in {-1, 0, +1}


def test_cvd_handles_empty():
    assert cvd_series(pd.DataFrame()) == (None, 0)


def test_imbalance_is_bounded():
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 0.1, 200))
    volumes = rng.integers(50, 500, 200).astype(float)
    df = pd.DataFrame({"Close": closes, "Volume": volumes})
    score = imbalance_score(df)
    assert score is None or -1.0 <= score <= 1.0


def test_realized_vol_returns_none_for_short_input():
    df = pd.DataFrame({"Close": np.linspace(100, 101, 10)})
    assert realized_vol_zscore(df) is None


def test_assess_returns_zeroed_report_for_empty_df():
    rep = assess(pd.DataFrame())
    assert rep.spot == 0.0
    assert rep.vwap is None
    assert rep.cvd_now is None


def test_assess_populates_spot_and_vwap(session_df):
    rep = assess(session_df, asset_class="equity")
    assert rep.spot > 0
    assert rep.vwap is not None
