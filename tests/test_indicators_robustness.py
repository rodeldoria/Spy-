"""Regression tests for indicators that previously crashed on odd input.

The watchlist page used to raise ``ValueError: If using all scalar values, you
must pass an index`` when yfinance handed back a multi-level column DataFrame
(e.g. ``df['Close']`` was a single-column DataFrame, not a Series). The
indicators now coerce inputs and never construct a scalar-only DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monte.data._normalize import normalize_ohlcv
from monte.indicators.regime import RegimeLabel, classify_regime
from monte.indicators.technical import bollinger, macd, rsi


def _close_series(n: int = 80, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        100.0 + rng.standard_normal(n).cumsum(),
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
        name="Close",
    )


def test_bollinger_accepts_single_column_dataframe():
    s = _close_series()
    df = s.to_frame()
    out = bollinger(df)
    assert list(out.columns) == ["mid", "upper", "lower", "bb_pctb"]
    assert len(out) == len(s)
    assert not np.isnan(out["bb_pctb"].iloc[-1])


def test_bollinger_accepts_multilevel_close_slice():
    s = _close_series()
    multi = pd.DataFrame(
        {("Close", "BTC-USD"): s, ("Volume", "BTC-USD"): s * 0 + 1.0}
    )
    close = multi["Close"]
    out = bollinger(close)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(s)


def test_rsi_macd_handle_empty_series():
    empty = pd.Series([], dtype=float)
    assert rsi(empty).empty
    m = macd(empty)
    assert list(m.columns) == ["macd", "signal", "hist"]
    assert m.empty


def test_bollinger_handles_short_series():
    s = pd.Series([100.0, 101.0, 99.5, 102.1, 100.8])
    out = bollinger(s, period=20)
    assert len(out) == len(s)
    assert not out["mid"].isna().all()


def test_normalize_flattens_multilevel_columns():
    s = _close_series()
    multi = pd.DataFrame(
        {
            ("Open", "BTC-USD"): s,
            ("High", "BTC-USD"): s + 1,
            ("Low", "BTC-USD"): s - 1,
            ("Close", "BTC-USD"): s,
            ("Volume", "BTC-USD"): s * 0 + 10,
        }
    )
    flat = normalize_ohlcv(multi, "BTC-USD")
    assert list(flat.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(flat["Close"], pd.Series)


def test_classify_regime_handles_empty_and_dataframe_close():
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    assert classify_regime(empty).regime == RegimeLabel.RANGING

    s = _close_series(120)
    df = pd.DataFrame({"Close": s, "High": s + 1, "Low": s - 1})
    res = classify_regime(df)
    assert res.regime in set(RegimeLabel)
    assert res.adx >= 0


@pytest.mark.parametrize("n", [0, 1, 5, 25, 200])
def test_indicators_never_raise_value_error(n):
    s = _close_series(max(n, 1)).iloc[:n] if n else pd.Series([], dtype=float)
    rsi(s)
    bollinger(s)
    macd(s)
