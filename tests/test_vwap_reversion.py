from __future__ import annotations

import pandas as pd
import pytest

from app.signals import vwap_reversion
from monte.strategies.signals import action_from_score


def test_scenarios_match_golden(vwap_reversion_scenarios):
    for case in vwap_reversion_scenarios:
        closes = pd.Series(case["closes"], dtype=float)
        volumes = pd.Series(case["volumes"], dtype=float)
        sig = vwap_reversion(closes, volumes)
        assert sig.meta["vwap"] == pytest.approx(case["expected_vwap"]), case[
            "name"
        ]
        assert sig.meta["deviation"] == pytest.approx(
            case["expected_deviation"]
        ), case["name"]
        assert sig.score == pytest.approx(case["expected_score"]), case["name"]
        assert (
            action_from_score(sig.score).value == case["expected_action"]
        ), case["name"]


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        vwap_reversion(pd.Series([1.0, 2.0]), pd.Series([1.0]))


def test_empty_series_raises():
    with pytest.raises(ValueError, match="at least 1 bar"):
        vwap_reversion(pd.Series([], dtype=float), pd.Series([], dtype=float))


def test_negative_volume_raises():
    with pytest.raises(ValueError, match="non-negative"):
        vwap_reversion(pd.Series([1.0, 2.0]), pd.Series([1.0, -1.0]))


def test_zero_total_volume_raises():
    with pytest.raises(ValueError, match="VWAP is undefined"):
        vwap_reversion(pd.Series([1.0, 2.0]), pd.Series([0.0, 0.0]))


def test_threshold_ordering_enforced():
    with pytest.raises(ValueError, match="weak < strong"):
        vwap_reversion(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 1.0]),
            weak_threshold=0.02,
            strong_threshold=0.005,
        )


def test_signal_metadata_carries_intermediates():
    sig = vwap_reversion(
        pd.Series([10, 10, 11], dtype=float),
        pd.Series([1, 1, 1], dtype=float),
    )
    assert sig.name == "VWAP reversion"
    assert sig.meta["vwap"] == pytest.approx(31 / 3)
    assert sig.meta["close"] == 11.0
    assert sig.meta["deviation"] == pytest.approx((11 - 31 / 3) / (31 / 3))
