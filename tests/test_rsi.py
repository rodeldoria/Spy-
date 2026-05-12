from __future__ import annotations

import pandas as pd
import pytest

from app.signals import rsi
from monte.strategies.signals import action_from_score


def test_scenarios_match_golden(rsi_scenarios):
    for case in rsi_scenarios:
        closes = pd.Series(case["closes"], dtype=float)
        sig = rsi(closes, period=case["period"])
        assert sig.meta["rsi"] == pytest.approx(
            case["expected_rsi"], abs=1e-9
        ), case["name"]
        assert sig.score == pytest.approx(case["expected_score"]), case["name"]
        assert (
            action_from_score(sig.score).value == case["expected_action"]
        ), case["name"]


def test_too_few_closes_raises():
    with pytest.raises(ValueError, match="at least"):
        rsi(pd.Series([1.0, 2.0, 3.0]), period=4)


def test_period_must_be_at_least_two():
    with pytest.raises(ValueError, match=">= 2"):
        rsi(pd.Series([1.0] * 10), period=1)


def test_threshold_ordering_enforced():
    with pytest.raises(ValueError, match="thresholds must satisfy"):
        rsi(
            pd.Series([1.0] * 10),
            period=4,
            oversold=70.0,
            overbought=30.0,
        )


def test_signal_metadata_carries_intermediates():
    sig = rsi(
        pd.Series([10, 11, 12, 13, 12], dtype=float), period=4
    )
    assert sig.name == "RSI"
    assert sig.meta["period"] == 4
    assert sig.meta["rsi"] == pytest.approx(75.0)
    assert sig.meta["avg_gain"] == pytest.approx(0.75)
    assert sig.meta["avg_loss"] == pytest.approx(0.25)
