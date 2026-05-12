from __future__ import annotations

import pandas as pd
import pytest

from app.signals import sma_crossover
from monte.strategies.signals import action_from_score


def test_scenarios_match_golden(sma_cross_scenarios):
    for case in sma_cross_scenarios:
        closes = pd.Series(case["closes"], dtype=float)
        sig = sma_crossover(closes, fast=case["fast"], slow=case["slow"])
        assert sig.score == pytest.approx(case["expected_score"]), case["name"]
        assert (
            action_from_score(sig.score).value == case["expected_action"]
        ), case["name"]


def test_too_few_closes_raises():
    with pytest.raises(ValueError, match="at least"):
        sma_crossover(pd.Series([1.0, 2.0, 3.0]), fast=2, slow=5)


def test_fast_must_be_less_than_slow():
    with pytest.raises(ValueError, match="must be"):
        sma_crossover(pd.Series([1.0] * 10), fast=5, slow=5)


def test_meta_carries_sma_values():
    sig = sma_crossover(
        pd.Series([1, 2, 3, 4, 5, 6, 7], dtype=float), fast=3, slow=5
    )
    assert sig.meta["fast"] == 3
    assert sig.meta["slow"] == 5
    assert sig.meta["fast_sma"] == pytest.approx(6.0)
    assert sig.meta["slow_sma"] == pytest.approx(5.0)
    assert sig.name == "SMA cross"
