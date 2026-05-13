"""Tests for Monte Edge — confluence, macro filter, tier classification."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monte.signals.dip_pump import Alert
from monte.strategy.monte_edge import (
    ACT_NOW_CONFIDENCE,
    ACT_NOW_CONFLUENCE,
    CONFLUENCE_MIN,
    EdgeTier,
    REASONING_LIBRARY,
    _count_confluence,
    _spy_above_200ma,
    confidence_scaled_risk_pct,
    evaluate,
    tier_from_signal,
)
from monte.strategies.signals import Action
from monte.signals.horizon import Horizon


# ---------- confluence ----------

def test_confluence_counts_same_direction_only():
    contribs = [
        {"name": "RSI", "score": 0.4},
        {"name": "MACD", "score": 0.3},
        {"name": "BB", "score": -0.5},  # opposes
        {"name": "Trend", "score": 0.2},
        {"name": "Regime", "score": 0.0},  # ignored, abs <= 0.05
    ]
    assert _count_confluence(contribs, score=0.3) == 3


def test_confluence_zero_when_score_zero():
    contribs = [{"name": "RSI", "score": 0.5}]
    assert _count_confluence(contribs, score=0.0) == 0


def test_confluence_filters_noise_under_threshold():
    contribs = [
        {"name": "RSI", "score": 0.03},   # below 0.05 noise floor
        {"name": "MACD", "score": 0.4},
    ]
    assert _count_confluence(contribs, score=0.3) == 1


# ---------- macro filter ----------

def test_spy_above_200ma_true():
    closes = list(range(100, 350))  # strongly rising
    df = pd.DataFrame({"Close": closes})
    assert _spy_above_200ma(df) is True


def test_spy_above_200ma_false():
    closes = list(range(350, 100, -1))  # strongly falling
    df = pd.DataFrame({"Close": closes})
    assert _spy_above_200ma(df) is False


def test_spy_above_200ma_unknown_when_insufficient_data():
    closes = list(range(100, 110))
    df = pd.DataFrame({"Close": closes})
    assert _spy_above_200ma(df) is None
    assert _spy_above_200ma(None) is None
    assert _spy_above_200ma(pd.DataFrame()) is None


# ---------- tier classification ----------

def _alert(action=Action.BUY, confidence=80, score=0.5):
    return Alert(
        symbol="BTC-USD",
        timeframe="1h",
        action=action,
        confidence=confidence,
        score=score,
        entry=100.0,
        stop=95.0,
        target=110.0,
        rr=2.0,
        horizon=Horizon.SWING,
    )


def test_hold_is_stand_down():
    a = _alert(action=Action.HOLD, confidence=0, score=0)
    assert tier_from_signal(a, confluence=5, macro_aligned=True, drawdown_halt=False) is EdgeTier.STAND_DOWN


def test_drawdown_halt_forces_stand_down():
    a = _alert(confidence=99, score=0.95)
    assert tier_from_signal(a, confluence=5, macro_aligned=True, drawdown_halt=True) is EdgeTier.STAND_DOWN


def test_macro_mismatch_demotes_unless_overwhelming():
    a = _alert(confidence=90, score=0.9)
    # mismatch, only 4 confluence → stand down
    assert tier_from_signal(a, confluence=4, macro_aligned=False, drawdown_halt=False) is EdgeTier.STAND_DOWN
    # mismatch, 5 confluence → still allowed (we trust full-stack agreement)
    t = tier_from_signal(a, confluence=5, macro_aligned=False, drawdown_halt=False)
    assert t in {EdgeTier.WATCH, EdgeTier.ACT_NOW}


def test_act_now_requires_confidence_and_confluence():
    a = _alert(confidence=ACT_NOW_CONFIDENCE + 1, score=0.9)
    t = tier_from_signal(
        a, confluence=ACT_NOW_CONFLUENCE, macro_aligned=True, drawdown_halt=False,
    )
    assert t is EdgeTier.ACT_NOW


def test_watch_when_confidence_mid():
    a = _alert(confidence=65, score=0.4)
    t = tier_from_signal(
        a, confluence=CONFLUENCE_MIN, macro_aligned=True, drawdown_halt=False,
    )
    assert t is EdgeTier.WATCH


def test_below_confluence_minimum_stand_down():
    a = _alert(confidence=90, score=0.9)
    t = tier_from_signal(
        a, confluence=CONFLUENCE_MIN - 1, macro_aligned=True, drawdown_halt=False,
    )
    assert t is EdgeTier.STAND_DOWN


# ---------- confidence-scaled risk ----------

def test_risk_at_floor_for_min_confidence():
    assert confidence_scaled_risk_pct(50.0) == pytest.approx(0.005, rel=0.01)


def test_risk_at_ceiling_for_max_confidence():
    assert confidence_scaled_risk_pct(100.0) == pytest.approx(0.015, rel=0.01)


def test_risk_halved_at_5pct_drawdown():
    base = confidence_scaled_risk_pct(80.0, drawdown_pct=0.0)
    half = confidence_scaled_risk_pct(80.0, drawdown_pct=-0.06)
    assert half == pytest.approx(base * 0.5, rel=0.05)


def test_risk_zero_at_10pct_drawdown():
    assert confidence_scaled_risk_pct(95.0, drawdown_pct=-0.10) == 0.0
    assert confidence_scaled_risk_pct(95.0, drawdown_pct=-0.20) == 0.0


# ---------- end-to-end evaluate ----------

def _synthetic_uptrend(n=300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    closes = np.cumsum(rng.normal(0.4, 1.0, n)) + 100.0
    closes = np.maximum(closes, 1.0)
    highs = closes + rng.uniform(0.5, 1.5, n)
    lows = closes - rng.uniform(0.5, 1.5, n)
    opens = closes - rng.normal(0.0, 0.5, n)
    vols = rng.integers(1000, 5000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )


def test_evaluate_returns_edge_signal_with_reasoning():
    df = _synthetic_uptrend()
    spy_df = _synthetic_uptrend(n=300)
    sig = evaluate(df, symbol="BTC-USD", timeframe="1h", spy_daily=spy_df)
    assert sig.symbol == "BTC-USD"
    assert sig.tier in set(EdgeTier)
    assert 0 <= sig.confluence <= 5
    assert isinstance(sig.reasoning, str)
    assert sig.macro_note  # something was set


def test_evaluate_with_unknown_macro_does_not_crash():
    df = _synthetic_uptrend()
    sig = evaluate(df, symbol="BTC-USD", timeframe="1h", spy_daily=None)
    assert sig.macro_aligned is None
