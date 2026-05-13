"""Coverage for horizon classification, detector decisiveness and journal."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monte.indicators.regime import RegimeLabel
from monte.signals.dip_pump import detect
from monte.signals.horizon import Horizon, classify_horizon
from monte.strategies.signals import Action


def _ramp_df(values: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(values), freq="1h")
    s = pd.Series(values, index=idx)
    return pd.DataFrame({"Open": s, "High": s * 1.001, "Low": s * 0.999, "Close": s, "Volume": 1.0})


def test_horizon_intraday_defaults_to_day_trade():
    call = classify_horizon("5m", RegimeLabel.RANGING, adx=18, score=0.3)
    assert call.horizon is Horizon.DAY_TRADE


def test_horizon_swing_with_strong_trend_upgrades_to_long_hold():
    call = classify_horizon("1h", RegimeLabel.TRENDING_UP, adx=30, score=0.5)
    assert call.horizon is Horizon.LONG_HOLD


def test_horizon_daily_in_ranging_regime_downgrades_to_swing():
    call = classify_horizon("1d", RegimeLabel.RANGING, adx=12, score=0.3)
    assert call.horizon is Horizon.SWING


def test_horizon_volatile_intraday_forces_day_trade():
    call = classify_horizon("1h", RegimeLabel.VOLATILE, adx=30, score=0.4)
    assert call.horizon is Horizon.DAY_TRADE


def test_detector_strong_uptrend_returns_buy():
    closes = list(np.linspace(100, 130, 220))
    alert = detect(_ramp_df(closes), symbol="TEST", timeframe="1h")
    assert alert.action in {Action.BUY, Action.STRONG_BUY}
    assert alert.score > 0
    # 5 weighted factors must populate
    assert len(alert.contributions) == 5
    assert set(alert.indicator_snapshot) >= {"rsi", "bb_pctb", "macd_hist", "adx", "atr_pct"}


def test_detector_strong_downtrend_returns_sell():
    closes = list(np.linspace(130, 100, 220))
    alert = detect(_ramp_df(closes), symbol="TEST", timeframe="1h")
    assert alert.action in {Action.SELL, Action.STRONG_SELL}
    assert alert.score < 0


def test_journal_records_and_finds_similar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MONTE_JOURNAL_PATH", str(tmp_path / "journal.jsonl"))
    from monte import journal

    snap = {"rsi": 28, "bb_pctb": 0.05, "macd_hist": -0.4, "adx": 22, "atr_pct": 0.02}
    e = journal.record_entry(
        symbol="BTC-USD", timeframe="1h", action="BUY", horizon="SWING",
        entry=100.0, stop=95.0, target=110.0,
        confidence=70.0, score=0.45, snapshot=snap,
    )
    journal.record_exit(e.id, exit_price=108.0, exit_reason="target")

    history = journal.similar_history(
        symbol="BTC-USD", action="BUY",
        snapshot={"rsi": 29, "bb_pctb": 0.06, "macd_hist": -0.35, "adx": 24, "atr_pct": 0.02},
        k=5,
    )
    assert history.count == 1
    assert history.win_rate == 100.0
    assert history.avg_pnl_pct > 0


def test_journal_summary_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MONTE_JOURNAL_PATH", str(tmp_path / "empty.jsonl"))
    from monte import journal

    s = journal.summary()
    assert s["closed"] == 0
    assert s["wins"] == 0


def test_perplexity_unconfigured_is_graceful(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    from monte.intel import perplexity

    brief = perplexity.fetch_news("BTC-USD", action="BUY")
    assert brief.configured is False
    assert brief.sentiment == "unknown"
    assert brief.aligns_with("BUY") == "neutral"
