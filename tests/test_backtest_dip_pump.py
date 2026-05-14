"""Replay the dip_pump engine over deterministic synthetic OHLCV.

We construct a ramp UP (100 → 130) followed by a ramp DOWN (130 → 95)
so the detector should fire BUY during the up-ramp and SELL during the
down-ramp. With small bar-to-bar moves, at least one stop hit and one
target hit are expected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monte.backtest import replay_dip_pump
from monte.backtest.config import BacktestConfig
from monte.backtest.store import BacktestStore


def _ohlcv(closes: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
    df = pd.DataFrame({
        "Open": closes,
        "High": closes * 1.005,
        "Low":  closes * 0.995,
        "Close": closes,
        "Volume": np.full(len(closes), 1000.0),
    }, index=idx)
    return df


@pytest.fixture
def ramp_then_drop():
    up = np.linspace(100.0, 130.0, 260)
    down = np.linspace(130.0, 95.0, 200)
    return _ohlcv(np.concatenate([up, down]))


def test_dip_pump_replay_records_wins_and_losses(bt_paths, monkeypatch, ramp_then_drop):
    monkeypatch.setattr(replay_dip_pump, "load_ohlcv",
                        lambda *a, **kw: ramp_then_drop)
    cfg = BacktestConfig(
        engine="dip_pump", symbols=("FAKE",), timeframes=("1h",),
        start_date="2024-01-01", end_date="2024-12-31", timeout_bars=24,
    )
    with BacktestStore.open(cfg) as store:
        run_id = store.start_run(cfg)
        n = replay_dip_pump.run(cfg, store, run_id)
        store.finish_run(run_id, n_trades=n)
        rows = store.conn().execute(
            "SELECT win_loss_scratch, direction FROM trades WHERE run_id = ?",
            (run_id,),
        ).fetchall()

    assert n > 0, "expected at least one BUY/SELL signal across the ramp"
    outcomes = {wl for wl, _ in rows}
    directions = {d for _, d in rows}
    assert "win" in outcomes or "loss" in outcomes
    assert directions & {"long", "short"}, "expected at least one directional trade"


def test_dip_pump_replay_skips_when_no_history(bt_paths, monkeypatch):
    monkeypatch.setattr(replay_dip_pump, "load_ohlcv",
                        lambda *a, **kw: pd.DataFrame())
    cfg = BacktestConfig(
        engine="dip_pump", symbols=("EMPTY",), timeframes=("1h",),
        start_date="2024-01-01", end_date="2024-01-31",
    )
    with BacktestStore.open(cfg) as store:
        run_id = store.start_run(cfg)
        n = replay_dip_pump.run(cfg, store, run_id)
        store.finish_run(run_id, n_trades=n)
    assert n == 0
