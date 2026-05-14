"""Replay triangulation under both fixture modes and verify the fixture
sources are correctly tagged in the snapshot."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from monte.backtest import replay_triangulation
from monte.backtest.config import BacktestConfig
from monte.backtest.runner import run_engine_or_all
from monte.backtest.store import BacktestStore


def _ohlcv(closes: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame({
        "Open": closes, "High": closes * 1.001, "Low": closes * 0.999,
        "Close": closes, "Volume": np.full(len(closes), 1000.0),
    }, index=idx)


def test_triangulation_replay_neutral_records_fixture_vote(bt_paths, monkeypatch):
    rng = np.random.default_rng(11)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.05, 130))
    df = _ohlcv(closes)
    monkeypatch.setattr(replay_triangulation, "load_ohlcv", lambda *a, **kw: df)
    cfg = BacktestConfig(
        engine="triangulation", symbols=("FAKE",), timeframes=("1h",),
        start_date="2024-01-01", end_date="2024-12-31",
        fixture_mode="neutral", seed=11,
    )
    with BacktestStore.open(cfg) as store:
        run_id = store.start_run(cfg)
        n = replay_triangulation.run(cfg, store, run_id)
        store.finish_run(run_id, n_trades=n)
        snap_jsons = [r[0] for r in store.conn().execute(
            "SELECT snapshot_json FROM trades WHERE run_id = ?", (run_id,)
        ).fetchall()]

    assert n > 0
    for sj in snap_jsons:
        votes = json.loads(sj)["votes"]
        assert set(votes) >= {"Crowd", "Patterns", "Influencers", "Session", "News"}
        # Fixture-injected votes are tagged source="fixture" by the replay
        assert votes["News"]["source"] == "fixture"
        assert votes["Influencers"]["source"] == "fixture"


def test_triangulation_runs_both_fixtures_side_by_side(bt_paths, monkeypatch):
    rng = np.random.default_rng(13)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.05, 130))
    df = _ohlcv(closes)
    monkeypatch.setattr(replay_triangulation, "load_ohlcv", lambda *a, **kw: df)

    base = BacktestConfig(
        engine="triangulation", symbols=("FAKE",), timeframes=("1h",),
        start_date="2024-01-01", end_date="2024-12-31",
        fixture_mode="neutral", seed=13,
    )
    results = run_engine_or_all(engines=["triangulation"], base_cfg=base)
    assert len(results) == 2
    modes = sorted([r.fixture_mode for r in results])
    assert modes == ["neutral", "seeded_random"]
    assert all(r.status == "ok" for r in results)
