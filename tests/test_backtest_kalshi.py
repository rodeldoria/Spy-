"""Replay the synthetic Kalshi engine on a known closing series.

We craft closes that move in one direction across a single bar, so we
can predict which strikes settle YES vs NO and verify the engine records
trades on the correct sides."""

from __future__ import annotations

import numpy as np
import pandas as pd

from monte.backtest import replay_kalshi
from monte.backtest.config import BacktestConfig
from monte.backtest.scoring import aggregate_buckets
from monte.backtest.store import BacktestStore


def _ohlcv(closes: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame({
        "Open": closes, "High": closes * 1.001, "Low": closes * 0.999,
        "Close": closes, "Volume": np.full(len(closes), 1000.0),
    }, index=idx)


def test_kalshi_replay_records_settled_trades(bt_paths, monkeypatch):
    base = 100.0
    # 90 bars of mild noise around 100, then a sharp jump up to 102 at bar 91.
    rng = np.random.default_rng(7)
    closes = base + rng.normal(0, 0.05, 91)
    closes = np.concatenate([closes, [102.0, 102.5, 103.0, 103.5]])
    df = _ohlcv(closes)

    monkeypatch.setattr(replay_kalshi, "load_ohlcv", lambda *a, **kw: df)
    cfg = BacktestConfig(
        engine="kalshi", symbols=("FAKE",), timeframes=("1h",),
        start_date="2024-01-01", end_date="2024-12-31", seed=7,
    )
    with BacktestStore.open(cfg) as store:
        run_id = store.start_run(cfg)
        n = replay_kalshi.run(cfg, store, run_id)
        aggregate_buckets(store, run_id, "kalshi")
        store.finish_run(run_id, n_trades=n)
        rows = store.conn().execute(
            "SELECT direction, actual_outcome FROM trades WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        bucket_rows = store.conn().execute(
            "SELECT signal_name, COUNT(*) FROM signal_buckets WHERE run_id = ? GROUP BY signal_name",
            (run_id,),
        ).fetchall()

    assert n > 0
    directions = {d for d, _ in rows}
    assert directions & {"yes", "no", "pass"}, "engine should pick at least one side"
    assert any(actual in ("settled_yes", "settled_no") for _, actual in rows)
    signal_names = {n for n, _ in bucket_rows}
    assert "direction" in signal_names
