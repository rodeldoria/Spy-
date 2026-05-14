"""The bootstrap bridge turns backtest trades into pattern-tracker rows.

We synthesise a tiny price series (sharp uptrend across one bar) so the
kalshi backtest produces a deterministic mix of YES-winning and
NO-winning trades, then run ``bootstrap_from_backtest`` and assert the
pattern tracker now sees settled outcomes with hit-rates that move at
least one signature off ×1.00.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monte.backtest import replay_kalshi
from monte.backtest.config import BacktestConfig
from monte.backtest.scoring import aggregate_buckets
from monte.backtest.store import BacktestStore
from monte.learning import pattern_tracker as ptrack
from monte.learning.bootstrap import bootstrap_from_backtest


def _ohlcv(closes: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame({
        "Open": closes, "High": closes * 1.001, "Low": closes * 0.999,
        "Close": closes, "Volume": np.full(len(closes), 1000.0),
    }, index=idx)


@pytest.fixture
def isolated_tracker(tmp_path, monkeypatch):
    """Point the pattern tracker at a tmp JSONL so tests don't pollute
    the user's ``~/.monte/pattern_tracker.jsonl``."""
    path = tmp_path / "pattern_tracker.jsonl"
    monkeypatch.setenv("MONTE_PATTERN_TRACKER_PATH", str(path))
    return path


def _run_synthetic_backtest(cfg: BacktestConfig, df: pd.DataFrame,
                            monkeypatch) -> str:
    monkeypatch.setattr(replay_kalshi, "load_ohlcv", lambda *a, **kw: df)
    with BacktestStore.open(cfg) as store:
        run_id = store.start_run(cfg)
        n = replay_kalshi.run(cfg, store, run_id)
        aggregate_buckets(store, run_id, "kalshi")
        store.finish_run(run_id, n_trades=n, status="ok")
    return run_id


def test_bootstrap_mirrors_kalshi_trades_into_tracker(
    bt_paths, isolated_tracker, monkeypatch,
):
    rng = np.random.default_rng(11)
    closes = 100.0 + rng.normal(0, 0.05, 91)
    closes = np.concatenate([closes, [102.0, 102.5, 103.0, 103.5]])
    df = _ohlcv(closes)

    cfg = BacktestConfig(
        engine="kalshi", symbols=("FAKE",), timeframes=("1h",),
        start_date="2024-01-01", end_date="2024-12-31", seed=11,
    )
    run_id = _run_synthetic_backtest(cfg, df, monkeypatch)

    # Before bootstrap, tracker is empty.
    rep_before = ptrack.build_report()
    assert rep_before.n_verdicts == 0
    assert rep_before.n_outcomes == 0

    summary = bootstrap_from_backtest(run_id=run_id, db_path=bt_paths["db"])
    assert summary["verdicts_written"] > 0
    assert summary["outcomes_written"] == summary["verdicts_written"]

    # After bootstrap, every backtest trade has a paired verdict + outcome.
    rep = ptrack.build_report()
    assert rep.n_verdicts == summary["verdicts_written"]
    assert rep.n_outcomes == summary["outcomes_written"]
    assert rep.overall_hit_rate is not None
    assert 0.0 <= rep.overall_hit_rate <= 1.0

    # At least one signature should accumulate enough settled outcomes to
    # move its confidence multiplier off the ×1.00 "no data" baseline.
    moved = [
        sig for sig, stats in rep.by_signature.items()
        if stats.n_settled >= 5
    ]
    assert moved, "expected ≥1 signature with ≥5 settled outcomes after bootstrap"
    for sig in moved:
        mult, label = ptrack.confidence_multiplier(sig)
        assert 0.6 <= mult <= 1.4
        assert "learned" in label


def test_bootstrap_is_idempotent(bt_paths, isolated_tracker, monkeypatch):
    """Re-running bootstrap on the same run_id should not double-count."""
    closes = np.concatenate([
        100.0 + np.linspace(-0.1, 0.1, 91),
        [102.0, 102.5, 103.0, 103.5],
    ])
    df = _ohlcv(closes)

    cfg = BacktestConfig(
        engine="kalshi", symbols=("FAKE",), timeframes=("1h",),
        start_date="2024-01-01", end_date="2024-12-31", seed=7,
    )
    run_id = _run_synthetic_backtest(cfg, df, monkeypatch)

    first = bootstrap_from_backtest(run_id=run_id, db_path=bt_paths["db"])
    second = bootstrap_from_backtest(run_id=run_id, db_path=bt_paths["db"])

    assert first["verdicts_written"] > 0
    assert second["verdicts_written"] == 0
    assert second["skipped_dup"] == first["verdicts_written"]


def test_bootstrap_with_no_db_returns_empty(tmp_path, isolated_tracker):
    """Missing backtest DB shouldn't crash — just return zeroes."""
    summary = bootstrap_from_backtest(
        run_id=None, db_path=tmp_path / "nonexistent.db",
    )
    assert summary["verdicts_written"] == 0
    assert summary["outcomes_written"] == 0


def test_bootstrap_latest_run_when_run_id_omitted(
    bt_paths, isolated_tracker, monkeypatch,
):
    """Calling without a run_id should pick the most recent ok kalshi run."""
    rng = np.random.default_rng(3)
    closes = 100.0 + rng.normal(0, 0.05, 91)
    closes = np.concatenate([closes, [102.0, 102.5, 103.0, 103.5]])
    df = _ohlcv(closes)
    cfg = BacktestConfig(
        engine="kalshi", symbols=("FAKE",), timeframes=("1h",),
        start_date="2024-01-01", end_date="2024-12-31", seed=3,
    )
    _run_synthetic_backtest(cfg, df, monkeypatch)

    summary = bootstrap_from_backtest(db_path=bt_paths["db"])
    assert summary["verdicts_written"] > 0
    assert summary["run_id"] is not None
