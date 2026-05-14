"""Smoke tests for the SQLite + JSONL mirror writer."""

from __future__ import annotations

import json

from monte.backtest.config import BacktestConfig
from monte.backtest.store import BacktestStore, TradeRow, classify_outcome


def test_classify_outcome_bands():
    assert classify_outcome(None) == "open"
    assert classify_outcome(0.5) == "win"
    assert classify_outcome(-0.5) == "loss"
    assert classify_outcome(0.0) == "scratch"
    assert classify_outcome(0.25) == "win"
    assert classify_outcome(-0.25) == "loss"


def test_store_writes_sqlite_and_jsonl_mirror(bt_paths):
    cfg = BacktestConfig(
        engine="dip_pump",
        symbols=("BTC",),
        timeframes=("1h",),
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    with BacktestStore.open(cfg) as store:
        run_id = store.start_run(cfg)
        for i, pnl in enumerate([1.0, -1.0, 0.05]):
            store.record_trade(TradeRow(
                run_id=run_id, engine="dip_pump", symbol="BTC", timeframe="1h",
                ts_entry=1_700_000_000.0 + i * 3600, ts_exit=1_700_000_000.0 + (i + 1) * 3600,
                action="BUY", direction="long",
                entry_price=100.0, exit_price=100.0 + pnl,
                stop_price=95.0, target_price=105.0,
                exit_reason="target" if pnl > 0 else "stop",
                pnl_pct=pnl,
                confidence=70.0, score=0.5,
                snapshot={"rsi": 30 + i, "regime": "trend"},
            ))
        store.finish_run(run_id, n_trades=3)

    with BacktestStore.open(cfg) as reopen:
        rows = reopen.conn().execute(
            "SELECT win_loss_scratch, pnl_pct FROM trades WHERE run_id = ? ORDER BY ts_entry",
            (run_id,),
        ).fetchall()
    assert rows == [("win", 1.0), ("loss", -1.0), ("scratch", 0.05)]

    jsonl_path = bt_paths["jsonl_dir"] / f"{run_id}.jsonl"
    assert jsonl_path.exists()
    lines = [json.loads(l) for l in jsonl_path.read_text().splitlines()]
    assert len(lines) == 3
    assert {l["win_loss_scratch"] for l in lines} == {"win", "loss", "scratch"}


def test_store_cascade_deletes_trades_with_run(bt_paths):
    cfg = BacktestConfig(engine="dip_pump", symbols=("BTC",), timeframes=("1h",),
                         start_date="2024-01-01", end_date="2024-01-31")
    with BacktestStore.open(cfg) as store:
        run_id = store.start_run(cfg)
        store.record_trade(TradeRow(
            run_id=run_id, engine="dip_pump", symbol="BTC", timeframe="1h",
            ts_entry=1.0, action="BUY", direction="long",
            entry_price=1.0, snapshot={},
        ))
        store.finish_run(run_id, n_trades=1)
        n_before = store.conn().execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        assert n_before == 1
        store.conn().execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        store.conn().commit()
        n_after = store.conn().execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    assert n_after == 0
