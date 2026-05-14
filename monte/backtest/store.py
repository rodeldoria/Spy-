"""SQLite + JSONL mirror writer for backtest runs and trades.

The SQLite DB is the authoritative store for queries/aggregation; the JSONL
mirror (one file per ``run_id``) is for human-readable audit and replay.
Every trade is written to both — the writer is the single chokepoint that
keeps them in sync.

Schema lives at module top-level (``SCHEMA_SQL``). Win/loss/scratch bands
mirror ``monte/journal/store.py`` (±0.25%) — keep them in lockstep.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from monte.backtest.config import (
    LOSS_THRESHOLD_PCT,
    WIN_THRESHOLD_PCT,
    BacktestConfig,
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    ts_started    REAL NOT NULL,
    ts_finished   REAL,
    engine        TEXT NOT NULL,
    fixture_mode  TEXT,
    symbols       TEXT NOT NULL,
    start_date    TEXT NOT NULL,
    end_date      TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    seed          INTEGER,
    status        TEXT NOT NULL,
    error         TEXT,
    n_trades      INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_runs_engine_ts ON runs(engine, ts_started DESC);

CREATE TABLE IF NOT EXISTS trades (
    trade_id         TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    engine           TEXT NOT NULL,
    fixture_mode     TEXT,
    symbol           TEXT NOT NULL,
    timeframe        TEXT,
    horizon          TEXT,
    ts_entry         REAL NOT NULL,
    ts_exit          REAL,
    action           TEXT NOT NULL,
    direction        TEXT NOT NULL,
    entry_price      REAL NOT NULL,
    stop_price       REAL,
    target_price     REAL,
    exit_price       REAL,
    exit_reason      TEXT,
    pnl_pct          REAL,
    expected_outcome TEXT,
    actual_outcome   TEXT,
    win_loss_scratch TEXT,
    confidence       REAL,
    score            REAL,
    snapshot_json    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_run    ON trades(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_engine ON trades(engine, symbol);
CREATE INDEX IF NOT EXISTS idx_trades_wl     ON trades(win_loss_scratch);

CREATE TABLE IF NOT EXISTS signal_buckets (
    run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    engine      TEXT NOT NULL,
    signal_name TEXT NOT NULL,
    bucket      TEXT NOT NULL,
    n           INTEGER NOT NULL,
    wins        INTEGER NOT NULL,
    losses      INTEGER NOT NULL,
    scratches   INTEGER NOT NULL,
    win_rate    REAL NOT NULL,
    avg_pnl_pct REAL NOT NULL,
    sharpe      REAL,
    PRIMARY KEY (run_id, signal_name, bucket)
);
"""


def classify_outcome(pnl_pct: float | None) -> str:
    """Reuse the journal's ±0.25% bands so wins/losses are comparable
    across live trading and backtested trades."""
    if pnl_pct is None:
        return "open"
    if pnl_pct >= WIN_THRESHOLD_PCT:
        return "win"
    if pnl_pct <= LOSS_THRESHOLD_PCT:
        return "loss"
    return "scratch"


@dataclass
class TradeRow:
    run_id: str
    engine: str
    symbol: str
    ts_entry: float
    action: str
    direction: str
    entry_price: float
    snapshot: dict[str, Any]
    fixture_mode: str | None = None
    timeframe: str | None = None
    horizon: str | None = None
    ts_exit: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl_pct: float | None = None
    expected_outcome: str | None = None
    actual_outcome: str | None = None
    confidence: float | None = None
    score: float | None = None
    trade_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def win_loss_scratch(self) -> str:
        return classify_outcome(self.pnl_pct)


class BacktestStore:
    """Single chokepoint for SQLite + JSONL mirror writes.

    Open via the context manager so the DB connection and JSONL file handle
    are released together::

        with BacktestStore.open(cfg) as store:
            run_id = store.start_run(cfg)
            store.record_trade(TradeRow(...))
            store.finish_run(run_id, n_trades=N)
    """

    def __init__(self, db_path: Path, jsonl_dir: Path):
        self.db_path = db_path
        self.jsonl_dir = jsonl_dir
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()
        self._jsonl_handles: dict[str, Any] = {}

    @classmethod
    @contextmanager
    def open(cls, cfg: BacktestConfig) -> "Iterator[BacktestStore]":
        store = cls(cfg.db_path, cfg.jsonl_dir)
        try:
            yield store
        finally:
            store.close()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)

    def _jsonl_for(self, run_id: str):
        if run_id not in self._jsonl_handles:
            path = self.jsonl_dir / f"{run_id}.jsonl"
            self._jsonl_handles[run_id] = path.open("a", encoding="utf-8")
        return self._jsonl_handles[run_id]

    def start_run(self, cfg: BacktestConfig) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO runs (run_id, ts_started, engine, fixture_mode, symbols,
                                  start_date, end_date, config_json, seed, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')
                """,
                (
                    run_id,
                    time.time(),
                    cfg.engine,
                    cfg.fixture_mode,
                    json.dumps(list(cfg.symbols)),
                    cfg.start_date,
                    cfg.end_date,
                    cfg.to_json(),
                    cfg.seed,
                ),
            )
        return run_id

    def record_trade(self, row: TradeRow) -> None:
        wl = row.win_loss_scratch()
        snapshot_json = json.dumps(row.snapshot, default=str, sort_keys=True)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO trades (
                    trade_id, run_id, engine, fixture_mode, symbol, timeframe, horizon,
                    ts_entry, ts_exit, action, direction, entry_price, stop_price,
                    target_price, exit_price, exit_reason, pnl_pct, expected_outcome,
                    actual_outcome, win_loss_scratch, confidence, score, snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.trade_id, row.run_id, row.engine, row.fixture_mode, row.symbol,
                    row.timeframe, row.horizon, row.ts_entry, row.ts_exit, row.action,
                    row.direction, row.entry_price, row.stop_price, row.target_price,
                    row.exit_price, row.exit_reason, row.pnl_pct, row.expected_outcome,
                    row.actual_outcome, wl, row.confidence, row.score, snapshot_json,
                ),
            )
        mirror = {**asdict(row), "win_loss_scratch": wl, "snapshot": row.snapshot}
        self._jsonl_for(row.run_id).write(json.dumps(mirror, default=str) + "\n")

    def finish_run(self, run_id: str, *, n_trades: int, status: str = "ok",
                   error: str | None = None) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE runs SET ts_finished = ?, n_trades = ?, status = ?, error = ?
                WHERE run_id = ?
                """,
                (time.time(), n_trades, status, error, run_id),
            )
        h = self._jsonl_handles.pop(run_id, None)
        if h is not None:
            h.flush()
            h.close()

    def upsert_signal_bucket(self, *, run_id: str, engine: str, signal_name: str,
                             bucket: str, n: int, wins: int, losses: int,
                             scratches: int, avg_pnl_pct: float,
                             sharpe: float | None = None) -> None:
        closed = wins + losses + scratches
        win_rate = (wins / closed * 100.0) if closed else 0.0
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO signal_buckets
                    (run_id, engine, signal_name, bucket, n, wins, losses,
                     scratches, win_rate, avg_pnl_pct, sharpe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, signal_name, bucket) DO UPDATE SET
                    n=excluded.n, wins=excluded.wins, losses=excluded.losses,
                    scratches=excluded.scratches, win_rate=excluded.win_rate,
                    avg_pnl_pct=excluded.avg_pnl_pct, sharpe=excluded.sharpe
                """,
                (run_id, engine, signal_name, bucket, n, wins, losses,
                 scratches, win_rate, avg_pnl_pct, sharpe),
            )

    def conn(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        for h in self._jsonl_handles.values():
            h.flush()
            h.close()
        self._jsonl_handles.clear()
        self._conn.close()
