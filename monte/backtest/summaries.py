"""KPI DataFrames consumed by ``app/pages/12_Backtest_Results.py``.

Helpers are pure SQL readers — they never write. Each returns a
``pandas.DataFrame`` so the Streamlit page can ``st.dataframe`` it
directly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from monte.backtest.config import default_db_path
from monte.backtest.store import SCHEMA_SQL

# Expected column sets for each reader. Used to return a well-shaped empty
# DataFrame when the underlying tables are missing or the query raises an
# OperationalError, so the Streamlit page can render its empty state instead
# of throwing a red traceback.
_RUNS_COLS = (
    "run_id", "engine", "fixture_mode", "ts_started", "ts_finished",
    "start_date", "end_date", "n_trades", "status", "error",
)
_OVERVIEW_COLS = (
    "engine", "fixture_mode", "n_trades", "wins", "losses",
    "avg_pnl_pct", "win_rate",
)
_SIGNAL_COLS = (
    "engine", "signal_name", "bucket", "n", "wins", "losses", "scratches",
    "win_rate", "avg_pnl_pct", "sharpe",
)
_LEDGER_COLS = (
    "trade_id", "run_id", "engine", "fixture_mode", "symbol", "timeframe",
    "horizon", "ts_entry", "ts_exit", "action", "direction", "entry_price",
    "exit_price", "pnl_pct", "win_loss_scratch", "exit_reason", "confidence",
)


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open the backtest SQLite DB and guarantee its schema exists.

    The reader path historically called raw ``sqlite3.connect``, which silently
    creates an empty database file when the path doesn't exist yet. Subsequent
    ``SELECT`` queries against ``runs`` / ``trades`` / ``signal_buckets`` then
    blew up with ``no such table: runs`` on every fresh deploy that hadn't
    written a backtest yet. Running the schema (``CREATE TABLE IF NOT EXISTS``,
    so cheap and idempotent) on every connection means the empty-state UI
    renders instead of a traceback.
    """
    path = Path(db_path) if db_path is not None else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_SQL)
    return conn


def _empty(cols: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def _is_missing_table(exc: BaseException) -> bool:
    """True if the exception is the specific 'no such table' SQLite error.

    We deliberately only swallow this narrow case so that unrelated DB
    faults (malformed SQL, corruption, locking) still surface with their
    real traceback instead of silently rendering as empty results.
    """
    msg = str(exc).lower()
    if "no such table" in msg:
        return True
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    return cause is not None and "no such table" in str(cause).lower()


def list_runs(*, db_path: Path | None = None, limit: int = 200) -> pd.DataFrame:
    try:
        with _connect(db_path) as conn:
            return pd.read_sql_query(
                "SELECT run_id, engine, fixture_mode, ts_started, ts_finished, "
                "       start_date, end_date, n_trades, status, error "
                "FROM runs ORDER BY ts_started DESC LIMIT ?",
                conn, params=(limit,),
            )
    except (sqlite3.OperationalError, pd.errors.DatabaseError) as exc:
        if _is_missing_table(exc):
            return _empty(_RUNS_COLS)
        raise


def overview_kpis(*, db_path: Path | None = None, run_ids: list[str] | None = None
                  ) -> pd.DataFrame:
    where, params = _where_runs(run_ids)
    try:
        with _connect(db_path) as conn:
            return pd.read_sql_query(
                f"""
                SELECT engine, fixture_mode,
                       COUNT(*) AS n_trades,
                       SUM(CASE WHEN win_loss_scratch='win' THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN win_loss_scratch='loss' THEN 1 ELSE 0 END) AS losses,
                       AVG(pnl_pct) AS avg_pnl_pct,
                       100.0 * SUM(CASE WHEN win_loss_scratch='win' THEN 1 ELSE 0 END) /
                           NULLIF(SUM(CASE WHEN win_loss_scratch IN ('win','loss','scratch') THEN 1 ELSE 0 END), 0)
                           AS win_rate
                FROM trades {where}
                GROUP BY engine, fixture_mode
                ORDER BY engine, fixture_mode
                """,
                conn, params=params,
            )
    except (sqlite3.OperationalError, pd.errors.DatabaseError) as exc:
        if _is_missing_table(exc):
            return _empty(_OVERVIEW_COLS)
        raise


def signal_table(*, db_path: Path | None = None, run_ids: list[str] | None = None,
                 engine: str | None = None) -> pd.DataFrame:
    clauses, params = [], []
    if run_ids:
        clauses.append(f"run_id IN ({','.join('?'*len(run_ids))})")
        params.extend(run_ids)
    if engine:
        clauses.append("engine = ?")
        params.append(engine)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    try:
        with _connect(db_path) as conn:
            return pd.read_sql_query(
                f"""
                SELECT engine, signal_name, bucket, n, wins, losses, scratches,
                       win_rate, avg_pnl_pct, sharpe
                FROM signal_buckets {where}
                ORDER BY win_rate DESC, n DESC
                """,
                conn, params=params,
            )
    except (sqlite3.OperationalError, pd.errors.DatabaseError) as exc:
        if _is_missing_table(exc):
            return _empty(_SIGNAL_COLS)
        raise


def trade_ledger(*, db_path: Path | None = None, run_ids: list[str] | None = None,
                 engine: str | None = None, symbol: str | None = None,
                 limit: int = 2000) -> pd.DataFrame:
    clauses, params = [], []
    if run_ids:
        clauses.append(f"run_id IN ({','.join('?'*len(run_ids))})")
        params.extend(run_ids)
    if engine:
        clauses.append("engine = ?")
        params.append(engine)
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    try:
        with _connect(db_path) as conn:
            return pd.read_sql_query(
                f"""
                SELECT trade_id, run_id, engine, fixture_mode, symbol, timeframe, horizon,
                       ts_entry, ts_exit, action, direction, entry_price, exit_price,
                       pnl_pct, win_loss_scratch, exit_reason, confidence
                FROM trades {where}
                ORDER BY ts_entry DESC LIMIT ?
                """,
                conn, params=params,
            )
    except (sqlite3.OperationalError, pd.errors.DatabaseError) as exc:
        if _is_missing_table(exc):
            return _empty(_LEDGER_COLS)
        raise


def _where_runs(run_ids: list[str] | None) -> tuple[str, list]:
    if not run_ids:
        return "", []
    return f"WHERE run_id IN ({','.join('?'*len(run_ids))})", list(run_ids)
