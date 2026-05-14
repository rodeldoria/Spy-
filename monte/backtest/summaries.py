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


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    return sqlite3.connect(db_path or default_db_path())


def list_runs(*, db_path: Path | None = None, limit: int = 200) -> pd.DataFrame:
    with _connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT run_id, engine, fixture_mode, ts_started, ts_finished, "
            "       start_date, end_date, n_trades, status, error "
            "FROM runs ORDER BY ts_started DESC LIMIT ?",
            conn, params=(limit,),
        )


def overview_kpis(*, db_path: Path | None = None, run_ids: list[str] | None = None
                  ) -> pd.DataFrame:
    where, params = _where_runs(run_ids)
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


def _where_runs(run_ids: list[str] | None) -> tuple[str, list]:
    if not run_ids:
        return "", []
    return f"WHERE run_id IN ({','.join('?'*len(run_ids))})", list(run_ids)
