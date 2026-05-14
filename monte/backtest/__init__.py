"""Backtesting pipeline: replay historical data through Monte's decision
engines, persist trade-level outcomes to SQLite + JSONL, and aggregate by
signal bucket so the Streamlit results page can show which setups have
actually made money.

The package is intentionally split into thin modules so individual engines
can be replayed in isolation:

- ``config``        — ``BacktestConfig`` dataclass + defaults
- ``data``          — historical OHLCV loader with on-disk parquet cache
- ``store``         — SQLite schema + JSONL mirror writer
- ``replay_*``      — one module per engine (dip_pump, kalshi, triangulation)
- ``fixtures``      — neutral / seeded-random vote providers for triangulation
- ``scoring``       — signal_buckets aggregation
- ``summaries``     — KPI DataFrames consumed by the Streamlit page
- ``runner``        — in-process driver shared by CLI and Streamlit form
- ``run``           — argparse CLI entrypoint
"""

from monte.backtest.config import BacktestConfig, EngineKind, FixtureMode
from monte.backtest.runner import RunResult, run_one
from monte.backtest.store import BacktestStore

__all__ = [
    "BacktestConfig",
    "BacktestStore",
    "EngineKind",
    "FixtureMode",
    "RunResult",
    "run_one",
]
