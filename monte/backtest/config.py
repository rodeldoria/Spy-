"""Configuration dataclass for a single backtest run."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal

EngineKind = Literal["dip_pump", "kalshi", "triangulation"]
FixtureMode = Literal["neutral", "seeded_random"]

DEFAULT_SYMBOLS: tuple[str, ...] = ("BTC", "ETH", "SOL")
DEFAULT_TIMEFRAMES: tuple[str, ...] = ("1h", "1d")
DEFAULT_TIMEOUT_BARS = 48
DEFAULT_MIN_EDGE_PP = 4.0
DEFAULT_MIN_EV_CENTS = 2.0
DEFAULT_SEED = 42

WIN_THRESHOLD_PCT = 0.25
LOSS_THRESHOLD_PCT = -0.25


def _home() -> Path:
    return Path(os.path.expanduser("~"))


def default_db_path() -> Path:
    return Path(os.environ.get("MONTE_BACKTEST_DB", _home() / ".monte" / "backtest.db"))


def default_jsonl_dir() -> Path:
    return Path(os.environ.get("MONTE_BACKTEST_JSONL_DIR", _home() / ".monte" / "bt_jsonl"))


def default_cache_dir() -> Path:
    return Path(os.environ.get("MONTE_BACKTEST_CACHE_DIR", _home() / ".monte" / "bt_cache"))


@dataclass(frozen=True)
class BacktestConfig:
    """Frozen description of one engine run.

    Triangulation runs are scheduled as TWO BacktestConfigs by the runner —
    one per FixtureMode — so the SQLite ``runs`` table holds them side by
    side and the results page can compare neutral vs seeded_random.
    """

    engine: EngineKind
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES
    start_date: str = "2024-01-01"
    end_date: str = field(default_factory=lambda: date.today().isoformat())
    seed: int = DEFAULT_SEED
    timeout_bars: int = DEFAULT_TIMEOUT_BARS
    min_edge_pp: float = DEFAULT_MIN_EDGE_PP
    min_ev_cents: float = DEFAULT_MIN_EV_CENTS
    fixture_mode: FixtureMode | None = None
    db_path: Path = field(default_factory=default_db_path)
    jsonl_dir: Path = field(default_factory=default_jsonl_dir)
    cache_dir: Path = field(default_factory=default_cache_dir)

    def __post_init__(self) -> None:
        if self.engine == "triangulation" and self.fixture_mode is None:
            raise ValueError("triangulation runs require fixture_mode (neutral|seeded_random)")
        for fld in ("start_date", "end_date"):
            datetime.fromisoformat(getattr(self, fld))

    def to_json(self) -> str:
        return json.dumps(
            {
                "engine": self.engine,
                "symbols": list(self.symbols),
                "timeframes": list(self.timeframes),
                "start_date": self.start_date,
                "end_date": self.end_date,
                "seed": self.seed,
                "timeout_bars": self.timeout_bars,
                "min_edge_pp": self.min_edge_pp,
                "min_ev_cents": self.min_ev_cents,
                "fixture_mode": self.fixture_mode,
            },
            sort_keys=True,
        )
