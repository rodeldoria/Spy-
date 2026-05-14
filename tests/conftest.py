from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sma_cross_scenarios() -> list[dict]:
    return json.loads((FIXTURES / "sma_cross_scenarios.json").read_text())


@pytest.fixture(scope="session")
def rsi_scenarios() -> list[dict]:
    return json.loads((FIXTURES / "rsi_scenarios.json").read_text())


@pytest.fixture(scope="session")
def vwap_reversion_scenarios() -> list[dict]:
    return json.loads((FIXTURES / "vwap_reversion_scenarios.json").read_text())


@pytest.fixture
def bt_paths(tmp_path, monkeypatch):
    """Isolated SQLite DB, JSONL mirror dir, and parquet cache for one
    backtest test. Tests depending on this fixture never touch ``~/.monte``."""
    db = tmp_path / "bt.db"
    jsonl_dir = tmp_path / "bt_jsonl"
    cache_dir = tmp_path / "bt_cache"
    monkeypatch.setenv("MONTE_BACKTEST_DB", str(db))
    monkeypatch.setenv("MONTE_BACKTEST_JSONL_DIR", str(jsonl_dir))
    monkeypatch.setenv("MONTE_BACKTEST_CACHE_DIR", str(cache_dir))
    return {"db": db, "jsonl_dir": jsonl_dir, "cache_dir": cache_dir}
