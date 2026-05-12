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
