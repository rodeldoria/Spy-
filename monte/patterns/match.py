"""Stub pattern match module."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from monte.patterns.vector_store import PatternStore


@dataclass
class MatchResult:
    cold_start: bool
    pattern_score: float
    win_rate: float
    mean_fwd_5: float
    std_fwd_5: float
    mean_fwd_20: float
    std_fwd_20: float
    k: int


def find_similar(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    k: int = 20,
    store: PatternStore | None = None,
    window: int = 60,
) -> MatchResult:
    if store is None:
        store = PatternStore()
    n = store.count(symbol, timeframe)
    if n == 0:
        return MatchResult(
            cold_start=True,
            pattern_score=0.0,
            win_rate=0.0,
            mean_fwd_5=0.0,
            std_fwd_5=0.0,
            mean_fwd_20=0.0,
            std_fwd_20=0.0,
            k=0,
        )
    return MatchResult(
        cold_start=False,
        pattern_score=0.0,
        win_rate=0.5,
        mean_fwd_5=0.0,
        std_fwd_5=0.01,
        mean_fwd_20=0.0,
        std_fwd_20=0.02,
        k=min(k, n),
    )
