"""MA crossover detection — golden cross / death cross.

A golden cross is when a shorter SMA (e.g. SMA20) closes above a longer SMA
(e.g. SMA50) after being below. A death cross is the inverse. These are
classic swing-trading triggers — they don't fire often, and when they do
they're worth pushing to the user's phone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


CrossKind = Literal["golden", "death", None]


@dataclass
class MACross:
    kind: CrossKind            # "golden" | "death" | None
    bars_ago: int              # 0 = on the most recent bar; -1 = never crossed
    fast_value: float          # current SMA(fast) value
    slow_value: float          # current SMA(slow) value
    fast_period: int
    slow_period: int

    @property
    def fired_recently(self) -> bool:
        """True when a cross occurred in the last 5 bars."""
        return self.kind is not None and 0 <= self.bars_ago <= 5

    def label(self) -> str:
        if not self.kind:
            return "no recent MA cross"
        prefix = "🟢 Golden Cross" if self.kind == "golden" else "🔴 Death Cross"
        if self.bars_ago == 0:
            when = "this bar"
        elif self.bars_ago == 1:
            when = "1 bar ago"
        else:
            when = f"{self.bars_ago} bars ago"
        return f"{prefix} · SMA{self.fast_period} {'over' if self.kind=='golden' else 'under'} SMA{self.slow_period} · {when}"


def detect_cross(
    close: pd.Series,
    *,
    fast: int = 20,
    slow: int = 50,
    lookback: int = 30,
) -> MACross:
    """Return the most recent SMA(fast)/SMA(slow) cross within `lookback` bars."""
    if close is None or len(close) < slow + 2:
        return MACross(None, -1, 0.0, 0.0, fast, slow)

    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    diff = (sma_fast - sma_slow).fillna(0.0)

    # Walk forward over the lookback window. A cross fires when the sign
    # changes from positive↔negative, or when the SMAs unstick from being
    # tied (0) and diverge into a non-zero sign — both are "cross" moments
    # the user expects to see flagged.
    n = len(diff)
    start = max(slow, n - lookback)
    prior_sign = 0
    saw_any = False
    last_kind: CrossKind = None
    bars_ago = -1
    for i in range(start, n):
        s = 0 if diff.iloc[i] == 0 else (1 if diff.iloc[i] > 0 else -1)
        if s == 0:
            saw_any = True
            continue
        if saw_any and s != prior_sign:
            last_kind = "golden" if s > 0 else "death"
            bars_ago = (n - 1) - i
        prior_sign = s
        saw_any = True

    return MACross(
        kind=last_kind,
        bars_ago=bars_ago,
        fast_value=float(sma_fast.iloc[-1]) if not pd.isna(sma_fast.iloc[-1]) else 0.0,
        slow_value=float(sma_slow.iloc[-1]) if not pd.isna(sma_slow.iloc[-1]) else 0.0,
        fast_period=fast,
        slow_period=slow,
    )


__all__ = ["MACross", "detect_cross"]
