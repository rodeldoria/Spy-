"""Stub pattern encoder."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class EncodedWindow:
    vector: list[float] = field(default_factory=list)


def encode_window(df: pd.DataFrame, window: int = 60) -> EncodedWindow:
    """Encode the last `window` bars as a normalised return vector."""
    try:
        close = df["Close"].tail(window)
        if len(close) < 2:
            return EncodedWindow(vector=[0.0] * window)
        returns = close.pct_change().fillna(0.0).tolist()
        vec = returns[-window:]
        if len(vec) < window:
            vec = [0.0] * (window - len(vec)) + vec
        return EncodedWindow(vector=vec)
    except Exception:
        return EncodedWindow(vector=[0.0] * window)
