"""Normalize OHLCV frames returned by data providers.

yfinance often returns a DataFrame with a `MultiIndex` columns axis (even for a
single ticker), e.g. columns like ``('Close', 'BTC-USD')``. Downstream code in
this project expects a flat schema with ``Open/High/Low/Close/Volume`` columns
and Series-typed selections. This helper guarantees that contract.
"""
from __future__ import annotations

import pandas as pd


_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def normalize_ohlcv(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Return a copy of `df` with flat OHLCV columns.

    - Drops a multi-level column axis by selecting the symbol level if present,
      otherwise the first level.
    - Coerces remaining OHLCV columns to numeric, rows with all-NaN are dropped.
    - Returns an empty frame with OHLCV columns if input is empty.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=_OHLCV_COLS)

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        levels = out.columns.nlevels
        flattened = None
        if symbol is not None:
            for lvl in range(levels):
                vals = out.columns.get_level_values(lvl)
                if symbol in vals:
                    flattened = out.xs(symbol, axis=1, level=lvl, drop_level=True)
                    break
        if flattened is None:
            # fall back: keep the outermost level (typically OHLCV names)
            outer = out.columns.get_level_values(0)
            inner = out.columns.get_level_values(levels - 1)
            chosen = outer if any(c in _OHLCV_COLS for c in outer) else inner
            new = pd.DataFrame(out.values, index=out.index, columns=chosen)
            flattened = new.loc[:, ~new.columns.duplicated()]
        out = flattened

    out.columns = [str(c) for c in out.columns]
    for col in _OHLCV_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "Close" in out.columns:
        out = out.dropna(subset=["Close"])
    return out
