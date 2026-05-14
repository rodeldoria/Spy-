"""Historical OHLCV loader with on-disk parquet cache.

yfinance applies aggressive 429s above ~2k requests/hour. Backtests over
months of data can easily request the same series many times, so cache
once to disk and serve subsequent requests from there.

Cache layout::

    ~/.monte/bt_cache/{SYMBOL}_{INTERVAL}.parquet

The cache stores the full series yfinance is willing to return at that
interval (per ``yfinance.download``'s ``period`` ceiling for each
resolution). ``load_ohlcv`` slices by ``[start_iso, end_iso]``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from monte.backtest.config import default_cache_dir
from monte.data._normalize import normalize_ohlcv

_MAX_PERIOD = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "1h": "730d",
    "1d": "max",
}


def cache_path(symbol: str, interval: str, cache_dir: Path | None = None) -> Path:
    base = cache_dir if cache_dir is not None else default_cache_dir()
    return base / f"{symbol.replace('/', '_')}_{interval}.parquet"


def fetch_full(symbol: str, interval: str) -> pd.DataFrame:
    """Pull the deepest history yfinance allows at this interval. Slow path
    — only called on cache miss."""
    import yfinance as yf

    period = _MAX_PERIOD.get(interval, "60d")
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
        group_by="column",
    )
    df = normalize_ohlcv(df, symbol)
    return df


def load_ohlcv(
    symbol: str,
    interval: str,
    start_iso: str,
    end_iso: str,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return cached OHLCV between ``[start_iso, end_iso]`` (inclusive on
    both ends, naive timestamps). Fetches and caches the full series on
    first miss; subsequent calls slice locally.

    Returns an empty DataFrame on fetch failure (callers should check
    ``df.empty`` rather than catch exceptions)."""
    path = cache_path(symbol, interval, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    if refresh or not path.exists():
        try:
            full = fetch_full(symbol, interval)
            if not full.empty:
                full.to_parquet(path)
        except Exception:
            if not path.exists():
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.read_parquet(path)
    if df.empty:
        return df

    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    start_ts = pd.Timestamp(start_iso)
    end_ts = pd.Timestamp(end_iso)
    if df.index.tz is not None:
        start_ts = start_ts.tz_localize(df.index.tz) if start_ts.tz is None else start_ts
        end_ts = end_ts.tz_localize(df.index.tz) if end_ts.tz is None else end_ts
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()


def realised_vol(closes: pd.Series, *, lookback_bars: int) -> float:
    """Per-bar realised volatility (stdev of log returns) over the last
    ``lookback_bars``. Used to seed the synthetic Kalshi GBM book."""
    if len(closes) < 3:
        return 0.0
    rets = closes.pct_change().dropna().tail(lookback_bars)
    if rets.empty:
        return 0.0
    return float(rets.std())


def iso_to_ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()
