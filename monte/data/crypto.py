"""Crypto data fetcher using yfinance."""
from __future__ import annotations

import pandas as pd

from monte.data._normalize import normalize_ohlcv


def get_candles(symbol: str, interval: str, lookback_bars: int = 300) -> pd.DataFrame:
    """Fetch OHLCV candles via yfinance, flattened to single-level columns."""
    try:
        import yfinance as yf
        period_map = {
            "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
            "1h": "730d", "1d": "5y",
        }
        period = period_map.get(interval, "60d")
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            group_by="column",
        )
        df = normalize_ohlcv(df, symbol)
        if df.empty:
            return _empty_df()
        return df.tail(lookback_bars)
    except Exception as e:
        raise RuntimeError(f"Could not fetch candles for {symbol}: {e}") from e


def live_price(symbol: str) -> tuple[float, str]:
    """Fetch the latest price via yfinance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        price = float(info.last_price or info.regular_market_price or 0.0)
        if price > 0:
            return price, "yfinance"
    except Exception:
        pass
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1]), "yfinance-history"
    except Exception:
        pass
    return 0.0, "unavailable"


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
