"""Stub stock price fetcher using yfinance."""
from __future__ import annotations

import pandas as pd


def get_daily(symbol: str, period: str = "2y") -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return _empty_df()
        return df
    except Exception as e:
        raise RuntimeError(f"Could not fetch daily prices for {symbol}: {e}") from e


def get_intraday(symbol: str, period: str = "60d", interval: str = "1h") -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return _empty_df()
        return df
    except Exception as e:
        raise RuntimeError(f"Could not fetch intraday prices for {symbol}: {e}") from e


def live_price(symbol: str) -> tuple[float, str]:
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        price = float(info.last_price or info.regular_market_price or 0.0)
        return price, "yfinance"
    except Exception:
        return 0.0, "error"


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
