"""Streamlit caching wrappers around Monte engine calls.

Mutable handles (PaperBook) are deliberately NOT cached — they're created per
request and synchronized via filelock. PatternStore is wrapped in
`@st.cache_resource` so the Chroma client is shared. Live prices are NOT
cached, by design — they're the freshness anchor.
"""

from __future__ import annotations

import streamlit as st

from monte.config import settings


def is_crypto(symbol: str) -> bool:
    s = symbol.upper()
    return "-" in s and s.split("-")[-1] in {"USD", "USDC", "USDT"}


@st.cache_resource(show_spinner=False)
def pattern_store():
    from monte.patterns.vector_store import PatternStore

    return PatternStore()


# Minimum bars to cover a 2-week lookback window per timeframe.
# Used for indicator stability and journal-similarity matching.
_BARS_PER_TWO_WEEKS = {
    # 1m caps at 7d on yfinance, so cap the bar request to keep us in-range.
    "1m":  7 * 24 * 60,
    "5m":  14 * 24 * 12,
    "15m": 14 * 24 * 4,
    "30m": 14 * 24 * 2,
    "1h":  14 * 24,
    "1d":  14,
}


def two_week_bars(interval: str, floor: int = 300) -> int:
    """How many bars we need to cover ~2 weeks of history at this timeframe."""
    return max(floor, _BARS_PER_TWO_WEEKS.get(interval, floor))


@st.cache_data(ttl=30, show_spinner=False)
def candles_short(symbol: str, interval: str, lookback_bars: int | None = None):
    """Fetch candles. Defaults to >= 2 weeks of bars for the chosen timeframe."""
    if lookback_bars is None:
        lookback_bars = two_week_bars(interval)
    if is_crypto(symbol):
        from monte.data import crypto

        return crypto.get_candles(symbol, interval, lookback_bars=lookback_bars)
    from monte.data import prices

    if interval == "1d":
        return prices.get_daily(symbol, period="2y")
    period_map = {"1m": "5d", "5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d"}
    df = prices.get_intraday(
        symbol, period=period_map.get(interval, "60d"), interval=interval
    )
    return df.tail(lookback_bars) if len(df) > lookback_bars else df


@st.cache_data(ttl=300, show_spinner=False)
def candles_long(symbol: str, interval: str, lookback_bars: int = 1000):
    return candles_short(symbol, interval, lookback_bars)


def live_price(symbol: str) -> tuple[float, str]:
    """Never cached — this is the freshness anchor."""
    if is_crypto(symbol):
        from monte.data import crypto

        return crypto.live_price(symbol)
    from monte.data import prices

    return prices.live_price(symbol)


def setup_page(title: str, icon: str = "📈") -> None:
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.title(f"{icon} {title}")


def sidebar_watchlists() -> tuple[list[str], list[str]]:
    with st.sidebar:
        st.subheader("Watchlists")
        crypto_text = st.text_input(
            "Crypto",
            value=",".join(settings.crypto_watchlist),
            help="Comma-separated, e.g. BTC-USD,ETH-USD,SOL-USD",
        )
        stock_text = st.text_input(
            "Stocks",
            value=",".join(settings.stock_watchlist),
            help="Comma-separated, e.g. SPY,QQQ",
        )
    crypto = [s.strip().upper() for s in crypto_text.split(",") if s.strip()]
    stocks = [s.strip().upper() for s in stock_text.split(",") if s.strip()]
    return crypto, stocks


def action_color(action: str) -> str:
    return {
        "STRONG_BUY": "#0a7d2a",
        "BUY": "#48a85a",
        "HOLD": "#888",
        "SELL": "#e07b3c",
        "STRONG_SELL": "#c73a3a",
    }.get(str(action), "#888")
