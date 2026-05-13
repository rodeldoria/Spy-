"""Spot price feed for Kalshi decisions.

Hybrid source: by default we pull from Coinbase's public spot endpoint
(no key required), but the user can override with a manual value on the
Streamlit page. Returns both the price and a short-window realised volatility
estimate so the decision engine can size probabilities for range markets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

COINBASE_SPOT_URL = "https://api.coinbase.com/v2/prices/{pair}/spot"
BINANCE_KLINES_URL = "https://api.binance.us/api/v3/klines"

_SYMBOL_TO_COINBASE_PAIR = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
}

_SYMBOL_TO_BINANCE_PAIR = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}


@dataclass(frozen=True)
class SpotQuote:
    symbol: str
    price: float
    ts: float
    source: str
    # Realised log-return stdev over the lookback window (per-minute).
    # 0.0 means "unknown" — decision engine will fall back to a default.
    sigma_per_min: float = 0.0


def get_spot_price(symbol: str, timeout: float = 5.0) -> SpotQuote:
    """Fetch a fresh spot quote for `symbol`.

    Tries Coinbase first (most permissive on Replit), falls back to Binance
    US klines. Raises on failure so the caller can show an explicit error
    and surface the manual-override input.
    """
    symbol = symbol.upper()
    pair = _SYMBOL_TO_COINBASE_PAIR.get(symbol)
    if not pair:
        raise ValueError(f"unsupported symbol: {symbol}")

    try:
        r = requests.get(COINBASE_SPOT_URL.format(pair=pair), timeout=timeout)
        r.raise_for_status()
        amount = float(r.json()["data"]["amount"])
        sigma = _realised_sigma_per_min(symbol, timeout=timeout)
        return SpotQuote(symbol=symbol, price=amount, ts=time.time(), source="coinbase", sigma_per_min=sigma)
    except Exception:
        # Fall through to Binance US — different geographies block one or the other.
        pass

    bpair = _SYMBOL_TO_BINANCE_PAIR.get(symbol)
    if not bpair:
        raise RuntimeError(f"all spot sources failed for {symbol}")
    r = requests.get(
        BINANCE_KLINES_URL,
        params={"symbol": bpair, "interval": "1m", "limit": 1},
        timeout=timeout,
    )
    r.raise_for_status()
    last = r.json()[-1]
    close = float(last[4])
    sigma = _realised_sigma_per_min(symbol, timeout=timeout)
    return SpotQuote(symbol=symbol, price=close, ts=time.time(), source="binance", sigma_per_min=sigma)


def manual_quote(symbol: str, price: float, sigma_per_min: float = 0.0) -> SpotQuote:
    """User-supplied spot price. Used when API fetch is blocked or stale."""
    return SpotQuote(
        symbol=symbol.upper(),
        price=float(price),
        ts=time.time(),
        source="manual",
        sigma_per_min=float(sigma_per_min),
    )


def _realised_sigma_per_min(symbol: str, lookback_min: int = 60, timeout: float = 5.0) -> float:
    """Estimate per-minute realised log-return stdev from Binance 1m candles.

    Returns 0.0 on any failure so the decision engine falls back to a
    reasonable default for the asset.
    """
    bpair = _SYMBOL_TO_BINANCE_PAIR.get(symbol.upper())
    if not bpair:
        return 0.0
    try:
        import math

        r = requests.get(
            BINANCE_KLINES_URL,
            params={"symbol": bpair, "interval": "1m", "limit": max(lookback_min, 10)},
            timeout=timeout,
        )
        r.raise_for_status()
        closes = [float(k[4]) for k in r.json()]
        if len(closes) < 5:
            return 0.0
        returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / max(1, len(returns) - 1)
        return math.sqrt(var)
    except Exception:
        return 0.0


# Reasonable fallback vols (per-minute log-return stdev) when the live
# estimate is unavailable. Calibrated from rough 2024-2025 realised vol.
DEFAULT_SIGMA_PER_MIN = {
    "BTC": 0.0012,   # ~5.4% daily
    "ETH": 0.0016,
    "SOL": 0.0022,
    "XRP": 0.0022,
}


def default_sigma_per_min(symbol: str) -> float:
    return DEFAULT_SIGMA_PER_MIN.get(symbol.upper(), 0.0015)
