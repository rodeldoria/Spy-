"""Kalshi public market-data client.

Uses the public REST endpoints under `/trade-api/v2/` for event/market
discovery and orderbook snapshots. Authentication is optional — read-only
market data is public; an API key/private-key signature is only needed for
placing trades, which this integration does NOT do.

Reference: https://trading-api.readme.io/reference/getmarkets
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests

DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi crypto series prefixes we care about. The platform groups markets
# into "series" (e.g. KXBTCD = BTC daily, KXETHD = ETH daily). These are the
# tickers that show up under the Crypto tab in the screenshots.
CRYPTO_SERIES = {
    "BTC": {
        "15min": "KXBTC",        # 15-minute Up/Down range (e.g. KXBTC-25MAY13H07)
        "hourly": "KXBTCD",      # hourly settlement, "BTC price today at Xpm"
        "daily": "KXBTCY",       # end-of-day settlement
        "weekly": "KXBTCW",      # Friday 5pm EDT
    },
    "ETH": {
        "15min": "KXETH",
        "hourly": "KXETHD",
        "daily": "KXETHY",
        "weekly": "KXETHW",
    },
    "SOL": {
        "15min": "KXSOL",
        "hourly": "KXSOLD",
        "daily": "KXSOLY",
        "weekly": "KXSOLW",
    },
    "XRP": {
        "15min": "KXXRP",
        "hourly": "KXXRPD",
        "daily": "KXXRPY",
        "weekly": "KXXRPW",
    },
}


@dataclass(frozen=True)
class KalshiMarket:
    """A single Kalshi market (one binary contract).

    Range markets (e.g. "$79,750 or above") have a strike + side. Up/Down
    15-min markets have direction = "up" or "down" with the strike implied
    by the event title.
    """

    ticker: str
    event_ticker: str
    title: str
    subtitle: str
    status: str  # "active", "settled", "closed"
    yes_bid: int  # cents (0-100). 0 if no bid.
    yes_ask: int  # cents (0-100). 100 if no ask.
    no_bid: int
    no_ask: int
    last_price: int  # cents. Last traded price for YES.
    volume: int
    open_interest: int
    close_time: float  # epoch seconds
    expiration_time: float  # epoch seconds
    strike_type: str | None  # "greater", "less", "between", or None for Up/Down
    floor_strike: float | None
    cap_strike: float | None
    raw: dict[str, Any]

    @property
    def yes_mid(self) -> float:
        """Midpoint of YES side, in cents (0-100)."""
        if self.yes_ask > 0 and self.yes_bid > 0:
            return (self.yes_bid + self.yes_ask) / 2.0
        return float(self.last_price or self.yes_bid or self.yes_ask or 50)

    @property
    def no_mid(self) -> float:
        if self.no_ask > 0 and self.no_bid > 0:
            return (self.no_bid + self.no_ask) / 2.0
        return 100.0 - self.yes_mid

    @property
    def implied_prob_yes(self) -> float:
        """Implied probability of YES resolving true (0-1)."""
        return max(0.0, min(1.0, self.yes_mid / 100.0))

    @property
    def implied_prob_no(self) -> float:
        return max(0.0, min(1.0, self.no_mid / 100.0))

    @property
    def seconds_to_close(self) -> float:
        return max(0.0, self.close_time - time.time())

    @property
    def payout_yes(self) -> float:
        """Payout multiplier if you buy YES at the ask and it resolves true."""
        price = max(1, self.yes_ask) / 100.0
        return 1.0 / price

    @property
    def payout_no(self) -> float:
        price = max(1, self.no_ask) / 100.0
        return 1.0 / price


@dataclass(frozen=True)
class KalshiOrderbook:
    """Top-of-book snapshot. Bids/asks are lists of (price_cents, size)."""

    market_ticker: str
    yes_bids: list[tuple[int, int]]
    yes_asks: list[tuple[int, int]]
    no_bids: list[tuple[int, int]]
    no_asks: list[tuple[int, int]]


class KalshiClient:
    """Thin wrapper around Kalshi's REST API.

    Read-only. No order placement, no authentication required for market
    data. If `KALSHI_API_KEY` is set the client will forward it as a bearer
    token — but the public endpoints we use don't require it.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 8.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("KALSHI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        api_key = os.getenv("KALSHI_API_KEY")
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.session.headers.setdefault("Accept", "application/json")
        self.session.headers.setdefault("User-Agent", "spy-/kalshi-crypto-integration")

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_markets(
        self,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        status: str = "open",
        limit: int = 100,
    ) -> list[KalshiMarket]:
        """List markets, optionally filtered by series or event.

        `status` may be a comma-separated string ("open,closed") per the API.
        """
        params: dict[str, Any] = {"limit": min(limit, 200), "status": status}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        data = self._get("/markets", params=params)
        return [_market_from_payload(m) for m in data.get("markets", [])]

    def get_market(self, ticker: str) -> KalshiMarket:
        data = self._get(f"/markets/{ticker}")
        return _market_from_payload(data["market"])

    def get_orderbook(self, ticker: str, depth: int = 5) -> KalshiOrderbook:
        data = self._get(f"/markets/{ticker}/orderbook", params={"depth": depth})
        ob = data.get("orderbook", {}) or {}

        def _normalize(side: list[Any] | None) -> list[tuple[int, int]]:
            if not side:
                return []
            return [(int(level[0]), int(level[1])) for level in side]

        return KalshiOrderbook(
            market_ticker=ticker,
            yes_bids=_normalize(ob.get("yes")),
            yes_asks=_normalize(ob.get("yes_ask") or ob.get("no")),
            no_bids=_normalize(ob.get("no")),
            no_asks=_normalize(ob.get("no_ask") or ob.get("yes")),
        )

    def get_events(self, series_ticker: str, status: str = "open", limit: int = 50) -> list[dict[str, Any]]:
        """Return events (groups of markets) for a series."""
        params = {"series_ticker": series_ticker, "status": status, "limit": limit}
        data = self._get("/events", params=params)
        return data.get("events", []) or []

    def crypto_markets(
        self,
        symbol: str,
        horizons: tuple[str, ...] = ("15min", "hourly", "daily", "weekly"),
        limit_per_series: int = 25,
    ) -> dict[str, list[KalshiMarket]]:
        """Fetch open crypto markets for `symbol` across the requested horizons.

        Returns a dict keyed by horizon. Symbol must be one of the keys in
        `CRYPTO_SERIES`. Network errors are surfaced as exceptions — the
        caller decides how to render them.
        """
        symbol = symbol.upper()
        series_map = CRYPTO_SERIES.get(symbol)
        if not series_map:
            raise ValueError(f"unknown crypto symbol for Kalshi: {symbol}")

        out: dict[str, list[KalshiMarket]] = {}
        for horizon in horizons:
            series = series_map.get(horizon)
            if not series:
                continue
            try:
                out[horizon] = self.get_markets(series_ticker=series, limit=limit_per_series)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    out[horizon] = []
                else:
                    raise
        return out


def _market_from_payload(m: dict[str, Any]) -> KalshiMarket:
    """Build a KalshiMarket from the API response shape.

    Kalshi returns prices in cents; close/expiration times are ISO-8601 UTC.
    Some fields differ slightly between event types, so we coalesce.
    """
    def _iso_to_epoch(s: str | None) -> float:
        if not s:
            return 0.0
        try:
            from datetime import datetime
            return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError):
            return 0.0

    floor_strike = m.get("floor_strike")
    cap_strike = m.get("cap_strike")
    return KalshiMarket(
        ticker=m.get("ticker", ""),
        event_ticker=m.get("event_ticker", ""),
        title=m.get("title", "") or m.get("yes_sub_title", ""),
        subtitle=m.get("subtitle", "") or m.get("yes_sub_title", ""),
        status=m.get("status", ""),
        yes_bid=int(m.get("yes_bid") or 0),
        yes_ask=int(m.get("yes_ask") or 0),
        no_bid=int(m.get("no_bid") or 0),
        no_ask=int(m.get("no_ask") or 0),
        last_price=int(m.get("last_price") or 0),
        volume=int(m.get("volume") or 0),
        open_interest=int(m.get("open_interest") or 0),
        close_time=_iso_to_epoch(m.get("close_time")),
        expiration_time=_iso_to_epoch(m.get("expiration_time") or m.get("close_time")),
        strike_type=m.get("strike_type"),
        floor_strike=float(floor_strike) if floor_strike is not None else None,
        cap_strike=float(cap_strike) if cap_strike is not None else None,
        raw=m,
    )
