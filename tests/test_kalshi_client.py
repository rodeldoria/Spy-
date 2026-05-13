"""Tests for the Kalshi REST client (offline; mocks HTTP)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.kalshi.client import CRYPTO_SERIES, KalshiClient, _market_from_payload


def test_crypto_series_covers_expected_symbols():
    assert {"BTC", "ETH", "SOL", "XRP"}.issubset(set(CRYPTO_SERIES))
    for sym, by_horizon in CRYPTO_SERIES.items():
        # Every supported symbol covers all four horizons.
        assert {"15min", "hourly", "daily", "weekly"}.issubset(set(by_horizon))


def test_market_from_payload_parses_iso_and_strikes():
    payload = {
        "ticker": "KXBTCD-25MAY13-79750",
        "event_ticker": "KXBTCD-25MAY13",
        "title": "Bitcoin price today at 5pm EDT?",
        "subtitle": "$79,750 or above",
        "status": "active",
        "yes_bid": 49,
        "yes_ask": 51,
        "no_bid": 49,
        "no_ask": 51,
        "last_price": 50,
        "volume": 1234,
        "open_interest": 567,
        "close_time": "2026-05-13T21:00:00Z",
        "expiration_time": "2026-05-13T21:00:00Z",
        "strike_type": "greater",
        "floor_strike": 79750.0,
        "cap_strike": None,
    }
    m = _market_from_payload(payload)
    assert m.ticker == "KXBTCD-25MAY13-79750"
    assert m.floor_strike == 79750.0
    assert m.cap_strike is None
    assert m.close_time > 0
    assert m.implied_prob_yes == pytest.approx(0.5, abs=0.01)
    assert m.payout_yes == pytest.approx(100.0 / 51, abs=1e-9)


def test_market_yes_mid_falls_back_to_last_price():
    payload = {
        "ticker": "T",
        "event_ticker": "E",
        "title": "x",
        "subtitle": "",
        "status": "active",
        "yes_bid": 0,
        "yes_ask": 0,
        "no_bid": 0,
        "no_ask": 0,
        "last_price": 73,
        "volume": 0,
        "open_interest": 0,
        "close_time": "",
        "expiration_time": "",
        "strike_type": None,
        "floor_strike": None,
        "cap_strike": None,
    }
    m = _market_from_payload(payload)
    assert m.yes_mid == 73


def test_crypto_markets_calls_series_per_horizon():
    client = KalshiClient()
    with patch.object(client, "get_markets") as mock_get:
        mock_get.return_value = []
        client.crypto_markets("BTC", horizons=("15min", "hourly"))
        called_series = {call.kwargs.get("series_ticker") for call in mock_get.call_args_list}
        assert called_series == {CRYPTO_SERIES["BTC"]["15min"], CRYPTO_SERIES["BTC"]["hourly"]}


def test_crypto_markets_swallows_404_per_horizon():
    """A missing series (e.g. SOL weekly not listed) should yield an empty list,
    not abort the whole symbol fetch."""
    import requests

    client = KalshiClient()
    resp = MagicMock(status_code=404)
    err = requests.HTTPError(response=resp)

    def fake(**kwargs):
        if kwargs["series_ticker"] == CRYPTO_SERIES["SOL"]["weekly"]:
            raise err
        return []

    with patch.object(client, "get_markets", side_effect=fake):
        out = client.crypto_markets("SOL", horizons=("daily", "weekly"))
    assert out["weekly"] == []
    assert out["daily"] == []


def test_crypto_markets_rejects_unknown_symbol():
    client = KalshiClient()
    with pytest.raises(ValueError):
        client.crypto_markets("DOGE")
