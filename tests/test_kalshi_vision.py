"""Tests for vision-parsed Kalshi markets.

The vision call itself is mocked — we only test the payload→ParsedMarket
shaping and the ParsedMarket→KalshiMarket bridge used by the decision engine.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.kalshi.decisions import score_market
from app.kalshi.parsed import market_from_parsed, score_parsed_markets
from app.kalshi.spot import SpotQuote
from app.kalshi.vision import (
    ParsedMarket,
    ParsedSide,
    _payload_to_markets,
    parse_screenshot,
)


def _mock_text_response(text: str) -> SimpleNamespace:
    """Build a minimal anthropic-shaped response with one text content block."""
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block])


def _eth_15min_payload() -> dict:
    return {
        "markets": [
            {
                "symbol": "ETH",
                "horizon": "15min",
                "title": "ETH 15 min · $2,261.91 target",
                "target_price": 2261.91,
                "sides": [
                    {
                        "label": "Up",
                        "prob_pct": 42,
                        "payout": 2.01,
                        "strike": 2261.91,
                        "strike_type": "up",
                    },
                    {
                        "label": "Down",
                        "prob_pct": 58,
                        "payout": 1.76,
                        "strike": 2261.91,
                        "strike_type": "down",
                    },
                ],
                "volume_usd": 7419,
                "time_remaining_seconds": 515,
            }
        ]
    }


def _btc_daily_range_payload() -> dict:
    return {
        "markets": [
            {
                "symbol": "BTC",
                "horizon": "daily",
                "title": "Bitcoin price today at 5pm EDT?",
                "target_price": None,
                "sides": [
                    {
                        "label": "$79,750 or above",
                        "prob_pct": 49,
                        "payout": 2.01,
                        "strike": 79750,
                        "strike_type": "greater",
                    },
                    {
                        "label": "$80,750 or above",
                        "prob_pct": 12,
                        "payout": 7.85,
                        "strike": 80750,
                        "strike_type": "greater",
                    },
                ],
                "volume_usd": 580513,
                "time_remaining_seconds": 24829,
            }
        ]
    }


# ---------------------------------------------------------------------------
# Payload → ParsedMarket
# ---------------------------------------------------------------------------

def test_payload_to_markets_eth_15min():
    parsed = _payload_to_markets(_eth_15min_payload())
    assert len(parsed) == 1
    pm = parsed[0]
    assert pm.symbol == "ETH"
    assert pm.horizon == "15min"
    assert pm.target_price == pytest.approx(2261.91)
    assert pm.time_remaining_seconds == 515
    assert len(pm.sides) == 2
    up = next(s for s in pm.sides if s.label == "Up")
    assert up.prob_pct == 42
    assert up.payout == pytest.approx(2.01)
    assert up.strike_type == "up"


def test_payload_to_markets_handles_nulls():
    parsed = _payload_to_markets({
        "markets": [{
            "symbol": "btc",
            "horizon": "hourly",
            "title": "x",
            "target_price": None,
            "sides": [{"label": "Y", "prob_pct": 50, "strike_type": "greater"}],
            "volume_usd": None,
            "time_remaining_seconds": None,
        }]
    })
    pm = parsed[0]
    assert pm.symbol == "BTC"  # uppercased
    assert pm.target_price is None
    assert pm.sides[0].payout is None
    assert pm.sides[0].strike is None
    assert pm.volume_usd is None


def test_payload_to_markets_empty_list():
    assert _payload_to_markets({"markets": []}) == []
    assert _payload_to_markets({}) == []


# ---------------------------------------------------------------------------
# parse_screenshot — mocked API
# ---------------------------------------------------------------------------

def test_parse_screenshot_calls_anthropic_with_image_and_returns_markets():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _mock_text_response(
        json.dumps(_eth_15min_payload())
    )

    result = parse_screenshot(b"fakeimagebytes", media_type="image/png", client=fake_client)

    fake_client.messages.create.assert_called_once()
    kwargs = fake_client.messages.create.call_args.kwargs
    # System prompt is sent as a cacheable block.
    assert isinstance(kwargs["system"], list)
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    # The image is base64-encoded in a vision content block.
    user_content = kwargs["messages"][0]["content"]
    image_block = next(b for b in user_content if b["type"] == "image")
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/png"
    # JSON schema is the output constraint.
    assert kwargs["output_config"]["format"]["type"] == "json_schema"

    assert len(result) == 1
    assert result[0].symbol == "ETH"


def test_parse_screenshot_raises_on_non_json_response():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _mock_text_response("sorry, can't help")
    with pytest.raises(ValueError):
        parse_screenshot(b"img", client=fake_client)


# ---------------------------------------------------------------------------
# ParsedMarket → KalshiMarket bridge
# ---------------------------------------------------------------------------

def test_market_from_parsed_up_side_maps_to_greater():
    pm = _payload_to_markets(_eth_15min_payload())[0]
    up_side = next(s for s in pm.sides if s.label == "Up")
    market = market_from_parsed(pm, up_side)
    assert market.strike_type == "greater"
    assert market.floor_strike == pytest.approx(2261.91)
    assert market.status == "active"
    # Payout 2.01 → ask ≈ 50¢
    assert 49 <= market.yes_ask <= 51


def test_market_from_parsed_down_side_maps_to_less():
    pm = _payload_to_markets(_eth_15min_payload())[0]
    down_side = next(s for s in pm.sides if s.label == "Down")
    market = market_from_parsed(pm, down_side)
    assert market.strike_type == "less"
    # Payout 1.76 → ask ≈ 57¢
    assert 55 <= market.yes_ask <= 59


def test_market_from_parsed_range_preserves_strike():
    pm = _payload_to_markets(_btc_daily_range_payload())[0]
    side = pm.sides[1]  # $80,750 or above
    market = market_from_parsed(pm, side)
    assert market.strike_type == "greater"
    assert market.floor_strike == pytest.approx(80750)
    # Payout 7.85 → ask ≈ 13¢
    assert 11 <= market.yes_ask <= 14


def test_market_from_parsed_uses_time_remaining_as_close_window():
    pm = _payload_to_markets(_eth_15min_payload())[0]
    side = pm.sides[0]
    market = market_from_parsed(pm, side)
    seconds_to_close = market.close_time - time.time()
    # parsed time_remaining_seconds is 515; allow a small slack.
    assert 500 <= seconds_to_close <= 520


def test_score_parsed_markets_runs_decision_engine():
    pm = _payload_to_markets(_btc_daily_range_payload())[0]
    spot = SpotQuote(symbol="BTC", price=79_800.0, ts=time.time(), source="test", sigma_per_min=0.0008)
    out = score_parsed_markets([pm], {"BTC": spot})
    assert len(out) == 1
    parsed_market, decisions = out[0]
    assert parsed_market is pm
    assert len(decisions) == 2
    for d in decisions:
        # Each decision is real — direction is one of the known labels and
        # confidence is bounded.
        assert d.direction in {"YES", "NO", "PASS"}
        assert 0.0 <= d.confidence_pct <= 99.0


def test_score_parsed_markets_skips_unknown_symbol():
    pm = ParsedMarket(
        symbol="DOGE",  # not in our spot map
        horizon="15min",
        title="DOGE test",
        sides=[ParsedSide(label="Up", prob_pct=50, payout=2.0, strike=0.1, strike_type="up")],
    )
    spot = SpotQuote(symbol="BTC", price=79_800.0, ts=time.time(), source="test")
    out = score_parsed_markets([pm], {"BTC": spot})
    assert out == [(pm, [])]


def test_score_parsed_yields_yes_when_spot_well_above_strike():
    """End-to-end: spot far above strike + low book prob → YES recommendation."""
    pm = ParsedMarket(
        symbol="BTC",
        horizon="daily",
        title="BTC daily test",
        target_price=None,
        sides=[
            ParsedSide(
                label="$79,000 or above",
                prob_pct=30,    # book undervaluing this side
                payout=3.0,
                strike=79_000,
                strike_type="greater",
            ),
        ],
        time_remaining_seconds=3600,
    )
    # Spot well above strike with low vol → model says YES ~ certain.
    spot = SpotQuote(symbol="BTC", price=80_000.0, ts=time.time(), source="test", sigma_per_min=0.0005)
    [(_, decisions)] = score_parsed_markets([pm], {"BTC": spot})
    assert decisions[0].direction == "YES"
    assert decisions[0].yes_side.edge > 0.04
