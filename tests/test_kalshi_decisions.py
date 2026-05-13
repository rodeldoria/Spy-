"""Tests for the Kalshi decision engine."""

from __future__ import annotations

import math
import time

import pytest

from app.kalshi.client import KalshiMarket
from app.kalshi.decisions import (
    DEFAULT_MIN_EDGE,
    DEFAULT_MIN_EV,
    Decision,
    gbm_prob_above,
    gbm_prob_between,
    model_prob_yes,
    score_market,
)
from app.kalshi.spot import SpotQuote


def _make_market(
    *,
    title: str = "BTC 15 min · $79,627.89 target",
    yes_bid: int = 50,
    yes_ask: int = 52,
    no_bid: int = 48,
    no_ask: int = 50,
    last_price: int = 51,
    seconds_to_close: float = 600.0,
    strike_type: str | None = None,
    floor_strike: float | None = 79627.89,
    cap_strike: float | None = None,
    status: str = "active",
    volume: int = 1000,
) -> KalshiMarket:
    return KalshiMarket(
        ticker="KXBTC-TEST",
        event_ticker="KXBTC-EVENT",
        title=title,
        subtitle="",
        status=status,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        last_price=last_price,
        volume=volume,
        open_interest=500,
        close_time=time.time() + seconds_to_close,
        expiration_time=time.time() + seconds_to_close,
        strike_type=strike_type,
        floor_strike=floor_strike,
        cap_strike=cap_strike,
        raw={},
    )


def _spot(price: float = 79_627.89, sigma: float = 0.0012, symbol: str = "BTC") -> SpotQuote:
    return SpotQuote(symbol=symbol, price=price, ts=time.time(), source="test", sigma_per_min=sigma)


# ---------------------------------------------------------------------------
# Probability model
# ---------------------------------------------------------------------------

def test_gbm_at_the_money_is_half():
    # Spot == strike, any positive sigma → P(above) should be ~50%.
    assert gbm_prob_above(spot=100.0, strike=100.0, sigma_total=0.02) == pytest.approx(0.5, abs=1e-6)


def test_gbm_zero_sigma_collapses_to_indicator():
    # No vol → degenerate: just compare spot vs strike.
    assert gbm_prob_above(100.0, 110.0, 0.0) == 0.0
    assert gbm_prob_above(110.0, 100.0, 0.0) == 1.0
    assert gbm_prob_above(100.0, 100.0, 0.0) == 1.0


def test_gbm_above_decreases_with_strike():
    # Monotonicity sanity check.
    probs = [gbm_prob_above(100.0, k, sigma_total=0.05) for k in (90, 95, 100, 105, 110)]
    assert probs == sorted(probs, reverse=True)


def test_gbm_between_within_zero_width_is_zero():
    assert gbm_prob_between(100.0, 100.0, 100.0, 0.05) == 0.0


def test_gbm_between_full_range_sums_with_above():
    spot, floor, cap, sigma = 100.0, 95.0, 105.0, 0.03
    p_between = gbm_prob_between(spot, floor, cap, sigma)
    p_above_floor = gbm_prob_above(spot, floor, sigma)
    p_above_cap = gbm_prob_above(spot, cap, sigma)
    assert p_between == pytest.approx(p_above_floor - p_above_cap, abs=1e-9)


# ---------------------------------------------------------------------------
# model_prob_yes — shape parsing
# ---------------------------------------------------------------------------

def test_model_prob_greater_uses_floor():
    market = _make_market(strike_type="greater", floor_strike=79_750.0)
    spot = _spot(price=79_627.89)
    p, _ = model_prob_yes(market, spot)
    # Spot is below strike, so model probability of YES (≥ strike) should be < 50%.
    assert p < 0.5


def test_model_prob_less_inverts_greater():
    market_g = _make_market(strike_type="greater", floor_strike=79_750.0)
    market_l = _make_market(strike_type="less", floor_strike=79_750.0)
    spot = _spot()
    p_g, _ = model_prob_yes(market_g, spot)
    p_l, _ = model_prob_yes(market_l, spot)
    assert p_g + p_l == pytest.approx(1.0, abs=1e-9)


def test_model_prob_between_inside_range():
    # Tight range straddling spot — probability should be sizeable, not 1.
    market = _make_market(
        strike_type="between",
        floor_strike=79_500.0,
        cap_strike=79_750.0,
    )
    spot = _spot(price=79_627.89, sigma=0.0006)
    p, _ = model_prob_yes(market, spot)
    assert 0.05 < p < 0.95


def test_model_prob_unknown_shape_returns_half_with_warning():
    market = _make_market(strike_type=None, floor_strike=None, title="weird market")
    spot = _spot()
    p, why = model_prob_yes(market, spot)
    assert p == 0.5
    assert "unrecognised" in why


def test_model_prob_up_market_uses_floor_strike():
    market = _make_market(
        title="ETH 15 min · $2,261.91 target Up",
        strike_type=None,
        floor_strike=2_261.91,
    )
    spot = _spot(price=2_300.0, symbol="ETH")
    p, _ = model_prob_yes(market, spot)
    # Spot is above the strike → P(Up = price ≥ strike) should be > 50%.
    assert p > 0.5


# ---------------------------------------------------------------------------
# score_market — direction, EV, confidence
# ---------------------------------------------------------------------------

def test_score_market_pass_when_book_matches_model():
    # Spot at strike, sigma generous → model ~50%. Book at 50/50 → no edge.
    market = _make_market(
        strike_type="greater",
        floor_strike=79_627.89,
        yes_bid=49,
        yes_ask=51,
        no_bid=49,
        no_ask=51,
    )
    spot = _spot(price=79_627.89, sigma=0.0012)
    d = score_market(market, spot)
    assert d.direction == "PASS"
    assert d.chosen is None
    # Confidence in PASS is bounded.
    assert 0.0 <= d.confidence_pct <= 100.0


def test_score_market_yes_when_model_far_above_book():
    # Spot far above strike → model says YES near-certain. Book mispriced low.
    market = _make_market(
        strike_type="greater",
        floor_strike=79_000.0,
        yes_bid=30,
        yes_ask=32,
        no_bid=68,
        no_ask=70,
    )
    spot = _spot(price=80_000.0, sigma=0.0005)
    d = score_market(market, spot)
    assert d.direction == "YES"
    assert d.yes_side.edge > DEFAULT_MIN_EDGE
    assert d.yes_side.ev_per_dollar > DEFAULT_MIN_EV
    assert 0 < d.confidence_pct <= 99.0


def test_score_market_no_when_model_far_below_book():
    # Spot far below strike → model says YES near-impossible. Book mispriced high.
    market = _make_market(
        strike_type="greater",
        floor_strike=80_500.0,
        yes_bid=68,
        yes_ask=70,
        no_bid=30,
        no_ask=32,
    )
    spot = _spot(price=79_000.0, sigma=0.0005)
    d = score_market(market, spot)
    assert d.direction == "NO"
    assert d.no_side.edge > DEFAULT_MIN_EDGE


def test_score_market_ev_math_per_dollar():
    # Hand-verifiable EV: model=0.7, ask=50¢ → payout 2x → EV = 0.7*2 - 1 = 0.4 ($/$1).
    market = _make_market(
        strike_type="greater",
        floor_strike=79_000.0,
        yes_bid=49,
        yes_ask=50,
        no_bid=49,
        no_ask=51,
    )
    # Pick spot/sigma so model ~ 0.7. Closed form: z = ln(K/S)/sigma_total.
    # Want gbm_prob_above = 0.7 → 1 - Phi(z) = 0.7 → Phi(z) = 0.3 → z ≈ -0.5244.
    # ln(K/S) = z*sigma_total. With sigma_total=0.05 → ln(K/S) ≈ -0.02622 → K/S ≈ 0.9741.
    # If K = 79000, S ≈ 81100. seconds_to_close=3600 (60 min), sigma_per_min so total=0.05:
    # sigma_per_min = 0.05 / sqrt(60) ≈ 0.006455.
    market = _make_market(
        strike_type="greater",
        floor_strike=79_000.0,
        yes_bid=49,
        yes_ask=50,
        no_bid=49,
        no_ask=51,
        seconds_to_close=3600,
    )
    spot = _spot(price=81_100.0, sigma=0.05 / math.sqrt(60))
    d = score_market(market, spot)
    # Approximate but should be near 0.7 model, 0.4 EV.
    assert d.yes_side.model_prob == pytest.approx(0.7, abs=0.02)
    expected_ev = d.yes_side.model_prob * (100.0 / 50) - 1.0
    assert d.yes_side.ev_per_dollar == pytest.approx(expected_ev, abs=1e-9)


def test_score_market_kelly_capped_at_quarter():
    # Even with a huge edge, Kelly fraction should never exceed 25%.
    market = _make_market(
        strike_type="greater",
        floor_strike=70_000.0,
        yes_bid=4,
        yes_ask=5,
        no_bid=94,
        no_ask=96,
    )
    spot = _spot(price=80_000.0, sigma=0.0003)
    d = score_market(market, spot)
    assert d.yes_side.kelly_fraction <= 0.25 + 1e-9


def test_score_market_warns_on_wide_spread():
    market = _make_market(
        strike_type="greater",
        floor_strike=79_000.0,
        yes_bid=10,
        yes_ask=30,   # 20¢ spread
        no_bid=70,
        no_ask=90,
    )
    spot = _spot(price=80_000.0, sigma=0.0005)
    d = score_market(market, spot)
    assert any("Wide spread" in w for w in d.warnings)


def test_score_market_warns_on_zero_volume():
    market = _make_market(
        strike_type="greater",
        floor_strike=79_000.0,
        volume=0,
        yes_bid=30,
        yes_ask=32,
        no_bid=68,
        no_ask=70,
    )
    spot = _spot(price=80_000.0, sigma=0.0005)
    d = score_market(market, spot)
    assert any("Zero volume" in w for w in d.warnings)


def test_confidence_pct_bounded():
    # Across a sweep of edges, confidence stays in [0, 99].
    for strike, side_prob in [(79_000, 70), (79_500, 50), (80_500, 30), (82_000, 10)]:
        market = _make_market(
            strike_type="greater",
            floor_strike=float(strike),
            yes_bid=side_prob - 1,
            yes_ask=side_prob + 1,
            no_bid=100 - side_prob - 1,
            no_ask=100 - side_prob + 1,
        )
        spot = _spot(price=80_000.0, sigma=0.0008)
        d = score_market(market, spot)
        assert 0.0 <= d.confidence_pct <= 99.0
        assert isinstance(d, Decision)
