"""Tests for the Kalshi Decision Council gate.

The council is a deterministic re-read of a `Decision` — it runs no new
analysis, so all five gates can be exercised by constructing decisions
with the right field values and asserting which checks pass.
"""

from __future__ import annotations

import time

from app.kalshi.client import KalshiMarket
from app.kalshi.council import (
    ARMED_THRESHOLD,
    LIQUIDITY_BAND_MAX_CENTS,
    LIQUIDITY_BAND_MIN_CENTS,
    MIN_CONVICTION_PCT,
    MIN_KELLY,
    evaluate,
)
from app.kalshi.decisions import score_market
from app.kalshi.spot import SpotQuote


def _make_market(
    *,
    yes_ask: int = 40,
    no_ask: int = 60,
    yes_bid: int | None = None,
    no_bid: int | None = None,
    strike_type: str | None = "greater",
    floor_strike: float | None = 79_000.0,
    seconds_to_close: float = 600.0,
    volume: int = 1000,
) -> KalshiMarket:
    return KalshiMarket(
        ticker="KXBTC-TEST",
        event_ticker="KXBTC-EVENT",
        title="BTC test market",
        subtitle="",
        status="active",
        yes_bid=yes_bid if yes_bid is not None else max(1, yes_ask - 1),
        yes_ask=yes_ask,
        no_bid=no_bid if no_bid is not None else max(1, no_ask - 1),
        no_ask=no_ask,
        last_price=yes_ask,
        volume=volume,
        open_interest=500,
        close_time=time.time() + seconds_to_close,
        expiration_time=time.time() + seconds_to_close,
        strike_type=strike_type,
        floor_strike=floor_strike,
        cap_strike=None,
        raw={},
    )


def _spot(price: float = 79_627.89, sigma: float = 0.0012) -> SpotQuote:
    return SpotQuote(
        symbol="BTC", price=price, ts=time.time(), source="test", sigma_per_min=sigma
    )


def test_pass_decisions_return_none():
    """No side chosen → no order button to gate → council is N/A."""
    # Book at 50/50, model at 50/50 → PASS.
    market = _make_market(yes_ask=51, no_ask=51, floor_strike=79_627.89)
    decision = score_market(market, _spot(price=79_627.89))
    assert decision.direction == "PASS"
    assert evaluate(decision, min_edge=0.04, min_ev=0.02) is None


def test_clean_setup_arms_button():
    """Healthy edge + EV + Kelly + conviction + mid-band ask → 5/5 armed."""
    # Spot far above strike → high model_p of YES; book at 30/70 → big YES edge.
    market = _make_market(
        yes_ask=30, no_ask=70, floor_strike=79_000.0, volume=1000
    )
    decision = score_market(market, _spot(price=80_500.0, sigma=0.0015))
    assert decision.direction == "YES"

    council = evaluate(decision, min_edge=0.04, min_ev=0.02)
    assert council is not None
    assert council.armed is True
    assert council.passed == council.total == 5
    assert {c.name for c in council.checks} == {
        "EDGE", "EV", "KELLY", "CONVICTION", "LIQUIDITY"
    }


def test_one_penny_ask_fails_liquidity_gate():
    """A 1¢ ask outside [5,95]¢ is the 'free money mirage' case."""
    market = _make_market(yes_ask=1, no_ask=99, floor_strike=79_000.0)
    decision = score_market(market, _spot(price=80_500.0, sigma=0.0015))
    assert decision.direction == "YES"

    council = evaluate(decision, min_edge=0.04, min_ev=0.02)
    assert council is not None
    liq = next(c for c in council.checks if c.name == "LIQUIDITY")
    assert liq.passed is False
    assert f"{LIQUIDITY_BAND_MIN_CENTS}" in liq.detail
    # 4 other gates likely pass on a 1¢ ask (huge edge, EV, Kelly), so the
    # armed status depends on whether 4-of-5 still clears. Verify the gate
    # at minimum suppresses one check.
    assert council.passed <= 4


def test_zero_volume_fails_liquidity_gate():
    """score_market appends a 'Zero volume' warning; council reads it."""
    market = _make_market(yes_ask=30, no_ask=70, floor_strike=79_000.0, volume=0)
    decision = score_market(market, _spot(price=80_500.0, sigma=0.0015))
    assert decision.direction == "YES"

    council = evaluate(decision, min_edge=0.04, min_ev=0.02)
    assert council is not None
    liq = next(c for c in council.checks if c.name == "LIQUIDITY")
    assert liq.passed is False
    assert "warning" in liq.detail.lower()


def test_unrecognised_shape_fails_liquidity_gate():
    """An unrecognised market defaults to model_p=50% and is gated off."""
    market = _make_market(
        strike_type=None,
        floor_strike=None,
        yes_ask=10,
        no_ask=90,
    )
    decision = score_market(market, _spot(price=80_000.0))
    # Direction may still be YES (huge phantom edge), but liquidity gate fails
    # because the shape-unrecognised warning is present.
    if decision.direction != "PASS":
        council = evaluate(decision, min_edge=0.04, min_ev=0.02)
        assert council is not None
        liq = next(c for c in council.checks if c.name == "LIQUIDITY")
        assert liq.passed is False


def test_armed_threshold_constant():
    """4 of 5 is the documented armed threshold — guard the constant."""
    assert ARMED_THRESHOLD == 4


def test_check_names_are_stable():
    """UI strings depend on these check names — guard against renames."""
    market = _make_market(yes_ask=30, no_ask=70, floor_strike=79_000.0)
    decision = score_market(market, _spot(price=80_500.0))
    council = evaluate(decision, min_edge=0.04, min_ev=0.02)
    assert council is not None
    names = [c.name for c in council.checks]
    assert names == ["EDGE", "EV", "KELLY", "CONVICTION", "LIQUIDITY"]


def test_score_label_format():
    market = _make_market(yes_ask=30, no_ask=70, floor_strike=79_000.0)
    decision = score_market(market, _spot(price=80_500.0))
    council = evaluate(decision, min_edge=0.04, min_ev=0.02)
    assert council is not None
    assert council.score_label.endswith("/5")


def test_failed_checks_helper():
    """Helper returns only the failing checks, in order."""
    market = _make_market(yes_ask=1, no_ask=99, volume=0, floor_strike=79_000.0)
    decision = score_market(market, _spot(price=80_500.0))
    council = evaluate(decision, min_edge=0.04, min_ev=0.02)
    assert council is not None
    failed = council.failed_checks
    assert all(not c.passed for c in failed)
    assert len(failed) == council.total - council.passed


def test_thresholds_match_documented_values():
    """The module constants are the source of truth — guard against drift."""
    assert MIN_KELLY == 0.01
    assert MIN_CONVICTION_PCT == 50.0
    assert LIQUIDITY_BAND_MIN_CENTS == 5
    assert LIQUIDITY_BAND_MAX_CENTS == 95
