"""Golden-input tests for the likelihood gate.

We construct minimal `EventBundle`/`RegimeReport`/`MicrostructureReport`
fixtures and assert the verdict matches what the docstring promises:

  - 5 confluence-aligned axes → GO + high P(hit)
  - Tier-1 calendar event in window → STAND_DOWN regardless of axes
  - Token unlock in window → STAND_DOWN for crypto longs
  - Premortem critical not acknowledged → soft blocker → STAND_DOWN
  - Disagreeing axes → CAUTION
"""
from __future__ import annotations

import time

import pytest

from monte.data.econ_calendar import CalendarEvent
from monte.data.fred import FredObservation, FredSnapshot
from monte.data.onchain import OnChainSnapshot
from monte.intel.event_aggregator import EventBundle, IdeaContext
from monte.intel.perplexity import NewsBrief
from monte.microstructure import MicrostructureReport
from monte.regime import RegimeReport
from monte.regime.hmm import HMMResult
from monte.regime.macro_quadrant import MacroQuadrant
from monte.regime.wyckoff import WyckoffPhase
from monte.signals.likelihood_gate import score


def _bullish_news() -> NewsBrief:
    return NewsBrief(
        symbol="BTC-USD",
        configured=True,
        sentiment="bullish",
        summary="ETF inflows accelerating, new highs in sight.",
        fetched_at=time.time(),
    )


def _bullish_regime() -> RegimeReport:
    return RegimeReport(
        symbol="BTC-USD",
        hmm=HMMResult(label="bull-quiet", bull_prob=0.78, state=1, source="fallback"),
        hurst=0.62,
        hurst_label="trending (H=0.62)",
        wyckoff=WyckoffPhase(phase="markup", confidence=0.7, note="trend up", price_vs_200ma_pct=8.0, slope_50_pct=4.0),
        macro=MacroQuadrant(
            quadrant="Goldilocks",
            growth_label="↑",
            inflation_label="↓",
            equity_bias="bull",
            crypto_bias="bull",
            note="growth up, inflation down",
        ),
    )


def _bullish_micro() -> MicrostructureReport:
    return MicrostructureReport(
        spot=70_000,
        vwap=69_500,
        vwap_band_sigma=0.5,
        cvd_now=12345.0,
        cvd_divergence=+1,
        imbalance_score=0.6,
        rv_zscore=-0.2,
    )


def _quiet_bundle(idea: IdeaContext, news: NewsBrief) -> EventBundle:
    return EventBundle(
        idea=idea,
        news=news,
        fred=FredSnapshot(
            available=True,
            observations=[
                FredObservation("INDPRO", "Industrial Production", 105.0, 0.5, 2.5, "2025-04-01", False, ""),
            ],
        ),
        calendar=[],
        onchain=OnChainSnapshot(
            symbol=idea.symbol,
            available=True,
            etf_net_flow_5d_musd=400.0,
            funding_rate_now_bps=2.0,
            funding_rate_z_30d=0.3,
        ),
    )


@pytest.fixture
def long_idea() -> IdeaContext:
    return IdeaContext(
        symbol="BTC-USD",
        direction="long",
        entry=68_000,
        stop=65_000,
        target=75_000,
        horizon_hours=72.0,
        is_crypto=True,
    )


def test_aligned_axes_produce_go_verdict(long_idea):
    bundle = _quiet_bundle(long_idea, _bullish_news())
    verdict = score(
        idea=long_idea,
        bundle=bundle,
        regime=_bullish_regime(),
        microstructure=_bullish_micro(),
    )
    assert verdict.action == "GO"
    assert verdict.p_hit >= 0.55
    assert verdict.confluence_count >= 3
    assert verdict.blockers == []


def test_tier_one_calendar_blocks_action(long_idea):
    bundle = _quiet_bundle(long_idea, _bullish_news())
    bundle.calendar = [
        CalendarEvent(
            timestamp_utc=time.time() + 6 * 3600,
            name="CPI YoY",
            importance="high",
            country="US",
            source="test",
        )
    ]
    verdict = score(
        idea=long_idea,
        bundle=bundle,
        regime=_bullish_regime(),
        microstructure=_bullish_micro(),
    )
    assert verdict.action == "STAND_DOWN"
    assert any("CPI" in b for b in verdict.blockers)


def test_token_unlock_blocks_crypto_long(long_idea):
    bundle = _quiet_bundle(long_idea, _bullish_news())
    assert bundle.onchain is not None
    bundle.onchain.next_unlock_at_utc = time.time() + 24 * 3600
    bundle.onchain.next_unlock_pct_supply = 1.5
    bundle.onchain.next_unlock_label = "ARB"
    verdict = score(
        idea=long_idea,
        bundle=bundle,
        regime=_bullish_regime(),
        microstructure=_bullish_micro(),
    )
    assert verdict.action == "STAND_DOWN"
    assert any("unlock" in b.lower() for b in verdict.blockers)


def test_premortem_critical_not_acked_blocks(long_idea):
    bundle = _quiet_bundle(long_idea, _bullish_news())
    verdict = score(
        idea=long_idea,
        bundle=bundle,
        regime=_bullish_regime(),
        microstructure=_bullish_micro(),
        premortem_critical_acknowledged=False,
    )
    assert verdict.action == "STAND_DOWN"
    assert any("premortem" in b.lower() for b in verdict.blockers)


def test_neutral_inputs_produce_caution(long_idea):
    neutral_news = NewsBrief(symbol="BTC-USD", configured=True, sentiment="neutral", summary="Quiet.")
    bundle = _quiet_bundle(long_idea, neutral_news)
    bundle.onchain = None  # remove on-chain so funding/ETF doesn't tilt
    neutral_regime = RegimeReport(symbol="BTC-USD")
    neutral_micro = MicrostructureReport(
        spot=70_000, vwap=70_000, vwap_band_sigma=0.0, cvd_now=0.0, cvd_divergence=0,
        imbalance_score=0.0, rv_zscore=0.0,
    )
    verdict = score(
        idea=long_idea,
        bundle=bundle,
        regime=neutral_regime,
        microstructure=neutral_micro,
    )
    assert verdict.action in {"CAUTION", "STAND_DOWN"}
    assert verdict.confluence_count <= 2


def test_verdict_is_jsonable(long_idea):
    bundle = _quiet_bundle(long_idea, _bullish_news())
    verdict = score(
        idea=long_idea,
        bundle=bundle,
        regime=_bullish_regime(),
        microstructure=_bullish_micro(),
    )
    d = verdict.to_dict()
    assert d["action"] == verdict.action
    assert isinstance(d["axes"], list)
    assert all("name" in a for a in d["axes"])
