"""Smoke tests for the event aggregator and the catalyst feed wrappers.

We monkey-patch the network paths so the tests run offline and quickly.
"""
from __future__ import annotations

import time

import pytest

from monte.data import econ_calendar
from monte.data.fred import FredObservation, FredSnapshot
from monte.data.onchain import OnChainSnapshot
from monte.intel import event_aggregator as ea
from monte.intel.perplexity import NewsBrief


@pytest.fixture(autouse=True)
def _clear_cal_cache():
    econ_calendar.clear_cache()
    yield
    econ_calendar.clear_cache()


def _fake_news(*_args, **_kwargs):
    return NewsBrief(
        symbol="BTC-USD",
        configured=True,
        sentiment="bullish",
        summary="ETF inflows accelerating; spot grinding higher.",
        headlines=["Spot ETF inflows hit weekly high"],
        catalysts=["FOMC next week"],
        fetched_at=time.time(),
    )


def _fake_fred():
    return FredSnapshot(
        available=True,
        observations=[
            FredObservation("INDPRO", "Industrial Production", 105.0, 0.5, 2.5, "2025-04-01", False, ""),
            FredObservation("CPIAUCSL", "CPI", 312.0, 0.3, 2.4, "2025-04-01", False, ""),
        ],
    )


def _fake_calendar(hours=24, **_kwargs):
    return [
        econ_calendar.CalendarEvent(
            timestamp_utc=time.time() + 4 * 3600,
            name="CPI YoY",
            importance="high",
            country="US",
            source="test",
        ),
    ]


def _fake_onchain(symbol, *, hold_window_hours=168.0):
    return OnChainSnapshot(
        symbol=symbol,
        available=True,
        fetched_at=time.time(),
        etf_net_flow_5d_musd=300.0,
        etf_last_day_musd=50.0,
        funding_rate_now_bps=2.5,
        funding_rate_z_30d=0.4,
        next_unlock_at_utc=None,
    )


def test_aggregator_runs_in_parallel_and_aggregates(monkeypatch):
    monkeypatch.setattr(ea, "fetch_news", _fake_news)
    monkeypatch.setattr(ea, "fred_snapshot", _fake_fred)
    monkeypatch.setattr(ea, "next_hours", _fake_calendar)
    monkeypatch.setattr(ea, "onchain_snapshot", _fake_onchain)

    idea = ea.IdeaContext(symbol="BTC-USD", direction="long", horizon_hours=72.0, is_crypto=True)
    bundle = ea.gather(idea, deadline_seconds=2.0)

    assert bundle.news is not None and bundle.news.sentiment == "bullish"
    assert bundle.fred is not None and bundle.fred.available
    assert bundle.calendar and bundle.calendar[0].name == "CPI YoY"
    assert bundle.onchain is not None and bundle.onchain.etf_net_flow_5d_musd == 300.0
    assert bundle.errors == {}
    assert bundle.elapsed_ms >= 0
    assert bundle.tier_1_in_window(), "CPI inside 72h hold should appear as tier-1"


def test_aggregator_skips_onchain_for_non_crypto(monkeypatch):
    monkeypatch.setattr(ea, "fetch_news", _fake_news)
    monkeypatch.setattr(ea, "fred_snapshot", _fake_fred)
    monkeypatch.setattr(ea, "next_hours", lambda *a, **kw: [])
    monkeypatch.setattr(ea, "onchain_snapshot",
                        lambda *a, **kw: pytest.fail("on-chain shouldn't be called for SPY"))

    idea = ea.IdeaContext(symbol="SPY", direction="long", is_crypto=False)
    bundle = ea.gather(idea, deadline_seconds=2.0)
    assert bundle.onchain is None
    assert bundle.tier_1_in_window() == []


def test_aggregator_captures_provider_errors(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(ea, "fetch_news", boom)
    monkeypatch.setattr(ea, "fred_snapshot", _fake_fred)
    monkeypatch.setattr(ea, "next_hours", _fake_calendar)

    idea = ea.IdeaContext(symbol="SPY", direction="long", is_crypto=False)
    bundle = ea.gather(idea, deadline_seconds=2.0)
    assert "news" in bundle.errors
    assert "RuntimeError" in bundle.errors["news"]
