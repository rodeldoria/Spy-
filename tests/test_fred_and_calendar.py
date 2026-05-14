"""Tests for the FRED and economic-calendar wrappers — no live network."""
from __future__ import annotations

import time

import pytest

from monte.data import econ_calendar, fred


@pytest.fixture(autouse=True)
def _clear_caches():
    fred.clear_cache()
    econ_calendar.clear_cache()
    yield
    fred.clear_cache()
    econ_calendar.clear_cache()


def test_fred_unavailable_when_key_missing(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    snap = fred.snapshot()
    assert snap.available is False
    assert "FRED_API_KEY" in snap.error


def test_fred_uses_provided_key(monkeypatch):
    captured: dict[str, str] = {}

    def fake_fetch(series_id, label, threshold, warn_direction, api_key):
        captured["called"] = api_key
        return fred.FredObservation(
            series_id, label, 1.0, 0.0, None, "2025-01-01", False, "ok"
        )

    monkeypatch.setattr(fred, "_fetch_one", fake_fetch)
    snap = fred.snapshot(api_key="abc123")
    assert snap.available is True
    assert captured["called"] == "abc123"
    assert snap.observations  # at least one


def test_econ_calendar_filters_to_window(monkeypatch):
    now = time.time()
    fake_events = [
        econ_calendar.CalendarEvent(now + 3600, "CPI", "high", "US", source="fake"),
        econ_calendar.CalendarEvent(now + 48 * 3600, "FOMC", "high", "US", source="fake"),
        econ_calendar.CalendarEvent(now - 3600, "Past", "high", "US", source="fake"),
    ]
    monkeypatch.setattr(econ_calendar, "_fetch_forexfactory", lambda: fake_events)
    out = econ_calendar.next_hours(hours=24, countries=("US",), provider="forexfactory")
    names = [e.name for e in out]
    assert "CPI" in names
    assert "FOMC" not in names
    assert "Past" not in names


def test_calendar_country_filter(monkeypatch):
    now = time.time()
    fake = [
        econ_calendar.CalendarEvent(now + 3600, "CPI", "high", "US", source="fake"),
        econ_calendar.CalendarEvent(now + 3600, "ECB rate decision", "high", "EUR", source="fake"),
    ]
    monkeypatch.setattr(econ_calendar, "_fetch_forexfactory", lambda: fake)
    only_us = econ_calendar.next_hours(hours=24, countries=("US",), provider="forexfactory")
    assert all(e.country == "US" for e in only_us)


def test_calendar_event_tier_1_keywords():
    e = econ_calendar.CalendarEvent(time.time() + 3600, "Core CPI YoY", "medium", "US")
    assert e.is_tier_1
    e_low = econ_calendar.CalendarEvent(time.time() + 3600, "Building permits", "low", "US")
    assert not e_low.is_tier_1


def test_forexfactory_xml_parser_handles_minimal_doc():
    body = b"""<weeklyevents>
        <event>
            <title>CPI YoY</title>
            <country>USD</country>
            <date>06-15-2025</date>
            <time>8:30am</time>
            <impact>High</impact>
            <forecast>3.0%</forecast>
            <previous>3.1%</previous>
        </event>
    </weeklyevents>"""
    out = econ_calendar._parse_forexfactory_xml(body)
    assert len(out) == 1
    assert out[0].name == "CPI YoY"
    assert out[0].country == "USD"
    assert out[0].importance == "high"
