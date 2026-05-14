"""Economic-calendar fetcher for the event-aware chat widget.

Returns the list of macroeconomic prints scheduled inside the next N hours,
ranked by importance, so the likelihood-gate can hard-block longs that
would otherwise sit through a tier-1 release (FOMC, CPI, NFP, etc.).

Two providers are supported:

- `forexfactory` (default, free, no key) — scrapes the public weekly XML
  feed at https://www.forexfactory.com/ffcal_week_this.xml
- `tradingeconomics` — opt-in via `TE_API_KEY`; richer schema and country
  filters but rate-limited on the free tier.

Both paths return the same `CalendarEvent` shape so callers don't branch.
We cache for 30 minutes per provider so the widget can run on every submit
without hammering anyone.
"""
from __future__ import annotations

import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from xml.etree import ElementTree as ET

_CACHE_TTL_SECONDS = 60 * 30
_CACHE: dict[str, tuple[float, list["CalendarEvent"]]] = {}

FOREX_FACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
TRADING_ECONOMICS_URL = "https://api.tradingeconomics.com/calendar"

# Hard list of "tier-1" event names — these get bumped to importance "high"
# regardless of what the source says. The likelihood-gate uses this to
# decide whether to STAND_DOWN longs that would sit through the print.
TIER_1_KEYWORDS: tuple[str, ...] = (
    "fomc", "cpi", "core cpi", "ppi", "nonfarm payrolls", "nfp",
    "fed chair", "rate decision", "interest rate", "powell",
    "gdp", "ism manufacturing", "pce", "core pce",
)


@dataclass
class CalendarEvent:
    timestamp_utc: float        # epoch seconds
    name: str
    importance: str             # "low" | "medium" | "high"
    country: str                # 2-letter or "US" / "EUR"
    forecast: str = ""
    previous: str = ""
    actual: str = ""
    source: str = ""

    @property
    def is_tier_1(self) -> bool:
        n = self.name.lower()
        return self.importance == "high" or any(k in n for k in TIER_1_KEYWORDS)

    @property
    def hours_from_now(self) -> float:
        return (self.timestamp_utc - time.time()) / 3600.0


def next_hours(
    hours: int = 24,
    *,
    countries: Optional[tuple[str, ...]] = ("US", "EUR", "GB"),
    provider: Optional[str] = None,
) -> list[CalendarEvent]:
    """Return calendar events occurring within the next `hours` hours.

    Filters to high-impact regions by default (US / Eurozone / UK) — the
    gate cares about events that move global risk assets. Pass
    `countries=None` to disable filtering.
    """
    provider = (provider or os.environ.get("MONTE_CALENDAR_PROVIDER", "forexfactory")).strip().lower()
    cache_key = f"{provider}::{','.join(countries or ())}"
    cached = _CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < _CACHE_TTL_SECONDS:
        events = cached[1]
    else:
        try:
            if provider == "tradingeconomics":
                events = _fetch_tradingeconomics()
            else:
                events = _fetch_forexfactory()
        except Exception:  # noqa: BLE001
            events = []
        _CACHE[cache_key] = (time.time(), events)

    now = time.time()
    horizon = now + hours * 3600.0
    out: list[CalendarEvent] = []
    for e in events:
        if e.timestamp_utc < now or e.timestamp_utc > horizon:
            continue
        if countries and e.country and e.country.upper() not in {c.upper() for c in countries}:
            continue
        out.append(e)
    out.sort(key=lambda e: (e.timestamp_utc, _importance_rank(e.importance)))
    return out


def clear_cache() -> None:
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Forex Factory provider (free, no key)
# ---------------------------------------------------------------------------

def _fetch_forexfactory() -> list[CalendarEvent]:
    req = urllib.request.Request(
        FOREX_FACTORY_URL,
        headers={"User-Agent": "spy-event-chat/1.0"},
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        body = resp.read()
    return _parse_forexfactory_xml(body)


def _parse_forexfactory_xml(body: bytes) -> list[CalendarEvent]:
    """Parse the FF "weekly" XML. Schema:
        <weeklyevents>
          <event>
            <title>...</title>
            <country>USD</country>
            <date>11-22-2025</date>     (US format)
            <time>8:30am</time>
            <impact>High|Medium|Low</impact>
            <forecast>...</forecast>
            <previous>...</previous>
          </event>
          ...
        </weeklyevents>
    """
    out: list[CalendarEvent] = []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return out
    for ev in root.findall("event"):
        title = (ev.findtext("title") or "").strip()
        if not title:
            continue
        date_s = (ev.findtext("date") or "").strip()
        time_s = (ev.findtext("time") or "").strip()
        ts = _parse_ff_datetime(date_s, time_s)
        if ts is None:
            continue
        country = (ev.findtext("country") or "").strip().upper()
        impact = (ev.findtext("impact") or "Low").strip().lower()
        out.append(
            CalendarEvent(
                timestamp_utc=ts,
                name=title,
                importance=impact,
                country=country,
                forecast=(ev.findtext("forecast") or "").strip(),
                previous=(ev.findtext("previous") or "").strip(),
                source="forexfactory",
            )
        )
    return out


_FF_DATE_RE = re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})$")
_FF_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})\s*(am|pm)$", re.IGNORECASE)


def _parse_ff_datetime(date_s: str, time_s: str) -> Optional[float]:
    dm = _FF_DATE_RE.match(date_s)
    if not dm:
        return None
    month, day, year = (int(g) for g in dm.groups())
    if not time_s or "all day" in time_s.lower() or "tentative" in time_s.lower():
        # Pin all-day events to noon UTC so they survive the next-N-hours filter.
        hour, minute = 12, 0
    else:
        tm = _FF_TIME_RE.match(time_s)
        if not tm:
            return None
        hour = int(tm.group(1)) % 12
        minute = int(tm.group(2))
        if tm.group(3).lower() == "pm":
            hour += 12
    try:
        # FF publishes in US Eastern by default. We treat the timestamp as
        # naive ET and convert to UTC by adding 5h (no DST handling — close
        # enough for a 30-minute cache and a 24h look-ahead).
        dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc) + timedelta(hours=5)
        return dt.timestamp()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# TradingEconomics provider (optional)
# ---------------------------------------------------------------------------

def _fetch_tradingeconomics() -> list[CalendarEvent]:
    api_key = os.environ.get("TE_API_KEY", "").strip()
    if not api_key:
        return []
    qs = urllib.parse.urlencode({"c": api_key, "f": "json"})
    url = f"{TRADING_ECONOMICS_URL}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "spy-event-chat/1.0"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        import json as _json
        data = _json.loads(resp.read().decode("utf-8", errors="ignore"))
    out: list[CalendarEvent] = []
    if not isinstance(data, list):
        return out
    for r in data:
        try:
            dt = datetime.strptime(r.get("Date", "")[:19], "%Y-%m-%dT%H:%M:%S")
            ts = dt.replace(tzinfo=timezone.utc).timestamp()
        except (ValueError, AttributeError):
            continue
        out.append(
            CalendarEvent(
                timestamp_utc=ts,
                name=str(r.get("Event", "")).strip(),
                importance=_te_importance(r.get("Importance")),
                country=str(r.get("Country", "")).upper()[:3],
                forecast=str(r.get("Forecast", "")),
                previous=str(r.get("Previous", "")),
                actual=str(r.get("Actual", "")),
                source="tradingeconomics",
            )
        )
    return out


def _te_importance(raw: object) -> str:
    try:
        n = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "low"
    if n >= 3:
        return "high"
    if n == 2:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _importance_rank(imp: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(imp.lower(), 3)


__all__ = [
    "CalendarEvent",
    "next_hours",
    "clear_cache",
    "TIER_1_KEYWORDS",
]
