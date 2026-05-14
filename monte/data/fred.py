"""FRED macro snapshot for the event-aware chat widget.

Pulls a small, curated panel of recession / risk indicators directly from
the public St. Louis Fed `series/observations` endpoint (no third-party
SDK — keeps the dependency surface tiny). When `FRED_API_KEY` is missing
we return an `unavailable=True` snapshot rather than raising; the UI shows
a single "FRED key not set" chip in that case.

Indicators tracked:
- SAHMREALTIME      — Sahm Rule recession indicator (>= 0.5 = warning)
- T10Y2Y            — 10Y minus 2Y treasury spread (negative = yield-curve inversion)
- T10Y3M            — 10Y minus 3M treasury spread (NY Fed's recession-prob input)
- NFCI              — Chicago Fed National Financial Conditions Index (>0 = tight)
- BAMLH0A0HYM2      — ICE BofA US HY OAS (credit-spread proxy, bps/100)
- USSLIND           — Conference Board's Leading Economic Index (state-level pulse)
- CPIAUCSL          — headline CPI (used for YoY inflation trend)
- INDPRO            — industrial production (growth proxy when LEI lags)

Each indicator returns the latest value, the 30-day change, and a label
telling the user whether the level is in a recession-warning zone. The
snapshot is cached in-process for 15 minutes so the chat widget can run on
every submit without blowing through any provider's rate limit.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

FRED_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"
_CACHE_TTL_SECONDS = 60 * 15
_CACHE: dict[str, tuple[float, "FredSnapshot"]] = {}

# Series we always pull. (series_id, friendly_label, recession_threshold, "warn-direction")
# warn_direction: "above" means values > threshold are warnings, "below" means
# values < threshold are warnings.
SERIES_PANEL: list[tuple[str, str, float, str]] = [
    ("SAHMREALTIME", "Sahm Rule", 0.5, "above"),
    ("T10Y2Y",       "10Y-2Y spread", 0.0, "below"),
    ("T10Y3M",       "10Y-3M spread", 0.0, "below"),
    ("NFCI",         "Financial conditions (NFCI)", 0.0, "above"),
    ("BAMLH0A0HYM2", "HY OAS (bps/100)", 5.0, "above"),
    ("USSLIND",      "Leading Indicator (USSLIND)", 0.0, "below"),
    ("CPIAUCSL",     "CPI (level)", 0.0, "above"),
    ("INDPRO",       "Industrial Production", 0.0, "below"),
]


@dataclass
class FredObservation:
    series_id: str
    label: str
    value: float
    delta_30d: float           # absolute change vs the closest reading 30 days ago
    yoy_pct: Optional[float]   # YoY % change (for level series like CPIAUCSL)
    last_date: str             # ISO date of the latest observation
    warning: bool              # True when the level is in a recession-warning zone
    note: str                  # one-line interpretation


@dataclass
class FredSnapshot:
    available: bool
    error: str = ""
    fetched_at: float = 0.0
    observations: list[FredObservation] = field(default_factory=list)

    def by_id(self, series_id: str) -> Optional[FredObservation]:
        for obs in self.observations:
            if obs.series_id == series_id:
                return obs
        return None

    @property
    def warnings(self) -> list[FredObservation]:
        return [o for o in self.observations if o.warning]


def snapshot(*, api_key: Optional[str] = None) -> FredSnapshot:
    """Return a `FredSnapshot` covering the standard indicator panel.

    Network failures, missing keys, or malformed responses become an
    `available=False` snapshot rather than raising.
    """
    api_key = (api_key if api_key is not None else os.environ.get("FRED_API_KEY", "")).strip()
    if not api_key:
        return FredSnapshot(
            available=False,
            error="FRED_API_KEY not set — add one (free) at https://fredaccount.stlouisfed.org/apikeys.",
            fetched_at=time.time(),
        )

    cache_key = f"panel::{api_key[:6]}"
    cached = _CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    obs_list: list[FredObservation] = []
    errors: list[str] = []
    for series_id, label, threshold, warn_direction in SERIES_PANEL:
        try:
            obs = _fetch_one(series_id, label, threshold, warn_direction, api_key)
            if obs is not None:
                obs_list.append(obs)
        except Exception as e:  # noqa: BLE001 — single-series failures are tolerated
            errors.append(f"{series_id}: {type(e).__name__}: {str(e)[:80]}")

    snap = FredSnapshot(
        available=bool(obs_list),
        error="; ".join(errors) if errors and not obs_list else "",
        fetched_at=time.time(),
        observations=obs_list,
    )
    _CACHE[cache_key] = (time.time(), snap)
    return snap


def clear_cache() -> None:
    """Drop the in-process FRED cache. Used by tests."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _fetch_one(
    series_id: str,
    label: str,
    threshold: float,
    warn_direction: str,
    api_key: str,
) -> Optional[FredObservation]:
    qs = urllib.parse.urlencode(
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 400,   # ~13mo of dailies, enough for a YoY comparison
        }
    )
    url = f"{FRED_ENDPOINT}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "spy-event-chat/1.0"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="ignore"))

    raw = payload.get("observations") or []
    points: list[tuple[datetime, float]] = []
    for r in raw:
        v = r.get("value")
        if v in (None, ".", ""):
            continue
        try:
            d = datetime.strptime(r["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            points.append((d, float(v)))
        except (KeyError, ValueError):
            continue
    if not points:
        return None

    # `sort_order=desc` already; sort defensively.
    points.sort(key=lambda t: t[0], reverse=True)
    latest_date, latest_value = points[0]

    delta_30d = 0.0
    cutoff_30 = latest_date.timestamp() - 30 * 86400
    for d, v in points[1:]:
        if d.timestamp() <= cutoff_30:
            delta_30d = latest_value - v
            break

    yoy_pct: Optional[float] = None
    cutoff_365 = latest_date.timestamp() - 365 * 86400
    for d, v in points[1:]:
        if d.timestamp() <= cutoff_365 and v != 0:
            yoy_pct = (latest_value - v) / abs(v) * 100.0
            break

    warning = (
        (warn_direction == "above" and latest_value > threshold)
        or (warn_direction == "below" and latest_value < threshold)
    )
    note = _interpret(label, latest_value, threshold, warn_direction, yoy_pct)

    return FredObservation(
        series_id=series_id,
        label=label,
        value=latest_value,
        delta_30d=delta_30d,
        yoy_pct=yoy_pct,
        last_date=latest_date.strftime("%Y-%m-%d"),
        warning=warning,
        note=note,
    )


def _interpret(
    label: str,
    value: float,
    threshold: float,
    warn_direction: str,
    yoy_pct: Optional[float],
) -> str:
    if warn_direction == "above":
        relation = "above" if value > threshold else "below"
    else:
        relation = "below" if value < threshold else "above"
    base = f"{label}: {value:.2f} ({relation} {threshold:.2f} threshold)"
    if yoy_pct is not None and label.startswith("CPI"):
        base += f" — YoY {yoy_pct:+.1f}%"
    return base


__all__ = ["FredSnapshot", "FredObservation", "snapshot", "clear_cache", "SERIES_PANEL"]
