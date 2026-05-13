"""Short-horizon price forecaster.

Uses the recent volatility profile of the asset to project a median price
and a 1-sigma band at a target time. Honest about uncertainty — we never
claim to predict, only to bound the likely range based on recent realised
volatility and drift.

Model: log-returns of the last `lookback` bars give a per-bar drift μ and
volatility σ. Over `n_bars` ahead, the projected price distribution is
log-normal with:
    median = spot * exp(μ * n_bars)
    upper  = spot * exp(μ * n_bars + σ * sqrt(n_bars))
    lower  = spot * exp(μ * n_bars − σ * sqrt(n_bars))

Good for Kalshi / range-bound markets where an honest probabilistic band is
more useful than a point prediction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd


# Timeframe → bar duration in minutes.
_TF_MINUTES = {
    "1m":   1, "2m": 2, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h":  60, "2h": 120, "3h": 180, "4h": 240, "6h": 360,
    "1d": 1440,
}


def timeframe_minutes(timeframe: str) -> int:
    return _TF_MINUTES.get(timeframe, 15)


@dataclass
class PriceProjection:
    label: str          # human label e.g. "15m" or "8 AM PST"
    target_dt: datetime # tz-aware UTC target
    minutes_ahead: float
    n_bars: float
    median: float       # projected median price
    lower: float        # 1-sigma lower band
    upper: float        # 1-sigma upper band
    drift_pct: float    # implied % move (median vs spot)
    range_pct: float    # half-width of band in % (σ at horizon)
    spot: float         # current price used as anchor

    @property
    def direction(self) -> str:
        if self.drift_pct > 0.05:
            return "up"
        if self.drift_pct < -0.05:
            return "down"
        return "flat"

    @property
    def target_pst(self) -> datetime:
        """Target time in Pacific (fixed UTC-8 — close enough for label use)."""
        return self.target_dt + timedelta(hours=-8)

    def label_pst(self) -> str:
        """Render target time as 'h:MMa PST' (e.g. '8:00a PST')."""
        pst = self.target_pst
        return pst.strftime("%-I:%M%p PST").replace("AM", "a").replace("PM", "p")

    def label_utc(self) -> str:
        return self.target_dt.strftime("%H:%M UTC")


def _drift_vol(close: pd.Series, lookback: int = 96) -> tuple[float, float]:
    """Per-bar log-return drift μ and volatility σ.

    Caps σ at 5%/bar to avoid wild bands when a fresh regime change has
    pushed std artificially high.
    """
    if len(close) < 10:
        return 0.0, 0.005
    series = close.tail(lookback + 1).astype(float)
    log_ret = np.log(series / series.shift(1)).dropna()
    if log_ret.empty:
        return 0.0, 0.005
    mu = float(log_ret.mean())
    sigma = float(log_ret.std())
    if not np.isfinite(mu):
        mu = 0.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.005
    return mu, min(sigma, 0.05)


def project_price(
    close: pd.Series,
    spot: float,
    minutes_ahead: float,
    timeframe: str,
    label: str,
    target_dt: datetime,
    lookback: int = 96,
) -> PriceProjection:
    """Project price `minutes_ahead` minutes from now using recent vol/drift."""
    bar_min = max(timeframe_minutes(timeframe), 1)
    n_bars = max(minutes_ahead / bar_min, 0.0)
    mu, sigma = _drift_vol(close, lookback=lookback)

    drift = mu * n_bars
    spread = sigma * math.sqrt(n_bars) if n_bars > 0 else 0.0

    median = spot * math.exp(drift)
    lower = spot * math.exp(drift - spread)
    upper = spot * math.exp(drift + spread)

    drift_pct = (median / spot - 1.0) * 100 if spot > 0 else 0.0
    range_pct = ((upper / median) - 1.0) * 100 if median > 0 else 0.0

    return PriceProjection(
        label=label,
        target_dt=target_dt,
        minutes_ahead=minutes_ahead,
        n_bars=n_bars,
        median=median,
        lower=lower,
        upper=upper,
        drift_pct=drift_pct,
        range_pct=range_pct,
        spot=spot,
    )


def _next_pst_8am(now_utc: datetime) -> datetime:
    """Next 8am Pacific Time (PST/PDT handled as fixed UTC-8 — close enough
    for a forecast horizon)."""
    pst_offset = timedelta(hours=-8)
    pst_now = now_utc + pst_offset
    target_pst = pst_now.replace(hour=8, minute=0, second=0, microsecond=0)
    if target_pst <= pst_now:
        target_pst += timedelta(days=1)
    return target_pst - pst_offset  # back to UTC


def _next_three_hour_mark(now_utc: datetime) -> datetime:
    """Next clock-aligned 3-hour mark in PST (00, 03, 06, 09, 12, 15, 18, 21)."""
    pst_offset = timedelta(hours=-8)
    pst_now = now_utc + pst_offset
    next_hour = ((pst_now.hour // 3) + 1) * 3
    if next_hour >= 24:
        target_pst = (pst_now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    else:
        target_pst = pst_now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
    return target_pst - pst_offset


def _pst_label(target_utc: datetime) -> str:
    pst = target_utc + timedelta(hours=-8)
    return pst.strftime("%-I %p PST")


def standard_horizons(
    close: pd.Series,
    spot: float,
    timeframe: str,
    now: Optional[datetime] = None,
) -> list[PriceProjection]:
    """Generate the standard forecast grid: 15m, 1h, 3h, next-8am-PST,
    next-3h-PST mark.
    """
    now = now or datetime.now(timezone.utc)

    horizons: list[tuple[str, datetime, float]] = [
        ("15 min",  now + timedelta(minutes=15),  15.0),
        ("1 hr",    now + timedelta(hours=1),     60.0),
        ("3 hr",    now + timedelta(hours=3),     180.0),
    ]

    next_8am = _next_pst_8am(now)
    minutes_to_8am = (next_8am - now).total_seconds() / 60
    if 30 <= minutes_to_8am <= 60 * 36:
        horizons.append(("8 AM PST", next_8am, minutes_to_8am))

    next_3h = _next_three_hour_mark(now)
    minutes_to_3h = (next_3h - now).total_seconds() / 60
    if minutes_to_3h >= 30 and abs(minutes_to_3h - 180) > 10:
        horizons.append((f"Next {_pst_label(next_3h)}", next_3h, minutes_to_3h))

    return [
        project_price(close, spot, mins, timeframe, label, dt)
        for label, dt, mins in horizons
    ]
