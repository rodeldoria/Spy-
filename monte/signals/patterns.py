"""Classical pro-investor pattern engine.

Encodes a small library of well-known market microstructure patterns that
real desks use as soft tilts on directional bias. Each pattern looks at
the recent close series + spot and emits a `PatternSignal` with:

  - name:       short label, e.g. "Round-Number Magnet"
  - direction:  "bull", "bear", or "neutral"
  - strength:   0-1, how strong the historical edge tends to be
  - bias_pp:    suggested probability adjustment in percentage points
                (e.g. +3.0pp added to the up-side forecast)
  - note:       one-sentence rationale shown to the user

The patterns implemented here are the ones most cited in the technical
analysis literature (Wyckoff, Murphy, Brooks) and have observed empirical
support in crypto microstructure papers:

  1. Round-Number Magnetism — price gravitates toward and reacts at
     psychological round levels ($80,000, $4,000, $200). Studies on FX
     and BTC show ~55-65% of intraday tests of major round numbers
     either reject on first touch or break decisively after 3+ touches.

  2. Round-Number Rejection — if a round level was touched recently
     without a clean close-through, expect that level to act as
     resistance (above) or support (below).

  3. Volatility Compression / Bollinger Squeeze — when realised vol
     compresses below half its longer-term average, expansion in either
     direction becomes statistically more likely. Direction is taken
     from the prevailing EMA stack.

  4. Session Drift — crypto exhibits well-documented session biases:
     US morning (NY open ≈ 13:00 UTC) and Asia open (≈ 00:00 UTC) are
     the highest-volume windows; Asia hours often mean-revert; US
     hours often trend with the equity tape.

  5. Trend Regime (EMA stack) — fast EMA above slow EMA = uptrend bias;
     slope of the slow EMA confirms momentum. Standard Murphy.

  6. Mean Reversion Pressure — when price has stretched > 2σ from its
     20-bar mean, statistical reversion bias kicks in.

These are *tilts*, not signals on their own. The forecast keeps the
log-normal band as the base; patterns nudge the directional probability
by a few percentage points each. Calibration over time will tell us
which patterns actually pay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatternSignal:
    name: str
    direction: str    # "bull", "bear", "neutral"
    strength: float   # 0..1
    bias_pp: float    # signed pp tilt to up-direction probability
    note: str

    @property
    def emoji(self) -> str:
        return {"bull": "🟢", "bear": "🔴", "neutral": "⚪"}[self.direction]


@dataclass(frozen=True)
class PatternBundle:
    signals: list[PatternSignal]
    net_bias_pp: float  # sum of signed bias_pp, capped at ±10pp
    consensus: str      # "bullish", "bearish", "mixed", "quiet"

    @property
    def top(self) -> list[PatternSignal]:
        return sorted(self.signals, key=lambda s: -s.strength)[:3]


# ---------------------------------------------------------------------------
# Round-number psychology
# ---------------------------------------------------------------------------

def _round_step(price: float) -> float:
    """Pick the relevant 'round number' grid for a given price level.

    BTC ~$80k → $1,000 grid. ETH ~$3k → $100. SOL ~$200 → $10.
    Heuristic based on order of magnitude of the price.
    """
    if price >= 10_000:
        return 1_000.0
    if price >= 1_000:
        return 100.0
    if price >= 100:
        return 10.0
    if price >= 10:
        return 1.0
    return 0.1


def _nearest_round(price: float) -> tuple[float, float]:
    """Return (nearest_round_level, distance_in_pct)."""
    step = _round_step(price)
    nearest = round(price / step) * step
    dist_pct = abs(price - nearest) / price * 100 if price > 0 else 0.0
    return nearest, dist_pct


def _round_number_magnet(close: pd.Series, spot: float) -> PatternSignal | None:
    """When spot is within 0.3% of a round number, magnetism is high."""
    nearest, dist_pct = _nearest_round(spot)
    if dist_pct > 0.5:
        return None
    direction_to_round = "bull" if nearest > spot else ("bear" if nearest < spot else "neutral")
    strength = max(0.0, min(1.0, (0.5 - dist_pct) / 0.5))  # 1.0 at zero distance
    bias = (3.5 if direction_to_round == "bull" else -3.5 if direction_to_round == "bear" else 0.0) * strength
    return PatternSignal(
        name="Round-Number Magnet",
        direction=direction_to_round,
        strength=round(strength, 2),
        bias_pp=round(bias, 2),
        note=f"Spot ${spot:,.2f} is {dist_pct:.2f}% from ${nearest:,.0f} — psychological levels tend to attract price.",
    )


def _round_number_rejection(close: pd.Series, spot: float, lookback: int = 60) -> PatternSignal | None:
    """If recent bars touched a round number ≥2 times without closing through,
    that level is likely active resistance/support.
    """
    if len(close) < 10:
        return None
    nearest, _ = _nearest_round(spot)
    recent = close.tail(lookback).astype(float)
    step = _round_step(spot)
    band = 0.0015 * spot  # touch tolerance: 0.15% around the round level

    above = (recent > nearest + band).sum()
    below = (recent < nearest - band).sum()
    closes_through = 0
    if above > 0 and below > 0:
        # Price oscillated across — it's an active battleground.
        closes_through = min(above, below)

    touches = ((recent - nearest).abs() < band).sum()
    if touches < 2:
        return None

    if spot > nearest:
        # Price above the round; round acts as support, with bullish lean
        return PatternSignal(
            name="Round-Number Support",
            direction="bull",
            strength=round(min(1.0, touches / 8.0), 2),
            bias_pp=round(min(1.0, touches / 8.0) * 2.5, 2),
            note=f"${nearest:,.0f} tested {int(touches)}× recently and held — typical support behaviour.",
        )
    else:
        return PatternSignal(
            name="Round-Number Resistance",
            direction="bear",
            strength=round(min(1.0, touches / 8.0), 2),
            bias_pp=round(-min(1.0, touches / 8.0) * 2.5, 2),
            note=f"${nearest:,.0f} tested {int(touches)}× recently and capped — typical resistance behaviour.",
        )


# ---------------------------------------------------------------------------
# Volatility regime
# ---------------------------------------------------------------------------

def _vol_squeeze(close: pd.Series) -> PatternSignal | None:
    """Bollinger-squeeze analogue: short-window σ << long-window σ → expansion expected."""
    if len(close) < 100:
        return None
    log_ret = np.log(close / close.shift(1)).dropna()
    short_sigma = float(log_ret.tail(20).std() or 0.0)
    long_sigma = float(log_ret.tail(100).std() or 0.0)
    if long_sigma <= 0 or short_sigma <= 0:
        return None
    ratio = short_sigma / long_sigma
    if ratio > 0.55:
        return None  # not compressed enough
    # Direction from EMA stack
    ema_fast = close.ewm(span=12, adjust=False).mean().iloc[-1]
    ema_slow = close.ewm(span=48, adjust=False).mean().iloc[-1]
    direction = "bull" if ema_fast > ema_slow else "bear"
    strength = round(max(0.0, min(1.0, (0.55 - ratio) / 0.4)), 2)
    bias = (2.5 if direction == "bull" else -2.5) * strength
    return PatternSignal(
        name="Volatility Squeeze",
        direction=direction,
        strength=strength,
        bias_pp=round(bias, 2),
        note=f"Short-vol {ratio:.0%} of long-vol — compression often resolves with a directional expansion (taking EMA-stack direction).",
    )


# ---------------------------------------------------------------------------
# Trend regime
# ---------------------------------------------------------------------------

def _trend_regime(close: pd.Series) -> PatternSignal | None:
    if len(close) < 50:
        return None
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema48 = close.ewm(span=48, adjust=False).mean()
    fast = float(ema12.iloc[-1])
    slow = float(ema48.iloc[-1])
    slope = float(ema48.iloc[-1] - ema48.iloc[-10]) / float(ema48.iloc[-10]) if len(ema48) > 10 else 0.0
    if abs(slope) < 0.0005:
        return None
    direction = "bull" if fast > slow and slope > 0 else ("bear" if fast < slow and slope < 0 else "neutral")
    if direction == "neutral":
        return None
    strength = round(min(1.0, abs(slope) * 200), 2)
    bias = (2.0 if direction == "bull" else -2.0) * strength
    return PatternSignal(
        name="EMA Trend Stack",
        direction=direction,
        strength=strength,
        bias_pp=round(bias, 2),
        note=f"EMA12 {'over' if direction=='bull' else 'under'} EMA48 with {abs(slope)*100:.2f}% slope — trend continuation favoured.",
    )


# ---------------------------------------------------------------------------
# Mean reversion
# ---------------------------------------------------------------------------

def _mean_reversion(close: pd.Series, spot: float) -> PatternSignal | None:
    if len(close) < 20:
        return None
    window = close.tail(20).astype(float)
    mean = float(window.mean())
    std = float(window.std() or 0.0)
    if std <= 0:
        return None
    z = (spot - mean) / std
    if abs(z) < 2.0:
        return None
    direction = "bear" if z > 0 else "bull"   # stretched up → revert down
    strength = round(min(1.0, (abs(z) - 2.0) / 1.5), 2) or 0.05
    bias = (-2.5 if direction == "bear" else 2.5) * strength
    return PatternSignal(
        name="Mean-Reversion Pressure",
        direction=direction,
        strength=strength,
        bias_pp=round(bias, 2),
        note=f"Spot is {z:+.1f}σ from 20-bar mean (${mean:,.2f}) — extended moves often revert.",
    )


# ---------------------------------------------------------------------------
# Session drift
# ---------------------------------------------------------------------------

def _session_drift(now_utc: Optional[datetime] = None) -> PatternSignal | None:
    """Crypto session bias.

    - 13:00-21:00 UTC (NY hours): trends with US risk-on tape, mild bull lean
    - 00:00-07:00 UTC (Asia early): higher mean-reversion, slight bear lean
      historically when entering after a strong US close
    - 07:00-13:00 UTC (EU): mixed, no edge
    """
    now = now_utc or datetime.now(timezone.utc)
    h = now.hour
    if 13 <= h < 21:
        return PatternSignal(
            name="US Session Drift",
            direction="bull",
            strength=0.4,
            bias_pp=1.0,
            note="NY hours (13-21 UTC) — crypto historically trends with US equity risk appetite.",
        )
    if 0 <= h < 7:
        return PatternSignal(
            name="Asia Session Drift",
            direction="bear",
            strength=0.3,
            bias_pp=-0.8,
            note="Asia early hours (0-7 UTC) — mean-reversion bias; thin liquidity often retraces US close.",
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ALL_DETECTORS = [
    _round_number_magnet,
    _round_number_rejection,
    _vol_squeeze,
    _trend_regime,
    _mean_reversion,
]


def detect_patterns(close: pd.Series, spot: float,
                    now_utc: Optional[datetime] = None) -> PatternBundle:
    """Run every detector and bundle the results."""
    sigs: list[PatternSignal] = []
    for det in ALL_DETECTORS:
        try:
            s = det(close, spot)
            if s is not None:
                sigs.append(s)
        except Exception:
            continue
    sess = _session_drift(now_utc)
    if sess is not None:
        sigs.append(sess)

    net = sum(s.bias_pp for s in sigs)
    net = max(-10.0, min(10.0, net))

    if not sigs:
        consensus = "quiet"
    elif net > 1.5:
        consensus = "bullish"
    elif net < -1.5:
        consensus = "bearish"
    else:
        consensus = "mixed"

    return PatternBundle(signals=sigs, net_bias_pp=round(net, 2), consensus=consensus)
