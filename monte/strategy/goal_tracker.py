"""Goal-aware math: target equity by a deadline → required returns + risk.

All functions are pure. They consume `PaperBook` and return scalars or
small dataclasses; no I/O, no state.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from monte.broker.paper_book import PaperBook
from monte.strategy.monte_edge import confidence_scaled_risk_pct


# ---------- Config ----------

DEFAULT_START_USD = float(os.environ.get("MONTE_GOAL_START_USD", "10000"))
DEFAULT_TARGET_USD = float(os.environ.get("MONTE_GOAL_TARGET_USD", "15000"))
DEFAULT_DEADLINE = os.environ.get("MONTE_GOAL_DEADLINE", "2026-11-13")


@dataclass(frozen=True)
class GoalConfig:
    start_usd: float
    target_usd: float
    deadline: date

    @classmethod
    def from_env(cls) -> "GoalConfig":
        try:
            deadline = datetime.strptime(DEFAULT_DEADLINE, "%Y-%m-%d").date()
        except ValueError:
            deadline = date(2026, 11, 13)
        return cls(
            start_usd=DEFAULT_START_USD,
            target_usd=DEFAULT_TARGET_USD,
            deadline=deadline,
        )


# ---------- Math helpers ----------

def _days_remaining(deadline: date, *, today: date | None = None) -> int:
    today = today or datetime.now(timezone.utc).date()
    return max(1, (deadline - today).days)


def required_total_return(start: float, target: float) -> float:
    """Return needed from start → target, as a decimal (e.g. 0.5 = +50%)."""
    if start <= 0:
        return 0.0
    return target / start - 1.0


def required_weekly_return(
    current_equity: float,
    target: float,
    deadline: date,
    *,
    today: date | None = None,
) -> float:
    """Compounded weekly return needed to hit `target` by `deadline`."""
    days = _days_remaining(deadline, today=today)
    weeks = max(1.0, days / 7.0)
    if current_equity <= 0:
        return 0.0
    ratio = target / current_equity
    if ratio <= 0:
        return 0.0
    return ratio ** (1.0 / weeks) - 1.0


def required_monthly_return(
    current_equity: float,
    target: float,
    deadline: date,
    *,
    today: date | None = None,
) -> float:
    """Compounded monthly return needed."""
    days = _days_remaining(deadline, today=today)
    months = max(1.0, days / 30.4)
    if current_equity <= 0:
        return 0.0
    ratio = target / current_equity
    if ratio <= 0:
        return 0.0
    return ratio ** (1.0 / months) - 1.0


# ---------- Pace status ----------

@dataclass
class GoalStatus:
    on_pace: bool
    pace_gap_pct: float      # how far ahead/behind, vs linear-pace expectation
    expected_equity: float   # what equity should be today on linear pace
    weekly_required: float
    monthly_required: float
    days_remaining: int


def on_pace(
    book_or_equity: PaperBook | float,
    cfg: GoalConfig,
    *,
    today: date | None = None,
) -> GoalStatus:
    """Compare current equity vs the linear-pace expectation."""
    today = today or datetime.now(timezone.utc).date()
    if isinstance(book_or_equity, PaperBook):
        eq = book_or_equity.mark_to_market({}).equity
    else:
        eq = float(book_or_equity)

    days = _days_remaining(cfg.deadline, today=today)
    weekly_req = required_weekly_return(eq, cfg.target_usd, cfg.deadline, today=today)
    monthly_req = required_monthly_return(eq, cfg.target_usd, cfg.deadline, today=today)
    # On pace iff the required monthly return to still hit target is <= 8%.
    # 8% is the high-end of plausible swing-trading monthly returns; past that,
    # the strategy needs to push risk up and we'd rather flag it as behind.
    on_pace_flag = monthly_req <= 0.08
    expected = cfg.start_usd  # neutral baseline; UI shows actual vs target
    gap = (eq - cfg.start_usd) / max(cfg.start_usd, 1.0)

    return GoalStatus(
        on_pace=on_pace_flag,
        pace_gap_pct=float(gap),
        expected_equity=float(expected),
        weekly_required=float(weekly_req),
        monthly_required=float(monthly_req),
        days_remaining=int(days),
    )


# ---------- Risk-per-trade integrating goal + drawdown ----------

def suggested_risk_pct(
    equity: float,
    starting: float,
    confidence: float,
    cfg: GoalConfig,
    *,
    today: date | None = None,
) -> float:
    """Risk per trade given drawdown and goal pressure.

    - Falls to 0 when current drawdown <= -10% (matches Monte Edge brake).
    - Halves when drawdown <= -5%.
    - Scales 0.5%–1.5% by confidence.
    - Capped at 1.5% regardless of how far behind pace we are — we never let
      "I'm behind" justify reckless sizing.
    """
    if starting <= 0:
        return 0.0
    drawdown = (equity - starting) / starting if equity < starting else 0.0
    return confidence_scaled_risk_pct(confidence, drawdown_pct=drawdown)


def projected_equity_curve(
    start: float,
    weekly_return: float,
    weeks: int,
) -> list[tuple[int, float]]:
    """Compound `start` at `weekly_return` for `weeks` weeks. Returns (week, equity)."""
    curve: list[tuple[int, float]] = []
    eq = start
    for w in range(weeks + 1):
        curve.append((w, eq))
        eq *= 1.0 + weekly_return
    return curve


def progress_pct(equity: float, cfg: GoalConfig) -> float:
    """0.0 at start, 1.0 at target. Clamped."""
    span = cfg.target_usd - cfg.start_usd
    if span <= 0:
        return 1.0 if equity >= cfg.target_usd else 0.0
    return max(0.0, min(1.0, (equity - cfg.start_usd) / span))


__all__ = [
    "DEFAULT_START_USD",
    "DEFAULT_TARGET_USD",
    "DEFAULT_DEADLINE",
    "GoalConfig",
    "GoalStatus",
    "on_pace",
    "progress_pct",
    "projected_equity_curve",
    "required_monthly_return",
    "required_total_return",
    "required_weekly_return",
    "suggested_risk_pct",
]
