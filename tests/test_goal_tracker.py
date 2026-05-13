"""Tests for the goal tracker math."""
from __future__ import annotations

from datetime import date

import pytest

from monte.strategy.goal_tracker import (
    GoalConfig,
    on_pace,
    progress_pct,
    projected_equity_curve,
    required_monthly_return,
    required_total_return,
    required_weekly_return,
    suggested_risk_pct,
)


def test_required_total_return():
    assert required_total_return(10_000, 15_000) == pytest.approx(0.5, rel=1e-6)


def test_required_weekly_return_basic():
    # $10k → $15k over ~26 weeks → ~1.57%/week compounded
    deadline = date(2026, 11, 13)
    today = date(2026, 5, 13)
    weekly = required_weekly_return(10_000, 15_000, deadline, today=today)
    assert 0.014 < weekly < 0.018


def test_required_monthly_return_basic():
    deadline = date(2026, 11, 13)
    today = date(2026, 5, 13)
    monthly = required_monthly_return(10_000, 15_000, deadline, today=today)
    # ~6 months of 50% compounded ≈ 6.9%/mo
    assert 0.06 < monthly < 0.08


def test_required_returns_zero_when_already_at_target():
    deadline = date(2026, 11, 13)
    today = date(2026, 5, 13)
    assert required_weekly_return(15_000, 15_000, deadline, today=today) == pytest.approx(0.0)
    assert required_monthly_return(15_000, 15_000, deadline, today=today) == pytest.approx(0.0)


def test_required_returns_handle_zero_equity_gracefully():
    deadline = date(2026, 11, 13)
    today = date(2026, 5, 13)
    assert required_weekly_return(0, 15_000, deadline, today=today) == 0.0
    assert required_monthly_return(0, 15_000, deadline, today=today) == 0.0


def test_on_pace_true_when_close_to_target():
    cfg = GoalConfig(start_usd=10_000, target_usd=15_000, deadline=date(2026, 11, 13))
    status = on_pace(14_500, cfg, today=date(2026, 11, 1))
    assert status.on_pace is True


def test_on_pace_false_when_far_behind():
    cfg = GoalConfig(start_usd=10_000, target_usd=15_000, deadline=date(2026, 6, 1))
    # Only 2 weeks left to go from 10k→15k → required monthly return huge
    status = on_pace(10_000, cfg, today=date(2026, 5, 15))
    assert status.on_pace is False
    assert status.monthly_required > 0.10


def test_progress_pct_clamped():
    cfg = GoalConfig(start_usd=10_000, target_usd=15_000, deadline=date(2026, 11, 13))
    assert progress_pct(10_000, cfg) == pytest.approx(0.0)
    assert progress_pct(15_000, cfg) == pytest.approx(1.0)
    assert progress_pct(20_000, cfg) == pytest.approx(1.0)
    assert progress_pct(5_000, cfg) == pytest.approx(0.0)
    assert progress_pct(12_500, cfg) == pytest.approx(0.5)


def test_projected_curve_compounds():
    curve = projected_equity_curve(10_000, weekly_return=0.02, weeks=4)
    assert len(curve) == 5
    assert curve[0] == (0, 10_000)
    # 10_000 * 1.02^4 ≈ 10_824.32
    assert curve[-1][1] == pytest.approx(10_824.32, rel=1e-3)


def test_suggested_risk_falls_to_zero_in_deep_drawdown():
    cfg = GoalConfig(start_usd=10_000, target_usd=15_000, deadline=date(2026, 11, 13))
    # Equity 9_000 = -10% drawdown → halt
    r = suggested_risk_pct(9_000, 10_000, confidence=95, cfg=cfg)
    assert r == 0.0


def test_suggested_risk_halved_in_moderate_drawdown():
    cfg = GoalConfig(start_usd=10_000, target_usd=15_000, deadline=date(2026, 11, 13))
    # Equity 9_400 = -6% drawdown → halved
    full = suggested_risk_pct(10_000, 10_000, confidence=80, cfg=cfg)
    halved = suggested_risk_pct(9_400, 10_000, confidence=80, cfg=cfg)
    assert halved == pytest.approx(full * 0.5, rel=0.05)


def test_suggested_risk_capped_at_1p5_pct():
    cfg = GoalConfig(start_usd=10_000, target_usd=15_000, deadline=date(2026, 11, 13))
    # No drawdown, max confidence → ceiling
    r = suggested_risk_pct(20_000, 10_000, confidence=100, cfg=cfg)
    assert r <= 0.015 + 1e-9
