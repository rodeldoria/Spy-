"""Tests for the new PaperBook PnL/drawdown methods."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from monte.broker.paper_book import PaperBook


@pytest.fixture
def book(tmp_path: Path) -> PaperBook:
    b = PaperBook(state_path=tmp_path)
    b.reset(budget=10_000.0)
    return b


def test_daily_pnl_zero_when_no_trades(book: PaperBook):
    assert book.daily_pnl() == 0.0
    assert book.weekly_pnl() == 0.0
    assert book.monthly_pnl() == 0.0


def test_realised_pnl_in_window(book: PaperBook):
    now = time.time()
    # Buy 10 @ $100, sell 10 @ $110 — both within the last hour.
    book.place_order("AAPL", "buy", 10, 100.0)
    book.place_order("AAPL", "sell", 10, 110.0)
    # Daily window includes both trades — PnL ≈ +$100.
    pnl = book.daily_pnl(now=now)
    assert 90 <= pnl <= 110


def test_old_pnl_excluded_from_recent_window(book: PaperBook, monkeypatch):
    # Simulate an old buy 40 days ago, then a recent sell.
    old_ts = time.time() - 40 * 86400
    book.place_order("AAPL", "buy", 10, 100.0)
    # mutate the just-appended trade ts to old.
    book._state["trades"][-1]["ts"] = old_ts
    book._save()
    book.place_order("AAPL", "sell", 10, 120.0)  # now
    weekly = book.weekly_pnl()
    assert weekly == pytest.approx(200.0, rel=0.05)


def test_current_drawdown_zero_when_above_starting(book: PaperBook):
    book.place_order("AAPL", "buy", 1, 100.0)
    # mark at $200 → book equity 9_900 cash + $200 = $10_100 > starting
    dd = book.current_drawdown(prices={"AAPL": 200.0})
    assert dd == 0.0


def test_current_drawdown_negative_when_below(book: PaperBook):
    book.place_order("AAPL", "buy", 10, 100.0)
    # mark at $80 → equity drops below starting
    dd = book.current_drawdown(prices={"AAPL": 80.0})
    assert dd < 0


def test_max_drawdown_tracks_realised_curve(book: PaperBook):
    book.place_order("AAPL", "buy", 10, 100.0)
    book.place_order("AAPL", "sell", 10, 80.0)   # realised loss
    book.place_order("AAPL", "buy", 10, 80.0)
    book.place_order("AAPL", "sell", 10, 95.0)   # partial recovery
    mdd = book.max_drawdown()
    assert mdd < 0
    assert mdd >= -0.05  # roughly 2% drawdown peak
