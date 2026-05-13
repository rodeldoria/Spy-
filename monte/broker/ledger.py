"""Trade-ledger analytics on top of `PaperBook.trades()`.

Pairs buys with subsequent sells (FIFO per symbol) to compute realised P&L per
sell, equity curve over time, and per-symbol aggregates. Pure functions — no
I/O — so this can be unit-tested without touching disk.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class LedgerRow:
    ts: float
    symbol: str
    side: str
    qty: float
    price: float
    note: str = ""
    journal_id: str | None = None
    realised_pnl: float = 0.0  # only meaningful on sells


@dataclass
class EquityPoint:
    ts: float
    realised_cum: float


@dataclass
class LedgerSummary:
    rows: list[LedgerRow] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    total_realised: float = 0.0
    wins: int = 0
    losses: int = 0

    @property
    def win_rate(self) -> float:
        closed = self.wins + self.losses
        return (self.wins / closed * 100) if closed else 0.0


def _to_row(t: dict) -> LedgerRow:
    return LedgerRow(
        ts=float(t.get("ts", 0.0)),
        symbol=str(t.get("symbol", "")),
        side=str(t.get("side", "")).lower(),
        qty=float(t.get("qty", 0.0)),
        price=float(t.get("price", 0.0)),
        note=str(t.get("note", "")),
        journal_id=t.get("journal_id"),
    )


def build_summary(trades: Iterable[dict]) -> LedgerSummary:
    """Pair buys -> sells FIFO per symbol and compute realised P&L per sell."""
    rows = [_to_row(t) for t in trades]
    rows.sort(key=lambda r: r.ts)

    fifo: dict[str, deque[tuple[float, float]]] = {}  # symbol -> [(qty, price), ...]
    cum = 0.0
    summary = LedgerSummary()
    for r in rows:
        if r.side == "buy":
            fifo.setdefault(r.symbol, deque()).append((r.qty, r.price))
        elif r.side == "sell":
            remaining = r.qty
            realised = 0.0
            queue = fifo.setdefault(r.symbol, deque())
            while remaining > 1e-9 and queue:
                lot_qty, lot_price = queue[0]
                take = min(lot_qty, remaining)
                realised += take * (r.price - lot_price)
                remaining -= take
                if take >= lot_qty - 1e-9:
                    queue.popleft()
                else:
                    queue[0] = (lot_qty - take, lot_price)
            r.realised_pnl = realised
            cum += realised
            if realised > 1e-9:
                summary.wins += 1
            elif realised < -1e-9:
                summary.losses += 1
        summary.rows.append(r)
        if r.side == "sell":
            summary.equity_curve.append(EquityPoint(ts=r.ts, realised_cum=cum))
    summary.total_realised = cum
    return summary


def monthly_realised(rows: Iterable[LedgerRow], *, ts_now: float) -> float:
    """Sum realised P&L from sells in the calendar month containing `ts_now`."""
    import datetime as _dt

    now = _dt.datetime.utcfromtimestamp(ts_now)
    month_start = _dt.datetime(now.year, now.month, 1).timestamp()
    return sum(
        r.realised_pnl
        for r in rows
        if r.side == "sell" and r.ts >= month_start
    )


__all__ = [
    "LedgerRow",
    "EquityPoint",
    "LedgerSummary",
    "build_summary",
    "monthly_realised",
]
