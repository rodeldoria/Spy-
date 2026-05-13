"""Stub paper trading book."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _configured_default_budget() -> float:
    try:
        return float(os.environ.get("MONTE_BUDGET_USD", "500"))
    except ValueError:
        return 500.0


class InsufficientFunds(Exception):
    pass


@dataclass
class MarkResult:
    equity: float
    market_value: float
    cash: float
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)


class PaperBook:
    def __init__(self, state_path: Path | str | None = None) -> None:
        self._path = Path(state_path) if state_path else Path.home() / ".monte" / "paper"
        self._path.mkdir(parents=True, exist_ok=True)
        self._state_file = self._path / "book.json"
        self._load()

    def _load(self) -> None:
        if self._state_file.exists():
            try:
                self._state = json.loads(self._state_file.read_text())
            except Exception:
                self._state = self._default_state()
        else:
            self._state = self._default_state()

    def _default_state(self) -> dict[str, Any]:
        budget = _configured_default_budget()
        return {"cash": budget, "starting": budget, "positions": {}, "trades": []}

    def _save(self) -> None:
        self._state_file.write_text(json.dumps(self._state, indent=2))

    def cash(self) -> float:
        return float(self._state.get("cash", 0.0))

    def starting_budget(self) -> float:
        return float(self._state.get("starting", _configured_default_budget()))

    def positions(self) -> dict[str, Any]:
        return self._state.get("positions", {})

    def trades(self) -> list[dict[str, Any]]:
        """Return the append-only trade ledger (oldest first)."""
        return list(self._state.get("trades", []))

    def reset(self, budget: float | None = None) -> None:
        if budget is None:
            budget = _configured_default_budget()
        self._state = self._default_state()
        self._state["cash"] = budget
        self._state["starting"] = budget
        self._save()

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        note: str = "",
        journal_id: str | None = None,
    ) -> dict[str, Any]:
        """Place a paper order and append it to the trade ledger.

        Returns the trade record (with ts) so callers can correlate with a
        journal entry. `journal_id` is stored on the trade for later lookup.
        """
        cost = qty * price * (1 + 0.0005)
        if side == "buy":
            if cost > self._state["cash"]:
                raise InsufficientFunds(
                    f"Need ${cost:,.2f} but only ${self._state['cash']:,.2f} available."
                )
            self._state["cash"] -= cost
            pos = self._state["positions"].get(symbol, {"qty": 0.0, "avg_cost": 0.0, "realized_pnl": 0.0})
            total_qty = pos["qty"] + qty
            pos["avg_cost"] = (pos["qty"] * pos["avg_cost"] + qty * price) / max(total_qty, 1e-9)
            pos["qty"] = total_qty
            self._state["positions"][symbol] = pos
        elif side == "sell":
            pos = self._state["positions"].get(symbol)
            if not pos or pos["qty"] < qty:
                raise InsufficientFunds(f"Not enough {symbol} to sell.")
            proceeds = qty * price * (1 - 0.0005)
            pnl = (price - pos["avg_cost"]) * qty
            pos["qty"] -= qty
            pos["realized_pnl"] = pos.get("realized_pnl", 0.0) + pnl
            self._state["cash"] += proceeds
            if pos["qty"] < 1e-9:
                del self._state["positions"][symbol]
            else:
                self._state["positions"][symbol] = pos

        trade = {
            "ts": time.time(),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "note": note,
            "journal_id": journal_id,
        }
        self._state.setdefault("trades", []).append(trade)
        self._save()
        return trade

    def mark_to_market(self, prices: dict[str, float]) -> MarkResult:
        positions_snap = {}
        market_value = 0.0
        for sym, pos in self._state.get("positions", {}).items():
            mark = prices.get(sym, pos.get("avg_cost", 0.0))
            qty = pos.get("qty", 0.0)
            mv = qty * mark
            market_value += mv
            positions_snap[sym] = {
                "qty": qty,
                "avg_cost": pos.get("avg_cost", 0.0),
                "mark": mark,
                "unrealized_pnl": (mark - pos.get("avg_cost", 0.0)) * qty,
                "realized_pnl": pos.get("realized_pnl", 0.0),
            }
        equity = self._state["cash"] + market_value
        return MarkResult(
            equity=equity,
            market_value=market_value,
            cash=self._state["cash"],
            positions=positions_snap,
        )

    # ---------- Time-bucketed P&L (realised) ----------

    def _realised_since(self, since_ts: float) -> float:
        """Sum of realised PnL from sell-side trades that closed since `since_ts`.

        We rebuild a running average cost per symbol from the start of the
        trade ledger; each sell crystallises (sell_price − avg_cost) × qty
        as realised PnL on that trade. Sells timestamped >= since_ts count.
        """
        avg_cost: dict[str, float] = {}
        held: dict[str, float] = {}
        total = 0.0
        for tr in self._state.get("trades", []):
            sym = tr.get("symbol", "")
            side = tr.get("side")
            qty = float(tr.get("qty", 0))
            price = float(tr.get("price", 0))
            ts = float(tr.get("ts", 0.0))
            if side == "buy":
                prev_qty = held.get(sym, 0.0)
                prev_avg = avg_cost.get(sym, 0.0)
                new_qty = prev_qty + qty
                if new_qty > 1e-9:
                    avg_cost[sym] = (prev_qty * prev_avg + qty * price) / new_qty
                held[sym] = new_qty
            elif side == "sell":
                basis = avg_cost.get(sym, price)
                pnl = (price - basis) * qty
                held[sym] = held.get(sym, 0.0) - qty
                if ts >= since_ts:
                    total += pnl
        return total

    def daily_pnl(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        return self._realised_since(now - 24 * 60 * 60)

    def weekly_pnl(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        return self._realised_since(now - 7 * 24 * 60 * 60)

    def monthly_pnl(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        return self._realised_since(now - 30 * 24 * 60 * 60)

    # ---------- Drawdown ----------

    def current_drawdown(
        self,
        prices: dict[str, float] | None = None,
    ) -> float:
        """Return drawdown as a negative decimal vs starting budget.

        E.g. -0.07 = book is down 7% from starting capital. 0.0 when at or
        above starting. We use `starting_budget` as the peak anchor — for a
        rolling all-time-high drawdown, see `max_drawdown`.
        """
        prices = prices or {}
        eq = self.mark_to_market(prices).equity
        start = self.starting_budget()
        if start <= 0:
            return 0.0
        if eq >= start:
            return 0.0
        return (eq - start) / start

    def max_drawdown(self) -> float:
        """Peak-to-trough drawdown using a synthetic equity curve.

        Each trade timestamp is a curve sample. We approximate equity at
        each point as (starting + cumulative realised PnL on closed legs).
        Good enough for a UI gauge; not a Sharpe-grade calc.
        """
        peak = self.starting_budget()
        eq = peak
        trough_pct = 0.0
        cash_by_sym: dict[str, float] = {}
        qty_by_sym: dict[str, float] = {}
        for tr in self._state.get("trades", []):
            sym = tr.get("symbol", "")
            side = tr.get("side")
            qty = float(tr.get("qty", 0))
            price = float(tr.get("price", 0))
            if side == "buy":
                cash_by_sym[sym] = cash_by_sym.get(sym, 0.0) - qty * price
                qty_by_sym[sym] = qty_by_sym.get(sym, 0.0) + qty
            else:
                cash_by_sym[sym] = cash_by_sym.get(sym, 0.0) + qty * price
                qty_by_sym[sym] = qty_by_sym.get(sym, 0.0) - qty
            realised = sum(
                cash_by_sym.get(s, 0.0)
                for s, q in qty_by_sym.items()
                if abs(q) < 1e-9
            )
            eq = self.starting_budget() + realised
            peak = max(peak, eq)
            if peak > 0:
                trough_pct = min(trough_pct, (eq - peak) / peak)
        return trough_pct
