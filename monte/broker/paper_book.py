"""Stub paper trading book."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
        return {"cash": 10000.0, "starting": 10000.0, "positions": {}, "trades": []}

    def _save(self) -> None:
        self._state_file.write_text(json.dumps(self._state, indent=2))

    def cash(self) -> float:
        return float(self._state.get("cash", 0.0))

    def starting_budget(self) -> float:
        return float(self._state.get("starting", 10000.0))

    def positions(self) -> dict[str, Any]:
        return self._state.get("positions", {})

    def reset(self, budget: float = 10000.0) -> None:
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
    ) -> None:
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

        self._state.setdefault("trades", []).append(
            {"ts": time.time(), "symbol": symbol, "side": side, "qty": qty, "price": price, "note": note}
        )
        self._save()

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
