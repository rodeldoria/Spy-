"""Auto-trade engine — executes ACT_NOW signals into the paper book.

When the user enables auto-trading, every new ACT_NOW signal that hasn't
been executed yet is automatically paper-bought (or sold). A persistent
JSON ledger at ~/.monte/paper/auto_trades.json tracks which alert hashes
have already been filed so we never double-fill on refresh.

This is simulation only — no real broker connection.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monte.broker.paper_book import InsufficientFunds, PaperBook
from monte.config import settings
from monte import journal
from monte.strategy.goal_tracker import GoalConfig, suggested_risk_pct


_LEDGER_PATH = Path.home() / ".monte" / "paper" / "auto_trades.json"


@dataclass
class AutoTradeResult:
    alert_hash: str
    symbol: str
    side: str
    qty: float
    price: float
    confidence: float
    tier: str
    journal_id: str
    ts: float
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _load_executed() -> set[str]:
    try:
        if _LEDGER_PATH.exists():
            data = json.loads(_LEDGER_PATH.read_text())
            return set(data.get("hashes", []))
    except Exception:
        pass
    return set()


def _save_executed(hashes: set[str]) -> None:
    try:
        _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LEDGER_PATH.write_text(json.dumps({"hashes": list(hashes)}, indent=2))
    except Exception:
        pass


def run_auto_trades(
    alerts: list[dict[str, Any]],
    book: PaperBook | None = None,
    max_per_run: int = 3,
) -> list[AutoTradeResult]:
    """Check `alerts` for new ACT_NOW signals and paper-execute them.

    Parameters
    ----------
    alerts:
        Alert dicts from ``tail_alerts()``.
    book:
        PaperBook to execute against. If None, uses the default path.
    max_per_run:
        Safety cap — never execute more than this many orders in one pass
        so a burst of signals doesn't wipe the whole paper book.

    Returns
    -------
    List of AutoTradeResult — one per order attempted (success or error).
    """
    if book is None:
        book = PaperBook(state_path=settings.paper_state_path)

    executed = _load_executed()
    cfg = GoalConfig.from_env()

    cash = book.cash()
    starting = book.starting_budget()
    equity = book.mark_to_market({}).equity

    results: list[AutoTradeResult] = []

    for r in alerts:
        if len(results) >= max_per_run:
            break

        tier = r.get("tier", "")
        if tier != "ACT_NOW":
            continue

        alert_hash = r.get("hash", "")
        if not alert_hash or alert_hash in executed:
            continue

        action = str(r.get("action", "")).upper()
        side = "buy" if "BUY" in action else "sell" if "SELL" in action else ""
        if not side:
            executed.add(alert_hash)
            continue

        symbol = r.get("symbol", "")
        spot = float(r.get("spot", 0) or 0)
        stop = float(r.get("stop", 0) or 0)
        confidence = float(r.get("confidence", 0) or 0)

        if spot <= 0 or stop <= 0:
            executed.add(alert_hash)
            continue

        risk_pct = suggested_risk_pct(equity, starting, confidence, cfg)
        risk_dollar = equity * risk_pct
        risk_per_share = abs(spot - stop)
        qty = round(risk_dollar / max(risk_per_share, 1e-6), 6)

        if qty <= 0:
            executed.add(alert_hash)
            continue

        error = ""
        jid = ""
        try:
            snap = r.get("indicator_snapshot") or {}
            entry = journal.record_entry(
                symbol=symbol,
                timeframe=r.get("timeframe", ""),
                action=action,
                horizon=r.get("horizon", ""),
                entry=spot,
                stop=stop,
                target=float(r.get("target", spot) or spot),
                confidence=confidence,
                score=float(r.get("score", 0) or 0),
                snapshot=snap,
                note=f"auto-trade alert {alert_hash}",
            )
            jid = entry.id
            book.place_order(symbol, side, qty, spot, note=f"auto {alert_hash}", journal_id=jid)
        except InsufficientFunds as e:
            error = f"insufficient funds: {e}"
        except Exception as e:
            error = str(e)

        executed.add(alert_hash)
        results.append(
            AutoTradeResult(
                alert_hash=alert_hash,
                symbol=symbol,
                side=side,
                qty=qty,
                price=spot,
                confidence=confidence,
                tier=tier,
                journal_id=jid,
                ts=time.time(),
                error=error,
            )
        )

    _save_executed(executed)
    return results


def load_auto_trade_log() -> list[dict[str, Any]]:
    """Return the full list of auto-trade results saved to disk."""
    path = Path.home() / ".monte" / "paper" / "auto_trade_log.jsonl"
    if not path.exists():
        return []
    results = []
    for line in path.read_text().splitlines():
        try:
            results.append(json.loads(line))
        except Exception:
            pass
    return results


def append_auto_trade_log(results: list[AutoTradeResult]) -> None:
    """Append new auto-trade results to the persistent log."""
    path = Path.home() / ".monte" / "paper" / "auto_trade_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in results:
            f.write(json.dumps({
                "ts": r.ts,
                "symbol": r.symbol,
                "side": r.side,
                "qty": r.qty,
                "price": r.price,
                "confidence": r.confidence,
                "tier": r.tier,
                "journal_id": r.journal_id,
                "alert_hash": r.alert_hash,
                "error": r.error,
            }) + "\n")
