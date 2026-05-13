"""Alerts engine — runs a Monte Edge scan and writes results.

`scan_once()` iterates the requested symbols, fetches candles, runs the
Monte Edge evaluator, persists each result to both `alerts.jsonl` (the
existing tail) and `playbook.jsonl` (Claude's tactics log), and pushes
ACT_NOW / WATCH tier signals to ntfy.sh.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from monte.config import settings


def tail_alerts(limit: int = 50) -> list[dict[str, Any]]:
    """Read the last `limit` alerts from the alerts log file."""
    path = settings.alerts_log_path
    if not path.exists():
        return []
    try:
        lines = path.read_text().strip().splitlines()
        rows = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) >= limit:
                break
        return rows
    except Exception:
        return []


def _alerts_path() -> Path:
    p = Path(settings.alerts_log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _append_alert(row: dict[str, Any]) -> None:
    with _alerts_path().open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _fetch_candles(symbol: str, timeframe: str):
    """Fetch candles for a symbol, choosing crypto vs equity by ticker shape."""
    s = symbol.upper()
    is_crypto = "-" in s and s.split("-")[-1] in {"USD", "USDC", "USDT"}
    if is_crypto:
        from monte.data import crypto

        return crypto.get_candles(s, timeframe, lookback_bars=400)
    from monte.data import prices

    if timeframe == "1d":
        return prices.get_daily(s, period="2y")
    period_map = {"1m": "5d", "5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d"}
    df = prices.get_intraday(s, period=period_map.get(timeframe, "60d"), interval=timeframe)
    return df.tail(400) if len(df) > 400 else df


def _spy_daily():
    """Fetch SPY daily candles for the macro filter. Cached briefly."""
    try:
        from monte.data import prices

        return prices.get_daily("SPY", period="2y")
    except Exception:
        return None


def scan_once(
    symbols: list[str],
    timeframes: list[str] | None = None,
    min_confidence: float = 65.0,
    *,
    push: bool = True,
    record_to_playbook: bool = True,
) -> list[dict[str, Any]]:
    """Run a Monte Edge scan across `symbols` × `timeframes`.

    Side effects:
      - appends each alert to `~/.monte/alerts.jsonl`
      - appends ACT_NOW / WATCH rows to `~/.monte/playbook.jsonl`
      - pushes ACT_NOW / WATCH rows to ntfy.sh (if MONTE_NTFY_TOPIC set)

    Returns the list of alerts that met `min_confidence`.
    """
    from monte.strategy.monte_edge import evaluate, EdgeTier
    from monte.strategy.playbook import record_signal
    from monte.broker.paper_book import PaperBook
    from monte.notify.ntfy import push_alert

    tfs = timeframes or ["1h"]

    try:
        book = PaperBook(state_path=settings.paper_state_path)
        drawdown = book.current_drawdown()
    except Exception:
        drawdown = 0.0

    spy_df = _spy_daily()

    out: list[dict[str, Any]] = []
    for sym in symbols:
        for tf in tfs:
            try:
                df = _fetch_candles(sym, tf)
            except Exception as exc:
                _append_alert({
                    "ts": time.time(),
                    "symbol": sym,
                    "timeframe": tf,
                    "action": "HOLD",
                    "confidence": 0.0,
                    "tier": EdgeTier.STAND_DOWN.value,
                    "error": str(exc),
                })
                continue
            if df is None or df.empty:
                continue

            try:
                sig = evaluate(
                    df,
                    symbol=sym,
                    timeframe=tf,
                    spy_daily=spy_df,
                    drawdown_pct=drawdown,
                )
            except Exception as exc:
                _append_alert({
                    "ts": time.time(),
                    "symbol": sym,
                    "timeframe": tf,
                    "action": "HOLD",
                    "confidence": 0.0,
                    "tier": EdgeTier.STAND_DOWN.value,
                    "error": str(exc),
                })
                continue

            options_ticket = None
            if sym.upper() == "SPY" and sig.tier is EdgeTier.ACT_NOW:
                try:
                    from monte.options import suggest_contract

                    direction = "long" if sig.score > 0 else "short"
                    options_ticket = suggest_contract(direction, sig.spot)
                except Exception:
                    options_ticket = None

            row = sig.to_dict()
            row["ts"] = time.time()
            row["spot"] = sig.entry
            if options_ticket:
                row["options_ticket"] = options_ticket
            _append_alert(row)

            if record_to_playbook and sig.tier is not EdgeTier.STAND_DOWN:
                try:
                    record_signal(sig, options_ticket=options_ticket)
                except Exception:
                    pass

            if push and sig.tier in {EdgeTier.ACT_NOW, EdgeTier.WATCH}:
                try:
                    push_alert(row)
                except Exception:
                    pass

            if sig.confidence >= min_confidence:
                out.append(row)
    return out
