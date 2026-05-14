"""Bootstrap the pattern tracker from synthetic backtest runs.

The live learning loop in ``pattern_tracker`` needs ≥5 settled outcomes per
8-bit signature before the council multipliers actually move off ×1.00.
Waiting for that to fill up from live Kalshi settlements alone takes
weeks. The kalshi backtest engine (``monte.backtest.replay_kalshi``)
already synthesises asset-pegged markets from years of OHLCV and settles
each one against the actual close — but it writes to its own SQLite/JSONL
store, separate from the pattern tracker.

This module is the bridge: it walks a finished backtest run's trades,
replays each one through ``decision_council.evaluate`` to get a fresh
8-bit signature, and writes paired verdict + outcome rows into the
pattern tracker JSONL so ``build_report`` / ``confidence_multiplier``
pick them up immediately. Every row written gets ``source="backtest"``
and the originating ``run_id`` so the UI can separate synthetic history
from live verdicts if needed.

Re-running ``bootstrap_from_backtest`` on the same ``run_id`` is a no-op:
each verdict row carries the originating ``trade_id`` as its ``id`` and
we skip rows that already exist.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

from monte.backtest.config import default_db_path
from monte.intel import decision_council as council
from monte.learning import pattern_tracker as ptrack


def _trades_for_run(db_path: Path, run_id: str | None) -> list[dict]:
    """Pull every Kalshi trade (and snapshot) for the given run, or the
    latest completed kalshi run if ``run_id`` is None."""
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if run_id is None:
            row = conn.execute(
                """
                SELECT run_id FROM runs
                WHERE engine = 'kalshi' AND status = 'ok'
                ORDER BY ts_finished DESC LIMIT 1
                """,
            ).fetchone()
            if row is None:
                return []
            run_id = row["run_id"]
        rows = conn.execute(
            """
            SELECT trade_id, run_id, symbol, timeframe, horizon, ts_entry, ts_exit,
                   action, direction, pnl_pct, actual_outcome, confidence,
                   snapshot_json
            FROM trades
            WHERE run_id = ? AND engine = 'kalshi'
            """,
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _existing_ids() -> set[str]:
    """Set of verdict ids already in the pattern tracker JSONL — used to
    keep the bootstrap idempotent if it's run twice on the same backtest."""
    p = ptrack._path()
    if not p.exists():
        return set()
    out: set[str] = set()
    try:
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "verdict" and rec.get("id"):
                out.add(rec["id"])
    except Exception:
        pass
    return out


def _bootstrap_row(trade: dict) -> tuple[Optional[dict], Optional[dict]]:
    """Turn a backtest trade row into a (verdict, outcome) pair, or
    ``(None, None)`` if it's a PASS / unrecoverable. Mirrors the schema
    that ``pattern_tracker.record_verdict`` and ``reconcile_outcomes``
    write so the existing readers (``build_report``) need no changes.
    """
    direction = (trade.get("direction") or "").upper()
    if direction not in ("YES", "NO"):
        return None, None  # PASS trades have no signature to learn from

    try:
        snap = json.loads(trade.get("snapshot_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        snap = {}

    edge = float(snap.get("edge") or 0.0)
    ev_per_dollar = float(snap.get("ev_per_dollar") or 0.0)
    kelly_fraction = float(snap.get("kelly_fraction") or 0.0)
    ask_cents = int(snap.get("ask_cents") or 0)
    if ask_cents <= 0:
        return None, None
    payout = 100.0 / max(1, ask_cents)
    confidence_pct = float(trade.get("confidence") or 0.0)

    # Replay through the council to get the same 8-bit signature the live
    # page would have produced. We omit calibration / triangulation inputs
    # so those two checkpoints fall through to their default "no data"
    # branches — exactly what happens on a fresh live page, so the
    # synthesised signatures aggregate cleanly with live ones.
    verdict = council.evaluate(
        direction=direction,
        edge=edge,
        ev_per_dollar=ev_per_dollar,
        kelly_fraction=kelly_fraction,
        confidence_pct=confidence_pct,
        payout=payout,
        ask_cents=ask_cents,
        warnings=[],
        bet_summary=f"backtest {trade.get('symbol')} {trade.get('horizon') or trade.get('timeframe')}",
        market_ticker=f"BT-{trade['trade_id']}",
        enable_ai=False,
    )

    # Use the backtest trade_id as the verdict id so re-running bootstrap
    # on the same run_id is a no-op.
    vid = trade["trade_id"]
    ts_entry = float(trade.get("ts_entry") or 0.0)
    ts_exit = float(trade.get("ts_exit") or ts_entry)

    verdict_row = {
        "kind": "verdict",
        "id": vid,
        "ts": ts_entry,
        "ticker": f"BT-{vid}",
        "symbol": trade.get("symbol"),
        "bet_summary": f"backtest {trade.get('symbol')} {trade.get('horizon')}",
        "direction": direction,
        "ask_cents": ask_cents,
        "edge_pp": round(edge * 100, 2),
        "ev_per_dollar": round(ev_per_dollar, 4),
        "kelly_fraction": round(kelly_fraction, 4),
        "confidence_pct": round(confidence_pct, 1),
        "trigger_score": round(verdict.trigger_score, 1),
        "mechanical_score": round(verdict.mechanical_score, 1),
        "ai_score": None,
        "verdict_label": verdict.verdict_label,
        "signature": verdict.signature,
        "checkpoint_names": [c.name for c in verdict.checkpoints],
        "close_time": ts_exit,
        "source": "backtest",
        "run_id": trade.get("run_id"),
    }

    pnl_pct = float(trade.get("pnl_pct") or 0.0)
    won = pnl_pct > 0
    actual = trade.get("actual_outcome") or ""
    outcome_yes = 1 if "yes" in actual.lower() else 0
    if won and ask_cents > 0:
        roi_pct = (100.0 - ask_cents) / ask_cents * 100.0
    else:
        roi_pct = -100.0

    outcome_row = {
        "kind": "outcome",
        "ts": ts_exit,
        "ticker": f"BT-{vid}",
        "verdict_id": vid,
        "verdict_ts": ts_entry,
        "signature": verdict.signature,
        "trigger_score": round(verdict.trigger_score, 1),
        "won": bool(won),
        "outcome_yes": outcome_yes,
        "roi_pct": round(roi_pct, 2),
        "settle_price": None,
        "source": "backtest",
        "run_id": trade.get("run_id"),
    }
    return verdict_row, outcome_row


def bootstrap_from_backtest(
    *,
    run_id: str | None = None,
    db_path: Path | None = None,
) -> dict:
    """Mirror a backtest's Kalshi trades into the pattern tracker so the
    council's learned multipliers reflect synthetic history immediately.

    Args:
        run_id: backtest run id to mirror. ``None`` picks the latest
            ``status='ok'`` kalshi run.
        db_path: override for the backtest SQLite path. Defaults to
            ``BacktestConfig().db_path`` so the test suite can point at
            a temporary DB.

    Returns:
        ``{"run_id": str | None, "verdicts_written": int,
           "outcomes_written": int, "skipped_pass": int, "skipped_dup": int}``
    """
    if db_path is None:
        db_path = default_db_path()

    trades = _trades_for_run(db_path, run_id)
    if not trades:
        return {
            "run_id": run_id, "verdicts_written": 0, "outcomes_written": 0,
            "skipped_pass": 0, "skipped_dup": 0,
        }

    existing = _existing_ids()
    summary = {
        "run_id": trades[0].get("run_id"),
        "verdicts_written": 0,
        "outcomes_written": 0,
        "skipped_pass": 0,
        "skipped_dup": 0,
    }
    p = ptrack._path()
    with ptrack._locked_append(p) as f:
        for trade in trades:
            verdict, outcome = _bootstrap_row(trade)
            if verdict is None:
                summary["skipped_pass"] += 1
                continue
            if verdict["id"] in existing:
                summary["skipped_dup"] += 1
                continue
            f.write(json.dumps(verdict) + "\n")
            f.write(json.dumps(outcome) + "\n")
            existing.add(verdict["id"])
            summary["verdicts_written"] += 1
            summary["outcomes_written"] += 1
    return summary


__all__ = ["bootstrap_from_backtest"]
