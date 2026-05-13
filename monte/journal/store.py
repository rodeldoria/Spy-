"""Append-only JSONL journal of entries and exits.

Each entry captures an indicator snapshot at trade time. When the dashboard
shows a fresh signal it queries this journal for the K closest historical
setups (Euclidean distance over the indicator vector) and reports the
realised win-rate + average return of those neighbours, which acts as a
"have I successfully traded this before?" confirmation badge.

Storage is a single JSONL file at `~/.monte/journal.jsonl`. We avoid sqlite
or vector DBs on purpose: the file is small, easy to inspect, and the rest
of the project already uses JSONL for `alerts.jsonl`.
"""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PATH = Path(os.path.expanduser("~/.monte/journal.jsonl"))

# Indicator keys we use as the similarity vector. Order is significant and
# must stay stable so historic entries remain comparable.
FEATURE_KEYS = ("rsi", "bb_pctb", "macd_hist", "adx", "atr_pct")


@dataclass
class JournalEntry:
    id: str
    symbol: str
    timeframe: str
    action: str
    horizon: str
    entry: float
    stop: float
    target: float
    confidence: float
    score: float
    snapshot: dict[str, float] = field(default_factory=dict)
    note: str = ""
    ts_entry: float = 0.0
    ts_exit: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_pct: float | None = None
    outcome: str = "open"  # open | win | loss | scratch


@dataclass
class SimilarHistory:
    count: int
    win_rate: float
    avg_pnl_pct: float
    best_pnl_pct: float
    worst_pnl_pct: float
    samples: list[JournalEntry] = field(default_factory=list)


def _path() -> Path:
    p = Path(os.environ.get("MONTE_JOURNAL_PATH", str(DEFAULT_PATH)))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_all() -> list[JournalEntry]:
    path = _path()
    if not path.exists():
        return []
    out: list[JournalEntry] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append(JournalEntry(**payload))
    return out


def _rewrite(entries: Iterable[JournalEntry]) -> None:
    path = _path()
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text(
        "\n".join(json.dumps(asdict(e)) for e in entries) + ("\n" if entries else "")
    )
    tmp.replace(path)


def record_entry(
    *,
    symbol: str,
    timeframe: str,
    action: str,
    horizon: str,
    entry: float,
    stop: float,
    target: float,
    confidence: float,
    score: float,
    snapshot: dict[str, float] | None = None,
    note: str = "",
) -> JournalEntry:
    """Log a fresh paper-trade entry and return the saved record."""
    e = JournalEntry(
        id=uuid.uuid4().hex[:12],
        symbol=symbol.upper(),
        timeframe=timeframe,
        action=action,
        horizon=horizon,
        entry=float(entry),
        stop=float(stop),
        target=float(target),
        confidence=float(confidence),
        score=float(score),
        snapshot={k: float(snapshot.get(k, 0.0)) for k in FEATURE_KEYS} if snapshot else {},
        note=note,
        ts_entry=time.time(),
    )
    with _path().open("a") as f:
        f.write(json.dumps(asdict(e)) + "\n")
    return e


def record_exit(
    entry_id: str,
    *,
    exit_price: float,
    exit_reason: str = "manual",
) -> JournalEntry | None:
    """Close an open journal entry and compute its P&L."""
    entries = _read_all()
    match = next((e for e in entries if e.id == entry_id), None)
    if match is None or match.outcome != "open":
        return None
    match.ts_exit = time.time()
    match.exit_price = float(exit_price)
    match.exit_reason = exit_reason
    direction = 1 if match.action.upper() in {"BUY", "STRONG_BUY"} else -1
    if match.entry > 0:
        match.pnl_pct = direction * (match.exit_price - match.entry) / match.entry * 100
    else:
        match.pnl_pct = 0.0
    if match.pnl_pct >= 0.25:
        match.outcome = "win"
    elif match.pnl_pct <= -0.25:
        match.outcome = "loss"
    else:
        match.outcome = "scratch"
    _rewrite(entries)
    return match


def open_entries(symbol: str | None = None) -> list[JournalEntry]:
    return [
        e
        for e in _read_all()
        if e.outcome == "open" and (symbol is None or e.symbol == symbol.upper())
    ]


def _feature_vector(snapshot: dict[str, float]) -> list[float]:
    return [float(snapshot.get(k, 0.0)) for k in FEATURE_KEYS]


def _distance(a: list[float], b: list[float]) -> float:
    # Per-feature scales so RSI (0-100) does not drown out ATR% (~0-0.1).
    scales = [40.0, 0.5, 0.01, 25.0, 0.05]
    return math.sqrt(
        sum(((x - y) / s) ** 2 for x, y, s in zip(a, b, scales))
    )


def similar_history(
    *,
    symbol: str,
    action: str,
    snapshot: dict[str, float],
    k: int = 5,
    same_symbol_only: bool = False,
) -> SimilarHistory:
    """Find the K closest CLOSED entries by indicator distance.

    Returns an empty `SimilarHistory` (count=0) when no matches exist.
    """
    target_vec = _feature_vector(snapshot)
    bucket: list[tuple[float, JournalEntry]] = []
    for e in _read_all():
        if e.outcome == "open" or e.pnl_pct is None:
            continue
        if e.action.upper() != action.upper():
            continue
        if same_symbol_only and e.symbol != symbol.upper():
            continue
        if not e.snapshot:
            continue
        bucket.append((_distance(target_vec, _feature_vector(e.snapshot)), e))

    if not bucket:
        return SimilarHistory(count=0, win_rate=0.0, avg_pnl_pct=0.0, best_pnl_pct=0.0, worst_pnl_pct=0.0)

    bucket.sort(key=lambda x: x[0])
    nearest = [e for _, e in bucket[:k]]
    pnls = [e.pnl_pct for e in nearest if e.pnl_pct is not None]
    wins = sum(1 for e in nearest if e.outcome == "win")
    return SimilarHistory(
        count=len(nearest),
        win_rate=wins / len(nearest) * 100 if nearest else 0.0,
        avg_pnl_pct=sum(pnls) / len(pnls) if pnls else 0.0,
        best_pnl_pct=max(pnls) if pnls else 0.0,
        worst_pnl_pct=min(pnls) if pnls else 0.0,
        samples=nearest,
    )


def summary() -> dict[str, Any]:
    entries = _read_all()
    closed = [e for e in entries if e.outcome != "open" and e.pnl_pct is not None]
    if not closed:
        return {"total": len(entries), "closed": 0, "wins": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0}
    wins = sum(1 for e in closed if e.outcome == "win")
    return {
        "total": len(entries),
        "closed": len(closed),
        "wins": wins,
        "win_rate": wins / len(closed) * 100,
        "avg_pnl_pct": sum(e.pnl_pct for e in closed) / len(closed),
    }


__all__ = [
    "FEATURE_KEYS",
    "JournalEntry",
    "SimilarHistory",
    "record_entry",
    "record_exit",
    "open_entries",
    "similar_history",
    "summary",
]
