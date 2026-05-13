"""Kalshi calibration / learning loop.

Goal: turn every settled market into a data point that improves the
*next* recommendation. We do this with three pieces:

1. **Snapshot** — every time the dashboard renders a Decision, append a
   compact JSON record (model_prob, implied_prob, edge, direction, etc.)
   keyed by ticker. Re-snapshots of the same (ticker, snapshot_minute)
   are deduplicated so we don't bloat the log on every refresh.

2. **Settle** — periodically re-fetch markets whose `close_time` has
   passed and back-fill the actual outcome (YES = 1, NO = 0). Kalshi
   reports this either via `m.status == "settled"` + `result` field,
   or via `last_price` collapsing to 0 / 100.

3. **Calibrate** — fit a 10-bin isotonic-style mapping from
   `model_prob → empirical hit-rate` across the settled history. At
   inference, look up the calibrated probability instead of trusting
   raw GBM. We also compute a Brier-score comparison vs. the Kalshi
   book itself, so the user can see whether the model is actually
   beating the market.

Storage: a single JSONL file under `data/kalshi_observations.jsonl`.
Each line is one snapshot or one settled outcome. We keep snapshots
small (<300 bytes each) so years of data stays under a few MB.

Nothing here is "machine learning" in the heavy sense — calibration
is a statistical sanity layer that automatically adapts as more
markets settle. Crucially, it answers: "when the model says 70%,
does it actually hit 70% of the time?".
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

DATA_DIR = Path(os.getenv("KALSHI_DATA_DIR", "data"))
LOG_PATH = DATA_DIR / "kalshi_observations.jsonl"
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Storage primitives
# ---------------------------------------------------------------------------

def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _append_jsonl(record: dict) -> None:
    _ensure_dir()
    with _LOCK, LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def _read_all() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    out: list[dict] = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _rewrite_all(records: list[dict]) -> None:
    _ensure_dir()
    tmp = LOG_PATH.with_suffix(".jsonl.tmp")
    with _LOCK:
        with tmp.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        tmp.replace(LOG_PATH)


# ---------------------------------------------------------------------------
# Snapshotting
# ---------------------------------------------------------------------------

def record_snapshot(decision, symbol: str) -> bool:
    """Snapshot a Decision. Deduped to one record per (ticker, minute).

    Returns True if a new record was written, False if it was a duplicate
    of an existing snapshot from the same minute.
    """
    now = time.time()
    minute_bucket = int(now // 60)
    snap_key = f"{decision.market_ticker}:{minute_bucket}"

    # Cheap dedupe: only check the last ~200 records (most recent snapshots
    # dominate). Avoids re-reading megabytes on every render.
    history = _read_all()
    recent_keys = {
        f"{r.get('ticker')}:{r.get('minute_bucket')}"
        for r in history[-400:]
        if r.get("kind") == "snapshot"
    }
    if snap_key in recent_keys:
        return False

    chosen = decision.chosen
    record = {
        "kind": "snapshot",
        "ts": now,
        "minute_bucket": minute_bucket,
        "ticker": decision.market_ticker,
        "symbol": symbol,
        "title": decision.title[:120],
        "bet_summary": decision.bet_summary[:140],
        "close_time": decision.close_time,
        "spot_price": round(decision.spot_price, 4),
        "sigma_per_min": round(decision.sigma_per_min, 6),
        "yes_implied": round(decision.yes_side.implied_prob, 4),
        "yes_model": round(decision.yes_side.model_prob, 4),
        "no_implied": round(decision.no_side.implied_prob, 4),
        "no_model": round(decision.no_side.model_prob, 4),
        "yes_ask": decision.yes_side.ask_cents,
        "no_ask": decision.no_side.ask_cents,
        "direction": decision.direction,
        "confidence_pct": round(decision.confidence_pct, 2),
        "edge": round(
            chosen.edge if chosen else max(decision.yes_side.edge, decision.no_side.edge),
            4,
        ),
        "ev_per_dollar": round(chosen.ev_per_dollar if chosen else 0.0, 4),
        "kelly_fraction": round(chosen.kelly_fraction if chosen else 0.0, 4),
    }
    _append_jsonl(record)
    return True


# ---------------------------------------------------------------------------
# Settlement back-fill
# ---------------------------------------------------------------------------

def _detect_outcome(market) -> int | None:
    """Return 1 (YES won), 0 (NO won), or None (not yet settled)."""
    raw = market.raw or {}
    result = (raw.get("result") or "").lower()
    if result in {"yes", "no"}:
        return 1 if result == "yes" else 0
    if (market.status or "").lower() == "settled":
        # last_price collapses to 0 or 100 at settlement
        if market.last_price >= 95:
            return 1
        if market.last_price <= 5:
            return 0
        # Settled but ambiguous (e.g. void market) — treat as None
        return None
    return None


def settle_pending(client, max_lookups: int = 12) -> dict:
    """Look up pending snapshots whose close_time has passed and try to settle.

    Walks the JSONL once, finds snapshots that don't yet have a corresponding
    'outcome' record AND whose close_time is in the past. Calls
    `client.get_market(ticker)` for up to `max_lookups` of them per call to
    avoid rate-limiting.

    Returns a dict summary: {settled: int, still_pending: int, errors: int}.
    """
    history = _read_all()
    settled_tickers = {
        r["ticker"] for r in history if r.get("kind") == "outcome"
    }

    # Find pending snapshots: most recent snapshot per ticker whose close_time
    # has passed and is not already settled.
    by_ticker: dict[str, dict] = {}
    for r in history:
        if r.get("kind") != "snapshot":
            continue
        t = r.get("ticker")
        if not t or t in settled_tickers:
            continue
        if (r.get("close_time") or 0) > time.time():
            continue
        prev = by_ticker.get(t)
        if not prev or r.get("ts", 0) > prev.get("ts", 0):
            by_ticker[t] = r

    summary = {"settled": 0, "still_pending": 0, "errors": 0}
    looked_up = 0
    for ticker, snap in by_ticker.items():
        if looked_up >= max_lookups:
            summary["still_pending"] += 1
            continue
        looked_up += 1
        try:
            market = client.get_market(ticker)
        except Exception:
            summary["errors"] += 1
            continue
        outcome = _detect_outcome(market)
        if outcome is None:
            summary["still_pending"] += 1
            continue
        record = {
            "kind": "outcome",
            "ts": time.time(),
            "ticker": ticker,
            "symbol": snap.get("symbol"),
            "outcome_yes": outcome,
            "settle_price": market.last_price,
            "yes_model": snap.get("yes_model"),
            "no_model": snap.get("no_model"),
            "yes_implied": snap.get("yes_implied"),
            "no_implied": snap.get("no_implied"),
            "direction": snap.get("direction"),
            "edge": snap.get("edge"),
            "ev_per_dollar": snap.get("ev_per_dollar"),
            "horizon_seconds_at_snap": (snap.get("close_time") or 0) - (snap.get("ts") or 0),
        }
        _append_jsonl(record)
        summary["settled"] += 1

    return summary


# ---------------------------------------------------------------------------
# Calibration math
# ---------------------------------------------------------------------------

@dataclass
class CalibrationReport:
    n_snapshots: int = 0
    n_settled: int = 0
    n_still_pending: int = 0
    brier_model: float | None = None
    brier_book: float | None = None
    log_loss_model: float | None = None
    log_loss_book: float | None = None
    yes_recommend_hit_rate: float | None = None
    no_recommend_hit_rate: float | None = None
    pass_hit_rate: float | None = None
    decile_hit_rates: list[tuple[float, float, int]] = field(default_factory=list)
    # list of (bin_midpoint_model_prob, empirical_hit_rate, sample_count)
    last_settle_summary: dict | None = None

    @property
    def beats_book(self) -> bool | None:
        if self.brier_model is None or self.brier_book is None:
            return None
        return self.brier_model < self.brier_book

    @property
    def edge_vs_book_brier(self) -> float | None:
        if self.brier_model is None or self.brier_book is None:
            return None
        return self.brier_book - self.brier_model


def _bin_index(p: float, n_bins: int = 10) -> int:
    return min(n_bins - 1, max(0, int(p * n_bins)))


def calibration_report(history: list[dict] | None = None) -> CalibrationReport:
    history = history if history is not None else _read_all()
    snaps = [r for r in history if r.get("kind") == "snapshot"]
    outcomes = [r for r in history if r.get("kind") == "outcome"]
    rep = CalibrationReport(
        n_snapshots=len({s.get("ticker") for s in snaps}),
        n_settled=len(outcomes),
    )
    if not outcomes:
        return rep

    # Brier + log-loss for model and book
    model_sq, book_sq = 0.0, 0.0
    model_ll, book_ll = 0.0, 0.0
    n = 0
    yes_recs = []
    no_recs = []
    pass_recs = []
    bins: dict[int, list[int]] = {}
    for o in outcomes:
        actual = o.get("outcome_yes")
        if actual is None:
            continue
        ym = o.get("yes_model")
        yi = o.get("yes_implied")
        if ym is None or yi is None:
            continue
        n += 1
        model_sq += (ym - actual) ** 2
        book_sq += (yi - actual) ** 2
        # Clamp for log-loss numerical safety
        ym_c = min(0.9999, max(0.0001, ym))
        yi_c = min(0.9999, max(0.0001, yi))
        model_ll += -(actual * math.log(ym_c) + (1 - actual) * math.log(1 - ym_c))
        book_ll += -(actual * math.log(yi_c) + (1 - actual) * math.log(1 - yi_c))

        # Per-direction outcomes (did our recommendation win?)
        d = (o.get("direction") or "").upper()
        if d == "YES":
            yes_recs.append(actual)
        elif d == "NO":
            no_recs.append(1 - actual)
        elif d == "PASS":
            # PASS isn't a bet, but we can still measure: did the side closer
            # to 50% (i.e. the "no edge" side) actually flip?
            pass_recs.append(actual)

        bins.setdefault(_bin_index(ym), []).append(actual)

    if n > 0:
        rep.brier_model = model_sq / n
        rep.brier_book = book_sq / n
        rep.log_loss_model = model_ll / n
        rep.log_loss_book = book_ll / n

    if yes_recs:
        rep.yes_recommend_hit_rate = sum(yes_recs) / len(yes_recs)
    if no_recs:
        rep.no_recommend_hit_rate = sum(no_recs) / len(no_recs)
    if pass_recs:
        rep.pass_hit_rate = sum(pass_recs) / len(pass_recs)

    # Decile hit rates
    rep.decile_hit_rates = []
    for i in range(10):
        b = bins.get(i, [])
        if not b:
            continue
        midpoint = (i + 0.5) / 10.0
        rep.decile_hit_rates.append((midpoint, sum(b) / len(b), len(b)))

    return rep


def calibrate_prob(model_prob: float, history: list[dict] | None = None,
                   min_samples: int = 30) -> tuple[float, str]:
    """Return (calibrated_prob, source_label).

    Looks up `model_prob` in the empirical decile-hit-rate table. If we have
    fewer than `min_samples` total settled outcomes, returns the raw
    model_prob and a label noting we don't yet have enough data.
    """
    history = history if history is not None else _read_all()
    rep = calibration_report(history)
    if rep.n_settled < min_samples:
        return model_prob, f"raw (only {rep.n_settled} settled, need {min_samples})"

    # Find the two decile bins this prob falls between and linearly
    # interpolate.
    if not rep.decile_hit_rates:
        return model_prob, "raw (no decile data)"

    points = sorted(rep.decile_hit_rates, key=lambda x: x[0])
    if model_prob <= points[0][0]:
        return points[0][1], f"calibrated ({rep.n_settled} samples)"
    if model_prob >= points[-1][0]:
        return points[-1][1], f"calibrated ({rep.n_settled} samples)"
    for (x1, y1, _), (x2, y2, _) in zip(points, points[1:]):
        if x1 <= model_prob <= x2:
            t = (model_prob - x1) / (x2 - x1) if x2 > x1 else 0.0
            return y1 + t * (y2 - y1), f"calibrated ({rep.n_settled} samples)"
    return model_prob, "raw"


# ---------------------------------------------------------------------------
# Bulk snapshot helper
# ---------------------------------------------------------------------------

def snapshot_decisions(decisions: Iterable, symbol: str) -> int:
    """Snapshot every decision; return number of new records written."""
    n = 0
    for d in decisions:
        try:
            if record_snapshot(d, symbol):
                n += 1
        except Exception:
            continue
    return n
