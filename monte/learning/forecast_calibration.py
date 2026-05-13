"""Forecast accuracy / learning loop.

Mirrors the Kalshi calibration pattern: snapshot every forecast we show,
then settle it once the horizon time has passed by reading the actual
spot price from the close series that's already in memory.

Three pieces:

1. **Snapshot** — for each PriceProjection rendered, log
   {symbol, label, target_dt_epoch, spot_at_snap, median, lower, upper}
   keyed by (symbol, target_dt_minute_bucket) so re-renders dedupe.

2. **Settle** — on each render, walk pending snapshots whose
   `target_dt_epoch <= now`. Find the actual close at-or-near the
   target time from the supplied close series. If we have it, write
   an `outcome` record with actual price, error %, and a hit flag
   (was actual within the projected ±1σ band?).

3. **Report** — aggregate {n_settled, hit_rate_within_band,
   mean_abs_error_pct, direction_accuracy, mean_bias_pct} per symbol.

Storage: `data/forecast_observations.jsonl` — separate file from
Kalshi calibration so they can grow independently.

This answers the user's question: "was the prediction right?" — every
time the page loads, the system looks back, marks past forecasts
right or wrong, and aggregates a hit-rate so you can see whether the
forecast bands are actually trustworthy.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

DATA_DIR = Path(os.getenv("KALSHI_DATA_DIR", "data"))
LOG_PATH = DATA_DIR / "forecast_observations.jsonl"
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _append(record: dict) -> None:
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


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def snapshot_projections(symbol: str, projections: Iterable) -> int:
    """Snapshot every projection. Dedup per (symbol, label, target_minute)."""
    history = _read_all()
    recent = {
        (r.get("symbol"), r.get("label"), r.get("target_minute"))
        for r in history[-800:]
        if r.get("kind") == "snapshot"
    }
    n = 0
    now = time.time()
    for p in projections:
        target_epoch = p.target_dt.timestamp()
        target_minute = int(target_epoch // 60)
        key = (symbol, p.label, target_minute)
        if key in recent:
            continue
        _append({
            "kind": "snapshot",
            "ts": now,
            "symbol": symbol,
            "label": p.label,
            "target_epoch": target_epoch,
            "target_minute": target_minute,
            "spot_at_snap": round(p.spot, 6),
            "median": round(p.median, 6),
            "lower": round(p.lower, 6),
            "upper": round(p.upper, 6),
            "drift_pct": round(p.drift_pct, 4),
            "range_pct": round(p.range_pct, 4),
        })
        n += 1
    return n


# ---------------------------------------------------------------------------
# Settle
# ---------------------------------------------------------------------------

def _actual_at(close: pd.Series, target_epoch: float) -> Optional[float]:
    """Find the close price at-or-just-before target_epoch.

    Tolerance: must be within 30 minutes of target. Otherwise the bar is
    too stale to count as a fair check.
    """
    if close is None or len(close) == 0:
        return None
    try:
        idx = close.index
        # Normalize timezone: localize naive → UTC; convert tz-aware → UTC
        if hasattr(idx, "tz"):
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")
        target_ts = pd.Timestamp(target_epoch, unit="s", tz="UTC")
        # Find last index <= target
        mask = idx <= target_ts
        if not mask.any():
            return None
        pos = mask.values.nonzero()[0][-1]
        bar_ts = idx[pos]
        delta_min = abs((target_ts - bar_ts).total_seconds()) / 60.0
        if delta_min > 30:
            return None
        return float(close.iloc[pos])
    except Exception:
        return None


def settle_pending(symbol: str, close: pd.Series) -> int:
    """Walk pending snapshots for `symbol` whose target time has passed,
    and write outcome rows when we can find the actual price.

    Returns number of new outcomes written.
    """
    history = _read_all()
    settled_keys = {
        (r.get("symbol"), r.get("label"), r.get("target_minute"))
        for r in history if r.get("kind") == "outcome"
    }

    now = time.time()
    pending: dict[tuple, dict] = {}
    for r in history:
        if r.get("kind") != "snapshot" or r.get("symbol") != symbol:
            continue
        if (r.get("target_epoch") or 0) > now:
            continue
        k = (r.get("symbol"), r.get("label"), r.get("target_minute"))
        if k in settled_keys:
            continue
        prev = pending.get(k)
        # Use the *earliest* snapshot for that target — that's the original prediction
        if not prev or (r.get("ts") or 0) < (prev.get("ts") or 0):
            pending[k] = r

    written = 0
    written_keys: set[tuple] = set()
    for k, snap in pending.items():
        # Re-check before each write — guards against concurrent reruns or
        # double-calls writing duplicate outcome rows for the same forecast.
        if k in settled_keys or k in written_keys:
            continue
        actual = _actual_at(close, snap.get("target_epoch") or 0)
        if actual is None:
            continue
        med = snap.get("median") or 0
        spot0 = snap.get("spot_at_snap") or 0
        lower = snap.get("lower") or 0
        upper = snap.get("upper") or 0
        if med <= 0 or spot0 <= 0:
            continue
        # Re-read tail of file under lock to catch outcomes another rerun
        # may have just written, before we ourselves write.
        with _LOCK:
            if LOG_PATH.exists():
                try:
                    with LOG_PATH.open("rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - 8192))
                        tail = f.read().decode("utf-8", errors="ignore")
                    for line in tail.splitlines():
                        line = line.strip()
                        if not line or '"kind":"outcome"' not in line:
                            continue
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        rk = (r.get("symbol"), r.get("label"), r.get("target_minute"))
                        if rk == k:
                            settled_keys.add(rk)
                            break
                except Exception:
                    pass
        if k in settled_keys:
            continue
        err_pct = (actual - med) / med * 100
        actual_move_pct = (actual - spot0) / spot0 * 100
        predicted_dir = "up" if med > spot0 else ("down" if med < spot0 else "flat")
        actual_dir = "up" if actual > spot0 else ("down" if actual < spot0 else "flat")
        within_band = lower <= actual <= upper
        _append({
            "kind": "outcome",
            "ts": now,
            "symbol": snap.get("symbol"),
            "label": snap.get("label"),
            "target_minute": snap.get("target_minute"),
            "actual": round(actual, 6),
            "predicted_median": round(med, 6),
            "spot_at_snap": round(spot0, 6),
            "err_pct": round(err_pct, 4),
            "actual_move_pct": round(actual_move_pct, 4),
            "predicted_dir": predicted_dir,
            "actual_dir": actual_dir,
            "within_band": within_band,
        })
        written_keys.add(k)
        written += 1
    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class ForecastReport:
    n_snapshots: int = 0
    n_settled: int = 0
    hit_rate_within_band: float | None = None  # % of outcomes that fell inside the ±1σ band
    mean_abs_error_pct: float | None = None
    direction_accuracy: float | None = None    # % where predicted dir == actual dir (excluding flat)
    mean_bias_pct: float | None = None         # signed mean of err_pct
    by_label: dict[str, dict] = field(default_factory=dict)


def report(symbol: Optional[str] = None) -> ForecastReport:
    history = _read_all()
    snaps = [r for r in history if r.get("kind") == "snapshot"
             and (symbol is None or r.get("symbol") == symbol)]
    outs = [r for r in history if r.get("kind") == "outcome"
            and (symbol is None or r.get("symbol") == symbol)]
    rep = ForecastReport(n_snapshots=len(snaps), n_settled=len(outs))
    if not outs:
        return rep

    hits = [1 if o.get("within_band") else 0 for o in outs]
    errs = [abs(o.get("err_pct") or 0) for o in outs]
    biases = [o.get("err_pct") or 0 for o in outs]
    dirs_dec = [(o.get("predicted_dir"), o.get("actual_dir")) for o in outs
                if o.get("predicted_dir") in ("up", "down") and o.get("actual_dir") in ("up", "down")]
    rep.hit_rate_within_band = sum(hits) / len(hits)
    rep.mean_abs_error_pct = sum(errs) / len(errs)
    rep.mean_bias_pct = sum(biases) / len(biases)
    if dirs_dec:
        rep.direction_accuracy = sum(1 for p, a in dirs_dec if p == a) / len(dirs_dec)

    # Per-label breakdown so we can see "15 min is great, 8am PST is shaky"
    by: dict[str, list[dict]] = {}
    for o in outs:
        by.setdefault(o.get("label") or "?", []).append(o)
    for lbl, items in by.items():
        h = [1 if x.get("within_band") else 0 for x in items]
        e = [abs(x.get("err_pct") or 0) for x in items]
        d = [(x.get("predicted_dir"), x.get("actual_dir")) for x in items
             if x.get("predicted_dir") in ("up", "down") and x.get("actual_dir") in ("up", "down")]
        rep.by_label[lbl] = {
            "n": len(items),
            "hit_band": sum(h) / len(h) if h else None,
            "mae_pct": sum(e) / len(e) if e else None,
            "dir_acc": (sum(1 for p, a in d if p == a) / len(d)) if d else None,
        }
    return rep
