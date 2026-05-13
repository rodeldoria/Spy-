"""Pattern tracker — record every AI Decision Council verdict and learn
from outcomes.

Each Council verdict is logged with its 8-bit *signature* (which of the
8 frameworks passed), so we can aggregate hit-rate per signature and
learn which checkpoint combinations actually pay. Once the underlying
Kalshi market settles, we join with the kalshi_calibration outcomes
log to attach win/loss + ROI to each verdict.

After ≥5 settled outcomes for a given signature, we compute a
*confidence multiplier* the council can use to bump or fade its raw
trigger score. If signature `11110011` has historically won 80% of the
time, every new verdict with that same signature gets +X confidence.
If a signature has been losing, the council fades it.

The Playbook page reads from this log to show:
  - rolling list of every verdict and its outcome,
  - per-signature hit-rate / avg ROI table,
  - a graph of trigger-score-vs-actual-outcome over time.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import fcntl  # POSIX-only; not available on Windows
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover
    _HAS_FCNTL = False

DEFAULT_PATH = Path(os.path.expanduser("~/.monte/pattern_tracker.jsonl"))


@contextmanager
def _locked_append(path: Path):
    """Append to a JSONL file under an exclusive flock so concurrent
    Streamlit re-renders can't interleave writes mid-line."""
    f = path.open("a")
    try:
        if _HAS_FCNTL:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield f
    finally:
        try:
            if _HAS_FCNTL:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


def _path() -> Path:
    p = Path(os.environ.get("MONTE_PATTERN_TRACKER_PATH", str(DEFAULT_PATH)))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def signature_from_checkpoints(checkpoints) -> str:
    """Convert a list of Checkpoint objects into a stable binary signature,
    e.g. '11010110'. Length matches the number of checkpoints (currently 8
    in decision_council.evaluate, but resilient if more are added)."""
    return "".join("1" if c.passed else "0" for c in checkpoints)


# Default labels for the current 8-checkpoint Council. If the council ever
# adds/removes checkpoints, signatures of a different length will fall back
# to generic CHK1/CHK2 labels rather than crashing.
_DEFAULT_CHECKPOINT_NAMES = [
    "EDGE", "KELL", "EV", "CONV", "LIQ", "CAL", "TRI", "PRE",
]


def signature_label(sig: str, checkpoints=None) -> str:
    """Render a binary signature like '11010110' as a compact human-readable
    string. Length-agnostic — whatever bits are present get a label."""
    if checkpoints:
        names = [c.name[:4].upper() for c in checkpoints]
    else:
        names = list(_DEFAULT_CHECKPOINT_NAMES)
    # Pad names if the signature is longer than our defaults
    while len(names) < len(sig):
        names.append(f"CHK{len(names) + 1}")
    return " ".join(
        f"{names[i]}{'✓' if bit == '1' else '✗'}"
        for i, bit in enumerate(sig)
    )


def record_verdict(
    *,
    market_ticker: str,
    symbol: str,
    bet_summary: str,
    direction: str,
    ask_cents: int,
    edge_pp: float,
    ev_per_dollar: float,
    kelly_fraction: float,
    confidence_pct: float,
    trigger_score: float,
    mechanical_score: float,
    ai_score: Optional[float],
    verdict_label: str,
    signature: str,
    checkpoint_names: list[str],
    close_time: Optional[float] = None,
    dedupe_window_sec: float = 300.0,
) -> bool:
    """Append a verdict row to the pattern tracker log.

    Dedupes: if we already recorded the same (ticker, signature) within
    the last `dedupe_window_sec`, skip — Streamlit re-renders fire a lot.
    Returns True if a new row was written, False if it was deduped.
    """
    # Round to ms so JSON round-trip stays stable for the join key
    now = round(time.time(), 3)
    p = _path()
    if p.exists():
        # Cheap tail-scan — only check the last ~50 lines for dedup
        try:
            lines = p.read_text().splitlines()[-50:]
            for ln in reversed(lines):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") != "verdict":
                    continue
                if (rec.get("ticker") == market_ticker
                        and rec.get("signature") == signature
                        and (now - float(rec.get("ts", 0))) < dedupe_window_sec):
                    return False
        except Exception:
            pass

    row = {
        "kind": "verdict",
        "id": uuid.uuid4().hex[:12],
        "ts": now,
        "ticker": market_ticker,
        "symbol": symbol,
        "bet_summary": bet_summary,
        "direction": direction,
        "ask_cents": int(ask_cents),
        "edge_pp": round(float(edge_pp), 2),
        "ev_per_dollar": round(float(ev_per_dollar), 4),
        "kelly_fraction": round(float(kelly_fraction), 4),
        "confidence_pct": round(float(confidence_pct), 1),
        "trigger_score": round(float(trigger_score), 1),
        "mechanical_score": round(float(mechanical_score), 1),
        "ai_score": round(float(ai_score), 1) if ai_score is not None else None,
        "verdict_label": verdict_label,
        "signature": signature,
        "checkpoint_names": list(checkpoint_names),
        "close_time": float(close_time) if close_time else None,
    }
    with _locked_append(p) as f:
        f.write(json.dumps(row) + "\n")
    return True


# ---------------------------------------------------------------------------
# Outcome reconciliation
# ---------------------------------------------------------------------------

def reconcile_outcomes() -> dict:
    """Join verdict rows against the kalshi_calibration outcomes log so
    every verdict whose market has now settled gets win/loss + ROI.

    Idempotent — only writes outcome rows for verdicts that don't already
    have one. Returns {"settled": N, "still_pending": M}.
    """
    from monte.learning.kalshi_calibration import _read_all as _read_kalshi
    summary = {"settled": 0, "still_pending": 0}

    p = _path()
    if not p.exists():
        return summary

    try:
        lines = p.read_text().splitlines()
    except Exception:
        return summary

    rows = []
    # Track settled verdicts by their unique id (preferred) AND by the
    # legacy (ticker, verdict_ts) key so we don't double-settle anything.
    settled_ids: set[str] = set()
    settled_keys: set[tuple[str, float]] = set()
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue
        rows.append(rec)
        if rec.get("kind") == "outcome":
            if rec.get("verdict_id"):
                settled_ids.add(rec["verdict_id"])
            settled_keys.add((rec.get("ticker"), float(rec.get("verdict_ts", 0))))

    # Look up the kalshi outcome for each pending verdict
    kalshi_outcomes = {
        r["ticker"]: r for r in _read_kalshi() if r.get("kind") == "outcome"
    }

    new_outcomes = []
    for r in rows:
        if r.get("kind") != "verdict":
            continue
        rid = r.get("id")
        key = (r.get("ticker"), float(r.get("ts", 0)))
        if (rid and rid in settled_ids) or key in settled_keys:
            continue
        kout = kalshi_outcomes.get(r.get("ticker"))
        if not kout:
            summary["still_pending"] += 1
            continue
        outcome_yes = int(kout.get("outcome_yes", 0))
        # Did our trade win?
        won = (
            (r.get("direction") == "YES" and outcome_yes == 1)
            or (r.get("direction") == "NO" and outcome_yes == 0)
        )
        ask_c = int(r.get("ask_cents") or 0)
        if won and ask_c > 0:
            roi_pct = (100.0 - ask_c) / ask_c * 100.0
        elif ask_c > 0:
            roi_pct = -100.0
        else:
            roi_pct = 0.0
        new_outcomes.append({
            "kind": "outcome",
            "ts": round(time.time(), 3),
            "ticker": r.get("ticker"),
            "verdict_id": rid,
            "verdict_ts": float(r.get("ts", 0)),
            "signature": r.get("signature"),
            "trigger_score": r.get("trigger_score"),
            "won": bool(won),
            "outcome_yes": outcome_yes,
            "roi_pct": round(roi_pct, 2),
            "settle_price": kout.get("settle_price"),
        })
        summary["settled"] += 1

    if new_outcomes:
        with _locked_append(p) as f:
            for o in new_outcomes:
                f.write(json.dumps(o) + "\n")
    return summary


# ---------------------------------------------------------------------------
# Aggregation / learning
# ---------------------------------------------------------------------------

@dataclass
class SignatureStats:
    signature: str
    n_total: int = 0
    n_settled: int = 0
    n_won: int = 0
    sum_roi_pct: float = 0.0
    avg_trigger_score: float = 0.0

    @property
    def hit_rate(self) -> Optional[float]:
        return self.n_won / self.n_settled if self.n_settled else None

    @property
    def avg_roi(self) -> Optional[float]:
        return self.sum_roi_pct / self.n_settled if self.n_settled else None


@dataclass
class TrackerReport:
    n_verdicts: int = 0
    n_outcomes: int = 0
    n_won: int = 0
    avg_roi_pct: Optional[float] = None
    by_signature: dict[str, SignatureStats] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    outcomes: list[dict] = field(default_factory=list)

    @property
    def overall_hit_rate(self) -> Optional[float]:
        return self.n_won / self.n_outcomes if self.n_outcomes else None


def build_report() -> TrackerReport:
    p = _path()
    rep = TrackerReport()
    if not p.exists():
        return rep
    try:
        lines = p.read_text().splitlines()
    except Exception:
        return rep

    score_sum_by_sig: dict[str, float] = {}
    count_by_sig: dict[str, int] = {}
    roi_sum_total = 0.0
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") == "verdict":
            rep.n_verdicts += 1
            rep.rows.append(rec)
            sig = rec.get("signature", "")
            stats = rep.by_signature.setdefault(sig, SignatureStats(signature=sig))
            stats.n_total += 1
            score_sum_by_sig[sig] = score_sum_by_sig.get(sig, 0.0) + float(rec.get("trigger_score") or 0)
            count_by_sig[sig] = count_by_sig.get(sig, 0) + 1
        elif rec.get("kind") == "outcome":
            rep.n_outcomes += 1
            rep.outcomes.append(rec)
            sig = rec.get("signature", "")
            stats = rep.by_signature.setdefault(sig, SignatureStats(signature=sig))
            stats.n_settled += 1
            if rec.get("won"):
                rep.n_won += 1
                stats.n_won += 1
            roi = float(rec.get("roi_pct") or 0)
            stats.sum_roi_pct += roi
            roi_sum_total += roi

    if rep.n_outcomes:
        rep.avg_roi_pct = roi_sum_total / rep.n_outcomes

    for sig, stats in rep.by_signature.items():
        if count_by_sig.get(sig):
            stats.avg_trigger_score = score_sum_by_sig[sig] / count_by_sig[sig]

    # Sort rows newest-first for UI
    rep.rows.sort(key=lambda r: -float(r.get("ts", 0)))
    rep.outcomes.sort(key=lambda r: -float(r.get("ts", 0)))
    return rep


def confidence_multiplier(
    signature: str,
    *,
    min_samples: int = 3,
    full_trust_at: int = 20,
) -> tuple[float, str]:
    """Return (multiplier, source_label) the council can use to adjust raw
    trigger scores based on this signature's history.

    Bayesian-shrinkage flavoured:
      • Below `min_samples` settled outcomes → 1.0 (no signal yet).
      • Between `min_samples` and `full_trust_at` → blend the raw multiplier
        toward 1.0 proportional to sample size, so a 0/3 doesn't nuke a
        play down to 0.6× on day one.
      • At/above `full_trust_at` → use the full [0.6, 1.4] range.

    multiplier > 1.0 → boost (signature has been winning),
    multiplier < 1.0 → fade (signature has been losing),
    multiplier = 1.0 → no change (not enough data yet).
    """
    rep = build_report()
    stats = rep.by_signature.get(signature)
    if not stats or stats.n_settled < min_samples:
        n = stats.n_settled if stats else 0
        return 1.0, f"raw (only {n}/{min_samples} settled for this signature)"
    hr = stats.hit_rate or 0.5
    raw_mult = 0.6 + (hr * 0.8)  # [0.6, 1.4]
    # Shrink toward 1.0 based on sample size
    trust = min(1.0, max(0.0, (stats.n_settled - min_samples)
                         / max(1, full_trust_at - min_samples)))
    mult = 1.0 + (raw_mult - 1.0) * trust
    return round(mult, 3), (
        f"learned ({stats.n_settled} settled, "
        f"hit rate {hr*100:.0f}%, avg ROI {stats.avg_roi:+.1f}%, "
        f"trust {trust*100:.0f}%)"
    )


__all__ = [
    "record_verdict",
    "reconcile_outcomes",
    "build_report",
    "confidence_multiplier",
    "signature_from_checkpoints",
    "signature_label",
    "SignatureStats",
    "TrackerReport",
]
