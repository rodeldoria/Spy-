"""Temporal performance report.

Slices the settled-outcome history from `kalshi_calibration` and
`forecast_calibration` along time-of-day and day-of-week (PST) to
answer questions like:

  - "Which session has the highest model hit-rate — US open, midday,
    or close?"
  - "Are Friday markets actually slower / harder to predict?"
  - "Does the model do better at 8am PST than at midnight?"

Sessions (PST):
  - "Pre-market"  : 04:00-06:30
  - "US Open"     : 06:30-09:30
  - "Midday Drag" : 09:30-12:00
  - "Afternoon"   : 12:00-13:00
  - "Close"       : 13:00-15:30
  - "After Hours" : 15:30-20:00
  - "Asia"        : 20:00-04:00 (next day)

Returns a `TemporalReport` with three breakdowns:
  by_session, by_hour_pst, by_dow (Mon..Sun)
each is a list of `Bucket(label, n, hit_rate, avg_edge_pp, avg_err_pct)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

from monte.learning import kalshi_calibration as kcal
from monte.learning import forecast_calibration as fcal

try:
    from zoneinfo import ZoneInfo
    PACIFIC = ZoneInfo("America/Los_Angeles")
except Exception:  # pragma: no cover - zoneinfo always available on 3.9+
    PACIFIC = timezone(timedelta(hours=-8))


def _to_pacific(epoch: float) -> datetime:
    """Convert a Unix epoch to Pacific local time (DST-aware)."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone(PACIFIC)


SESSION_ORDER = [
    "Pre-market", "US Open", "Midday Drag", "Afternoon",
    "Close", "After Hours", "Asia",
]

DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _session_for_pst_hour(h: float) -> str:
    if 4 <= h < 6.5:
        return "Pre-market"
    if 6.5 <= h < 9.5:
        return "US Open"
    if 9.5 <= h < 12:
        return "Midday Drag"
    if 12 <= h < 13:
        return "Afternoon"
    if 13 <= h < 15.5:
        return "Close"
    if 15.5 <= h < 20:
        return "After Hours"
    return "Asia"


@dataclass
class Bucket:
    label: str
    n: int = 0
    hits: int = 0
    edge_sum: float = 0.0     # sum of |edge| pp at snapshot
    err_sum: float = 0.0      # sum of |forecast err| pct
    n_with_err: int = 0

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.n if self.n else None

    @property
    def avg_edge_pp(self) -> float | None:
        return self.edge_sum / self.n if self.n else None

    @property
    def avg_err_pct(self) -> float | None:
        return self.err_sum / self.n_with_err if self.n_with_err else None


@dataclass
class TemporalReport:
    n_kalshi: int = 0
    n_forecast: int = 0
    by_session: dict[str, Bucket] = field(default_factory=dict)
    by_hour_pst: dict[int, Bucket] = field(default_factory=dict)
    by_dow: dict[str, Bucket] = field(default_factory=dict)
    best_session: str | None = None
    worst_session: str | None = None


def _bucket(d: dict, key, label: str) -> Bucket:
    if key not in d:
        d[key] = Bucket(label=label)
    return d[key]


def _ingest_kalshi_outcome(
    o: dict, rep: TemporalReport, snap_ts_by_ticker: dict[str, float]
) -> None:
    """Bucket a settled Kalshi outcome by the *decision time* (snapshot ts),
    not the settlement-sweep ts. Falls back to outcome.ts if no snap found."""
    try:
        ticker = o.get("ticker")
        # Prefer when the bet was placed (decision time), not when we
        # detected settlement. This is what tells us "the model is sharp
        # at US Open" vs "we happen to run settlement sweeps at noon".
        decision_ts = snap_ts_by_ticker.get(ticker) or float(o.get("ts") or 0)
        if not decision_ts:
            return
        pac = _to_pacific(decision_ts)
        h = pac.hour + pac.minute / 60.0
        sess = _session_for_pst_hour(h)
        dow = DOW_ORDER[pac.weekday()]

        direction = (o.get("direction") or "").upper()
        actual_yes = o.get("outcome_yes")
        if actual_yes is None:
            return
        won = (
            (direction == "YES" and actual_yes == 1)
            or (direction == "NO" and actual_yes == 0)
        )
        edge_pp = abs(float(o.get("edge") or 0)) * 100

        for store, key, label in [
            (rep.by_session, sess, sess),
            (rep.by_hour_pst, int(pac.hour), f"{int(pac.hour):02d}:00 PT"),
            (rep.by_dow, dow, dow),
        ]:
            b = _bucket(store, key, label)
            b.n += 1
            if won:
                b.hits += 1
            b.edge_sum += edge_pp
    except (TypeError, ValueError, OverflowError, OSError):
        # One malformed row should never poison the whole report.
        return


def _ingest_forecast_outcome(o: dict, rep: TemporalReport) -> None:
    """Bucket a forecast outcome by its target time (when the prediction
    was *for*) — for forecasts that's the right comparison point."""
    try:
        target_epoch = o.get("target_epoch") or 0
        if not target_epoch:
            tm = o.get("target_minute")
            if tm:
                target_epoch = tm * 60
        if not target_epoch:
            return
        pac = _to_pacific(target_epoch)
        h = pac.hour + pac.minute / 60.0
        sess = _session_for_pst_hour(h)
        dow = DOW_ORDER[pac.weekday()]
        err = abs(float(o.get("err_pct") or 0))

        for store, key, label in [
            (rep.by_session, sess, sess),
            (rep.by_hour_pst, int(pac.hour), f"{int(pac.hour):02d}:00 PT"),
            (rep.by_dow, dow, dow),
        ]:
            b = _bucket(store, key, label)
            b.err_sum += err
            b.n_with_err += 1
    except (TypeError, ValueError, OverflowError, OSError):
        return


def build_report() -> TemporalReport:
    rep = TemporalReport()
    try:
        k_history = kcal._read_all()
    except Exception:
        k_history = []
    try:
        f_history = fcal._read_all()
    except Exception:
        f_history = []

    k_outcomes = [r for r in k_history if r.get("kind") == "outcome"]
    f_outcomes = [r for r in f_history if r.get("kind") == "outcome"]

    # Map ticker → earliest snapshot ts so we bucket Kalshi outcomes by the
    # decision moment rather than the settlement-sweep moment.
    snap_ts_by_ticker: dict[str, float] = {}
    for r in k_history:
        if r.get("kind") != "snapshot":
            continue
        t = r.get("ticker")
        if not t:
            continue
        ts = r.get("ts") or 0
        prev = snap_ts_by_ticker.get(t)
        if prev is None or ts < prev:
            snap_ts_by_ticker[t] = float(ts)

    rep.n_kalshi = len(k_outcomes)
    rep.n_forecast = len(f_outcomes)

    for o in k_outcomes:
        _ingest_kalshi_outcome(o, rep, snap_ts_by_ticker)
    for o in f_outcomes:
        _ingest_forecast_outcome(o, rep)

    if rep.by_session:
        # Best/worst sessions by Kalshi hit rate (require ≥3 samples)
        candidates = [
            (s, b.hit_rate) for s, b in rep.by_session.items()
            if b.n >= 3 and b.hit_rate is not None
        ]
        if candidates:
            rep.best_session = max(candidates, key=lambda x: x[1])[0]
            rep.worst_session = min(candidates, key=lambda x: x[1])[0]
    return rep


def session_table(rep: TemporalReport) -> list[dict]:
    """Render-friendly rows in canonical session order."""
    out = []
    for s in SESSION_ORDER:
        b = rep.by_session.get(s)
        out.append({
            "Session (PST)": s,
            "n": b.n if b else 0,
            "Kalshi hit-rate": f"{b.hit_rate*100:.0f}%" if (b and b.hit_rate is not None) else "—",
            "Avg edge": f"{b.avg_edge_pp:.1f}pp" if (b and b.avg_edge_pp is not None) else "—",
            "Forecast MAE": f"{b.avg_err_pct:.2f}%" if (b and b.avg_err_pct is not None) else "—",
        })
    return out


def dow_table(rep: TemporalReport) -> list[dict]:
    out = []
    for d in DOW_ORDER:
        b = rep.by_dow.get(d)
        out.append({
            "Day (PST)": d,
            "n": b.n if b else 0,
            "Kalshi hit-rate": f"{b.hit_rate*100:.0f}%" if (b and b.hit_rate is not None) else "—",
            "Avg edge": f"{b.avg_edge_pp:.1f}pp" if (b and b.avg_edge_pp is not None) else "—",
            "Forecast MAE": f"{b.avg_err_pct:.2f}%" if (b and b.avg_err_pct is not None) else "—",
        })
    return out
