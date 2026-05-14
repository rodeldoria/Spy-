"""Aggregate trade rows into ``signal_buckets`` so the results page can
answer "which setups make money."

For each engine we extract a few interpretable buckets:

- **dip_pump**       — RSI band (10-wide), top contribution name, regime.
- **kalshi**         — sign of ``edge`` (pos vs neg), ask-cents band (10c).
- **triangulation**  — dominant vote name, fixture_mode.

Buckets are intentionally coarse: we want enough N in each row to be
meaningful, not a chart with 200 single-trade buckets.
"""

from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean, pstdev
from typing import Iterable

from monte.backtest.store import BacktestStore


def _band(value: float, width: float) -> str:
    lo = int((value // width) * width)
    hi = lo + int(width)
    return f"{lo}-{hi}"


def _iter_trades(store: BacktestStore, run_id: str) -> Iterable[dict]:
    cur = store.conn().execute(
        "SELECT engine, snapshot_json, pnl_pct, win_loss_scratch, direction, fixture_mode "
        "FROM trades WHERE run_id = ?",
        (run_id,),
    )
    for engine, snap_json, pnl, wl, direction, fix in cur:
        try:
            snap = json.loads(snap_json or "{}")
        except json.JSONDecodeError:
            snap = {}
        yield {"engine": engine, "snap": snap, "pnl": pnl or 0.0,
               "wl": wl, "direction": direction, "fixture": fix}


def _bucketize(t: dict) -> list[tuple[str, str]]:
    """Return ``[(signal_name, bucket), ...]`` for a single trade."""
    engine = t["engine"]
    snap = t["snap"]
    out: list[tuple[str, str]] = []

    if engine == "dip_pump":
        rsi = (snap.get("indicators") or {}).get("rsi")
        if isinstance(rsi, (int, float)):
            out.append(("rsi", _band(float(rsi), 10)))
        contribs = snap.get("contributions") or []
        if contribs:
            top = max(contribs, key=lambda c: abs(c.get("score", 0)), default=None)
            if top and top.get("name"):
                out.append(("top_factor", str(top["name"])))
        regime = snap.get("regime")
        if regime:
            out.append(("regime", str(regime)))

    elif engine == "kalshi":
        edge = snap.get("edge")
        if isinstance(edge, (int, float)):
            out.append(("edge_sign", "pos" if edge >= 0 else "neg"))
        ask = snap.get("ask_cents")
        if isinstance(ask, (int, float)):
            out.append(("ask_cents", _band(float(ask), 10)))
        st = snap.get("strike_type")
        if st:
            out.append(("strike_type", str(st)))
        out.append(("direction", t["direction"] or "pass"))

    elif engine == "triangulation":
        votes = (snap.get("votes") or {})
        if votes:
            bull = sum(1 for v in votes.values() if v.get("verdict") == "BULL")
            bear = sum(1 for v in votes.values() if v.get("verdict") == "BEAR")
            dom = "BULL" if bull > bear else "BEAR" if bear > bull else "MIXED"
            out.append(("dominant_vote", dom))
        if t["fixture"]:
            out.append(("fixture_mode", t["fixture"]))
        out.append(("direction", t["direction"] or "pass"))

    return out


def aggregate_buckets(store: BacktestStore, run_id: str, engine: str) -> int:
    """Compute and persist ``signal_buckets`` rows for ``run_id``. Returns
    the number of buckets written."""
    by_bucket: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n": 0, "wins": 0, "losses": 0, "scratches": 0, "pnls": []}
    )
    for t in _iter_trades(store, run_id):
        for signal_name, bucket in _bucketize(t):
            b = by_bucket[(signal_name, bucket)]
            b["n"] += 1
            wl = t["wl"]
            if wl == "win":
                b["wins"] += 1
            elif wl == "loss":
                b["losses"] += 1
            elif wl == "scratch":
                b["scratches"] += 1
            b["pnls"].append(t["pnl"])

    for (signal_name, bucket), agg in by_bucket.items():
        pnls = agg["pnls"]
        avg = mean(pnls) if pnls else 0.0
        sharpe = (avg / pstdev(pnls)) if len(pnls) >= 2 and pstdev(pnls) > 0 else None
        store.upsert_signal_bucket(
            run_id=run_id, engine=engine, signal_name=signal_name, bucket=bucket,
            n=agg["n"], wins=agg["wins"], losses=agg["losses"],
            scratches=agg["scratches"], avg_pnl_pct=avg, sharpe=sharpe,
        )
    return len(by_bucket)
