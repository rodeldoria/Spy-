"""Replay logged alerts vs realised forward returns — see hit-rate by
confidence bucket so you know whether to trust the score."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app._shared import candles_long, is_crypto, setup_page
from monte.alerts.engine import tail_alerts


def _forward_return(symbol: str, timeframe: str, ts: float, horizon_bars: int = 20) -> float | None:
    try:
        df = candles_long(symbol, timeframe, lookback_bars=2000)
    except Exception:
        return None
    if df.empty:
        return None
    target_ts = pd.Timestamp(ts, unit="s", tz="UTC") if is_crypto(symbol) else pd.Timestamp(ts, unit="s")
    try:
        if df.index.tz is not None and target_ts.tz is None:
            target_ts = target_ts.tz_localize("UTC")
        if df.index.tz is None and target_ts.tz is not None:
            target_ts = target_ts.tz_convert(None)
    except Exception:
        pass
    pos = df.index.searchsorted(target_ts)
    if pos >= len(df) - horizon_bars:
        return None
    entry = float(df["Close"].iloc[pos])
    fwd = float(df["Close"].iloc[pos + horizon_bars])
    return (fwd - entry) / entry


def main() -> None:
    setup_page("Alert Backtest", icon="🧪")
    horizon = st.sidebar.slider("Forward horizon (bars)", 5, 60, 20)
    rows = tail_alerts(limit=500)
    if not rows:
        st.info("No alerts logged yet.")
        return

    out = []
    for r in rows:
        fwd = _forward_return(r.get("symbol"), r.get("timeframe"), r.get("ts", 0), horizon)
        if fwd is None:
            continue
        side = 1 if str(r.get("action", "")).endswith("BUY") else -1 if str(r.get("action", "")).endswith("SELL") else 0
        if side == 0:
            continue
        pnl = side * fwd
        out.append(
            {
                "symbol": r.get("symbol"),
                "timeframe": r.get("timeframe"),
                "action": r.get("action"),
                "confidence": r.get("confidence", 0),
                "fwd_ret": fwd,
                "pnl": pnl,
                "hit": pnl > 0,
            }
        )
    if not out:
        st.info("No alerts have enough forward data yet to evaluate.")
        return

    df = pd.DataFrame(out)
    cols = st.columns(4)
    cols[0].metric("Alerts", len(df))
    cols[1].metric("Hit rate", f"{df['hit'].mean():.0%}")
    cols[2].metric("Avg fwd P&L", f"{df['pnl'].mean():+.2%}")
    cols[3].metric("Median conf", f"{df['confidence'].median():.0f}%")

    df["bucket"] = pd.cut(df["confidence"], bins=[0, 60, 70, 80, 90, 101], labels=["<60", "60-70", "70-80", "80-90", "90+"])
    by_bucket = df.groupby("bucket", observed=True).agg(
        n=("hit", "size"), hit_rate=("hit", "mean"), avg_pnl=("pnl", "mean")
    )
    st.subheader("Hit-rate by confidence bucket")
    st.dataframe(by_bucket, use_container_width=True)

    fig = px.scatter(df, x="confidence", y="pnl", color="action", hover_data=["symbol", "timeframe"])
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    st.plotly_chart(fig, use_container_width=True)


main()
