"""Replay logged alerts vs realised forward returns — see hit-rate by
confidence bucket so you know whether to trust the score."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app._shared import candles_long, is_crypto, setup_page
from app._ui import inject_global_css, loading, status_pill
from monte.alerts.engine import tail_alerts


def _forward_return(symbol: str, timeframe: str, ts: float, horizon_bars: int = 20) -> float | None:
    try:
        df = candles_long(symbol, timeframe, lookback_bars=2000)
    except Exception:
        return None
    if df.empty:
        return None
    try:
        # Build a tz-aware UTC timestamp then match the index's tz
        target_ts = pd.Timestamp(ts, unit="s", tz="UTC")
        if df.index.tz is None:
            target_ts = target_ts.tz_localize(None)
        elif str(df.index.tz) != "UTC":
            target_ts = target_ts.tz_convert(df.index.tz)

        # Pandas 3.x: DatetimeIndex carries an explicit unit (ns/us/ms/s).
        # searchsorted raises "Cannot losslessly convert units" if they differ.
        idx_unit = getattr(df.index, "unit", "ns")
        target_ts = target_ts.as_unit(idx_unit)

        pos = df.index.searchsorted(target_ts)
    except Exception:
        return None
    if pos >= len(df) - horizon_bars:
        return None
    entry = float(df["Close"].iloc[pos])
    fwd = float(df["Close"].iloc[pos + horizon_bars])
    return (fwd - entry) / entry


def main() -> None:
    setup_page("Alert Backtest", icon="🧪")
    inject_global_css()
    horizon = st.sidebar.slider("Forward horizon (bars)", 5, 60, 20)

    with loading("Loading alerts log…"):
        rows = tail_alerts(limit=500)
    if not rows:
        st.markdown(
            status_pill("no alerts logged yet — run a scan from the home page", "muted"),
            unsafe_allow_html=True,
        )
        return

    out = []
    progress = st.progress(0.0, text="Computing forward returns…")
    for i, r in enumerate(rows, 1):
        progress.progress(i / len(rows), text=f"Computing forward returns ({i}/{len(rows)})")
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
    progress.empty()
    if not out:
        st.info(
            "No alerts have enough forward data yet to evaluate. "
            "(Data is flowing — but each alert needs `horizon` bars after "
            "its timestamp before it can be scored.)"
        )
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

    fig = px.scatter(
        df, x="confidence", y="pnl", color="action",
        hover_data=["symbol", "timeframe"],
        title="Confidence vs realised forward P&L",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(tickformat=".1%", gridcolor="rgba(127,127,127,0.15)"),
        xaxis=dict(gridcolor="rgba(127,127,127,0.15)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Pattern P&L Delta Bundle ──────────────────────────────────────────────
    st.subheader("Pattern P&L delta bundle")
    st.caption(
        "If you had paper-traded every signal the system fired — how much would "
        "each pattern cluster have earned? Each row assumes $100 risked per trade."
    )

    risk_per_trade = st.sidebar.number_input(
        "Hypothetical $ per trade (bundle calc)",
        min_value=10.0, max_value=10000.0, value=100.0, step=10.0,
    )

    # Per-symbol bundle
    by_sym = df.groupby("symbol").agg(
        signals=("pnl", "size"),
        hit_rate=("hit", "mean"),
        avg_pnl=("pnl", "mean"),
        total_pnl=("pnl", "sum"),
    ).reset_index()
    by_sym["delta_$"] = by_sym["total_pnl"] * risk_per_trade
    by_sym["hit_rate"] = by_sym["hit_rate"].map("{:.0%}".format)
    by_sym["avg_pnl"] = by_sym["avg_pnl"].map("{:+.2%}".format)
    by_sym["delta_$"] = by_sym["delta_$"].map("${:+,.2f}".format)
    by_sym = by_sym.rename(columns={
        "symbol": "Symbol", "signals": "Signals",
        "hit_rate": "Win rate", "avg_pnl": "Avg P&L/trade", "delta_$": f"Delta (${risk_per_trade:.0f}/trade)",
    })

    # Per-action bundle (BUY vs SELL)
    by_action = df.groupby("action").agg(
        signals=("pnl", "size"),
        hit_rate=("hit", "mean"),
        avg_pnl=("pnl", "mean"),
        total_pnl=("pnl", "sum"),
    ).reset_index()
    by_action["delta_$"] = by_action["total_pnl"] * risk_per_trade

    tab1, tab2, tab3 = st.tabs(["By symbol", "By action", "Equity curve"])

    with tab1:
        st.dataframe(by_sym.drop(columns=["total_pnl"], errors="ignore"), use_container_width=True)
        best = df.groupby("symbol")["pnl"].sum().idxmax() if not df.empty else None
        if best:
            best_delta = df[df["symbol"] == best]["pnl"].sum() * risk_per_trade
            st.success(
                f"Best pattern cluster: **{best}** — "
                f"${best_delta:+,.2f} cumulative delta if you followed every signal"
            )

    with tab2:
        for _, row in by_action.iterrows():
            color = "#0a7d2a" if row["delta_$"] >= 0 else "#a8261f"
            st.markdown(
                f"""
                <div style='border-left:4px solid {color};background:rgba(127,127,127,0.07);
                    border-radius:8px;padding:10px 14px;margin-bottom:8px;'>
                  <strong style='font-size:1rem;color:{color};'>{row["action"]}</strong>
                  <span style='color:#888;font-size:0.85rem;margin-left:8px;'>
                    {int(row["signals"])} signals · {row["hit_rate"]:.0%} win rate
                  </span>
                  <div style='font-size:1.3rem;font-weight:700;color:{color};margin-top:4px;'>
                    ${row["delta_$"]:+,.2f}
                    <span style='font-size:0.82rem;font-weight:400;color:#888;'>
                      cumulative delta · avg {row["avg_pnl"]:+.2%}/trade
                    </span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab3:
        df_sorted = df.copy()
        df_sorted["trade_n"] = range(1, len(df_sorted) + 1)
        df_sorted["cumulative_delta"] = df_sorted["pnl"].cumsum() * risk_per_trade
        fig2 = px.line(
            df_sorted, x="trade_n", y="cumulative_delta",
            color="action", markers=True,
            title=f"Cumulative P&L delta (${risk_per_trade:.0f}/trade)",
            labels={"trade_n": "Signal #", "cumulative_delta": "Cumulative $ delta"},
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="grey")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(tickprefix="$", gridcolor="rgba(127,127,127,0.15)"),
            xaxis=dict(gridcolor="rgba(127,127,127,0.15)"),
        )
        st.plotly_chart(fig2, use_container_width=True)
        total = df_sorted["cumulative_delta"].iloc[-1] if not df_sorted.empty else 0
        st.caption(
            f"If you had followed every signal with ${risk_per_trade:.0f} per trade, "
            f"your paper account would have moved **${total:+,.2f}** over these {len(df_sorted)} signals."
        )


main()
