"""Pattern Explorer — query the vector store for windows similar to the
current one and plot the forward-return distribution of the neighbours.

Even when the store is cold (no vectors yet), the page still shows the
recent close so you can verify data is flowing into the page.
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app._shared import candles_short, pattern_store, setup_page, sidebar_watchlists
from app._ui import (
    freshness_pill,
    inject_global_css,
    loading,
    status_pill,
)
from monte.config import settings
from monte.patterns.match import find_similar


def _vec_for(sym: str, timeframe: str, df):
    from monte.patterns.encoder import encode_window

    return encode_window(df, window=settings.pattern_window).vector


def _close_chart(closes) -> go.Figure:
    fig = go.Figure(go.Scatter(x=closes.index, y=closes.values, mode="lines"))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(127,127,127,0.15)"),
    )
    return fig


def main() -> None:
    setup_page("Pattern Explorer", icon="🔍")
    inject_global_css()

    crypto, stocks = sidebar_watchlists()
    symbols = crypto + stocks
    if not symbols:
        st.warning("Add a symbol in the sidebar.")
        return

    sym = st.sidebar.selectbox("Symbol", symbols)
    timeframe = st.sidebar.selectbox("Timeframe", ["1h", "15m", "5m", "1d"], index=0)
    k = st.sidebar.slider("Neighbours (k)", 5, 100, 20)

    with loading(f"Connecting to vector store for {sym} {timeframe}…"):
        store = pattern_store()
        n = store.count(sym, timeframe)

    st.markdown(
        f"{status_pill(f'{n} vectors in store', 'info' if n > 0 else 'warn')} "
        f"{status_pill(f'{sym} · {timeframe}', 'muted')}",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Backfill via `python -m monte.patterns.ingest {sym} {timeframe} --years 1`."
    )

    try:
        with loading(f"Fetching {sym} candles…"):
            df = candles_short(sym, timeframe)
    except Exception as e:
        st.error(f"Data error: {e}")
        return

    if df.empty:
        st.warning("No candles returned. Try a different timeframe.")
        return

    last_ts = df.index[-1] if len(df.index) else None
    st.markdown(freshness_pill(last_ts), unsafe_allow_html=True)

    st.subheader("Recent close")
    st.plotly_chart(_close_chart(df["Close"].tail(180)), use_container_width=True)

    if n == 0:
        st.info(
            "Cold start — no vectors in the store yet for this "
            "(symbol, timeframe). Data is still flowing (chart above is live)."
        )
        return

    try:
        with loading(f"Searching for {k} similar windows…"):
            sim = find_similar(
                sym, timeframe, df, k=k, store=store, window=settings.pattern_window
            )
    except Exception as e:
        st.error(f"Pattern match error: {e}")
        return

    if sim.cold_start:
        st.info("Cold start — no vectors yet for this (symbol, timeframe).")
        return

    cols = st.columns(5)
    cols[0].metric("Pattern score", f"{sim.pattern_score:+.2f}")
    cols[1].metric("Win rate (fwd-20)", f"{sim.win_rate:.0%}")
    cols[2].metric("Mean fwd-5", f"{sim.mean_fwd_5:+.2%}", f"σ {sim.std_fwd_5:.2%}")
    cols[3].metric("Mean fwd-20", f"{sim.mean_fwd_20:+.2%}", f"σ {sim.std_fwd_20:.2%}")
    cols[4].metric("k", str(sim.k))

    with loading("Pulling neighbour metadata…"):
        res = store.query(sym, timeframe, _vec_for(sym, timeframe, df), k=k)
        fwd20 = [float(m.get("fwd_20_ret", 0.0)) for m in res.metadatas]

    if fwd20:
        fig = px.histogram(
            x=fwd20,
            nbins=25,
            title="Forward 20-bar return distribution of neighbours",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="grey")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, tickformat=".1%"),
            yaxis=dict(showgrid=True, gridcolor="rgba(127,127,127,0.15)"),
        )
        st.plotly_chart(fig, use_container_width=True)


main()
