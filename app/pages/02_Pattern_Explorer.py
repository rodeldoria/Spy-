"""Pattern Explorer — query Chroma for windows similar to the current one,
plot the forward-return distribution of the neighbours."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from app._shared import candles_short, pattern_store, setup_page, sidebar_watchlists
from monte.config import settings
from monte.patterns.match import find_similar


def main() -> None:
    setup_page("Pattern Explorer", icon="🔍")
    crypto, stocks = sidebar_watchlists()
    symbols = crypto + stocks
    if not symbols:
        st.warning("Add a symbol in the sidebar.")
        return

    sym = st.sidebar.selectbox("Symbol", symbols)
    timeframe = st.sidebar.selectbox("Timeframe", ["1h", "15m", "5m", "1d"], index=0)
    k = st.sidebar.slider("Neighbours (k)", 5, 100, 20)

    store = pattern_store()
    n = store.count(sym, timeframe)
    st.caption(
        f"Collection **{sym} {timeframe}** has **{n}** vectors. "
        f"Backfill via `python -m monte.patterns.ingest {sym} {timeframe} --years 1`."
    )

    try:
        df = candles_short(sym, timeframe)
    except Exception as e:
        st.error(f"Data error: {e}")
        return

    try:
        sim = find_similar(sym, timeframe, df, k=k, store=store, window=settings.pattern_window)
    except Exception as e:
        st.error(f"Pattern match error: {e}")
        return

    if sim.cold_start:
        st.info("Cold start — no vectors in the store yet for this (symbol, timeframe).")
        return

    cols = st.columns(5)
    cols[0].metric("Pattern score", f"{sim.pattern_score:+.2f}")
    cols[1].metric("Win rate (fwd-20)", f"{sim.win_rate:.0%}")
    cols[2].metric("Mean fwd-5", f"{sim.mean_fwd_5:+.2%}", f"σ {sim.std_fwd_5:.2%}")
    cols[3].metric("Mean fwd-20", f"{sim.mean_fwd_20:+.2%}", f"σ {sim.std_fwd_20:.2%}")
    cols[4].metric("k", str(sim.k))

    res = store.query(sym, timeframe, _vec_for(sym, timeframe, df), k=k)
    fwd20 = [float(m.get("fwd_20_ret", 0.0)) for m in res.metadatas]
    if fwd20:
        fig = px.histogram(
            x=fwd20, nbins=25, title="Forward 20-bar return distribution of neighbours"
        )
        fig.add_vline(x=0, line_dash="dash", line_color="grey")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent close")
    st.line_chart(df["Close"].tail(120))


def _vec_for(sym: str, timeframe: str, df):
    from monte.patterns.encoder import encode_window

    return encode_window(df, window=settings.pattern_window).vector


main()
