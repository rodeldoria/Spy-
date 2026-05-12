"""Watchlist dashboard — indicators per symbol, color-coded action chip."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app._shared import (
    action_color,
    candles_short,
    live_price,
    setup_page,
    sidebar_watchlists,
)
from monte.indicators.regime import classify_regime
from monte.indicators.technical import bollinger, macd, rsi
from monte.signals.dip_pump import detect


def _sparkline(closes) -> go.Figure:
    fig = go.Figure(go.Scatter(x=closes.index, y=closes.values, mode="lines"))
    fig.update_layout(
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def main() -> None:
    setup_page("Crypto + SPY Watchlist", icon="📊")
    crypto, stocks = sidebar_watchlists()
    timeframe = st.sidebar.selectbox("Timeframe", ["1h", "15m", "5m", "1d"], index=0)
    symbols = crypto + stocks
    if not symbols:
        st.warning("Add at least one symbol in the sidebar.")
        return

    cols = st.columns(2)
    for i, sym in enumerate(symbols):
        with cols[i % 2]:
            with st.container(border=True):
                try:
                    df = candles_short(sym, timeframe)
                except Exception as e:
                    st.error(f"{sym}: data error — {e}")
                    continue
                try:
                    spot, src = live_price(sym)
                except Exception:
                    spot, src = float(df["Close"].iloc[-1]), "candle"

                close = df["Close"]
                last_rsi = float(rsi(close).iloc[-1])
                bb = bollinger(close).iloc[-1]
                macd_hist = float(macd(close)["hist"].iloc[-1])
                regime = classify_regime(df)
                try:
                    alert = detect(df, symbol=sym, timeframe=timeframe)
                except Exception as e:
                    st.error(f"{sym}: detector error — {e}")
                    continue

                color = action_color(alert.action.value)
                st.markdown(
                    f"### {sym} <span style='color:{color}'>{alert.action.value}</span>"
                    f" &nbsp;<small>conf {alert.confidence:.0f}%</small>",
                    unsafe_allow_html=True,
                )
                m = st.columns(5)
                m[0].metric("Spot", f"${spot:,.2f}", help=f"source: {src}")
                m[1].metric("RSI(14)", f"{last_rsi:.0f}")
                m[2].metric("BB %b", f"{bb['bb_pctb']:.2f}")
                m[3].metric("MACD hist", f"{macd_hist:+.3f}")
                m[4].metric("Regime", regime.regime.value, f"ADX {regime.adx:.0f}")
                st.plotly_chart(_sparkline(close.tail(120)), use_container_width=True)
                st.caption(
                    f"Entry ${alert.entry:,.2f} · Stop ${alert.stop:,.2f} · "
                    f"Target ${alert.target:,.2f} · R:R {alert.rr:.2f}"
                )


main()
