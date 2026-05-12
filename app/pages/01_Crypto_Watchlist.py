"""Watchlist dashboard — always shows live data per symbol so you can
verify the pipeline is flowing, even when no actionable pattern is firing.

Each card renders:
  - Spot price + freshness badge (updated Xs/m/h ago)
  - RSI(14), Bollinger %b, MACD histogram, regime + ADX
  - A sparkline of the last ~120 closes
  - The dip/pump detector's current verdict (HOLD shown explicitly)
  - Composite signals (SMA cross / RSI / VWAP rev) — n/a is shown when
    there isn't enough history rather than swallowing the symbol
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app._shared import (
    candles_short,
    live_price,
    setup_page,
    sidebar_watchlists,
)
from app._ui import (
    action_pill,
    freshness_pill,
    inject_global_css,
    loading,
    status_pill,
)
from app.signals import rsi as rsi_signal
from app.signals import sma_crossover, vwap_reversion
from monte.indicators.regime import classify_regime
from monte.indicators.technical import bollinger, macd, rsi
from monte.signals.dip_pump import detect
from monte.strategies.signals import action_from_score


def _sparkline(closes) -> go.Figure:
    fig = go.Figure(go.Scatter(x=closes.index, y=closes.values, mode="lines"))
    fig.update_layout(
        height=110,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_traces(line=dict(width=2))
    return fig


def _render_card(sym: str, timeframe: str) -> None:
    placeholder = st.empty()
    placeholder.markdown(
        f"<div class='spy-meta'>Loading {sym}…</div>",
        unsafe_allow_html=True,
    )

    try:
        with loading(f"Fetching {sym} ({timeframe}) candles…"):
            df = candles_short(sym, timeframe)
    except Exception as e:
        placeholder.empty()
        st.markdown(
            f"### {sym} {status_pill('data error', 'err')}",
            unsafe_allow_html=True,
        )
        st.caption(f"`{e}`")
        return

    if df.empty or "Close" not in df.columns:
        placeholder.empty()
        st.markdown(
            f"### {sym} {status_pill('no candles', 'warn')}",
            unsafe_allow_html=True,
        )
        st.caption("Provider returned no rows. Try a different timeframe.")
        return

    close = df["Close"]
    last_close = float(close.iloc[-1])
    last_ts = close.index[-1] if len(close.index) else None

    try:
        spot, src = live_price(sym)
        if not spot:
            spot, src = last_close, "candle"
    except Exception:
        spot, src = last_close, "candle"

    last_rsi = float(rsi(close).iloc[-1])
    bb_row = bollinger(close).iloc[-1]
    macd_hist = float(macd(close)["hist"].iloc[-1])
    regime = classify_regime(df)

    try:
        alert = detect(df, symbol=sym, timeframe=timeframe)
        action_label = alert.action.value
        confidence = alert.confidence
        entry, stop, target, rr = alert.entry, alert.stop, alert.target, alert.rr
        detector_ok = True
    except Exception as e:
        action_label = "HOLD"
        confidence = 0.0
        entry, stop, target, rr = last_close, last_close, last_close, 0.0
        detector_ok = False
        st.caption(f"detector fallback: {e}")

    placeholder.empty()

    header = (
        f"<div class='spy-card-header'>"
        f"<h3>{sym}<span class='spy-meta' style='margin-left:8px'>{timeframe}</span></h3>"
        f"<div>{action_pill(action_label, confidence)} "
        f"{freshness_pill(last_ts)}</div>"
        f"</div>"
    )
    st.markdown(header, unsafe_allow_html=True)

    m = st.columns(5)
    m[0].metric("Spot", f"${spot:,.2f}", help=f"source: {src}")
    m[1].metric("RSI(14)", f"{last_rsi:.0f}", help="<30 oversold · >70 overbought")
    m[2].metric("BB %b", f"{float(bb_row['bb_pctb']):.2f}", help="0=lower band, 1=upper")
    m[3].metric(
        "MACD hist",
        f"{macd_hist:+.3f}",
        help="positive = bullish momentum",
    )
    m[4].metric("Regime", regime.regime.value, f"ADX {regime.adx:.0f}")

    st.plotly_chart(_sparkline(close.tail(120)), use_container_width=True)

    if detector_ok and action_label != "HOLD":
        st.caption(
            f"Entry **${entry:,.2f}** · Stop **${stop:,.2f}** · "
            f"Target **${target:,.2f}** · R:R **{rr:.2f}**"
        )
    else:
        st.markdown(
            f"{status_pill('no actionable pattern · data flowing ok', 'muted')}",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='spy-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Composite signals**")
    volume = df["Volume"] if "Volume" in df.columns else None
    runners = [
        ("SMA cross", lambda: sma_crossover(close, timeframe=timeframe)),
        ("RSI", lambda: rsi_signal(close, timeframe=timeframe)),
        (
            "VWAP rev",
            lambda: (
                vwap_reversion(close, volume, timeframe=timeframe)
                if volume is not None
                else None
            ),
        ),
    ]
    sig_cols = st.columns(3)
    for sig_col, (label, runner) in zip(sig_cols, runners):
        with sig_col:
            try:
                sig = runner()
            except ValueError as e:
                st.markdown(
                    f"**{label}** {status_pill('n/a', 'muted')}",
                    unsafe_allow_html=True,
                )
                st.caption(str(e))
                continue
            if sig is None:
                st.markdown(
                    f"**{label}** {status_pill('no volume', 'muted')}",
                    unsafe_allow_html=True,
                )
                continue
            sig_action = action_from_score(sig.score)
            st.markdown(
                f"**{sig.name}** {action_pill(sig_action.value)} "
                f"<span class='spy-meta'>score {sig.score:+.2f}</span>",
                unsafe_allow_html=True,
            )
            st.caption(sig.rationale)


def main() -> None:
    setup_page("Crypto + SPY Watchlist", icon="📊")
    inject_global_css()

    crypto, stocks = sidebar_watchlists()
    timeframe = st.sidebar.selectbox("Timeframe", ["1h", "15m", "5m", "1d"], index=0)
    symbols = crypto + stocks

    st.caption(
        "Live indicator dashboard. Every symbol shows its data even when no "
        "pattern is actionable — that way you can verify the pipeline is "
        "flowing. **HOLD** = nothing to do right now, not a failure."
    )

    if not symbols:
        st.warning("Add at least one symbol in the sidebar.")
        return

    cols = st.columns(2)
    for i, sym in enumerate(symbols):
        with cols[i % 2]:
            with st.container(border=True):
                _render_card(sym, timeframe)


main()
