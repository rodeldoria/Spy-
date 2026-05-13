"""Watchlist dashboard — surfaces a clear, actionable signal per symbol.

Each card now renders:
  - Spot price + freshness badge
  - The triangulated detector verdict as a big BUY / SELL / HOLD pill
  - A "horizon" badge: DAY TRADE / SWING / LONG HOLD
  - RSI, %b, MACD histogram, regime + ADX with a sparkline of ~2 weeks
  - Per-factor contributions (RSI/MACD/BB/Trend/Regime) with their score
  - Optional Perplexity news brief that confirms or conflicts with the call
  - Pattern-journal lookup: "X similar setups before → Y% wins, avg +Z%"
  - One-click "Log paper entry" / "Close open entries" so successful trades
    can be remembered for the next confirmation pass
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
from monte import journal
from monte.indicators.regime import classify_regime
from monte.indicators.technical import bollinger, macd, rsi
from monte.intel import perplexity
from monte.signals.dip_pump import detect
from monte.signals.horizon import HORIZON_HOLD_HINT, HORIZON_LABEL
from monte.strategies.signals import action_from_score


def _sparkline(closes) -> go.Figure:
    fig = go.Figure(go.Scatter(x=closes.index, y=closes.values, mode="lines"))
    fig.update_layout(
        height=120,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_traces(line=dict(width=2))
    return fig


def _horizon_pill(horizon_value: str) -> str:
    label = HORIZON_LABEL.get(horizon_value, horizon_value)
    palette = {
        "DAY_TRADE": ("#1d4ed8", "#e8efff"),
        "SWING":     ("#7c3aed", "#f0e8ff"),
        "LONG_HOLD": ("#0a7d2a", "#e8f7ec"),
    }
    fg, bg = palette.get(horizon_value, ("#5b6470", "#f1f3f5"))
    return (
        f"<span class='spy-pill' style='color:{fg};background:{bg};'>"
        f"⏱ {label}</span>"
    )


def _render_contributions(contribs: list[dict]) -> None:
    if not contribs:
        return
    cols = st.columns(len(contribs))
    for col, c in zip(cols, contribs):
        score = float(c.get("score", 0.0))
        sign = "+" if score >= 0 else ""
        col.markdown(
            f"<div style='text-align:center;'>"
            f"<div class='spy-meta'>{c.get('name')}</div>"
            f"<div style='font-weight:700;color:{'#0a7d2a' if score>0 else ('#a8261f' if score<0 else '#5b6470')};'>"
            f"{sign}{score:.2f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_news(sym: str, action_label: str, news_enabled: bool) -> None:
    if not news_enabled:
        return
    brief = perplexity.fetch_news(sym, action=action_label)
    if not brief.configured:
        st.caption(f"🔎 News: {brief.summary}")
        return
    alignment = brief.aligns_with(action_label)
    kind = {"confirms": "ok", "conflicts": "err", "neutral": "muted"}[alignment]
    badge = status_pill(f"news {alignment} · {brief.sentiment}", kind)
    st.markdown(f"🔎 **News check** {badge}", unsafe_allow_html=True)
    if brief.summary:
        st.caption(brief.summary)
    if brief.headlines:
        st.markdown("\n".join(f"- {h}" for h in brief.headlines))
    if brief.catalysts:
        st.caption("Watch for: " + " · ".join(brief.catalysts))


def _render_journal(sym: str, action_label: str, snapshot: dict[str, float]) -> None:
    history = journal.similar_history(
        symbol=sym, action=action_label, snapshot=snapshot, k=5
    )
    if history.count == 0:
        st.caption(
            "🧠 Journal: no similar past setups yet. Log entries on this card "
            "to teach future confirmation."
        )
        return
    win_kind = "ok" if history.win_rate >= 60 else ("warn" if history.win_rate >= 40 else "err")
    st.markdown(
        "🧠 **Journal** "
        + status_pill(
            f"{history.count} similar · {history.win_rate:.0f}% wins · avg {history.avg_pnl_pct:+.2f}%",
            win_kind,
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Best {history.best_pnl_pct:+.2f}% · worst {history.worst_pnl_pct:+.2f}% "
        "across nearest neighbours by indicator distance."
    )


def _render_journal_controls(
    sym: str,
    timeframe: str,
    alert,
) -> None:
    open_for_sym = journal.open_entries(symbol=sym)
    cols = st.columns([1, 1, 2])
    if alert.action.value not in {"HOLD"} and cols[0].button(
        "Log paper entry", key=f"log-{sym}-{timeframe}"
    ):
        e = journal.record_entry(
            symbol=sym,
            timeframe=timeframe,
            action=alert.action.value,
            horizon=alert.horizon.value,
            entry=alert.entry,
            stop=alert.stop,
            target=alert.target,
            confidence=alert.confidence,
            score=alert.score,
            snapshot=alert.indicator_snapshot,
            note=f"auto: {alert.horizon_rationale}",
        )
        st.success(f"Logged entry {e.id} at ${e.entry:,.2f}")

    if open_for_sym and cols[1].button(
        f"Close {len(open_for_sym)} open", key=f"close-{sym}-{timeframe}"
    ):
        spot, _ = live_price(sym)
        for e in open_for_sym:
            journal.record_exit(e.id, exit_price=spot or alert.entry, exit_reason="manual-close")
        st.success(f"Closed {len(open_for_sym)} open {sym} entries at ${spot:,.2f}")

    if open_for_sym:
        cols[2].caption(
            f"Open: " + " · ".join(
                f"{e.action} ${e.entry:,.2f}" for e in open_for_sym[:3]
            )
        )


def _render_card(sym: str, timeframe: str, news_enabled: bool) -> None:
    placeholder = st.empty()
    placeholder.markdown(
        f"<div class='spy-meta'>Loading {sym}…</div>", unsafe_allow_html=True
    )

    try:
        with loading(f"Fetching {sym} ({timeframe}) candles · 2-week window…"):
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

    alert = detect(df, symbol=sym, timeframe=timeframe)

    placeholder.empty()

    header = (
        f"<div class='spy-card-header'>"
        f"<h3>{sym}<span class='spy-meta' style='margin-left:8px'>{timeframe}</span></h3>"
        f"<div>{action_pill(alert.action.value, alert.confidence)} "
        f"{_horizon_pill(alert.horizon.value)} "
        f"{freshness_pill(last_ts)}</div>"
        f"</div>"
    )
    st.markdown(header, unsafe_allow_html=True)
    st.caption(
        f"{alert.horizon_rationale} — "
        f"{HORIZON_HOLD_HINT.get(alert.horizon, '')}"
    )

    m = st.columns(5)
    m[0].metric("Spot", f"${spot:,.2f}", help=f"source: {src}")
    m[1].metric("RSI(14)", f"{last_rsi:.0f}", help="<30 oversold · >70 overbought")
    m[2].metric("BB %b", f"{float(bb_row['bb_pctb']):.2f}", help="0=lower, 1=upper")
    m[3].metric("MACD hist", f"{macd_hist:+.3f}", help="positive = bullish momentum")
    m[4].metric("Regime", regime.regime.value, f"ADX {regime.adx:.0f}")

    st.plotly_chart(_sparkline(close.tail(400)), use_container_width=True)

    if alert.action.value != "HOLD":
        direction = "Long" if alert.score > 0 else "Short"
        st.success(
            f"**{direction} setup · {alert.action.value.replace('_', ' ')}** — "
            f"Entry **${alert.entry:,.2f}** · Stop **${alert.stop:,.2f}** · "
            f"Target **${alert.target:,.2f}** · R:R **{alert.rr:.2f}**"
        )
    else:
        st.markdown(
            status_pill("no actionable pattern · data flowing ok", "muted"),
            unsafe_allow_html=True,
        )

    st.markdown("<div class='spy-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Factor breakdown**")
    _render_contributions(alert.contributions)

    st.markdown("<div class='spy-divider'></div>", unsafe_allow_html=True)
    _render_news(sym, alert.action.value, news_enabled)
    _render_journal(sym, alert.action.value, alert.indicator_snapshot)
    _render_journal_controls(sym, timeframe, alert)

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
    timeframe = st.sidebar.selectbox(
        "Timeframe", ["1h", "15m", "5m", "1d"], index=0,
        help="Pulls ~2 weeks of bars at the selected timeframe.",
    )
    news_enabled = st.sidebar.checkbox(
        "News confirmation (Perplexity)",
        value=True,
        help="Calls Perplexity for a 48h news brief per symbol. "
        "Falls back gracefully when PERPLEXITY_API_KEY is missing.",
    )
    symbols = crypto + stocks

    st.caption(
        "Live indicator dashboard. Every card shows a clear BUY / SELL / HOLD "
        "with a trade horizon (Day Trade / Swing / Long-Term Hold). "
        "**HOLD** = nothing to do right now, not a failure."
    )

    stats = journal.summary()
    if stats["closed"]:
        st.markdown(
            status_pill(
                f"journal · {stats['closed']} closed trades · "
                f"{stats['win_rate']:.0f}% wins · avg {stats['avg_pnl_pct']:+.2f}%",
                "ok" if stats["win_rate"] >= 50 else "muted",
            ),
            unsafe_allow_html=True,
        )

    if not symbols:
        st.warning("Add at least one symbol in the sidebar.")
        return

    cols = st.columns(2)
    for i, sym in enumerate(symbols):
        with cols[i % 2]:
            with st.container(border=True):
                _render_card(sym, timeframe, news_enabled)


main()
