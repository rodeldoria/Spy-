"""Watchlist dashboard — surfaces a clear, actionable signal per symbol.

Each card renders:
  - Spot price + freshness badge
  - Pulsing BUY NOW / SELL NOW banner when Monte Edge says ACT_NOW
  - Triangulated verdict as a big BUY / SELL / HOLD pill
  - Horizon badge: DAY TRADE / SWING / LONG HOLD
  - RSI, %b, MACD histogram, regime + ADX with a sparkline
  - Per-factor contributions (RSI/MACD/BB/Trend/Regime) with score
  - Perplexity news brief that confirms or conflicts with the call
  - Option plays for stocks/ETFs; futures guidance for crypto
  - Pattern-journal lookup and one-click paper entry logging
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app._chart import build_chart
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
    signal_banner,
    status_pill,
    tier_pill,
)
from app.signals import rsi as rsi_signal
from app.signals import sma_crossover, vwap_reversion
from app._shared import pattern_store
from monte import journal
from monte.indicators.ma_cross import detect_cross
from monte.indicators.regime import classify_regime
from monte.indicators.technical import bollinger, macd, rsi
from monte.intel import perplexity
from monte.options import suggest_contract as suggest_option
from monte.patterns.match import find_similar
from monte.signals.dip_pump import detect
from monte.signals.forecast import standard_horizons
from monte.signals.patterns import detect_patterns
from monte.learning import forecast_calibration
from monte.signals.horizon import HORIZON_HOLD_HINT, HORIZON_LABEL
from monte.strategies.signals import action_from_score
from monte.strategy.monte_edge import EdgeTier, evaluate as edge_evaluate


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


def _render_forecast_grid(close, spot: float, timeframe: str, sym: str) -> None:
    """Show a forecast grid: predicted price/range at 15m, 1h, 3h, key times.

    Useful for sizing Kalshi calls and short-term option entries — every
    card answers "where is this likely to be at the top of the next hour?"
    based on recent realised volatility (NOT a guarantee, just a calibrated
    range).
    """
    try:
        projections = standard_horizons(close, spot, timeframe)
    except Exception as e:
        st.caption(f"⚠️ Forecast unavailable: {e}")
        return
    if not projections:
        return

    # ---- Pattern engine (pro-investor frameworks) ------------------------
    try:
        bundle = detect_patterns(close, spot)
    except Exception:
        bundle = None

    # ---- Forecast accuracy learning loop ---------------------------------
    # First: settle anything pending whose horizon time has passed, using
    # the close series we already have in memory. Then snapshot today's
    # projections so they can be settled later.
    try:
        forecast_calibration.settle_pending(sym, close)
        forecast_calibration.snapshot_projections(sym, projections)
        fc_report = forecast_calibration.report(sym)
    except Exception:
        fc_report = None

    cells = []
    for p in projections:
        if p.drift_pct > 0.05:
            color = "#0a7d2a"
            arrow = "▲"
        elif p.drift_pct < -0.05:
            color = "#a8261f"
            arrow = "▼"
        else:
            color = "#6b7280"
            arrow = "→"

        cells.append(
            f"<div class='spy-fc-cell'>"
            f"<div class='spy-fc-label'>{p.label}</div>"
            f"<div class='spy-fc-time'>by {p.label_pst()} · {p.label_utc()}</div>"
            f"<div class='spy-fc-price' style='color:{color};'>{arrow} ${p.median:,.2f}</div>"
            f"<div class='spy-fc-delta' style='color:{color};'>{p.drift_pct:+.2f}%</div>"
            f"<div class='spy-fc-range'>±${(p.upper - p.median):,.2f} ({p.range_pct:.2f}%)</div>"
            f"<div class='spy-fc-band'>${p.lower:,.2f} → ${p.upper:,.2f}</div>"
            f"</div>"
        )

    st.markdown(
        "<div class='spy-meta' style='margin:8px 0 4px 0;'>"
        f"📈 <strong>Forecast grid</strong> — projected price &amp; 1σ band at each horizon "
        "<span style='color:#888;'>(based on recent volatility — for Kalshi/options sizing, not a guarantee)</span>"
        "<br/><span style='color:#888;font-size:0.78rem;'>"
        "Legend: <strong>▲</strong> projected up · <strong>▼</strong> projected down · "
        "<strong>→</strong> projected flat. The bottom row "
        "<code>$lo → $hi</code> is the 1σ band — about 68% of the time the "
        "actual price should land inside this range."
        "</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='spy-fc-grid'>{''.join(cells)}</div>",
        unsafe_allow_html=True,
    )

    # ---- Pattern strip ---------------------------------------------------
    if bundle and bundle.signals:
        consensus_color = {
            "bullish": "#0a7d2a",
            "bearish": "#a8261f",
            "mixed": "#6b7280",
            "quiet": "#6b7280",
        }[bundle.consensus]
        chips = []
        for s in bundle.top:
            chip_color = {"bull": "#0a7d2a", "bear": "#a8261f", "neutral": "#6b7280"}[s.direction]
            chips.append(
                f"<span title=\"{s.note}\" "
                f"style='display:inline-block;padding:3px 8px;margin:2px 4px 2px 0;"
                f"border-radius:10px;background:rgba(107,114,128,0.10);"
                f"font-size:0.78rem;color:{chip_color};font-weight:600;'>"
                f"{s.emoji} {s.name} · {s.bias_pp:+.1f}pp</span>"
            )
        st.markdown(
            f"<div style='margin:6px 0 4px 0;font-size:0.82rem;'>"
            f"🧠 <strong>Patterns active</strong> · "
            f"<span style='color:{consensus_color};font-weight:700;'>"
            f"{bundle.consensus.upper()}</span> "
            f"<span style='color:#888;'>(net {bundle.net_bias_pp:+.1f}pp tilt to up-side)</span><br/>"
            f"{''.join(chips)}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ---- Forecast accuracy panel ----------------------------------------
    if fc_report and fc_report.n_settled > 0:
        with st.expander(
            f"🎯 Forecast accuracy ({fc_report.n_settled} past predictions verified)",
            expanded=False,
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Inside ±1σ band",
                f"{(fc_report.hit_rate_within_band or 0)*100:.0f}%",
                help="Honest: a calibrated 1σ band should hit ~68% of the time.",
            )
            c2.metric(
                "Direction accuracy",
                f"{(fc_report.direction_accuracy or 0)*100:.0f}%"
                if fc_report.direction_accuracy is not None else "—",
                help="When forecast said up, did it actually go up? (50% = coin flip)",
            )
            c3.metric(
                "Mean abs error",
                f"{fc_report.mean_abs_error_pct:.2f}%"
                if fc_report.mean_abs_error_pct is not None else "—",
            )
            c4.metric(
                "Bias",
                f"{fc_report.mean_bias_pct:+.2f}%"
                if fc_report.mean_bias_pct is not None else "—",
                help="Positive = forecast tends to overshoot, negative = undershoot.",
            )
            if fc_report.by_label:
                st.caption("Per-horizon breakdown:")
                rows = []
                for lbl, d in sorted(fc_report.by_label.items()):
                    band = f"{(d['hit_band'] or 0)*100:.0f}%" if d.get("hit_band") is not None else "—"
                    mae = f"{d['mae_pct']:.2f}%" if d.get("mae_pct") is not None else "—"
                    dac = f"{(d['dir_acc'] or 0)*100:.0f}%" if d.get("dir_acc") is not None else "—"
                    rows.append(f"- **{lbl}** ({d['n']} settled) — in band {band}, dir {dac}, MAE {mae}")
                st.markdown("\n".join(rows))
            st.caption(
                "Predictions self-verify: each render checks past forecasts against "
                "the actual close at that minute, then logs the outcome. Numbers grow "
                "more reliable as the sample size builds."
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
    if brief.error:
        st.caption(f"⚠️ {brief.summary}")
        return
    if brief.summary:
        st.caption(brief.summary)
    if brief.headlines:
        st.markdown("\n".join(f"- {h}" for h in brief.headlines))
    if brief.catalysts:
        st.caption("Watch for: " + " · ".join(brief.catalysts))


def _render_options(sym: str, alert, edge, timeframe: str) -> None:
    """Show option / futures play for any symbol when signal is actionable."""
    if alert.action.value == "HOLD":
        return
    if edge is not None and edge.tier not in {EdgeTier.ACT_NOW, EdgeTier.WATCH}:
        return

    direction_key = "long" if alert.score > 0 else "short"
    try:
        ticket = suggest_option(direction_key, alert.entry, symbol=sym)
    except Exception:
        ticket = None

    if ticket is None:
        return

    horizon = str(alert.horizon.value) if hasattr(alert.horizon, "value") else str(alert.horizon)
    horizon_label = HORIZON_LABEL.get(horizon, horizon.replace("_", " ").title())

    st.markdown("---")
    st.markdown(f"### 📈 Suggested Play · {horizon_label}")

    if ticket.get("is_crypto_note"):
        st.info(ticket.get("rationale", ""))
    else:
        cols = st.columns(4)
        cols[0].metric("Side", ticket["side"])
        cols[1].metric("Strike", f"${ticket['strike']:.0f}")
        cols[2].metric("Expiry", ticket["expiry"])
        cols[3].metric("Premium", f"${ticket['premium']:.2f}")

        c2 = st.columns(3)
        c2[0].metric("Breakeven", f"${ticket['breakeven']:.2f}")
        c2[1].metric("Max Risk/contract", f"${ticket['max_risk_per_contract']:.0f}")
        c2[2].metric("Est. Delta", f"{ticket['est_delta']:.2f}")

        if ticket.get("iv"):
            st.caption(f"Implied volatility: {ticket['iv']*100:.1f}%")
        st.caption(ticket.get("rationale", ""))


def _render_journal(
    sym: str,
    action_label: str,
    snapshot: dict[str, float],
    df,
    timeframe: str,
) -> None:
    history = journal.similar_history(
        symbol=sym, action=action_label, snapshot=snapshot, k=5
    )
    if history.count == 0:
        st.caption(
            "🧠 Journal: no similar past setups yet. Log entries on this card "
            "to teach future confirmation."
        )
    else:
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

    try:
        match = find_similar(sym, timeframe, df, k=20, store=pattern_store())
    except Exception as e:
        st.caption(f"📚 Pattern library unavailable: {e}")
        return
    if match.cold_start:
        st.caption(
            "📚 Pattern library cold-start — ingest history with "
            "`python -m monte.patterns.ingest <symbol> <interval> --years 1`."
        )
        return
    lib_win_pct = match.win_rate * 100
    lib_kind = "ok" if lib_win_pct >= 55 else ("warn" if lib_win_pct >= 45 else "err")
    st.markdown(
        "📚 **Pattern library** "
        + status_pill(
            f"K={match.k} similar · {lib_win_pct:.0f}% wins · mean fwd-20 {match.mean_fwd_20*100:+.2f}%",
            lib_kind,
        ),
        unsafe_allow_html=True,
    )

    if history.count and abs(history.win_rate - lib_win_pct) > 25:
        st.markdown(
            status_pill("mixed evidence — journal & pattern library disagree", "warn"),
            unsafe_allow_html=True,
        )


def _render_journal_controls(sym: str, timeframe: str, alert) -> None:
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
            "Open: " + " · ".join(
                f"{e.action} ${e.entry:,.2f}" for e in open_for_sym[:3]
            )
        )


def _render_card(sym: str, timeframe: str, news_enabled: bool) -> None:
    placeholder = st.empty()
    placeholder.markdown(
        f"<div class='spy-meta'>Loading {sym}…</div>", unsafe_allow_html=True
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

    alert = detect(df, symbol=sym, timeframe=timeframe)

    spy_df = None
    if sym.upper() != "SPY":
        try:
            from monte.data import prices as _prices
            spy_df = _prices.get_daily("SPY", period="2y")
        except Exception:
            spy_df = None
    else:
        spy_df = df if timeframe == "1d" else None
    try:
        edge = edge_evaluate(df, symbol=sym, timeframe=timeframe, spy_daily=spy_df)
    except Exception:
        edge = None

    placeholder.empty()

    # ── Pulsing ACT NOW banner — impossible to miss ──
    if edge is not None and edge.tier is EdgeTier.ACT_NOW:
        direction_key = "long" if alert.score > 0 else "short"
        ticket = None
        try:
            ticket = suggest_option(direction_key, alert.entry, symbol=sym)
        except Exception:
            pass
        signal_banner({
            "symbol": sym,
            "action": alert.action.value,
            "tier": edge.tier.value,
            "confidence": edge.confidence,
            "spot": spot,
            "stop": alert.stop,
            "target": alert.target,
            "rr": alert.rr,
            "horizon": alert.horizon.value,
            "reasoning": edge.reasoning,
            "options_ticket": ticket,
        })

    tier_block = ""
    if edge is not None:
        tier_block = tier_pill(edge.tier.value, edge.confidence) + " "
    header = (
        f"<div class='spy-card-header'>"
        f"<h3>{sym}<span class='spy-meta' style='margin-left:8px'>{timeframe}</span></h3>"
        f"<div>{tier_block}{action_pill(alert.action.value, alert.confidence)} "
        f"{_horizon_pill(alert.horizon.value)} "
        f"{freshness_pill(last_ts)}</div>"
        f"</div>"
    )
    st.markdown(header, unsafe_allow_html=True)
    st.caption(
        f"{alert.horizon_rationale} — "
        f"{HORIZON_HOLD_HINT.get(alert.horizon, '')}"
    )
    if edge is not None and edge.reasoning and edge.tier is not EdgeTier.ACT_NOW:
        st.info(f"💡 **Why this works:** {edge.reasoning}")
    if edge is not None and edge.macro_note:
        st.caption(f"📊 Macro: {edge.macro_note} · confluence {edge.confluence}/5")

    cross = detect_cross(close, fast=20, slow=50)
    if cross.fired_recently:
        kind = "ok" if cross.kind == "golden" else "err"
        st.markdown(status_pill(cross.label(), kind), unsafe_allow_html=True)

    _render_forecast_grid(close, spot, timeframe, sym)

    import html as _html

    regime_short = regime.regime.value.replace("TRENDING_", "TREND ").replace("_", " ")
    if regime.adx >= 25:
        adx_color = "#0a7d2a"
        adx_arrow = "▲"
    elif regime.adx <= 15:
        adx_color = "#a8261f"
        adx_arrow = "▼"
    else:
        adx_color = "#6b7280"
        adx_arrow = "→"
    adx_sub = (
        f"<span style='color:{adx_color};font-weight:600;'>"
        f"{adx_arrow} ADX {regime.adx:.0f}</span>"
    )

    metrics_cells = [
        ("Spot", f"${spot:,.2f}", f"source: {src}", None),
        ("RSI(14)", f"{last_rsi:.0f}", "<30 oversold · >70 overbought", None),
        ("BB %b", f"{float(bb_row['bb_pctb']):.2f}", "0 = lower · 1 = upper", None),
        ("MACD", f"{macd_hist:+.3f}", "positive = bullish momentum", None),
        ("Regime", regime_short, "trend strength below", adx_sub),
    ]
    metrics_html = "".join(
        f"<div class='spy-metric-cell' title='{_html.escape(tooltip)}'>"
        f"<div class='spy-metric-label'>{_html.escape(label)}</div>"
        f"<div class='spy-metric-value'>{_html.escape(value)}</div>"
        f"<div class='spy-metric-sub'>{sub_html if sub_html else _html.escape(tooltip)}</div>"
        f"</div>"
        for label, value, tooltip, sub_html in metrics_cells
    )
    st.markdown(
        f"<div class='spy-metric-strip'>{metrics_html}</div>",
        unsafe_allow_html=True,
    )

    chart_df = df.tail(400)
    st.plotly_chart(
        build_chart(chart_df, ma_cross=cross, height=560),
        use_container_width=True,
        key=f"chart-{sym}-{timeframe}",
    )

    if alert.action.value != "HOLD":
        is_buy = alert.score > 0
        direction = "Long" if is_buy else "Short"
        color = "#0a7d2a" if is_buy else "#a8261f"
        bg = "#e8f7ec" if is_buy else "#fbe9e7"
        st.markdown(
            f"<div style='padding:12px 16px;border-radius:10px;background:{bg};"
            f"border-left:4px solid {color};margin:6px 0;'>"
            f"<strong style='color:{color};font-size:1.05rem;'>"
            f"{'🚀' if is_buy else '🔻'} {direction} setup · "
            f"{alert.action.value.replace('_', ' ')}"
            f"</strong><br/>"
            f"Entry <strong>${alert.entry:,.2f}</strong> · "
            f"Stop <strong>${alert.stop:,.2f}</strong> · "
            f"Target <strong>${alert.target:,.2f}</strong> · "
            f"R:R <strong>{alert.rr:.2f}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )
        # Option / futures play for all symbols
        _render_options(sym, alert, edge, timeframe)
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
    _render_journal(sym, alert.action.value, alert.indicator_snapshot, df, timeframe)
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

    from streamlit_autorefresh import st_autorefresh

    crypto, stocks = sidebar_watchlists()
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "30m", "1h", "1d"],
        index=2,
        help="1m = highest resolution (yfinance limit: last 7 days only).",
    )
    refresh_secs = st.sidebar.select_slider(
        "Auto-refresh",
        options=[5, 15, 30, 60, 120, 300],
        value=30,
        help="How often the cards re-pull candles and live prices.",
    )
    st_autorefresh(interval=refresh_secs * 1000, key="watchlist_refresh")
    news_enabled = st.sidebar.checkbox(
        "News confirmation (Perplexity)",
        value=True,
        help="Calls Perplexity for a 48h news brief per symbol. "
        "Falls back gracefully when PERPLEXITY_API_KEY is missing.",
    )
    symbols = crypto + stocks

    st.caption(
        "Live indicator dashboard. Cards with a **pulsing green/red banner** = ACT NOW signal. "
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
