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


_REGIME_SHORT = {
    "TRENDING_UP": ("Uptrend", "Buyers are in control"),
    "TRENDING_DOWN": ("Downtrend", "Sellers are in control"),
    "RANGING": ("Range", "Stuck between support & resistance"),
    "VOLATILE": ("Volatile", "Choppy — no clear direction"),
}


def _regime_display(regime_value: str) -> tuple[str, str]:
    return _REGIME_SHORT.get(str(regime_value).upper(), (str(regime_value).title(), ""))


def _render_metric_chips(
    spot: float,
    src: str,
    last_rsi: float,
    bb_pctb: float,
    macd_hist: float,
    regime_value: str,
    adx: float,
) -> None:
    """Mobile-friendly chip grid — wraps on phones, no value truncation."""
    regime_label, _ = _regime_display(regime_value)
    html = (
        "<div class='spy-chips'>"
        f"<div class='spy-chip'><div class='spy-chip-label'>Spot</div>"
        f"<div class='spy-chip-value'>${spot:,.2f}</div>"
        f"<div class='spy-chip-sub'>source: {src}</div></div>"
        f"<div class='spy-chip'><div class='spy-chip-label'>RSI(14)</div>"
        f"<div class='spy-chip-value'>{last_rsi:.0f}</div>"
        f"<div class='spy-chip-sub'>&lt;30 oversold · &gt;70 overbought</div></div>"
        f"<div class='spy-chip'><div class='spy-chip-label'>BB %b</div>"
        f"<div class='spy-chip-value'>{bb_pctb:.2f}</div>"
        f"<div class='spy-chip-sub'>0 = lower · 1 = upper</div></div>"
        f"<div class='spy-chip'><div class='spy-chip-label'>MACD hist</div>"
        f"<div class='spy-chip-value'>{macd_hist:+.3f}</div>"
        f"<div class='spy-chip-sub'>+ = bullish momentum</div></div>"
        f"<div class='spy-chip'><div class='spy-chip-label'>Regime</div>"
        f"<div class='spy-chip-value'>{regime_label}</div>"
        f"<div class='spy-chip-sub'>ADX {adx:.0f}</div></div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _plain_english_guide(
    action: str,
    confidence: float,
    last_rsi: float,
    bb_pctb: float,
    macd_hist: float,
    regime_value: str,
    alert,
    bundle=None,
    fc_report=None,
) -> None:
    """Friendly 'should I buy?' explainer in everyday language."""
    act = str(action).upper()
    regime_label, regime_blurb = _regime_display(regime_value)

    if act in {"STRONG_BUY", "BUY"}:
        verdict = "✅ Lean BUY"
        headline = (
            "The signals are leaning bullish — there's a setup the model "
            "wants to take."
        )
    elif act in {"STRONG_SELL", "SELL"}:
        verdict = "🔻 Lean SELL / short"
        headline = (
            "The signals are leaning bearish — the model sees more downside "
            "than upside here."
        )
    else:
        verdict = "⏸ HOLD — do nothing"
        headline = (
            "No clean setup right now. Sitting out is the trade. "
            "Waiting beats forcing it."
        )

    bullets: list[str] = []

    if last_rsi <= 30:
        bullets.append(f"**RSI {last_rsi:.0f}** — oversold; bounces often start here.")
    elif last_rsi >= 70:
        bullets.append(f"**RSI {last_rsi:.0f}** — overbought; pullback risk is high.")
    else:
        bullets.append(f"**RSI {last_rsi:.0f}** — neutral, no extreme to fade.")

    if bb_pctb <= 0.1:
        bullets.append(f"**BB %b {bb_pctb:.2f}** — pressed against the lower band (cheap-ish).")
    elif bb_pctb >= 0.9:
        bullets.append(f"**BB %b {bb_pctb:.2f}** — pinned to the upper band (extended).")
    else:
        bullets.append(f"**BB %b {bb_pctb:.2f}** — mid-channel, no edge.")

    if macd_hist > 0.01:
        bullets.append(f"**MACD {macd_hist:+.3f}** — momentum is positive.")
    elif macd_hist < -0.01:
        bullets.append(f"**MACD {macd_hist:+.3f}** — momentum is negative.")
    else:
        bullets.append(f"**MACD {macd_hist:+.3f}** — flat momentum.")

    bullets.append(f"**Regime: {regime_label}** — {regime_blurb}.")

    if act not in {"HOLD"}:
        bullets.append(
            f"**Plan:** entry ~${alert.entry:,.2f} · stop ${alert.stop:,.2f} · "
            f"target ${alert.target:,.2f} · risk/reward {alert.rr:.2f}× · "
            f"confidence {confidence:.0f}%."
        )
    else:
        bullets.append(
            "**What to do:** keep this on the watchlist. A clean RSI extreme, "
            "a band tag with MACD flipping, or a 20/50 cross is what wakes it up."
        )

    li = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(
        f"<div class='spy-plain'>"
        f"<h4>🧭 Should I buy? — {verdict}</h4>"
        f"<div style='font-size:0.9rem;'>{headline}</div>"
        f"<ul>{li}</ul>"
        f"<div class='spy-chip-sub' style='margin-top:6px;'>"
        f"This is educational, not advice. Sizing &amp; stops are on you."
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ---- Pattern strip ---------------------------------------------------
    if bundle and bundle.signals:
        consensus_color = {
            "bullish": "#16a34a",
            "bearish": "#dc2626",
            "mixed": "#6b7280",
            "quiet": "#6b7280",
        }[bundle.consensus]
        # Use rgba backgrounds + matching solid borders so chips stay legible
        # against both light and dark themes (WCAG AA on either).
        _chip_palette = {
            "bull":    ("#16a34a", "rgba(22,163,74,0.18)",  "rgba(22,163,74,0.55)"),
            "bear":    ("#dc2626", "rgba(220,38,38,0.18)",  "rgba(220,38,38,0.55)"),
            "neutral": ("#6b7280", "rgba(107,114,128,0.18)","rgba(107,114,128,0.45)"),
        }
        chips = []
        for s in bundle.top:
            text_col, bg_col, border_col = _chip_palette[s.direction]
            chips.append(
                f"<span class='spy-pattern-chip' title=\"{s.note}\" "
                f"style='color:{text_col};background:{bg_col};"
                f"border:1px solid {border_col};'>"
                f"{s.emoji} {s.name} · {s.bias_pp:+.1f}pp</span>"
            )
        st.markdown(
            f"<div class='spy-pattern-row'>"
            f"🧠 <strong>Patterns active</strong> · "
            f"<span style='color:{consensus_color};font-weight:800;'>"
            f"{bundle.consensus.upper()}</span> "
            f"<span class='spy-meta'>(net {bundle.net_bias_pp:+.1f}pp tilt to up-side)</span>"
            f"<div class='spy-pattern-chips'>{''.join(chips)}</div>"
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


_FACTOR_TIPS = {
    "RSI": "Relative Strength Index. + = oversold bounce setup, − = overbought fade.",
    "MACD": "Trend-momentum cross. + = bullish momentum, − = bearish.",
    "BB %b": "Position within Bollinger bands. + = near upper band, − = near lower.",
    "Bollinger": "Position vs Bollinger bands. + = upper band, − = lower band.",
    "Trend": "Slope of the moving averages. + = up-trend, − = down-trend.",
    "Regime": "Macro regime tilt. + = risk-on, − = risk-off.",
    "Volume": "Volume vs typical. + = unusual buying, − = unusual selling.",
    "Momentum": "Rate-of-change. + = accelerating, − = decelerating.",
    "ROC": "Rate-of-change. + = accelerating, − = decelerating.",
}


def _render_contributions(contribs: list[dict]) -> None:
    if not contribs:
        return
    pills = []
    for c in contribs:
        name = str(c.get("name") or "?")
        score = float(c.get("score", 0.0))
        sign = "+" if score >= 0 else ""
        if score > 0.05:
            color = "#16a34a"
        elif score < -0.05:
            color = "#dc2626"
        else:
            color = "#6b7280"
        tip = _FACTOR_TIPS.get(name, f"{name} factor score (range −1 to +1).")
        pills.append(
            f"<div class='spy-factor-pill' title=\"{tip}\">"
            f"<div class='label'>{name}</div>"
            f"<div class='value' style='color:{color};'>{sign}{score:.2f}</div>"
            f"</div>"
        )
    st.markdown(
        f"<div class='spy-factor-card'>"
        f"<div class='spy-factor-head'>📊 Factor breakdown</div>"
        f"<div class='spy-factor-help'>"
        f"Each factor scores −1 to +1. Green = bullish push, red = bearish push, "
        f"grey = neutral. Tap a tile for what it measures."
        f"</div>"
        f"<div class='spy-factor-grid'>{''.join(pills)}</div>"
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

    bb_pctb = float(bb_row["bb_pctb"])
    _render_metric_chips(
        spot=spot,
        src=src,
        last_rsi=last_rsi,
        bb_pctb=bb_pctb,
        macd_hist=macd_hist,
        regime_value=regime.regime.value,
        adx=regime.adx,
    )

    try:
        bundle = detect_patterns(close, spot)
    except Exception:
        bundle = None

    try:
        projections = standard_horizons(close, spot, timeframe)
    except Exception:
        projections = []
    try:
        forecast_calibration.settle_pending(sym, close)
        if projections:
            forecast_calibration.snapshot_projections(sym, projections)
        fc_report = forecast_calibration.report(sym)
    except Exception:
        fc_report = None

    _plain_english_guide(
        action=alert.action.value,
        confidence=float(alert.confidence or 0),
        last_rsi=last_rsi,
        bb_pctb=bb_pctb,
        macd_hist=macd_hist,
        regime_value=regime.regime.value,
        alert=alert,
        bundle=bundle,
        fc_report=fc_report,
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
        # Bright accent for the heading; darker translucent background +
        # high-contrast body text so the box is readable on both light and
        # dark themes.
        if is_buy:
            accent = "#22c55e"
            bg = "rgba(34,197,94,0.12)"
            border = "rgba(34,197,94,0.55)"
        else:
            accent = "#ef4444"
            bg = "rgba(239,68,68,0.12)"
            border = "rgba(239,68,68,0.55)"
        body_color = "#e2e8f0"   # light grey — readable on both themes
        st.markdown(
            f"<div style='padding:12px 16px;border-radius:10px;background:{bg};"
            f"border:1px solid {border};border-left:4px solid {accent};margin:6px 0;'>"
            f"<strong style='color:{accent};font-size:1.05rem;'>"
            f"{'🚀' if is_buy else '🔻'} {direction} setup · "
            f"{alert.action.value.replace('_', ' ')}"
            f"</strong><br/>"
            f"<span style='color:{body_color};'>"
            f"Entry <strong style='color:#fff;'>${alert.entry:,.2f}</strong> · "
            f"Stop <strong style='color:#fff;'>${alert.stop:,.2f}</strong> · "
            f"Target <strong style='color:#fff;'>${alert.target:,.2f}</strong> · "
            f"R:R <strong style='color:#fff;'>{alert.rr:.2f}</strong>"
            f"</span></div>",
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
        index=1,  # default 5m
        help="1m = highest resolution (yfinance limit: last 7 days only).",
    )
    refresh_secs = st.sidebar.select_slider(
        "Auto-refresh (streaming)",
        options=[5, 15, 30, 60, 120, 300],
        value=5,
        help="Lower = more stream-like. 5s pulls candles & live prices "
        "almost continuously.",
    )
    st.sidebar.markdown(
        f"<span class='spy-stream'>● live · streaming every {refresh_secs}s "
        f"({timeframe} candles)</span>",
        unsafe_allow_html=True,
    )
    st_autorefresh(interval=refresh_secs * 1000, key="watchlist_refresh")
    news_enabled = st.sidebar.checkbox(
        "News confirmation (Perplexity)",
        value=True,
        help="Calls Perplexity for a 48h news brief per symbol. "
        "Falls back gracefully when PERPLEXITY_API_KEY is missing.",
    )
    symbols = crypto + stocks

    st.markdown(
        f"<span class='spy-stream'>● live · streaming every {refresh_secs}s "
        f"({timeframe} candles)</span>",
        unsafe_allow_html=True,
    )
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
