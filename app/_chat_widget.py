"""Sidebar chat widget — drop a chart screenshot, get a confirmation gate.

Drives the full event-aware stack:

  1. ChartVision reads the uploaded screenshot (Claude Haiku).
  2. Aggregates news (Perplexity), macro (FRED), economic calendar
     (Forex Factory or TradingEconomics), and on-chain flows (Farside +
     CoinGlass + TokenUnlocks).
  3. Runs the regime + microstructure assessment from the symbol's daily
     and intraday OHLCV.
  4. Scores everything through the likelihood gate.
  5. Surfaces an inline premortem so the user can stress-test the trade
     before clicking "Notify me".

Lives in the sidebar (collapsible) so it doesn't compete with the live
signal feed for the main page real estate.
"""
from __future__ import annotations

import time
from typing import Optional

import streamlit as st

from app._premortem_panel import render_premortem_panel
from app._shared import candles_short, is_crypto
from monte.intel.chart_vision import ChartRead, read_chart
from monte.intel.event_aggregator import EventBundle, IdeaContext, gather
from monte.microstructure import MicrostructureReport, assess as microstructure_assess
from monte.notify.ntfy import push as ntfy_push
from monte.regime import RegimeReport, assess as regime_assess
from monte.signals.likelihood_gate import GateVerdict, score as gate_score

_HORIZON_HOURS = {
    "intraday": 6.0,
    "swing": 168.0,
    "position": 720.0,
    "long": 2160.0,
}


def render_event_chat_widget(*, key_prefix: str = "home") -> None:
    """Render the sidebar chat widget. Idempotent across reruns."""
    state_key = f"event_chat::{key_prefix}"
    state = st.session_state.setdefault(
        state_key,
        {
            "chart_read": None,
            "verdict": None,
            "bundle": None,
            "regime": None,
            "micro": None,
            "ticker_override": "",
            "target_text": "",
            "horizon": "swing",
            "direction": "long",
        },
    )

    with st.sidebar:
        with st.expander("🧭 Event chat — drop a chart, get a verdict", expanded=False):
            st.caption(
                "Upload a TradingView (or any) screenshot. The widget reads it with "
                "Claude vision, pulls the news/macro/calendar/on-chain catalysts in "
                "parallel, and runs a multi-axis confirmation gate before you risk "
                "capital."
            )
            uploaded = st.file_uploader(
                "Chart screenshot",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"{key_prefix}::upload",
                help="Drop a chart image. The vision call is best with a single timeframe visible.",
            )
            ticker_override = st.text_input(
                "Ticker (override / fallback)",
                value=state.get("ticker_override", ""),
                placeholder="e.g. BTC-USD or SPY",
                key=f"{key_prefix}::ticker",
            )
            target_text = st.text_area(
                "Idea (entry / stop / target)",
                value=state.get("target_text", ""),
                placeholder="e.g. Long BTC at 68k, stop 65k, target 75k.",
                height=80,
                key=f"{key_prefix}::target",
            )
            cols = st.columns(2)
            with cols[0]:
                horizon = st.selectbox(
                    "Horizon",
                    list(_HORIZON_HOURS.keys()),
                    index=list(_HORIZON_HOURS.keys()).index(state.get("horizon", "swing")),
                    key=f"{key_prefix}::horizon",
                )
            with cols[1]:
                direction = st.selectbox(
                    "Direction",
                    ["long", "short"],
                    index=0 if state.get("direction", "long") == "long" else 1,
                    key=f"{key_prefix}::direction",
                )
            run = st.button(
                "🔍 Analyze",
                type="primary",
                use_container_width=True,
                disabled=(uploaded is None and not ticker_override.strip() and not target_text.strip()),
                key=f"{key_prefix}::run",
            )

            if run:
                state["ticker_override"] = ticker_override
                state["target_text"] = target_text
                state["horizon"] = horizon
                state["direction"] = direction
                _run_pipeline(state, uploaded)

            chart_read: Optional[ChartRead] = state.get("chart_read")
            verdict: Optional[GateVerdict] = state.get("verdict")
            bundle: Optional[EventBundle] = state.get("bundle")
            regime: Optional[RegimeReport] = state.get("regime")
            micro: Optional[MicrostructureReport] = state.get("micro")

            if chart_read is not None:
                _render_chart_read(chart_read)
            if verdict is not None:
                _render_verdict(verdict)
            if bundle is not None:
                _render_catalysts(bundle)
            if regime is not None or micro is not None:
                _render_regime_micro(regime, micro)

            if verdict is not None:
                idea_summary = state.get("ticker_override") or (chart_read.ticker if chart_read else "?")
                _render_premortem(
                    key_prefix=key_prefix,
                    idea_summary=idea_summary,
                    target_text=state.get("target_text", ""),
                    horizon=state.get("horizon", "swing"),
                    verdict=verdict,
                    bundle=bundle,
                    regime=regime,
                    micro=micro,
                )
                _render_notify_button(key_prefix, idea_summary, verdict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(state: dict, uploaded) -> None:
    chart_read: Optional[ChartRead] = None
    if uploaded is not None:
        with st.spinner("Reading the chart with Claude vision…"):
            try:
                chart_read = read_chart(uploaded.getvalue(), media_type=uploaded.type or "image/png")
            except Exception as e:  # noqa: BLE001
                chart_read = ChartRead(source="heuristic", error=f"{type(e).__name__}: {e}")
    state["chart_read"] = chart_read

    ticker = (state.get("ticker_override") or "").strip().upper()
    if not ticker and chart_read and chart_read.ticker:
        ticker = chart_read.ticker.upper()
    if not ticker:
        st.error("No ticker — set a ticker override or upload a chart that includes the symbol.")
        state["verdict"] = None
        return

    horizon_label = state.get("horizon", "swing")
    horizon_hours = _HORIZON_HOURS.get(horizon_label, 168.0)
    direction = state.get("direction", "long")
    crypto_flag = is_crypto(ticker)

    idea = IdeaContext(
        symbol=ticker,
        direction=direction,
        horizon_hours=horizon_hours,
        is_crypto=crypto_flag,
        note=state.get("target_text", ""),
    )

    with st.spinner("Pulling catalysts (news / FRED / calendar / on-chain) in parallel…"):
        bundle = gather(idea)
    state["bundle"] = bundle

    regime_report = None
    micro_report = None
    try:
        df_daily = candles_short(ticker, "1d")
        if df_daily is not None and not df_daily.empty:
            regime_report = regime_assess(ticker, df_daily, fred_snapshot=bundle.fred)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Daily candles unavailable for regime ({type(e).__name__}: {e})")

    try:
        intraday_interval = "5m" if not crypto_flag else "15m"
        df_intra = candles_short(ticker, intraday_interval)
        if df_intra is not None and not df_intra.empty:
            micro_report = microstructure_assess(
                df_intra, asset_class="crypto" if crypto_flag else "equity"
            )
    except Exception as e:  # noqa: BLE001
        st.warning(f"Intraday candles unavailable for microstructure ({type(e).__name__}: {e})")

    state["regime"] = regime_report
    state["micro"] = micro_report

    verdict = gate_score(
        idea=idea,
        bundle=bundle,
        regime=regime_report,
        microstructure=micro_report,
    )
    state["verdict"] = verdict


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _render_chart_read(read: ChartRead) -> None:
    src_color = "#8b5cf6" if read.source == "ai" else "#94a3b8"
    src_label = f"🤖 vision · {read.model or 'haiku'}" if read.source == "ai" else "🧮 heuristic"
    st.markdown(
        f"<div style='margin-top:8px;padding:8px 10px;background:#0b1220;border-radius:6px;'>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.78rem;'>"
        f"<strong style='color:#e2e8f0;'>Chart read</strong>"
        f"<span style='color:{src_color};'>{src_label}</span></div>"
        f"<div style='color:#cbd5e1;font-size:0.82rem;margin-top:4px;'>"
        f"<strong>{read.ticker or '—'}</strong> · {read.timeframe or '?'} · "
        f"<em>{read.suspected_pattern or 'no clear pattern'}</em></div>"
        f"<div style='color:#94a3b8;font-size:0.78rem;'>{read.setup or '—'}</div>"
        + (
            f"<div style='color:#94a3b8;font-size:0.74rem;margin-top:2px;'>"
            f"levels: {', '.join(f'{x:g}' for x in read.key_levels)}</div>"
            if read.key_levels
            else ""
        )
        + (
            f"<div style='color:#f97316;font-size:0.72rem;margin-top:2px;'>vision error: {read.error}</div>"
            if read.error
            else ""
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_verdict(verdict: GateVerdict) -> None:
    color = {"GO": "#0a7d2a", "CAUTION": "#a16207", "STAND_DOWN": "#a8261f"}.get(verdict.action, "#5b6470")
    bg = {"GO": "#d6f5dc", "CAUTION": "#fef9c3", "STAND_DOWN": "#fbe9e7"}.get(verdict.action, "#f1f3f5")
    icon = {"GO": "🟢", "CAUTION": "🟡", "STAND_DOWN": "🔴"}.get(verdict.action, "⚪")
    st.markdown(
        f"<div style='margin-top:10px;padding:10px 12px;background:{bg};border-radius:8px;"
        f"border-left:6px solid {color};'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<div style='color:{color};font-weight:800;font-size:1.05rem;'>{icon} {verdict.action.replace('_', ' ')}</div>"
        f"<div style='color:{color};font-weight:700;'>P(hit) = {verdict.p_hit*100:.0f}%</div>"
        f"</div>"
        f"<div style='color:#1f2937;font-size:0.78rem;margin-top:4px;'>"
        f"Confluence: {verdict.confluence_count} of 5 axes · weighted score {verdict.score:+.2f}</div>"
        f"<div style='color:#374151;font-size:0.78rem;margin-top:2px;'>{verdict.note}</div>"
        + (
            "<ul style='color:#a8261f;font-size:0.76rem;margin:6px 0 0 18px;padding:0;'>"
            + "".join(f"<li>{b}</li>" for b in verdict.blockers)
            + "</ul>"
            if verdict.blockers
            else ""
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    chips = []
    for axis in verdict.axes:
        d = axis.direction
        emoji = "🟢" if d > 0 else ("🔴" if d < 0 else "⚪")
        chips.append(
            f"<div style='padding:4px 8px;background:#0b1220;border-radius:6px;margin:2px 0;'>"
            f"<div style='color:#e2e8f0;font-size:0.74rem;'><strong>{emoji} {axis.name}</strong> · "
            f"strength {axis.strength:.2f}</div>"
            f"<div style='color:#94a3b8;font-size:0.7rem;'>{axis.detail}</div>"
            f"</div>"
        )
    st.markdown("".join(chips), unsafe_allow_html=True)


def _render_catalysts(bundle: EventBundle) -> None:
    st.markdown(
        "<div style='color:#94a3b8;font-size:0.72rem;font-weight:700;margin-top:10px;'>"
        "📡 CATALYSTS</div>",
        unsafe_allow_html=True,
    )

    if bundle.news and bundle.news.configured:
        sent_color = {
            "bullish": "#0a7d2a", "bearish": "#a8261f",
        }.get(bundle.news.sentiment, "#5b6470")
        st.markdown(
            f"<div style='padding:6px 8px;background:#0b1220;border-radius:6px;margin-top:4px;'>"
            f"<strong style='color:{sent_color};font-size:0.76rem;'>News · {bundle.news.sentiment}</strong>"
            f"<div style='color:#cbd5e1;font-size:0.74rem;'>{bundle.news.summary[:200]}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif bundle.news:
        st.caption(f"News: {bundle.news.summary[:120]}")

    if bundle.fred and bundle.fred.available:
        warn = bundle.fred.warnings
        warn_html = ""
        if warn:
            warn_html = (
                "<div style='color:#a16207;font-size:0.72rem;'>warnings: "
                + ", ".join(f"{w.label}={w.value:.2f}" for w in warn[:3])
                + "</div>"
            )
        st.markdown(
            "<div style='padding:6px 8px;background:#0b1220;border-radius:6px;margin-top:4px;'>"
            "<strong style='color:#e2e8f0;font-size:0.76rem;'>Macro (FRED)</strong>"
            "<div style='color:#cbd5e1;font-size:0.72rem;'>"
            + ", ".join(f"{o.label} {o.value:.2f}" for o in bundle.fred.observations[:5])
            + "</div>"
            + warn_html
            + "</div>",
            unsafe_allow_html=True,
        )
    elif bundle.fred:
        st.caption(f"FRED: {bundle.fred.error or 'unavailable'}")

    if bundle.calendar:
        rows = "".join(
            f"<li><strong>{e.name[:50]}</strong> "
            f"<span style='color:#a16207;'>{e.importance}</span> · "
            f"{e.country} · in {e.hours_from_now:.0f}h</li>"
            for e in bundle.calendar[:6]
        )
        st.markdown(
            f"<div style='padding:6px 8px;background:#0b1220;border-radius:6px;margin-top:4px;'>"
            f"<strong style='color:#e2e8f0;font-size:0.76rem;'>Calendar</strong>"
            f"<ul style='color:#cbd5e1;font-size:0.72rem;margin:4px 0 0 16px;padding:0;'>{rows}</ul>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if bundle.onchain and bundle.onchain.available:
        unlock_html = ""
        if bundle.onchain.next_unlock_at_utc:
            hrs = (bundle.onchain.next_unlock_at_utc - time.time()) / 3600.0
            unlock_html = (
                f"<div style='color:#cbd5e1;font-size:0.72rem;'>"
                f"next unlock: {bundle.onchain.next_unlock_label} in {hrs:.0f}h "
                f"({(bundle.onchain.next_unlock_pct_supply or 0):.2f}% supply)</div>"
            )
        st.markdown(
            f"<div style='padding:6px 8px;background:#0b1220;border-radius:6px;margin-top:4px;'>"
            f"<strong style='color:#e2e8f0;font-size:0.76rem;'>On-chain</strong>"
            f"<div style='color:#cbd5e1;font-size:0.72rem;'>"
            f"ETF: {bundle.onchain.etf_label()} · funding: {bundle.onchain.funding_label()}"
            + (f" · z={bundle.onchain.funding_rate_z_30d:+.1f}" if bundle.onchain.funding_rate_z_30d is not None else "")
            + "</div>"
            + unlock_html
            + "</div>",
            unsafe_allow_html=True,
        )

    if bundle.errors:
        with st.expander("provider errors", expanded=False):
            for k, v in bundle.errors.items():
                st.caption(f"{k}: {v}")


def _render_regime_micro(
    regime: Optional[RegimeReport], micro: Optional[MicrostructureReport]
) -> None:
    st.markdown(
        "<div style='color:#94a3b8;font-size:0.72rem;font-weight:700;margin-top:10px;'>"
        "🎚️ REGIME &amp; MICROSTRUCTURE</div>",
        unsafe_allow_html=True,
    )
    if regime is not None:
        bits: list[str] = []
        if regime.hmm:
            bits.append(f"HMM {regime.hmm.label} ({regime.hmm.bull_prob:.0%} bull, src {regime.hmm.source})")
        if regime.bocpd_changepoint_prob is not None:
            bits.append(f"BOCPD P(change)={regime.bocpd_changepoint_prob:.2f}")
        if regime.hurst is not None:
            bits.append(regime.hurst_label)
        if regime.vr_label:
            bits.append(regime.vr_label)
        if regime.wyckoff:
            bits.append(f"Wyckoff: {regime.wyckoff.phase} ({regime.wyckoff.confidence:.0%})")
        if regime.macro:
            bits.append(f"{regime.macro.quadrant} · equity {regime.macro.equity_bias} / crypto {regime.macro.crypto_bias}")
        st.markdown(
            "<div style='padding:6px 8px;background:#0b1220;border-radius:6px;font-size:0.74rem;color:#cbd5e1;'>"
            + "<br/>".join(bits)
            + "</div>",
            unsafe_allow_html=True,
        )
        if regime.errors:
            with st.expander("regime errors", expanded=False):
                for k, v in regime.errors.items():
                    st.caption(f"{k}: {v}")
    if micro is not None:
        micro_bits: list[str] = [
            f"VWAP: {micro.vwap_relation()}",
            f"CVD div: {micro.cvd_divergence:+d}",
        ]
        if micro.imbalance_score is not None:
            micro_bits.append(f"imbalance: {micro.imbalance_score:+.2f}")
        if micro.rv_zscore is not None:
            micro_bits.append(f"RV-z: {micro.rv_zscore:+.1f}")
        st.markdown(
            "<div style='padding:6px 8px;background:#0b1220;border-radius:6px;margin-top:4px;font-size:0.74rem;color:#cbd5e1;'>"
            + " · ".join(micro_bits)
            + "</div>",
            unsafe_allow_html=True,
        )


def _render_premortem(
    *,
    key_prefix: str,
    idea_summary: str,
    target_text: str,
    horizon: str,
    verdict: GateVerdict,
    bundle: Optional[EventBundle],
    regime: Optional[RegimeReport],
    micro: Optional[MicrostructureReport],
) -> None:
    context = {
        "verdict": verdict.to_dict(),
        "regime": regime.to_dict() if regime else None,
        "microstructure": (
            {
                "vwap_band_sigma": micro.vwap_band_sigma,
                "cvd_divergence": micro.cvd_divergence,
                "rv_zscore": micro.rv_zscore,
            }
            if micro
            else None
        ),
        "tier_1_calendar": [
            {"name": e.name, "in_hours": round(e.hours_from_now, 1)}
            for e in (bundle.tier_1_in_window() if bundle else [])
        ],
    }
    horizon_for_pm = horizon if horizon in {"intraday", "swing", "position", "long"} else "swing"
    render_premortem_panel(
        key_prefix=f"{key_prefix}::pm",
        default_title=f"{idea_summary} ({verdict.action})",
        default_plan=target_text,
        default_horizon=horizon_for_pm,  # type: ignore[arg-type]
        context=context,
        compact=True,
        expanded=False,
        intro="Premortem the idea with the gate's full context attached — Claude sees the verdict, regime, microstructure, and any blocking catalysts when generating failure modes.",
    )


def _render_notify_button(key_prefix: str, idea_summary: str, verdict: GateVerdict) -> None:
    import os
    if verdict.action != "GO":
        return
    if not os.environ.get("MONTE_NTFY_TOPIC"):
        st.caption("Set `MONTE_NTFY_TOPIC` to enable phone push.")
        return
    if st.button("📲 Notify me to act", key=f"{key_prefix}::notify", use_container_width=True):
        ok = ntfy_push(
            title=f"GO · {idea_summary}",
            body=f"P(hit) {verdict.p_hit*100:.0f}% · {verdict.confluence_count}/5 axes",
            priority="high",
            tags=["rotating_light"],
        )
        if ok:
            st.toast("Push sent.", icon="📲")
        else:
            st.toast("Push failed (check `MONTE_NTFY_TOPIC`).", icon="⚠️")


__all__ = ["render_event_chat_widget"]
