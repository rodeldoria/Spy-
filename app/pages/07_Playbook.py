"""Claude's Playbook — pattern tracker + alert log.

Two halves:

1. **AI Decision Council pattern tracker** — every verdict the council
   has made on the Kalshi Crypto page is logged here with its 8-framework
   signature. Once underlying markets settle we also see win/loss + ROI
   per signature, so the user can watch which patterns actually pay.

2. **Monte Edge alerts log** — the existing tail of EdgeSignal WATCH /
   ACT_NOW signals from the scanner.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import streamlit as st

from app._shared import setup_page
from app._ui import inject_global_css, status_pill, tier_pill
from monte.learning import pattern_tracker as ptrack
from monte.strategy.playbook import list_playbook


def _ago(ts: float) -> str:
    if not ts:
        return ""
    age = max(0.0, time.time() - float(ts))
    if age < 60:
        return f"{int(age)}s ago"
    if age < 3600:
        return f"{int(age // 60)}m ago"
    if age < 86400:
        return f"{int(age // 3600)}h ago"
    return f"{int(age // 86400)}d ago"


@st.cache_data(ttl=60, show_spinner=False)
def _cached_report(_t: int):
    return ptrack.build_report()


@st.cache_data(ttl=300, show_spinner=False)
def _cached_reconcile(_t: int) -> dict:
    try:
        return ptrack.reconcile_outcomes()
    except Exception:
        return {"settled": 0, "still_pending": 0}


def _render_pattern_tracker() -> None:
    """The AI Decision Council learning view — verdicts, signatures, outcomes."""
    # Try to settle any verdicts whose underlying Kalshi markets have closed.
    # Reconcile every 5 min, rebuild the report every 60 s.
    _cached_reconcile(int(time.time() // 300))
    rep = _cached_report(int(time.time() // 60))

    st.header("🚦 AI Decision Council — pattern tracker")
    if rep.n_verdicts == 0:
        st.info(
            "No council verdicts logged yet. Open the **Kalshi Crypto** "
            "page — every actionable opportunity scored by the council is "
            "logged here automatically with its 8-framework signature. "
            "Once those markets settle, this page learns which signatures "
            "actually win and feeds that confidence back into the score."
        )
        return

    # Top-line metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdicts logged", f"{rep.n_verdicts:,}")
    c2.metric("Outcomes settled", f"{rep.n_outcomes:,}")
    if rep.overall_hit_rate is not None:
        c3.metric("Overall hit rate", f"{rep.overall_hit_rate*100:.0f}%")
    else:
        c3.metric("Overall hit rate", "—",
                  help="Need at least 1 settled outcome.")
    if rep.avg_roi_pct is not None:
        c4.metric("Avg ROI / play", f"{rep.avg_roi_pct:+.1f}%")
    else:
        c4.metric("Avg ROI / play", "—")

    st.caption(
        "Verdicts are logged from the Kalshi Crypto page (deduped to once "
        "per market per 5 min). Outcomes are joined automatically when the "
        "underlying Kalshi market settles. After ≥5 settled outcomes for a "
        "given signature, the council multiplies new scores by the learned "
        "hit-rate multiplier (range 0.6× to 1.4×)."
    )

    # ----- Signature aggregation table -----------------------------------
    st.subheader("📐 Pattern signatures — which combinations actually pay")
    sig_rows = sorted(
        rep.by_signature.values(),
        key=lambda s: (-(s.n_settled), -s.n_total),
    )
    if sig_rows:
        st.caption(
            "Each signature is the 8-bit pattern of which checkpoints passed: "
            "**EDGE · KELLY · EV · CONV · LIQ · CAL · TRI · PRE-MORTEM**. "
            "1 = passed, 0 = failed. Hit rate and avg ROI shown for signatures "
            "with at least 1 settled outcome."
        )
        # Build a table-like display with mini cards per signature.
        for s in sig_rows[:15]:
            hr_str = f"{s.hit_rate*100:.0f}%" if s.hit_rate is not None else "—"
            roi_str = f"{s.avg_roi:+.1f}%" if s.avg_roi is not None else "—"
            mult, label = ptrack.confidence_multiplier(s.signature)
            if mult > 1.05:
                bar_color = "#22c55e"
            elif mult < 0.95:
                bar_color = "#ef4444"
            else:
                bar_color = "#94a3b8"
            st.markdown(
                f"<div style='padding:8px 12px;margin:4px 0;background:#0f172a;"
                f"border-left:3px solid {bar_color};border-radius:6px;'>"
                f"<div style='display:flex;justify-content:space-between;'>"
                f"<code style='color:#e2e8f0;font-size:0.92rem;'>{s.signature}</code>"
                f"<span style='color:{bar_color};font-weight:700;font-size:0.85rem;'>"
                f"×{mult:.2f}</span></div>"
                f"<div style='color:#94a3b8;font-size:0.74rem;margin-top:3px;'>"
                f"{ptrack.signature_label(s.signature)}</div>"
                f"<div style='color:#cbd5e1;font-size:0.8rem;margin-top:4px;'>"
                f"{s.n_total} verdict(s) · {s.n_settled} settled · "
                f"hit rate <strong>{hr_str}</strong> · "
                f"avg ROI <strong>{roi_str}</strong> · "
                f"avg trigger score <strong>{s.avg_trigger_score:.0f}/100</strong>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No signatures aggregated yet.")

    # ----- Recent verdicts log -------------------------------------------
    st.subheader("📋 Recent verdicts log")
    st.caption(
        "Newest first. Outcome shows up automatically when the Kalshi "
        "market settles (we re-check on every page load)."
    )

    # Build outcome lookup so we can show win/loss next to each verdict.
    outcome_by_key = {}
    for o in rep.outcomes:
        key = (o.get("ticker"), float(o.get("verdict_ts", 0)))
        outcome_by_key[key] = o

    show_n = st.slider("Rows", 10, 200, 40, key="ptrack_show_n")
    for r in rep.rows[:show_n]:
        key = (r.get("ticker"), float(r.get("ts", 0)))
        outcome = outcome_by_key.get(key)
        ts_iso = datetime.fromtimestamp(
            float(r.get("ts", 0)), tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M")
        verdict_label = r.get("verdict_label", "?")
        verdict_color = {
            "PULL THE TRIGGER": "#16a34a",
            "ALMOST THERE": "#eab308",
            "WATCH ONLY": "#f97316",
            "STAND DOWN": "#ef4444",
        }.get(verdict_label, "#6b7280")

        if outcome is None:
            outcome_html = (
                "<span style='color:#94a3b8;font-size:0.78rem;'>⏳ pending settlement</span>"
            )
        elif outcome.get("won"):
            outcome_html = (
                f"<span style='color:#22c55e;font-weight:700;font-size:0.82rem;'>"
                f"✅ WIN · ROI {outcome.get('roi_pct', 0):+.1f}%</span>"
            )
        else:
            outcome_html = (
                f"<span style='color:#ef4444;font-weight:700;font-size:0.82rem;'>"
                f"❌ LOSS · ROI {outcome.get('roi_pct', 0):+.1f}%</span>"
            )

        st.markdown(
            f"<div style='padding:8px 12px;margin:3px 0;background:#0f172a;"
            f"border:1px solid #1e293b;border-radius:6px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<div style='color:#e2e8f0;font-size:0.88rem;font-weight:600;'>"
            f"{r.get('symbol', '?')} · {r.get('bet_summary', r.get('ticker', '?'))[:80]}</div>"
            f"<div style='color:{verdict_color};font-weight:700;font-size:0.82rem;'>"
            f"{verdict_label} · {r.get('trigger_score', 0):.0f}/100</div>"
            f"</div>"
            f"<div style='color:#94a3b8;font-size:0.74rem;margin-top:3px;'>"
            f"{ts_iso} UTC · {_ago(r.get('ts', 0))} · "
            f"BUY {r.get('direction', '?')} @ {r.get('ask_cents', 0)}¢ · "
            f"edge {r.get('edge_pp', 0):+.1f}pp · "
            f"Kelly {r.get('kelly_fraction', 0)*100:.1f}% · "
            f"sig <code style='color:#e2e8f0;'>{r.get('signature', '????????')}</code>"
            f"</div>"
            f"<div style='margin-top:5px;'>{outcome_html}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_alerts_log() -> None:
    """The existing Monte Edge alerts tail (EdgeSignal-driven WATCH/ACT_NOW)."""
    with st.sidebar:
        st.subheader("Filter alerts")
        symbol = st.text_input("Symbol (optional)", value="", key="pb_symbol")
        tier_choice = st.selectbox(
            "Tier", ["all", "ACT_NOW", "WATCH"], index=0, key="pb_tier")
        limit = st.slider("Rows", 10, 500, 100, key="pb_rows")

    rows = list_playbook(
        limit=limit,
        symbol=symbol.strip().upper() or None,
        tier=None if tier_choice == "all" else tier_choice,
    )
    if not rows:
        st.info(
            "No EdgeSignal alerts logged yet. Run a scan from the home "
            "page or trigger the worker (`python -m monte.alerts.engine`) "
            "to populate this section."
        )
        return

    by_tier: dict[str, int] = {}
    for r in rows:
        by_tier[r.tier] = by_tier.get(r.tier, 0) + 1
    chips = " ".join(
        tier_pill(t, None) + f" <span class='spy-meta'>×{by_tier[t]}</span>"
        for t in ("ACT_NOW", "WATCH") if t in by_tier
    )
    st.markdown(chips, unsafe_allow_html=True)

    for r in rows:
        with st.container(border=True):
            head_cols = st.columns([3, 1, 1, 1])
            ts_iso = datetime.fromtimestamp(
                r.ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            head_cols[0].markdown(
                f"### {r.symbol} <span class='spy-meta'>{r.timeframe} · "
                f"{ts_iso} UTC · {_ago(r.ts)}</span>",
                unsafe_allow_html=True,
            )
            head_cols[1].markdown(tier_pill(r.tier, r.confidence),
                                  unsafe_allow_html=True)
            head_cols[2].metric("Score", f"{r.score:+.2f}")
            head_cols[3].metric("R:R", f"{r.rr:.2f}")

            st.markdown(f"**Action:** {r.action.replace('_',' ')} · **{r.horizon}**")
            st.info(f"💡 {r.reasoning}")
            st.caption(
                f"Entry **\\${r.entry:,.2f}** · Stop **\\${r.stop:,.2f}** · "
                f"Target **\\${r.target:,.2f}** · Confluence **{r.confluence}/5** · "
                f"Macro: {r.macro_note}"
            )
            snap = r.indicator_snapshot or {}
            if snap:
                cells = []
                for k in ("rsi", "bb_pctb", "macd_hist", "adx", "atr_pct"):
                    if k in snap:
                        cells.append(f"{k.upper()} {snap[k]:+.3f}")
                if cells:
                    st.caption(" · ".join(cells))
            if r.options_ticket:
                opt = r.options_ticket
                st.caption(
                    f"📈 Options: **{opt.get('side','')} \\${opt.get('strike',0):.0f}** "
                    f"{opt.get('expiry','')} · premium ~\\${opt.get('premium',0):.2f} · "
                    f"breakeven \\${opt.get('breakeven',0):.2f} · "
                    f"max risk \\${opt.get('max_risk_per_contract',0):.0f}/contract"
                )


def main() -> None:
    setup_page("Claude's Playbook", icon="📓")
    inject_global_css()

    st.markdown(
        "Pattern memory + signal log. The **AI Decision Council** logs every "
        "verdict it makes on the Kalshi Crypto page along with its 8-framework "
        "signature. Once outcomes settle we learn which signatures actually pay "
        "and feed that confidence back into the council's score."
    )

    tab_tracker, tab_alerts = st.tabs(
        ["🚦 Pattern tracker (Council)", "📡 EdgeSignal alerts"])

    with tab_tracker:
        _render_pattern_tracker()
    with tab_alerts:
        _render_alerts_log()


main()
