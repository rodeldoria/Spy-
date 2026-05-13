"""Shared UI helpers — pills, freshness badges, loading skeletons.

Keeps the Streamlit pages free of inline CSS and one-off HTML so the look
stays consistent. Everything here is pure presentation; no I/O.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd
import streamlit as st


_ACTION_STYLE = {
    "STRONG_BUY": ("#0a7d2a", "#e8f7ec"),
    "BUY": ("#1d8237", "#eef6f0"),
    "HOLD": ("#5b6470", "#f1f3f5"),
    "SELL": ("#b65a25", "#fdf1e8"),
    "STRONG_SELL": ("#a8261f", "#fbe9e7"),
}


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        :root { --pill-radius: 999px; }

        /* ── Core pills ── */
        .spy-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 2px 10px;
            border-radius: var(--pill-radius);
            font-size: 0.78rem;
            font-weight: 600;
            line-height: 1.4;
            border: 1px solid rgba(0,0,0,0.06);
            white-space: nowrap;
        }
        .spy-pill.dot::before {
            content: "";
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: currentColor;
            opacity: 0.85;
        }

        /* ── Card headers ── */
        .spy-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 4px;
        }
        .spy-card-header h3 {
            margin: 0;
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: 0.2px;
        }
        .spy-meta {
            font-size: 0.78rem;
            color: #6b7280;
        }

        /* ── Skeleton loader ── */
        .spy-skeleton {
            display: block;
            width: 100%;
            height: 14px;
            border-radius: 6px;
            background: linear-gradient(90deg,
                rgba(127,127,127,0.10) 0%,
                rgba(127,127,127,0.22) 50%,
                rgba(127,127,127,0.10) 100%);
            background-size: 200% 100%;
            animation: spy-shimmer 1.2s infinite linear;
            margin: 6px 0;
        }
        @keyframes spy-shimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        .spy-divider {
            height: 1px;
            background: rgba(127,127,127,0.18);
            margin: 8px 0 6px 0;
        }
        .spy-chips {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 8px;
            margin: 6px 0 10px 0;
        }
        .spy-chip {
            padding: 8px 10px;
            border: 1px solid rgba(127,127,127,0.22);
            border-radius: 10px;
            min-width: 0;
        }
        .spy-chip .spy-chip-label {
            font-size: 0.72rem;
            color: #6b7280;
            letter-spacing: 0.2px;
            text-transform: uppercase;
        }
        .spy-chip .spy-chip-value {
            font-weight: 700;
            font-size: 1.05rem;
            line-height: 1.25;
            word-break: break-word;
            white-space: normal;
        }
        .spy-chip .spy-chip-sub {
            font-size: 0.72rem;
            color: #6b7280;
            margin-top: 2px;
        }
        .spy-plain {
            border: 1px solid rgba(127,127,127,0.22);
            border-left: 4px solid #1d4ed8;
            border-radius: 10px;
            padding: 10px 12px;
            margin: 8px 0;
            background: rgba(29,78,216,0.04);
        }
        .spy-plain h4 {
            margin: 0 0 4px 0;
            font-size: 0.95rem;
            font-weight: 700;
        }
        .spy-plain ul { margin: 4px 0 0 0; padding-left: 18px; }
        .spy-plain li { margin: 2px 0; font-size: 0.9rem; }
        /* ── Alert card (Live Signals + HOLD rows) ── */
        .spy-alert-card {
            border: 1px solid rgba(127,127,127,0.22);
            border-radius: 12px;
            padding: 12px 14px;
            margin: 8px 0 10px 0;
            background: rgba(127,127,127,0.04);
        }
        .spy-alert-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 8px;
        }
        .spy-alert-symbol {
            font-weight: 700;
            font-size: 1.05rem;
        }
        .spy-alert-pills {
            display: inline-flex;
            gap: 4px;
            flex-wrap: wrap;
        }
        .spy-alert-metrics {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 6px;
            margin: 6px 0 8px 0;
        }
        .spy-alert-metric {
            border: 1px solid rgba(127,127,127,0.20);
            border-radius: 8px;
            padding: 6px 8px;
            text-align: center;
            background: rgba(127,127,127,0.04);
            min-width: 0;
            cursor: help;
        }
        .spy-alert-metric .label {
            font-size: 0.68rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            margin-bottom: 2px;
        }
        .spy-alert-metric .value {
            font-weight: 700;
            font-size: 0.92rem;
            line-height: 1.1;
            word-break: break-word;
        }
        .spy-alert-body {
            font-size: 0.85rem;
            line-height: 1.45;
            margin-top: 4px;
            opacity: 0.92;
        }
        .spy-alert-foot {
            margin-top: 6px;
            font-size: 0.78rem;
            color: #6b7280;
            line-height: 1.35;
        }
        /* ── Factor breakdown grid (replaces st.columns wall) ── */
        .spy-factor-card {
            border: 1px solid rgba(127,127,127,0.22);
            border-radius: 10px;
            padding: 10px 12px;
            margin: 6px 0 10px 0;
            background: rgba(127,127,127,0.03);
        }
        .spy-factor-head {
            font-weight: 700;
            font-size: 0.9rem;
            margin-bottom: 2px;
        }
        .spy-factor-help {
            font-size: 0.74rem;
            color: #6b7280;
            margin-bottom: 8px;
        }
        .spy-factor-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 6px;
        }
        .spy-factor-pill {
            border: 1px solid rgba(127,127,127,0.22);
            border-radius: 8px;
            padding: 6px 8px;
            text-align: center;
            background: rgba(127,127,127,0.04);
            min-width: 0;
            cursor: help;
        }
        .spy-factor-pill .label {
            font-size: 0.66rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        .spy-factor-pill .value {
            font-weight: 700;
            font-size: 0.9rem;
            line-height: 1.15;
        }
        /* ── Pattern chips (Patterns active strip) ── */
        .spy-pattern-row {
            margin: 8px 0 4px 0;
            font-size: 0.85rem;
        }
        .spy-pattern-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 4px 4px;
            margin-top: 4px;
        }
        .spy-pattern-chip {
            display: inline-flex;
            align-items: center;
            padding: 3px 9px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            cursor: help;
        }
        /* ── Forecast probability bar (Kalshi range buckets) ── */
        .spy-prob-bar {
            padding: 9px 12px;
            margin: 4px 0;
            border-radius: 8px;
            border: 1px solid rgba(127,127,127,0.20);
            background: rgba(127,127,127,0.05);
        }
        .spy-prob-bar.is-best {
            border: 2px solid #22c55e;
            background: rgba(34,197,94,0.10);
        }
        .spy-prob-bar .row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
            gap: 8px;
        }
        .spy-prob-bar .label {
            font-size: 0.88rem;
            font-weight: 600;
        }
        .spy-prob-bar .pct {
            font-weight: 800;
            font-size: 1.0rem;
            font-variant-numeric: tabular-nums;
        }
        .spy-prob-bar .track {
            background: rgba(127,127,127,0.22);
            border-radius: 4px;
            height: 8px;
            overflow: hidden;
        }
        .spy-prob-bar .fill {
            height: 8px;
            border-radius: 4px;
            transition: width 0.4s;
        }
        .spy-stream {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.78rem;
            color: #0a7d2a;
            font-weight: 600;
        }
        .spy-stream::before {
            content: "";
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #0a7d2a;
            box-shadow: 0 0 0 0 rgba(10,125,42,0.6);
            animation: spy-pulse 1.4s infinite;
        }
        @keyframes spy-pulse {
            0%   { box-shadow: 0 0 0 0 rgba(10,125,42,0.55); }
            70%  { box-shadow: 0 0 0 8px rgba(10,125,42,0.0); }
            100% { box-shadow: 0 0 0 0 rgba(10,125,42,0.0); }
        }
        /* --- Mobile (≤ 640px): stack the 2-up card grid and shrink chips --- */
        @media (max-width: 640px) {
            [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                gap: 8px !important;
            }
            [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }
            .spy-chips {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .spy-card-header {
                flex-wrap: wrap;
            }
            .spy-card-header h3 { font-size: 1rem; }
            .spy-pill { font-size: 0.72rem; padding: 2px 8px; }
            /* Keep these grids as grids on mobile — independent of column stack */
            .spy-alert-metrics {
                grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
            }
            .spy-factor-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            }
            .spy-alert-metric .value { font-size: 0.88rem; }
            .spy-alert-card { padding: 10px 12px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def action_pill(action: str, confidence: float | None = None) -> str:
    fg, bg = _ACTION_STYLE.get(str(action).upper(), _ACTION_STYLE["HOLD"])
    label = str(action).replace("_", " ")
    conf = f" · {confidence:.0f}%" if confidence is not None else ""
    return (
        f"<span class='spy-pill dot' "
        f"style='color:{fg};background:{bg};'>{label}{conf}</span>"
    )


def status_pill(text: str, kind: str = "info") -> str:
    palette = {
        "ok": ("#0a7d2a", "#e8f7ec"),
        "warn": ("#a16207", "#fef9c3"),
        "err": ("#a8261f", "#fbe9e7"),
        "info": ("#1f4ed8", "#e8efff"),
        "muted": ("#5b6470", "#f1f3f5"),
    }
    fg, bg = palette.get(kind, palette["info"])
    return (
        f"<span class='spy-pill' style='color:{fg};background:{bg};'>{text}</span>"
    )


def freshness_pill(last_ts) -> str:
    if last_ts is None:
        return status_pill("no data", "err")
    if isinstance(last_ts, pd.Timestamp):
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        ts = last_ts.timestamp()
    elif isinstance(last_ts, datetime):
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        ts = last_ts.timestamp()
    else:
        try:
            ts = float(last_ts)
        except (TypeError, ValueError):
            return status_pill("no data", "err")

    age = max(0.0, time.time() - ts)
    if age < 90:
        text, kind = f"updated {int(age)}s ago", "ok"
    elif age < 60 * 60:
        text, kind = f"updated {int(age // 60)}m ago", "ok"
    elif age < 24 * 60 * 60:
        text, kind = f"updated {int(age // 3600)}h ago", "warn"
    else:
        text, kind = f"updated {int(age // 86400)}d ago", "warn"
    return status_pill(text, kind)


def metric_chip(label: str, value: str, hint: str | None = None) -> str:
    hint_html = f"<div class='spy-meta'>{hint}</div>" if hint else ""
    return (
        "<div style='padding:6px 10px;border:1px solid rgba(127,127,127,0.18);"
        "border-radius:8px;min-width:80px;'>"
        f"<div class='spy-meta'>{label}</div>"
        f"<div style='font-weight:600;font-size:1rem;'>{value}</div>"
        f"{hint_html}</div>"
    )


def render_skeleton(rows: int = 3) -> None:
    st.markdown(
        "".join("<div class='spy-skeleton'></div>" for _ in range(rows)),
        unsafe_allow_html=True,
    )


@contextmanager
def loading(message: str):
    with st.spinner(message):
        yield


_TIER_STYLE = {
    "ACT_NOW":    ("#0a7d2a", "#d6f5dc", "🟢", "ACT NOW"),
    "WATCH":      ("#a16207", "#fef9c3", "🟡", "Watch"),
    "STAND_DOWN": ("#5b6470", "#f1f3f5", "⚪", "Stand down"),
}


def tier_pill(tier: str, confidence: float | None = None) -> str:
    fg, bg, emoji, label = _TIER_STYLE.get(str(tier).upper(), _TIER_STYLE["STAND_DOWN"])
    conf = f" · {confidence:.0f}%" if confidence is not None else ""
    return (
        f"<span class='spy-pill' style='color:{fg};background:{bg};"
        f"font-size:0.9rem;padding:4px 12px;'>{emoji} {label}{conf}</span>"
    )


def signal_banner(row: dict) -> None:
    """Pulsing full-width BUY NOW / SELL NOW banner for ACT_NOW signals."""
    sym = row.get("symbol", "?")
    action = str(row.get("action", "?")).replace("_", " ")
    spot = float(row.get("spot", row.get("entry", 0)) or 0)
    stop = float(row.get("stop", 0) or 0)
    target = float(row.get("target", 0) or 0)
    rr = float(row.get("rr", 0) or 0)
    conf = float(row.get("confidence", 0) or 0)
    horizon = str(row.get("horizon", "")).replace("_", " ").title()
    reasoning = row.get("reasoning", "")
    options = row.get("options_ticket")
    is_buy = "BUY" in str(row.get("action", "")).upper()
    cls = "buy-now-banner" if is_buy else "sell-now-banner"
    icon = "🚀" if is_buy else "🔻"

    opts_html = ""
    if options and not options.get("is_crypto_note"):
        opts_html = (
            f"<div class='signal-banner-options'>"
            f"📈 Options: <strong>{options.get('side','')} ${options.get('strike',0):.0f}</strong> "
            f"exp. {options.get('expiry','')} · "
            f"premium ~<strong>${options.get('premium',0):.2f}</strong> · "
            f"max risk <strong>${options.get('max_risk_per_contract',0):.0f}</strong>/contract"
            f"</div>"
        )
    elif options and options.get("is_crypto_note"):
        opts_html = (
            f"<div class='signal-banner-options'>"
            f"💡 {options.get('rationale','')}"
            f"</div>"
        )

    st.markdown(
        f"""
        <div class='{cls}'>
          <div class='signal-banner-label'>{icon} ACT NOW · {conf:.0f}% conviction · {horizon}</div>
          <div class='signal-banner-title'>{sym} {action} @ ${spot:,.2f}</div>
          <div class='signal-banner-meta'>
            Stop <strong>${stop:,.2f}</strong> · Target <strong>${target:,.2f}</strong> · R:R <strong>{rr:.2f}</strong>
          </div>
          {f"<div class='signal-banner-why'>💡 {reasoning}</div>" if reasoning else ""}
          {opts_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def act_now_banner(row: dict) -> None:
    """Alias kept for backward compatibility."""
    signal_banner(row)


def signal_guide() -> None:
    """Render a collapsible signal-tier reference card for new users."""
    tiers = [
        {
            "label": "ACT NOW",
            "color": "#0a7d2a",
            "bg": "rgba(10,125,42,0.10)",
            "border": "#0a7d2a",
            "icon": "🟢",
            "headline": "Enter now — high-conviction setup",
            "body": (
                "4 or more indicators agree, confidence ≥ 75%, and macro is on-side. "
                "This is the window you've been waiting for. The signal shows an exact "
                "<strong>entry price</strong>, a <strong>stop-loss</strong> (where you exit if wrong), "
                "and a <strong>target</strong> (where you take profit). "
                "Size your position using the Budget page — never risk more than the suggested amount. "
                "<em>Move quickly — ACT NOW windows are typically short.</em>"
            ),
            "do": "Open a paper trade. Note entry, stop, target.",
            "dont": "Risk more than the suggested size. Chase if price has already moved far.",
        },
        {
            "label": "WATCH",
            "color": "#b45309",
            "bg": "rgba(180,83,9,0.10)",
            "border": "#b45309",
            "icon": "🟡",
            "headline": "Setup forming — prepare but wait",
            "body": (
                "3+ indicators are aligned and confidence ≥ 60%, but the signal isn't fully "
                "confirmed yet. Think of this as the <strong>pre-game</strong>: conditions are "
                "good but you haven't seen the starting gun. Review the stop/target levels, "
                "decide your position size, and have capital ready. <em>Do not enter yet.</em>"
            ),
            "do": "Note the symbol, set a price alert, calculate your risk.",
            "dont": "Enter early — the setup may reverse before confirmation.",
        },
        {
            "label": "HOLD",
            "color": "#4b5563",
            "bg": "rgba(75,85,99,0.10)",
            "border": "#6b7280",
            "icon": "⬜",
            "headline": "No edge detected — stay flat",
            "body": (
                "Indicators are mixed or near-neutral. There is no clear directional signal. "
                "If you are <strong>not in this trade</strong>: stay out. "
                "If you are <strong>already in this trade</strong>: HOLD does <em>not</em> mean sell — "
                "it means no new information has arrived. Manage your existing stop and target."
            ),
            "do": "Hold existing positions. Manage stops.",
            "dont": "Open new positions. HOLD is not a buy or sell instruction.",
        },
        {
            "label": "STAND DOWN",
            "color": "#991b1b",
            "bg": "rgba(153,27,27,0.10)",
            "border": "#dc2626",
            "icon": "⚪",
            "headline": "Conditions are against you — wait",
            "body": (
                "Fewer than 3 indicators agree, macro is misaligned (SPY below its 200-day, "
                "broad selling pressure), or confidence is too low. "
                "The market environment works against a clean trade right now. "
                "<strong>Do not open new positions.</strong> This is the system protecting you "
                "from low-probability setups. Patience here is a position."
            ),
            "do": "Watch and wait. Re-scan when conditions shift.",
            "dont": "Force a trade. STAND DOWN is a protection, not a failure.",
        },
    ]

    with st.expander("🚦 Signal Guide — what do the tiers mean?", expanded=False):
        st.markdown(
            "<p style='color:var(--text-muted,#888);font-size:0.85rem;margin-bottom:0.75rem;'>"
            "New here? This guide explains every signal level the system can show you."
            "</p>",
            unsafe_allow_html=True,
        )
        for t in tiers:
            st.markdown(
                f"""
                <div style='
                    border-left:4px solid {t["border"]};
                    background:{t["bg"]};
                    border-radius:8px;
                    padding:12px 16px;
                    margin-bottom:10px;
                '>
                  <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
                    <span style='font-size:1.1rem;'>{t["icon"]}</span>
                    <strong style='color:{t["color"]};font-size:1rem;letter-spacing:0.04em;'>{t["label"]}</strong>
                    <span style='color:#888;font-size:0.82rem;'>— {t["headline"]}</span>
                  </div>
                  <div style='font-size:0.88rem;line-height:1.55;margin-bottom:6px;'>{t["body"]}</div>
                  <div style='display:flex;gap:24px;font-size:0.82rem;margin-top:4px;'>
                    <span><strong style='color:#0a7d2a;'>✓ Do:</strong> {t["do"]}</span>
                    <span><strong style='color:#a8261f;'>✗ Don't:</strong> {t["dont"]}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            "<p style='color:#888;font-size:0.78rem;margin-top:4px;'>"
            "All signals are paper-trade simulations only — no real money is at risk. "
            "Confidence is built from 7 technical factors: RSI, MACD, Bollinger %b, Trend, "
            "Regime, Volume Surge, and Momentum ROC."
            "</p>",
            unsafe_allow_html=True,
        )


def pnl_strip(daily: float, weekly: float, monthly: float, ytd: float) -> None:
    def _fmt(v: float) -> str:
        sign = "+" if v >= 0 else "−"
        return f"{sign}${abs(v):,.2f}"

    def _color(v: float) -> str:
        return "#0a7d2a" if v >= 0 else "#a8261f"

    cells = [
        ("Today", daily),
        ("Week", weekly),
        ("Month", monthly),
        ("YTD", ytd),
    ]
    cells_html = "".join(
        f"<div class='spy-pnl-cell'>"
        f"<div class='spy-meta'>{label} P&amp;L</div>"
        f"<div style='font-weight:700;color:{_color(v)};font-size:1.05rem;'>{_fmt(v)}</div>"
        f"</div>"
        for label, v in cells
    )
    st.markdown(f"<div class='spy-pnl-strip'>{cells_html}</div>", unsafe_allow_html=True)


def drawdown_gauge(current_dd: float, max_dd: float = -0.10) -> None:
    pct = min(0.0, current_dd)
    width = min(100.0, abs(pct / max_dd) * 100.0)
    color = "#0a7d2a" if pct >= -0.03 else ("#a16207" if pct >= -0.07 else "#a8261f")
    st.markdown(
        f"""
        <div style='margin:6px 0 10px 0;'>
          <div class='spy-meta'>Drawdown · {pct*100:.1f}% (brake at {max_dd*100:.0f}%)</div>
          <div style='background:rgba(127,127,127,0.12);border-radius:6px;height:10px;overflow:hidden;'>
            <div style='width:{width:.1f}%;height:100%;background:{color};'></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def target_progress(realised_pnl: float, target: float = 4000.0) -> None:
    target = max(float(target), 1.0)
    progress = max(0.0, min(1.0, float(realised_pnl) / target))
    kind = "ok" if progress >= 1.0 else ("info" if realised_pnl >= 0 else "warn")
    st.progress(
        progress,
        text=f"Monthly P&L · ${realised_pnl:,.2f} / ${target:,.0f} target",
    )
    st.markdown(
        status_pill(
            f"{progress * 100:.0f}% of target · informational, not a guarantee",
            kind,
        ),
        unsafe_allow_html=True,
    )
