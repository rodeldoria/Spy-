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

        /* ── Alert card ── */
        .spy-alert-card {
            padding: 12px 16px;
            border-radius: 10px;
            border: 1px solid rgba(127,127,127,0.18);
            margin: 6px 0;
        }
        .spy-alert-top {
            display: flex;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 6px;
        }
        .spy-alert-symbol {
            font-size: 1.1rem;
            font-weight: 700;
            flex: 1 1 auto;
            min-width: 80px;
        }
        .spy-alert-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            align-items: center;
        }
        .spy-alert-metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 6px 0;
        }
        .spy-alert-metric {
            padding: 4px 10px;
            border: 1px solid rgba(127,127,127,0.18);
            border-radius: 8px;
            min-width: 70px;
            flex: 1 1 auto;
        }
        .spy-alert-metric .label { font-size: 0.72rem; color: #6b7280; }
        .spy-alert-metric .value { font-size: 0.95rem; font-weight: 700; }
        .spy-alert-body {
            font-size: 0.82rem;
            color: #94a3b8;
            line-height: 1.5;
            margin-top: 4px;
        }

        /* ── PnL strip — horizontal on desktop, wraps on mobile ── */
        .spy-pnl-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 4px 0 8px 0;
        }
        .spy-pnl-cell {
            flex: 1 1 100px;
            padding: 8px 12px;
            border: 1px solid rgba(127,127,127,0.22);
            border-radius: 10px;
        }

        /* ── BUY / SELL pulse animations ── */
        @keyframes buy-pulse {
            0%   { box-shadow: 0 0 0 0   rgba(10,125,42,0.70); }
            50%  { box-shadow: 0 0 0 14px rgba(10,125,42,0.00); }
            100% { box-shadow: 0 0 0 0   rgba(10,125,42,0.70); }
        }
        @keyframes sell-pulse {
            0%   { box-shadow: 0 0 0 0   rgba(168,38,31,0.70); }
            50%  { box-shadow: 0 0 0 14px rgba(168,38,31,0.00); }
            100% { box-shadow: 0 0 0 0   rgba(168,38,31,0.70); }
        }
        .buy-now-banner {
            background: linear-gradient(135deg, #0a7d2a 0%, #12a03a 100%);
            color: #fff;
            padding: 18px 22px;
            border-radius: 14px;
            margin: 10px 0 14px 0;
            animation: buy-pulse 1.8s infinite;
        }
        .sell-now-banner {
            background: linear-gradient(135deg, #a8261f 0%, #c43028 100%);
            color: #fff;
            padding: 18px 22px;
            border-radius: 14px;
            margin: 10px 0 14px 0;
            animation: sell-pulse 1.8s infinite;
        }
        .signal-banner-label {
            font-size: 0.72rem;
            letter-spacing: 1.6px;
            text-transform: uppercase;
            opacity: 0.85;
            margin-bottom: 2px;
        }
        .signal-banner-title {
            font-size: 1.6rem;
            font-weight: 900;
            letter-spacing: 0.3px;
            margin: 2px 0 4px 0;
        }
        .signal-banner-meta {
            font-size: 0.92rem;
            opacity: 0.93;
        }
        .signal-banner-why {
            font-size: 0.82rem;
            margin-top: 8px;
            opacity: 0.88;
            line-height: 1.5;
        }
        .signal-banner-options {
            font-size: 0.82rem;
            margin-top: 6px;
            padding: 6px 10px;
            background: rgba(255,255,255,0.18);
            border-radius: 8px;
            opacity: 0.95;
        }

        /* ══════════════════════════════════════════════
           MOBILE RESPONSIVE  (≤ 768 px)
           ══════════════════════════════════════════════ */
        @media (max-width: 768px) {

            /* Tighter main padding */
            .main .block-container {
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
                padding-top: 0.75rem !important;
                max-width: 100% !important;
            }

            /* Stack ALL Streamlit columns into a single column */
            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 0 !important;
            }
            [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                flex: 1 1 100% !important;
                min-width: 0 !important;
                width: 100% !important;
            }

            /* Exception: allow 2-wide pairs (2 cols of roughly equal weight) */
            [data-testid="stHorizontalBlock"].spy-2col > [data-testid="column"] {
                flex: 1 1 48% !important;
            }

            /* Metric — reduce label size on very small screens */
            [data-testid="stMetric"] label {
                font-size: 0.75rem !important;
            }
            [data-testid="stMetricValue"] {
                font-size: 1.1rem !important;
            }

            /* Signal banners — tighter on mobile */
            .buy-now-banner, .sell-now-banner {
                padding: 12px 14px !important;
                border-radius: 10px !important;
            }
            .signal-banner-title {
                font-size: 1.25rem !important;
            }
            .signal-banner-label {
                font-size: 0.68rem !important;
                letter-spacing: 1.2px !important;
            }
            .signal-banner-meta {
                font-size: 0.82rem !important;
            }

            /* Pills — smaller on mobile */
            .spy-pill {
                font-size: 0.72rem !important;
                padding: 2px 8px !important;
            }

            /* PnL strip — 2-up grid */
            .spy-pnl-strip {
                gap: 6px !important;
            }
            .spy-pnl-cell {
                flex: 1 1 calc(50% - 6px) !important;
                padding: 6px 10px !important;
            }

            /* Kalshi probability bars — fit labels */
            .spy-meta {
                font-size: 0.72rem !important;
            }

            /* Sidebar — reasonable width on small screens */
            [data-testid="stSidebar"] {
                min-width: 240px !important;
                max-width: 80vw !important;
            }

            /* Tables and charts full-width */
            [data-testid="stDataFrame"],
            [data-testid="stPlotlyChart"] {
                width: 100% !important;
                overflow-x: auto !important;
            }

            /* Buttons — bigger tap targets */
            .stButton > button {
                min-height: 44px !important;
                font-size: 0.9rem !important;
                width: 100% !important;
            }

            /* Headings — scale down */
            h1 { font-size: 1.6rem !important; }
            h2 { font-size: 1.3rem !important; }
            h3 { font-size: 1.1rem !important; }
        }

        @media (max-width: 480px) {
            .signal-banner-title { font-size: 1.05rem !important; }
            .buy-now-banner, .sell-now-banner {
                padding: 10px 12px !important;
            }
            /* PnL strip — single column on very small phones */
            .spy-pnl-cell { flex: 1 1 100% !important; }
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


def pnl_strip(daily: float, weekly: float, monthly: float, ytd: float) -> None:
    def _fmt(v: float) -> str:
        sign = "+" if v >= 0 else "−"
        return f"{sign}${abs(v):,.2f}"

    def _color(v: float) -> str:
        return "#0a7d2a" if v >= 0 else "#a8261f"

    cells = [
        ("Today", daily),
        ("This week", weekly),
        ("This month", monthly),
        ("Year-to-date", ytd),
    ]
    html = "<div style='display:flex;gap:10px;margin:4px 0 8px 0;'>"
    for label, v in cells:
        html += (
            f"<div style='flex:1;padding:8px 12px;border:1px solid rgba(127,127,127,0.22);"
            f"border-radius:10px;'>"
            f"<div class='spy-meta'>{label} P&amp;L</div>"
            f"<div style='font-weight:700;color:{_color(v)};font-size:1.05rem;'>{_fmt(v)}</div>"
            f"</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


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
