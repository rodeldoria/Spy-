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
    """Inject a small CSS bundle so the app stops looking like the default demo.

    Idempotent — calling it multiple times in a session is fine because
    Streamlit only renders the most recent block.
    """
    st.markdown(
        """
        <style>
        :root {
            --pill-radius: 999px;
        }
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
        }
        .spy-pill.dot::before {
            content: "";
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: currentColor;
            opacity: 0.85;
        }
        .spy-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
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
    """Render an 'updated Xs ago' pill from a timestamp / Timestamp / datetime."""
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
    """Spinner that always shows even when work is fast — feels less demo-y."""
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


def act_now_banner(row: dict) -> None:
    """Render a full-width sticky banner for an ACT_NOW signal."""
    sym = row.get("symbol", "?")
    action = str(row.get("action", "?")).replace("_", " ")
    spot = float(row.get("spot", row.get("entry", 0)) or 0)
    stop = float(row.get("stop", 0) or 0)
    target = float(row.get("target", 0) or 0)
    rr = float(row.get("rr", 0) or 0)
    conf = float(row.get("confidence", 0) or 0)
    reasoning = row.get("reasoning", "")
    options = row.get("options_ticket")

    st.markdown(
        f"""
        <div style='background:linear-gradient(90deg,#0a7d2a,#138a3a);
                    color:#fff;padding:14px 18px;border-radius:12px;
                    margin:8px 0 12px 0;box-shadow:0 2px 6px rgba(10,125,42,0.25);'>
          <div style='font-size:0.78rem;letter-spacing:1.2px;opacity:0.85;'>🟢 ACT NOW · {conf:.0f}% conviction</div>
          <div style='font-size:1.35rem;font-weight:800;margin:2px 0;'>{sym} {action} @ ${spot:,.2f}</div>
          <div style='font-size:0.92rem;opacity:0.95;'>
            Stop ${stop:,.2f} · Target ${target:,.2f} · R:R {rr:.2f}
          </div>
          <div style='font-size:0.85rem;margin-top:6px;opacity:0.9;'>{reasoning}</div>
          {("<div style='margin-top:6px;font-size:0.85rem;'>Options: " +
            options.get("side","") + " $" + str(options.get("strike",0)) + " " +
            options.get("expiry","") + " · premium ~$" +
            f"{options.get('premium',0):.2f}" + " · max risk $" +
            f"{options.get('max_risk_per_contract',0):.0f}" + "/contract" +
            "</div>") if options else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def pnl_strip(daily: float, weekly: float, monthly: float, ytd: float) -> None:
    """Render a four-up DoD / WoW / MoM / YTD realised-PnL strip."""
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
    """Render a drawdown gauge with the 10% brake line annotated."""
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
    """Render a 'realised $X / $4,000 this month' progress bar.

    Informational only — the dashboard does not execute trades, so this is a
    motivator, not a guarantee.
    """
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
