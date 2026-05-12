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
