"""Kalshi Prediction Markets — financial & macro event probabilities.

Pulls live Kalshi market data for crypto, equities, and macro events.
Each market shows the crowd-sourced implied probability (real-money odds).
No authentication required — public read-only market data only.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from app._shared import setup_page
from app._ui import inject_global_css

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
TIMEOUT = 8.0

FINANCIAL_SERIES: list[dict[str, Any]] = [
    {
        "label": "₿ Bitcoin",
        "emoji": "₿",
        "category": "Crypto",
        "series": [
            {"ticker": "KXBTC", "label": "BTC price range today", "type": "range"},
            {"ticker": "KXBTCMAX150", "label": "When will BTC hit $150k?", "type": "milestone"},
            {"ticker": "KXBTCMAXM", "label": "BTC high this month", "type": "milestone"},
        ],
    },
    {
        "label": "⬨ Ethereum",
        "emoji": "⬨",
        "category": "Crypto",
        "series": [
            {"ticker": "KXETH", "label": "ETH price range today", "type": "range"},
            {"ticker": "KXETHMAXY", "label": "ETH high this year", "type": "milestone"},
        ],
    },
    {
        "label": "📈 S&P 500",
        "emoji": "📈",
        "category": "Equities",
        "series": [
            {"ticker": "KXINX", "label": "S&P 500 range today", "type": "range"},
            {"ticker": "KXINXPOS", "label": "S&P 500 positive this year?", "type": "binary"},
        ],
    },
    {
        "label": "🏛 Federal Reserve",
        "emoji": "🏛",
        "category": "Macro",
        "series": [
            {"ticker": "KXFED", "label": "Fed funds rate — next meeting", "type": "range"},
            {"ticker": "KXRATECUT", "label": "Fed rate cut before 2027?", "type": "binary"},
            {"ticker": "KXRATECUTCOUNT", "label": "Number of rate cuts", "type": "range"},
        ],
    },
    {
        "label": "📊 Inflation (CPI)",
        "emoji": "📊",
        "category": "Macro",
        "series": [
            {"ticker": "KXCPIYOY", "label": "CPI YoY — next print", "type": "range"},
            {"ticker": "CPICOREYOY", "label": "Core CPI YoY", "type": "range"},
        ],
    },
    {
        "label": "⚠️ Recession",
        "emoji": "⚠️",
        "category": "Macro",
        "series": [
            {"ticker": "KXRECSSNBER", "label": "NBER Recession", "type": "binary"},
        ],
    },
]

CATEGORY_COLORS = {
    "Crypto": "#f7931a",
    "Equities": "#1a6ef7",
    "Macro": "#7a5af8",
}


# ── API helpers ──────────────────────────────────────────────────────────────

def _sess() -> requests.Session:
    s = requests.Session()
    s.headers["Accept"] = "application/json"
    s.headers["User-Agent"] = "spy-/kalshi-markets"
    return s


@st.cache_resource(show_spinner=False)
def _session() -> requests.Session:
    return _sess()


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_events(series_ticker: str, limit: int = 4) -> list[dict]:
    url = f"{BASE_URL}/events"
    params = {"series_ticker": series_ticker, "status": "open", "limit": limit}
    try:
        r = _session().get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json().get("events", []) or []
    except Exception:
        return []


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_event_detail(event_ticker: str) -> dict | None:
    url = f"{BASE_URL}/events/{event_ticker}"
    try:
        r = _session().get(url, params={"with_nested_markets": "true"}, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json().get("event") or None
    except Exception:
        return None


def _dollars_to_prob(val: str | float | None) -> float:
    if val is None:
        return 0.0
    try:
        return max(0.0, min(1.0, float(val)))
    except (TypeError, ValueError):
        return 0.0


def _mid_prob(market: dict) -> float:
    bid = _dollars_to_prob(market.get("yes_bid_dollars"))
    ask = _dollars_to_prob(market.get("yes_ask_dollars"))
    if ask > 0 and bid >= 0:
        return (bid + ask) / 2.0
    return bid or ask


def _volume(market: dict) -> float:
    v = market.get("volume_fp") or market.get("volume") or 0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _prob_bar(prob: float, label: str, highlight: bool = False) -> str:
    pct = round(prob * 100)
    if prob >= 0.65:
        bar_color = "#22c55e"
        text_color = "#166534"
    elif prob >= 0.40:
        bar_color = "#f59e0b"
        text_color = "#92400e"
    else:
        bar_color = "#ef4444"
        text_color = "#991b1b"

    border = "border:2px solid #22c55e;border-radius:8px;" if highlight else "border-radius:8px;"
    bg = "background:#f0fdf4;" if highlight else "background:#1e2433;"
    star = "⭐ " if highlight else ""
    return f"""
<div style="{bg}padding:8px 12px;margin:3px 0;{border}">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
    <span style="font-size:0.88rem;color:#cbd5e1;">{star}{label}</span>
    <span style="font-weight:700;font-size:1.0rem;color:{text_color if highlight else bar_color};">{pct}%</span>
  </div>
  <div style="background:#374151;border-radius:4px;height:8px;overflow:hidden;">
    <div style="background:{bar_color};width:{pct}%;height:8px;border-radius:4px;transition:width 0.4s;"></div>
  </div>
</div>"""


def _render_range_event(event: dict, max_markets: int = 8) -> None:
    """Render a price-range event: show all buckets sorted by probability."""
    markets = event.get("markets") or []
    if not markets:
        st.caption("No markets found.")
        return

    active = [m for m in markets if m.get("status") == "active"]
    if not active:
        active = markets

    probs = [(m, _mid_prob(m)) for m in active]
    probs.sort(key=lambda x: x[1], reverse=True)
    top = probs[:max_markets]

    best_prob = top[0][1] if top else 0.0
    html_parts = []
    for m, prob in top:
        subtitle = m.get("subtitle") or m.get("yes_sub_title") or "?"
        is_best = prob == best_prob and prob > 0.05
        html_parts.append(_prob_bar(prob, subtitle, highlight=is_best))

    st.markdown("".join(html_parts), unsafe_allow_html=True)

    total_vol = sum(_volume(m) for m, _ in probs)
    if total_vol > 0:
        st.caption(f"Total volume: ${total_vol:,.0f}")


def _render_binary_event(event: dict) -> None:
    """Render a single YES/NO event."""
    markets = event.get("markets") or []
    if not markets:
        st.caption("No markets found.")
        return

    for m in markets[:1]:
        prob = _mid_prob(m)
        subtitle = m.get("subtitle") or m.get("yes_sub_title") or "YES"
        st.markdown(_prob_bar(prob, f"YES — {subtitle}", highlight=prob >= 0.5), unsafe_allow_html=True)
        no_prob = 1.0 - prob
        st.markdown(_prob_bar(no_prob, "NO", highlight=no_prob >= 0.5), unsafe_allow_html=True)
        vol = _volume(m)
        if vol > 0:
            st.caption(f"Volume: ${vol:,.0f}")


def _render_milestone_event(event: dict, max_markets: int = 6) -> None:
    """Render a milestone/when-will-X event: show most probable time buckets."""
    _render_range_event(event, max_markets=max_markets)


def _close_label(event: dict) -> str:
    ct = event.get("close_time") or event.get("end_date") or ""
    if not ct:
        return ""
    try:
        dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = dt - now
        secs = int(diff.total_seconds())
        if secs <= 0:
            return "closed"
        if secs < 3600:
            return f"closes in {secs // 60}m"
        if secs < 86400:
            return f"closes in {secs // 3600}h {(secs % 3600) // 60}m"
        return f"closes in {secs // 86400}d"
    except Exception:
        return ""


# ── Main page ─────────────────────────────────────────────────────────────────

def main() -> None:
    setup_page("Kalshi Markets", icon="🔮")
    inject_global_css()

    with st.sidebar:
        st.subheader("🔮 Kalshi Markets")
        refresh_secs = st.select_slider(
            "Auto-refresh",
            options=[15, 30, 60, 120, 300],
            value=60,
            help="How often to pull fresh Kalshi data.",
        )
        st_autorefresh(interval=refresh_secs * 1000, key="kalshi_markets_refresh")

        categories = st.multiselect(
            "Categories",
            ["Crypto", "Equities", "Macro"],
            default=["Crypto", "Equities", "Macro"],
        )

        st.divider()
        st.caption(
            "Probabilities are mid-market (bid+ask)/2 from real-money Kalshi "
            "contracts. Advisory only — this does not place trades."
        )

    st.title("🔮 Kalshi Prediction Markets")
    st.caption(
        f"Live crowd-sourced event probabilities · refreshes every {refresh_secs}s · "
        f"last update {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
    )

    groups = [g for g in FINANCIAL_SERIES if g["category"] in categories]
    if not groups:
        st.info("Select at least one category in the sidebar.")
        return

    for group in groups:
        cat = group["category"]
        cat_color = CATEGORY_COLORS.get(cat, "#64748b")
        st.markdown(
            f"<h3 style='color:{cat_color};margin-top:1.2em;'>{group['label']}</h3>",
            unsafe_allow_html=True,
        )

        for series_cfg in group["series"]:
            series_ticker = series_cfg["ticker"]
            series_label = series_cfg["label"]
            market_type = series_cfg["type"]

            events = _fetch_events(series_ticker, limit=2)
            if not events:
                with st.expander(f"{series_label} — no open markets", expanded=False):
                    st.caption("No open Kalshi events found for this series right now.")
                continue

            event = events[0]
            event_ticker = event.get("event_ticker", "")
            title = event.get("title") or series_label
            close_lbl = _close_label(event)

            header = f"**{series_label}** — {title}"
            if close_lbl:
                header += f"  \n<span style='font-size:0.8rem;color:#94a3b8;'>{close_lbl}</span>"

            with st.expander(f"{series_label}", expanded=True):
                st.markdown(header, unsafe_allow_html=True)

                detail = _fetch_event_detail(event_ticker)
                if detail and detail.get("markets"):
                    if market_type == "range":
                        _render_range_event(detail)
                    elif market_type == "binary":
                        _render_binary_event(detail)
                    elif market_type == "milestone":
                        _render_milestone_event(detail)
                else:
                    st.caption("Loading market data…")

    st.divider()
    st.caption(
        "Source: [Kalshi](https://kalshi.com) public market data API · "
        "Probabilities reflect real-money market consensus, not model forecasts."
    )


if __name__ == "__main__":
    main()
