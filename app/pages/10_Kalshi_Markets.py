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


def _seconds_to_close(event: dict) -> int | None:
    ct = event.get("close_time") or event.get("end_date") or ""
    if not ct:
        return None
    try:
        dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        return int((dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


def _close_label(event: dict) -> str:
    secs = _seconds_to_close(event)
    if secs is None:
        return ""
    if secs <= 0:
        return "closed"
    if secs < 3600:
        return f"closes in {secs // 60}m"
    if secs < 86400:
        return f"closes in {secs // 3600}h {(secs % 3600) // 60}m"
    return f"closes in {secs // 86400}d"


# ── Insight + ranking ─────────────────────────────────────────────────────────

def _event_insight(event: dict, market_type: str) -> tuple[str, str, float]:
    """Return (headline, narrative, play_score) for a fetched event detail.

    play_score: 0..100 — higher = more interesting play right now. Combines
      - conviction: how lopsided the top market is (closer to 100% / 0% = stronger signal)
      - urgency: closer to close = faster payout / more time pressure
      - liquidity: log-scaled total volume so we can actually trade it
    """
    markets = event.get("markets") or []
    active = [m for m in markets if m.get("status") == "active"] or markets
    if not active:
        return ("", "No active markets.", 0.0)

    probs = [(m, _mid_prob(m)) for m in active]
    probs.sort(key=lambda x: x[1], reverse=True)
    top_m, top_p = probs[0]
    top_label = top_m.get("subtitle") or top_m.get("yes_sub_title") or "?"
    total_vol = sum(_volume(m) for m, _ in probs)

    # Conviction = distance from 50/50, scaled to 0..1
    conviction = abs(top_p - 0.5) * 2.0

    # Urgency: 1.0 if <1h to close, decaying to ~0 at 7+ days
    secs = _seconds_to_close(event) or (60 * 60 * 24 * 30)  # default 30d if unknown
    if secs <= 0:
        urgency = 0.0
    elif secs < 3600:
        urgency = 1.0
    elif secs < 86400:
        urgency = 0.85 - (secs / 86400) * 0.4   # 24h → ~0.45
    else:
        urgency = max(0.0, 0.45 - (secs / 86400) * 0.05)  # 9d → 0

    # Liquidity: log10($vol)/5 capped at 1 (i.e. $100k+ = 1.0)
    import math
    liq = min(1.0, math.log10(total_vol + 1) / 5) if total_vol > 0 else 0.0

    # Composite score: conviction matters most, urgency second, liquidity gates
    score = 100.0 * (0.55 * conviction + 0.30 * urgency + 0.15 * liq)

    # Narrative — explain the imbalance in plain English
    if market_type == "binary":
        no_p = 1.0 - top_p
        if top_p >= 0.7:
            why = (
                f"Crowd is paying ${top_p:.2f} for YES — "
                f"strong consensus that this happens. Edge plays: buy NO if you "
                f"think the market is overconfident (paying ${no_p:.2f} to risk "
                f"$1 of payout)."
            )
        elif top_p <= 0.3:
            why = (
                f"Crowd is paying just ${top_p:.2f} for YES — strong consensus "
                f"this DOES NOT happen. NO is priced at ${no_p:.2f}. Edge plays: "
                f"buy YES contrarian if you think the market is underestimating it."
            )
        else:
            why = (
                f"Toss-up — YES ${top_p:.2f} / NO ${no_p:.2f}. "
                f"No strong consensus; trade only with a real informational edge."
            )
        headline = f"{int(round(top_p * 100))}% YES — {('strong' if conviction > 0.4 else 'leaning')}"
    elif market_type in ("range", "milestone"):
        if len(probs) > 1:
            second_p = probs[1][1]
            spread = top_p - second_p
            if spread > 0.3:
                why = (
                    f"Heavy concentration on **{top_label}** ({int(top_p*100)}%). "
                    f"Next bucket is far behind ({int(second_p*100)}%) — the market "
                    f"thinks this outcome is much more likely than alternatives."
                )
            elif spread > 0.1:
                why = (
                    f"**{top_label}** is the favourite at {int(top_p*100)}%, but "
                    f"the second bucket ({int(second_p*100)}%) is in striking "
                    f"distance. Watch for shifts before close."
                )
            else:
                why = (
                    f"Tight race — {top_label} only barely leads "
                    f"({int(top_p*100)}% vs {int(second_p*100)}%). Crowd is "
                    f"genuinely uncertain; small news could flip the favourite."
                )
        else:
            why = f"Single bucket: **{top_label}** at {int(top_p*100)}%."
        headline = f"{int(round(top_p * 100))}% on {top_label}"
    else:
        why = ""
        headline = f"{int(round(top_p * 100))}%"

    return (headline, why, score)


def _insight_box(headline: str, why: str, score: float, close_lbl: str) -> str:
    score_color = "#22c55e" if score >= 60 else ("#f59e0b" if score >= 40 else "#6b7280")
    score_badge = (
        f"<span style='background:{score_color};color:white;padding:2px 8px;"
        f"border-radius:10px;font-size:0.72rem;font-weight:700;'>"
        f"play score {score:.0f}</span>"
    )
    close_badge = ""
    if close_lbl:
        close_color = "#ef4444" if "m" in close_lbl and "h" not in close_lbl else "#94a3b8"
        close_badge = (
            f" <span style='color:{close_color};font-size:0.78rem;'>⏱ {close_lbl}</span>"
        )
    return (
        f"<div style='padding:8px 12px;margin:4px 0 8px 0;background:#0f172a;"
        f"border-left:3px solid {score_color};border-radius:6px;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<strong style='color:#e2e8f0;font-size:0.86rem;'>{headline}</strong>"
        f"<div>{score_badge}{close_badge}</div></div>"
        f"<div style='color:#94a3b8;font-size:0.8rem;margin-top:4px;'>"
        f"<strong>Why:</strong> {why}</div></div>"
    )


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

    # Pre-fetch all events + details so we can rank top plays before rendering.
    # _fetch_event_detail is cached 30s so re-calling it during render is free.
    pre: list[dict] = []
    for group in groups:
        for series_cfg in group["series"]:
            evs = _fetch_events(series_cfg["ticker"], limit=2)
            if not evs:
                continue
            ev = evs[0]
            detail = _fetch_event_detail(ev.get("event_ticker", "")) or ev
            headline, why, score = _event_insight(detail, series_cfg["type"])
            pre.append({
                "group": group, "series_cfg": series_cfg,
                "event": ev, "detail": detail,
                "headline": headline, "why": why, "score": score,
                "close_lbl": _close_label(ev),
                "secs_to_close": _seconds_to_close(ev) or 10**9,
            })

    # ── 🔥 Top plays right now ───────────────────────────────────────────
    if pre:
        top_n = 5
        ranked = sorted(pre, key=lambda x: x["score"], reverse=True)[:top_n]
        closing = sorted(
            [p for p in pre if 0 < p["secs_to_close"] < 3600 * 24],
            key=lambda x: x["secs_to_close"],
        )[:5]

        st.markdown(
            "<h3 style='margin-top:0.4em;'>🔥 Top plays right now</h3>"
            "<p style='color:#94a3b8;font-size:0.85rem;margin:-6px 0 8px 0;'>"
            "Ranked by <strong>conviction</strong> (how lopsided the price is) · "
            "<strong>urgency</strong> (closer to close = faster payout) · "
            "<strong>liquidity</strong> (volume you can actually trade against)."
            "</p>",
            unsafe_allow_html=True,
        )
        for p in ranked:
            st.markdown(
                f"<div style='font-size:0.82rem;color:#cbd5e1;margin:4px 0 -2px 0;'>"
                f"{p['group']['emoji']} <strong>{p['series_cfg']['label']}</strong></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                _insight_box(p["headline"], p["why"], p["score"], p["close_lbl"]),
                unsafe_allow_html=True,
            )

        if closing:
            st.markdown(
                "<h4 style='margin-top:0.8em;color:#ef4444;'>⏱ Closing within 24h</h4>",
                unsafe_allow_html=True,
            )
            for p in closing:
                st.markdown(
                    f"<div style='font-size:0.82rem;color:#cbd5e1;margin:4px 0 -2px 0;'>"
                    f"{p['group']['emoji']} <strong>{p['series_cfg']['label']}</strong></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _insight_box(p["headline"], p["why"], p["score"], p["close_lbl"]),
                    unsafe_allow_html=True,
                )

        st.divider()

    # ── Full breakdown by category ────────────────────────────────────────
    by_group: dict[str, list[dict]] = {}
    for p in pre:
        by_group.setdefault(p["group"]["label"], []).append(p)

    for group in groups:
        cat = group["category"]
        cat_color = CATEGORY_COLORS.get(cat, "#64748b")
        st.markdown(
            f"<h3 style='color:{cat_color};margin-top:1.2em;'>{group['label']}</h3>",
            unsafe_allow_html=True,
        )
        items = by_group.get(group["label"], [])
        if not items:
            st.caption("No open Kalshi events for this group right now.")
            continue

        for p in items:
            series_cfg = p["series_cfg"]
            series_label = series_cfg["label"]
            market_type = series_cfg["type"]
            event = p["event"]
            detail = p["detail"]
            title = event.get("title") or series_label
            close_lbl = p["close_lbl"]

            with st.expander(f"{series_label}", expanded=True):
                header = f"**{series_label}** — {title}"
                if close_lbl:
                    header += f"  \n<span style='font-size:0.8rem;color:#94a3b8;'>{close_lbl}</span>"
                st.markdown(header, unsafe_allow_html=True)

                # Insight box explains *why* this market sits where it does.
                if p["headline"]:
                    st.markdown(
                        _insight_box(p["headline"], p["why"], p["score"], close_lbl),
                        unsafe_allow_html=True,
                    )

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
