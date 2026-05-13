"""Kalshi Crypto — decision dashboard.

Hybrid spot source: pulls Kalshi market odds automatically, while letting
the user override the spot price if Coinbase/Binance are blocked or stale.
Two render modes:

- **Ranges table** (default): groups markets by event and shows one row per
  strike, sorted, with model probability + edge + EV for each.
- **Detail cards**: per-market cards with full reasoning.

Also includes a **photo upload** path: the user uploads a Kalshi screenshot,
Claude vision extracts the markets + odds, and the same decision engine
scores them — useful when the API is blocked or the screenshot has markets
not yet in the auto-pulled series list.

No automated execution. The user decides whether to bet; this just flags
where the book and the vol model disagree.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from app._shared import setup_page
from app._ui import inject_global_css, status_pill
from app.kalshi import (
    Decision,
    KalshiClient,
    KalshiMarket,
    ParsedMarket,
    get_spot_price,
    parse_screenshot,
    score_event,
    score_parsed_markets,
)
from app.kalshi.spot import SpotQuote, default_sigma_per_min, manual_quote

SYMBOLS = ["BTC", "ETH", "SOL"]
HORIZONS = ["15min", "hourly", "daily", "weekly"]
HORIZON_LABELS = {
    "15min": "15-min Up/Down",
    "hourly": "Hourly settlement",
    "daily": "Daily settlement",
    "weekly": "Weekly (Friday 5pm EDT)",
    "longer": "Longer-dated",
}


@st.cache_resource(show_spinner=False)
def _client() -> KalshiClient:
    return KalshiClient()


@st.cache_data(ttl=8, show_spinner=False)
def _fetch_markets(symbol: str, horizons: tuple[str, ...]) -> dict[str, list[KalshiMarket]]:
    return _client().crypto_markets(symbol, horizons=horizons)


@st.cache_data(ttl=8, show_spinner=False)
def _fetch_spot(symbol: str) -> SpotQuote:
    return get_spot_price(symbol)


def _direction_pill(direction: str, confidence: float) -> str:
    palette = {
        "YES": ("#0a7d2a", "#e0f5e6", "🟢"),
        "NO": ("#a8261f", "#fbe9e7", "🔴"),
        "PASS": ("#5b6470", "#f1f3f5", "⚪"),
    }
    fg, bg, emoji = palette.get(direction, palette["PASS"])
    return (
        f"<span class='spy-pill' style='color:{fg};background:{bg};"
        f"font-size:0.95rem;padding:4px 12px;font-weight:700;'>"
        f"{emoji} {direction} · {confidence:.0f}%</span>"
    )


def _countdown(seconds: float) -> str:
    if seconds <= 0:
        return "closed"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h {(s % 3600) // 60:02d}m"
    return f"{s // 86400}d {(s % 86400) // 3600:02d}h"


def _decision_row(d: Decision) -> dict[str, object]:
    """Flatten a Decision into a row for a DataFrame (ranges grid)."""
    return {
        "Side / strike": d.title.split("·")[-1].strip() if "·" in d.title else d.title,
        "Direction": d.direction,
        "Conf %": round(d.confidence_pct, 0),
        "Book YES ¢": d.yes_side.ask_cents,
        "Model YES %": round(d.yes_side.model_prob * 100, 1),
        "YES edge pp": round(d.yes_side.edge * 100, 1),
        "YES EV ¢/$": round(d.yes_side.ev_per_dollar * 100, 1),
        "Book NO ¢": d.no_side.ask_cents,
        "NO edge pp": round(d.no_side.edge * 100, 1),
        "NO EV ¢/$": round(d.no_side.ev_per_dollar * 100, 1),
    }


def _render_ranges_table(title: str, decisions: list[Decision]) -> None:
    """Compact table view: one row per market/strike, sorted by best edge."""
    if not decisions:
        return
    decisions = sorted(
        decisions,
        key=lambda d: -max(d.yes_side.edge, d.no_side.edge),
    )
    rows = [_decision_row(d) for d in decisions]
    df = pd.DataFrame(rows)

    def _style_direction(val: object) -> str:
        if val == "YES":
            return "color: #0a7d2a; font-weight: 700;"
        if val == "NO":
            return "color: #a8261f; font-weight: 700;"
        return "color: #5b6470;"

    def _style_edge(val: object) -> str:
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if v >= 4:
            return "background: #e0f5e6;"
        if v <= -4:
            return "background: #fbe9e7;"
        return ""

    styler = (
        df.style
        .applymap(_style_direction, subset=["Direction"])
        .applymap(_style_edge, subset=["YES edge pp", "NO edge pp"])
    )
    st.markdown(f"**{title}**")
    st.dataframe(styler, hide_index=True, use_container_width=True)


def _render_decision_card(d: Decision) -> None:
    """Full per-market card with reasoning + warnings."""
    with st.container(border=True):
        cols = st.columns([3.0, 1.4, 1.4, 1.6])
        cols[0].markdown(
            f"**{d.title}**  \n"
            f"<span class='spy-meta'>Closes in {_countdown(d.horizon_seconds)} · "
            f"Spot ${d.spot_price:,.2f} ({d.spot_source}) · "
            f"σ {d.sigma_per_min*100:.3f}%/min</span>",
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            _direction_pill(d.direction, d.confidence_pct),
            unsafe_allow_html=True,
        )

        yes = d.yes_side
        ev_color = "#0a7d2a" if yes.ev_per_dollar > 0 else "#a8261f"
        cols[2].markdown(
            f"**YES** @ {yes.ask_cents}¢ ({yes.payout:.2f}x)  \n"
            f"<span class='spy-meta'>book {yes.implied_prob*100:.1f}% · "
            f"model {yes.model_prob*100:.1f}% · edge {yes.edge*100:+.1f}pp</span>  \n"
            f"<span style='color:{ev_color};font-weight:700;'>"
            f"EV {yes.ev_per_dollar*100:+.1f}¢/$1</span>"
            + (f" · Kelly {yes.kelly_fraction*100:.1f}%" if yes.kelly_fraction > 0 else ""),
            unsafe_allow_html=True,
        )

        no = d.no_side
        ev_color = "#0a7d2a" if no.ev_per_dollar > 0 else "#a8261f"
        cols[3].markdown(
            f"**NO** @ {no.ask_cents}¢ ({no.payout:.2f}x)  \n"
            f"<span class='spy-meta'>book {no.implied_prob*100:.1f}% · "
            f"model {no.model_prob*100:.1f}% · edge {no.edge*100:+.1f}pp</span>  \n"
            f"<span style='color:{ev_color};font-weight:700;'>"
            f"EV {no.ev_per_dollar*100:+.1f}¢/$1</span>"
            + (f" · Kelly {no.kelly_fraction*100:.1f}%" if no.kelly_fraction > 0 else ""),
            unsafe_allow_html=True,
        )

        st.caption(d.reasoning)
        for w in d.warnings:
            st.markdown(status_pill(w, "warn"), unsafe_allow_html=True)


def _resolve_spot(symbol: str) -> tuple[SpotQuote | None, str | None]:
    override = st.session_state.get(f"kalshi_spot_override_{symbol}")
    if override and override.get("price"):
        sigma_override = override.get("sigma") or default_sigma_per_min(symbol)
        return manual_quote(symbol, float(override["price"]), float(sigma_override)), None
    try:
        return _fetch_spot(symbol), None
    except Exception as e:
        return None, str(e)


def _spot_block(symbol: str, spot: SpotQuote | None, err: str | None) -> None:
    spot_col, override_col = st.columns([2.4, 1.6])
    if spot:
        age = max(0, int(time.time() - spot.ts))
        spot_col.markdown(
            f"**{symbol} spot ${spot.price:,.2f}**  "
            f"<span class='spy-meta'>source: {spot.source} · {age}s old · "
            f"σ {spot.sigma_per_min*100:.3f}%/min</span>",
            unsafe_allow_html=True,
        )
    else:
        spot_col.markdown(
            status_pill(f"{symbol} spot fetch failed: {err}", "err"),
            unsafe_allow_html=True,
        )

    with override_col:
        with st.expander(f"{symbol} manual spot override", expanded=not spot):
            default_price = float(spot.price) if spot else 0.0
            new_price = st.number_input(
                f"{symbol} spot ($)",
                min_value=0.0,
                value=default_price,
                step=0.01,
                key=f"manual_price_{symbol}",
                format="%.2f",
            )
            default_sigma_pct = float(
                (spot.sigma_per_min if spot else default_sigma_per_min(symbol)) * 100
            )
            new_sigma = st.number_input(
                "σ per minute (%, optional)",
                min_value=0.0,
                value=default_sigma_pct,
                step=0.001,
                key=f"manual_sigma_{symbol}",
                format="%.4f",
            )
            c1, c2 = st.columns(2)
            if c1.button("Use override", key=f"use_override_{symbol}"):
                st.session_state[f"kalshi_spot_override_{symbol}"] = {
                    "price": new_price,
                    "sigma": new_sigma / 100.0,
                }
                st.rerun()
            if c2.button("Clear override", key=f"clear_override_{symbol}"):
                st.session_state.pop(f"kalshi_spot_override_{symbol}", None)
                st.rerun()


def _render_symbol(
    symbol: str,
    horizons: tuple[str, ...],
    *,
    view_mode: str,
    min_edge: float,
    min_ev: float,
) -> None:
    st.subheader(symbol)
    spot, err = _resolve_spot(symbol)
    _spot_block(symbol, spot, err)
    if not spot:
        return

    try:
        markets_by_horizon = _fetch_markets(symbol, horizons)
    except Exception as e:
        st.error(f"Kalshi API error for {symbol}: {e}")
        return

    any_rendered = False
    for horizon in horizons:
        markets = [m for m in markets_by_horizon.get(horizon, []) if m.status == "active"]
        if not markets:
            continue
        any_rendered = True
        st.markdown(f"#### {HORIZON_LABELS[horizon]}")

        # Group by event_ticker so multiple strikes of one event sit together.
        by_event: dict[str, list[KalshiMarket]] = {}
        for m in markets:
            by_event.setdefault(m.event_ticker, []).append(m)

        for event_ticker, event_markets in by_event.items():
            decisions = score_event(event_markets, spot, min_edge=min_edge, min_ev=min_ev)
            if view_mode == "Ranges table":
                event_title = event_markets[0].title or event_ticker
                _render_ranges_table(event_title, decisions)
            else:
                actionable = [d for d in decisions if d.direction != "PASS"]
                passes = [d for d in decisions if d.direction == "PASS"]
                for d in actionable:
                    _render_decision_card(d)
                if passes:
                    with st.expander(f"PASS markets ({len(passes)}) — book in line with model"):
                        for d in passes[:8]:
                            _render_decision_card(d)

    if not any_rendered:
        st.info(f"No active {symbol} markets in the selected horizons right now.")


# ---------------------------------------------------------------------------
# Photo upload — vision parsing path
# ---------------------------------------------------------------------------

def _render_photo_upload(min_edge: float, min_ev: float) -> None:
    st.markdown("### 📸 Upload a Kalshi screenshot")
    st.caption(
        "Drop a screenshot of the Kalshi crypto tab or a single market detail page. "
        "Claude vision extracts the markets + odds, then we score them against the "
        "current spot. Useful when the Kalshi API is blocked, or for markets not in "
        "the auto-pulled series."
    )
    uploaded = st.file_uploader(
        "Image",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        key="kalshi_screenshot",
    )
    if uploaded is None:
        return

    media_type = uploaded.type or "image/png"
    image_bytes = uploaded.getvalue()
    st.image(image_bytes, caption=uploaded.name, width=320)

    # Cache parsed result keyed by the image bytes so refreshes don't re-spend
    # API tokens on the same screenshot.
    cache_key = ("kalshi_parsed", hash(image_bytes))
    parsed: list[ParsedMarket] | None = st.session_state.get(cache_key)

    cols = st.columns([1, 1, 4])
    if cols[0].button("Parse with Claude vision", type="primary", key="parse_screenshot"):
        with st.spinner("Reading screenshot…"):
            try:
                parsed = parse_screenshot(image_bytes, media_type=media_type)
            except Exception as e:
                st.error(f"Vision parse failed: {e}")
                return
        st.session_state[cache_key] = parsed
    if cols[1].button("Clear", key="clear_screenshot"):
        st.session_state.pop(cache_key, None)
        st.session_state.pop("kalshi_screenshot", None)
        st.rerun()

    if not parsed:
        return

    if not parsed:
        st.warning("No crypto markets detected in this screenshot.")
        return

    symbols_in_screenshot = {p.symbol for p in parsed}
    spot_by_symbol: dict[str, SpotQuote] = {}
    for symbol in sorted(symbols_in_screenshot):
        spot, err = _resolve_spot(symbol)
        if spot:
            spot_by_symbol[symbol] = spot
        else:
            st.warning(f"{symbol}: spot unavailable ({err}). Set a manual override below to score.")

    if not spot_by_symbol:
        # Still let the user set overrides via the per-symbol blocks below.
        for symbol in sorted(symbols_in_screenshot):
            _spot_block(symbol, None, None)
        return

    scored = score_parsed_markets(parsed, spot_by_symbol, min_edge=min_edge, min_ev=min_ev)
    for pm, decisions in scored:
        if not decisions:
            st.info(f"Skipped {pm.title}: no spot for {pm.symbol}.")
            continue
        st.markdown(f"#### {pm.title}  \n*{HORIZON_LABELS.get(pm.horizon, pm.horizon)} · {pm.symbol}*")
        _render_ranges_table("Ranges from screenshot", decisions)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_page("Kalshi Crypto", icon="🪙")
    inject_global_css()

    with st.sidebar:
        st.subheader("Kalshi")
        symbols = st.multiselect(
            "Symbols",
            SYMBOLS,
            default=SYMBOLS,
            help="Crypto series to pull from Kalshi.",
        )
        horizons = st.multiselect(
            "Horizons",
            HORIZONS,
            default=HORIZONS,
            format_func=lambda h: HORIZON_LABELS[h],
        )
        view_mode = st.radio(
            "View",
            ["Ranges table", "Detail cards"],
            index=0,
            help="Ranges table: one row per strike, sorted by best edge. "
            "Detail cards: per-market reasoning + warnings.",
        )
        min_edge_pp = st.slider(
            "Min edge (pp)",
            0,
            20,
            4,
            help="Minimum (model − book) probability gap, in percentage points, "
            "before we recommend YES or NO. Smaller edges are dominated by "
            "model error.",
        )
        min_ev_cents = st.slider(
            "Min EV (¢ per $1)",
            0,
            20,
            2,
            help="Minimum expected return in cents per $1 staked. Acts as a "
            "fee + slippage cushion.",
        )
        refresh_secs = st.select_slider(
            "Auto-refresh",
            options=[5, 10, 30, 60, 120, 300],
            value=30,
        )
        st_autorefresh(interval=refresh_secs * 1000, key="kalshi_refresh")

    st.caption(
        "Hybrid: Kalshi orderbook auto-pulled, spot price auto-pulled "
        "(Coinbase → Binance) with manual override. Direction is based on "
        "edge vs. a driftless GBM model using realised 1-min vol over "
        "time-to-close. **Advisory only — no orders are placed.**"
    )
    st.markdown(
        f"{status_pill(f'auto-refresh {refresh_secs}s', 'info')} "
        f"{status_pill(f'edge ≥ {min_edge_pp}pp · EV ≥ {min_ev_cents}¢', 'muted')} "
        f"{status_pill(view_mode, 'muted')} "
        f"{status_pill(datetime.now(timezone.utc).strftime('%H:%M:%S UTC'), 'muted')}",
        unsafe_allow_html=True,
    )

    min_edge = min_edge_pp / 100.0
    min_ev = min_ev_cents / 100.0

    with st.expander("📸 Upload a Kalshi screenshot (vision parser)", expanded=False):
        _render_photo_upload(min_edge=min_edge, min_ev=min_ev)

    if not symbols or not horizons:
        st.warning("Pick at least one symbol and one horizon in the sidebar.")
        return

    for symbol in symbols:
        _render_symbol(
            symbol,
            tuple(horizons),
            view_mode=view_mode,
            min_edge=min_edge,
            min_ev=min_ev,
        )


if __name__ == "__main__":
    main()
