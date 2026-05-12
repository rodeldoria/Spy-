"""Budget input → suggested position sizing per active alert + simulated paper
book P&L. No real broker calls — everything goes to `~/.monte/paper/`."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app._shared import live_price, setup_page
from app._ui import inject_global_css, loading, status_pill
from monte.alerts.engine import tail_alerts
from monte.broker.paper_book import InsufficientFunds, PaperBook
from monte.config import settings


def _book(state_path) -> PaperBook:
    return PaperBook(state_path=state_path)


def main() -> None:
    setup_page("Budget & Paper Portfolio", icon="💰")
    inject_global_css()

    with st.sidebar:
        st.subheader("Budget")
        new_budget = st.number_input(
            "Reset budget to ($)",
            min_value=100.0,
            value=float(settings.budget_usd),
            step=500.0,
        )
        if st.button("Reset paper book"):
            _book(settings.paper_state_path).reset(budget=new_budget)
            st.success(f"Reset to ${new_budget:,.2f}")

    book = _book(settings.paper_state_path)
    cash = book.cash()
    starting = book.starting_budget()

    # mark to market against current live prices
    positions = book.positions()
    prices: dict[str, float] = {}
    if positions:
        with loading(f"Marking {len(positions)} position(s) to market…"):
            for sym in positions:
                try:
                    p, _ = live_price(sym)
                    prices[sym] = p
                except Exception:
                    prices[sym] = positions[sym].avg_cost or 1.0
    eq = book.mark_to_market(prices) if positions else None

    cols = st.columns(4)
    cols[0].metric("Starting", f"${starting:,.2f}")
    cols[1].metric("Cash", f"${cash:,.2f}")
    cols[2].metric(
        "Equity",
        f"${eq.equity:,.2f}" if eq else f"${cash:,.2f}",
        f"{((eq.equity if eq else cash) / starting - 1):+.2%}",
    )
    cols[3].metric("Positions", str(len(positions)))

    st.subheader("Open positions")
    if eq and eq.positions:
        rows = []
        for sym, snap in eq.positions.items():
            rows.append(
                {
                    "Symbol": sym,
                    "Qty": snap["qty"],
                    "Avg cost": snap["avg_cost"],
                    "Mark": snap["mark"],
                    "Unrealized P&L": snap["unrealized_pnl"],
                    "Realized P&L": snap["realized_pnl"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.caption("No positions yet.")

    st.subheader("Suggested orders from active alerts")
    with loading("Reading alerts log…"):
        rows = tail_alerts(limit=20)
    if not rows:
        st.markdown(
            status_pill("no alerts logged yet — run a scan from the home page", "muted"),
            unsafe_allow_html=True,
        )
        return

    risk_per = settings.risk_per_trade
    free = cash + (eq.market_value if eq else 0.0)

    for r in rows:
        spot = float(r.get("spot", 0))
        stop = float(r.get("stop", 0))
        if spot <= 0 or stop <= 0:
            continue
        risk_dollar = free * risk_per
        risk_per_share = abs(spot - stop)
        qty = round(risk_dollar / max(1e-6, risk_per_share), 6)
        with st.container(border=True):
            c = st.columns([2, 1, 1, 1, 2])
            c[0].markdown(f"**{r.get('symbol')}** {r.get('timeframe')} — {r.get('action')}")
            c[1].metric("Confidence", f"{r.get('confidence', 0):.0f}%")
            c[2].metric("Spot", f"${spot:,.2f}")
            c[3].metric("Suggested qty", f"{qty}")
            with c[4]:
                action_buy = st.button(f"Paper-buy {qty} {r.get('symbol')}", key=f"b-{r.get('hash','')}-{spot}")
                if action_buy:
                    try:
                        book.place_order(
                            r.get("symbol"),
                            "buy",
                            qty,
                            spot,
                            note=f"alert {r.get('hash','')}",
                        )
                        st.success("Filled.")
                        st.rerun()
                    except InsufficientFunds as e:
                        st.error(str(e))


main()
