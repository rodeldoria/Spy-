"""Budget input → suggested position sizing per active alert + simulated paper
book P&L. No real broker calls — everything goes to `~/.monte/paper/`."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app._shared import live_price, setup_page
from app._ui import (
    drawdown_gauge,
    inject_global_css,
    loading,
    pnl_strip,
    status_pill,
    target_progress,
    tier_pill,
)
from monte import journal
from monte.alerts.engine import tail_alerts
from monte.broker.paper_book import InsufficientFunds, PaperBook
from monte.config import settings
from monte.strategy.goal_tracker import (
    GoalConfig,
    on_pace,
    progress_pct,
    suggested_risk_pct,
)


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
    # Auto-migrate legacy $10k books to the new $500 default — but only when
    # there are no trades yet, so we never wipe an in-flight book.
    if (
        not book.trades()
        and not book.positions()
        and book.starting_budget() >= 9_999
        and float(settings.budget_usd) <= 1_000
    ):
        book.reset(budget=float(settings.budget_usd))
        st.toast(f"Migrated legacy $10k paper book to ${settings.budget_usd:,.0f}.")

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

    # DoD / WoW / MoM / YTD realised PnL strip + drawdown gauge.
    import time as _time
    from datetime import datetime, timezone

    now_ts = _time.time()
    year_start = datetime(datetime.now(timezone.utc).year, 1, 1, tzinfo=timezone.utc)
    pnl_strip(
        daily=book.daily_pnl(now_ts),
        weekly=book.weekly_pnl(now_ts),
        monthly=book.monthly_pnl(now_ts),
        ytd=book._realised_since(year_start.timestamp()),
    )
    current_eq = eq.equity if eq else cash
    current_dd = (current_eq - starting) / starting if starting > 0 and current_eq < starting else 0.0
    drawdown_gauge(current_dd)

    # Goal tracker — required return + on-pace status.
    cfg = GoalConfig.from_env()
    status = on_pace(current_eq, cfg)
    g_cols = st.columns(4)
    g_cols[0].metric(
        "Target",
        f"${cfg.target_usd:,.0f}",
        f"by {cfg.deadline.isoformat()}",
    )
    g_cols[1].metric(
        "Required / week",
        f"{status.weekly_required * 100:.2f}%",
        f"{status.monthly_required * 100:.2f}%/mo",
    )
    g_cols[2].metric("Days left", f"{status.days_remaining}")
    g_cols[3].metric(
        "Progress",
        f"{progress_pct(current_eq, cfg) * 100:.1f}%",
        ("on pace" if status.on_pace else "behind pace"),
    )
    pace_kind = "ok" if status.on_pace else "warn"
    st.markdown(
        status_pill(
            f"goal · ${cfg.start_usd:,.0f} → ${cfg.target_usd:,.0f} · "
            f"{'on pace' if status.on_pace else 'requires push'}",
            pace_kind,
        ),
        unsafe_allow_html=True,
    )

    # Monthly P&L vs the dashboard target.
    from monte.broker.ledger import build_summary, monthly_realised
    summary = build_summary(book.trades())
    realised_month = monthly_realised(summary.rows, ts_now=_time.time())
    target_progress(realised_month, target=float(settings.monthly_target_usd))

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

    free = cash + (eq.market_value if eq else 0.0)

    for r in rows:
        spot = float(r.get("spot", 0))
        stop = float(r.get("stop", 0))
        if spot <= 0 or stop <= 0:
            continue
        confidence = float(r.get("confidence", 0))
        risk_per = suggested_risk_pct(current_eq, starting, confidence, cfg)
        risk_dollar = free * risk_per
        risk_per_share = abs(spot - stop)
        qty = round(risk_dollar / max(1e-6, risk_per_share), 6)
        tier = r.get("tier", "")
        with st.container(border=True):
            c = st.columns([2, 1, 1, 1, 2])
            head = f"**{r.get('symbol')}** {r.get('timeframe')} — {r.get('action')}"
            if tier:
                head += f"<br/>{tier_pill(tier, confidence)}"
            c[0].markdown(head, unsafe_allow_html=True)
            c[1].metric("Risk per trade", f"{risk_per * 100:.2f}%", f"${risk_dollar:,.2f}")
            c[2].metric("Spot", f"${spot:,.2f}")
            c[3].metric("Suggested qty", f"{qty}")
            with c[4]:
                if r.get("reasoning"):
                    st.caption(f"💡 {r['reasoning']}")
                action_buy = st.button(f"Paper-buy {qty} {r.get('symbol')}", key=f"b-{r.get('hash','')}-{spot}")
                if action_buy:
                    try:
                        snap = r.get("indicator_snapshot") or {}
                        entry = journal.record_entry(
                            symbol=r.get("symbol"),
                            timeframe=r.get("timeframe", ""),
                            action=r.get("action", "BUY"),
                            horizon=r.get("horizon", ""),
                            entry=spot,
                            stop=stop,
                            target=float(r.get("target", spot)),
                            confidence=float(r.get("confidence", 0)),
                            score=float(r.get("score", 0)),
                            snapshot=snap,
                            note=f"alert {r.get('hash','')}",
                        )
                        book.place_order(
                            r.get("symbol"),
                            "buy",
                            qty,
                            spot,
                            note=f"alert {r.get('hash','')}",
                            journal_id=entry.id,
                        )
                        st.success(f"Filled · journal entry {entry.id}")
                        st.rerun()
                    except InsufficientFunds as e:
                        st.error(str(e))


main()
