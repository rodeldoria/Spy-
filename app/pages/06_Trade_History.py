"""Trade History — buy/sell ledger, equity curve, and pattern outcomes.

This is the "what have I actually traded, and what worked?" view. It pulls
from two stores that already exist on this branch:

  - `monte.broker.paper_book.PaperBook` — the simulated book; every
    place_order is appended to `book.trades()`.
  - `monte.journal.store` — the pattern journal that records the indicator
    snapshot at trade time and the realised pnl_pct on exit.

Realised P&L per sell is computed FIFO over the trade ledger
(`monte.broker.ledger.build_summary`). The page is read-only — no order
entry happens here.
"""
from __future__ import annotations

import time as _time
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app._premortem_panel import render_premortem_panel
from app._shared import live_price, setup_page
from app._ui import filter_chip_row, inject_global_css, status_pill, target_progress
from monte import journal
from monte.broker.ledger import build_summary, monthly_realised
from monte.broker.paper_book import InsufficientFunds, PaperBook
from monte.config import settings


def _format_ts(ts: float) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _equity_curve(equity_curve, starting: float) -> go.Figure:
    if not equity_curve:
        fig = go.Figure()
        fig.update_layout(
            height=240,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(text="No closed trades yet — equity curve will appear after your first sell.",
                     showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
            ],
        )
        return fig
    xs = [datetime.fromtimestamp(p.ts, tz=timezone.utc) for p in equity_curve]
    ys = [starting + p.realised_cum for p in equity_curve]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.add_hline(y=starting, line_dash="dash",
                  annotation_text=f"Starting ${starting:,.0f}", line_color="#5b6470")
    fig.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Equity ($)",
        xaxis_title="",
    )
    return fig


def _close_open_entries(symbol: str) -> int:
    """Mark every open journal entry for `symbol` as closed at the live mark."""
    open_for_sym = journal.open_entries(symbol=symbol)
    if not open_for_sym:
        return 0
    try:
        spot, _ = live_price(symbol)
    except Exception:
        spot = open_for_sym[0].entry
    for e in open_for_sym:
        journal.record_exit(e.id, exit_price=spot or e.entry, exit_reason="history-close")
    return len(open_for_sym)


def main() -> None:
    setup_page("Trade History", icon="📒")
    inject_global_css()

    book = PaperBook(state_path=settings.paper_state_path)
    summary = build_summary(book.trades())
    starting = book.starting_budget()

    realised_month = monthly_realised(summary.rows, ts_now=_time.time())

    # ── Top-row metrics ────────────────────────────────────────────────────
    closed_pnls = [r.realised_pnl for r in summary.rows if r.side == "sell"]
    wins = [p for p in closed_pnls if p > 0]
    losses = [p for p in closed_pnls if p < 0]
    cols = st.columns(5)
    cols[0].metric("Starting", f"${starting:,.2f}")
    cols[1].metric("Realised P&L", f"${summary.total_realised:,.2f}")
    cols[2].metric("Win rate", f"{summary.win_rate:.0f}%",
                   f"{len(wins)}W / {len(losses)}L")
    cols[3].metric("Best trade", f"${max(closed_pnls):,.2f}" if closed_pnls else "—")
    cols[4].metric("Worst trade", f"${min(closed_pnls):,.2f}" if closed_pnls else "—")

    target_progress(realised_month, target=float(settings.monthly_target_usd))

    # ── Page-level filter chips (apply to ledger + journal tabs) ──────────
    outcome_labels = ["Wins", "Losses", "Scratches", "Open"]
    outcome_chips = filter_chip_row(
        "Outcome", outcome_labels, state_key="th_outcome_chips", mode="multi",
        default=[],
    )
    date_window = filter_chip_row(
        "Window", ["7d", "30d", "90d", "All"], state_key="th_date_window",
        mode="single", default="All",
    )
    chip_outcomes = {
        "Wins": "win", "Losses": "loss", "Scratches": "scratch", "Open": "open",
    }
    selected_outcomes = {chip_outcomes[c] for c in outcome_chips}
    now_ts = _time.time()
    window_seconds = {"7d": 7 * 86400, "30d": 30 * 86400, "90d": 90 * 86400}.get(date_window)
    cutoff_ts = now_ts - window_seconds if window_seconds else 0.0

    tab_equity, tab_ledger, tab_journal, tab_premortem, tab_close = st.tabs(
        ["Equity", "Ledger", "Pattern journal", "Premortem", "Close positions"]
    )

    with tab_equity:
        st.plotly_chart(_equity_curve(summary.equity_curve, starting),
                        use_container_width=True)

    with tab_ledger:
        ledger_rows = list(summary.rows)
        if cutoff_ts:
            ledger_rows = [r for r in ledger_rows if r.ts >= cutoff_ts]
        if selected_outcomes:
            def _row_outcome(r) -> str:
                if r.side != "sell":
                    return "open"
                if r.realised_pnl > 0:
                    return "win"
                if r.realised_pnl < 0:
                    return "loss"
                return "scratch"
            ledger_rows = [r for r in ledger_rows if _row_outcome(r) in selected_outcomes]
        if not ledger_rows:
            st.caption("No paper trades match the current filters. Loosen the chips "
                       "above or open Budget & Paper Portfolio to record some.")
        else:
            df = pd.DataFrame([
                {
                    "ts": _format_ts(r.ts),
                    "symbol": r.symbol,
                    "side": r.side.upper(),
                    "qty": round(r.qty, 6),
                    "price": round(r.price, 4),
                    "realised P&L": f"${r.realised_pnl:,.2f}" if r.side == "sell" else "",
                    "note": r.note,
                    "journal_id": r.journal_id or "",
                }
                for r in reversed(ledger_rows)
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_journal:
        horizon_filter = st.multiselect(
            "Horizon",
            ["DAY_TRADE", "SWING", "LONG_HOLD"],
            default=["DAY_TRADE", "SWING", "LONG_HOLD"],
        )
        outcomes_for_query = (
            list(selected_outcomes) if selected_outcomes else ["win", "loss", "scratch", "open"]
        )
        filtered = journal.list_entries(
            outcomes=outcomes_for_query or None,
            horizons=horizon_filter or None,
        )
        if cutoff_ts:
            filtered = [e for e in filtered if (e.ts_entry or 0) >= cutoff_ts]
        if not filtered:
            st.caption("No journal entries match the current filter.")
        else:
            rows = []
            for e in reversed(filtered):
                rows.append({
                    "ts_entry": _format_ts(e.ts_entry),
                    "symbol": e.symbol,
                    "action": e.action,
                    "horizon": e.horizon,
                    "entry": round(e.entry, 4),
                    "exit": round(e.exit_price, 4) if e.exit_price else "",
                    "pnl %": f"{e.pnl_pct:+.2f}%" if e.pnl_pct is not None else "",
                    "outcome": e.outcome,
                    "rsi": round(e.snapshot.get("rsi", 0), 1) if e.snapshot else "",
                    "adx": round(e.snapshot.get("adx", 0), 1) if e.snapshot else "",
                    "note": e.note,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            closed = [e for e in filtered if e.outcome in {"win", "loss", "scratch"} and e.pnl_pct is not None]
            if closed:
                by_horizon: dict[str, list[float]] = {}
                for e in closed:
                    by_horizon.setdefault(e.horizon or "UNKNOWN", []).append(e.pnl_pct)
                st.markdown("**Horizon aggregates**")
                cols = st.columns(len(by_horizon))
                for col, (h, pnls) in zip(cols, by_horizon.items()):
                    wins_h = sum(1 for p in pnls if p > 0)
                    col.metric(
                        h.replace("_", " ").title(),
                        f"{wins_h / len(pnls) * 100:.0f}% wins",
                        f"avg {sum(pnls) / len(pnls):+.2f}% · n={len(pnls)}",
                    )

    positions = book.positions()
    with tab_premortem:
        if positions:
            sym_choices = list(positions.keys())
            chosen_sym = st.selectbox(
                "Pre-fill from open position",
                ["(blank)"] + sym_choices,
                index=0,
                help="Pick a symbol to pre-fill the plan with the current cost basis / "
                "live mark — or leave blank and type the idea yourself.",
            )
            pre_title = ""
            pre_plan = ""
            if chosen_sym != "(blank)":
                pos = positions[chosen_sym]
                try:
                    mark, _ = live_price(chosen_sym)
                except Exception:
                    mark = pos["avg_cost"]
                pnl_pct = (mark - pos["avg_cost"]) / pos["avg_cost"] * 100 if pos["avg_cost"] else 0.0
                pre_title = f"Open position: {chosen_sym}"
                pre_plan = (
                    f"Holding {pos['qty']} {chosen_sym} at avg cost ${pos['avg_cost']:,.2f}. "
                    f"Live mark ${mark:,.2f} ({pnl_pct:+.2f}%). "
                    f"State the invalidation, the time-stop, and the original edge before deciding "
                    f"to add, hold, or close."
                )
            render_premortem_panel(
                key_prefix="trade-history-premortem",
                default_title=pre_title,
                default_plan=pre_plan,
                default_horizon="swing",
                compact=True,
                expanded=True,
                intro=(
                    "Before you add to or close a position, run a quick premortem on the "
                    "thesis as it stands now. Same drill as before entry — the trade has "
                    "already failed, why?"
                ),
            )
        else:
            render_premortem_panel(
                key_prefix="trade-history-premortem",
                default_horizon="swing",
                compact=True,
                expanded=True,
                intro="Try a premortem on the next idea you're considering — paste the plan below.",
            )

    with tab_close:
        if not positions:
            st.caption("No open paper positions.")
        else:
            for sym, pos in positions.items():
                with st.container(border=True):
                    c = st.columns([2, 1, 1, 2])
                    c[0].markdown(f"**{sym}** · {pos['qty']} @ ${pos['avg_cost']:,.2f}")
                    try:
                        spot, _ = live_price(sym)
                    except Exception:
                        spot = pos["avg_cost"]
                    c[1].metric("Mark", f"${spot:,.2f}")
                    c[2].metric(
                        "Unrealised",
                        f"${(spot - pos['avg_cost']) * pos['qty']:,.2f}",
                    )
                    if c[3].button(f"Paper-sell all {sym}", key=f"sell-{sym}"):
                        try:
                            book.place_order(sym, "sell", pos["qty"], spot, note="history-close")
                            n = _close_open_entries(sym)
                            msg = f"Sold {pos['qty']} {sym} @ ${spot:,.2f}"
                            if n:
                                msg += f" · closed {n} journal entr{'y' if n == 1 else 'ies'}"
                            st.success(msg)
                            st.rerun()
                        except InsufficientFunds as e:
                            st.error(str(e))


main()
