"""Spy- home — live signal feed.

Tails `~/.monte/alerts.jsonl` (the file written by `monte.alerts.engine`) and
optionally runs a one-shot scan in-process so you can use the dashboard
without a separate worker.
"""

from __future__ import annotations

import time

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from app._shared import setup_page, sidebar_watchlists
from app._ui import (
    act_now_banner,
    action_pill,
    drawdown_gauge,
    freshness_pill,
    inject_global_css,
    loading,
    pnl_strip,
    status_pill,
    target_progress,
    tier_pill,
)
from monte.alerts.engine import scan_once, tail_alerts
from monte.broker.ledger import build_summary, monthly_realised
from monte.broker.paper_book import PaperBook
from monte.config import settings


def _render_alert_row(r: dict) -> None:
    with st.container(border=True):
        cols = st.columns([2.6, 1, 1, 1, 3])
        action = r.get("action", "HOLD")
        tier = r.get("tier", "")
        head = (
            f"### {r.get('symbol')} "
            f"<span class='spy-meta'>{r.get('timeframe')}</span><br/>"
        )
        if tier:
            head += (
                f"{tier_pill(tier, r.get('confidence', 0))} "
                f"{action_pill(action)} "
            )
        else:
            head += f"{action_pill(action, r.get('confidence', 0))} "
        head += freshness_pill(r.get("ts"))
        cols[0].markdown(head, unsafe_allow_html=True)
        cols[1].metric("Confidence", f"{r.get('confidence', 0):.0f}%")
        cols[2].metric("Spot", f"${r.get('spot', 0):,.2f}")
        cols[3].metric("R:R", f"{r.get('rr', 0):.2f}")
        with cols[4]:
            st.caption(
                f"Entry **${r.get('entry', 0):,.2f}** · "
                f"Stop **${r.get('stop', 0):,.2f}** · "
                f"Target **${r.get('target', 0):,.2f}**"
            )
            reasoning = r.get("reasoning")
            if reasoning:
                st.caption(f"💡 {reasoning}")
            contribs = r.get("contributions", [])
            if contribs:
                chips = " · ".join(
                    f"{c.get('name')} {c.get('score', 0):+.2f}"
                    for c in contribs
                    if abs(c.get("score", 0)) > 0.05
                )
                st.caption(chips or "All contributions near zero.")
            options = r.get("options_ticket")
            if options:
                st.caption(
                    f"📈 Options: **{options.get('side','')} ${options.get('strike',0):.0f}** "
                    f"{options.get('expiry','')} · premium ~${options.get('premium',0):.2f} "
                    f"(max risk ${options.get('max_risk_per_contract',0):.0f}/contract)"
                )


def main() -> None:
    setup_page("Live Signals", icon="🎯")
    inject_global_css()

    crypto, stocks = sidebar_watchlists()

    with st.sidebar:
        st.subheader("Scan")
        timeframe = st.selectbox(
            "Timeframe",
            ["1m", "5m", "15m", "30m", "1h", "1d"],
            index=2,  # default 15m — fastest yfinance offers reliably
            help="1m = ultra-short. yfinance caps 1m history at 7 days.",
        )
        refresh_secs = st.select_slider(
            "Auto-refresh",
            options=[5, 15, 30, 60, 120],
            value=30,
            help="How often the page polls for new alerts.",
        )
        st_autorefresh(interval=refresh_secs * 1000, key="signals_refresh")
        min_conf = st.slider(
            "Min confidence",
            0,
            100,
            int(settings.min_confidence_alert),
            help="Alerts below this confidence are suppressed.",
        )
        run_inline = st.button("Run scan now", type="primary")
        show_holds = st.checkbox(
            "Show HOLD / non-actionable rows",
            value=True,
            help="Even when no pattern fires, show the symbol so you can "
            "see data is flowing.",
        )
        show_errors = st.checkbox(
            "Show fetch errors",
            value=True,
            help="Surface rows where the data provider (yfinance) failed.",
        )

    st.markdown(
        f"{status_pill(f'auto-refresh {refresh_secs}s', 'info')} "
        f"{status_pill(f'watching {len(crypto) + len(stocks)} symbols', 'muted')} "
        f"{status_pill(f'{timeframe} · min conf {min_conf}%', 'muted')}",
        unsafe_allow_html=True,
    )
    st.caption(
        "Mean-reversion dip/pump detector triangulating RSI / MACD / Bollinger /"
        " regime / Monte Carlo zone / vector-pattern similarity. **No automated"
        " execution — signals only.**"
    )

    # Realised P&L strip (DoD / WoW / MoM / YTD) + drawdown gauge.
    try:
        from datetime import datetime, timezone

        book = PaperBook(state_path=settings.paper_state_path)
        now = time.time()
        year_start = datetime(datetime.now(timezone.utc).year, 1, 1, tzinfo=timezone.utc)
        pnl_strip(
            daily=book.daily_pnl(now),
            weekly=book.weekly_pnl(now),
            monthly=book.monthly_pnl(now),
            ytd=book._realised_since(year_start.timestamp()),
        )
        drawdown_gauge(book.current_drawdown())
    except Exception:
        pass

    # Monthly P&L vs target — informational tracker.
    try:
        book = PaperBook(state_path=settings.paper_state_path)
        summary = build_summary(book.trades())
        realised_month = monthly_realised(summary.rows, ts_now=time.time())
        target_progress(realised_month, target=float(settings.monthly_target_usd))
    except Exception:
        # If the paper book can't be read for any reason, skip the tracker —
        # it's a motivator, not a critical path.
        pass

    # Auto-trigger a scan on first load when nothing has been logged yet,
    # so the user sees data immediately instead of an empty-state message.
    existing = tail_alerts(limit=1)
    auto_scan = (not existing) and not st.session_state.get("auto_scan_done")
    if auto_scan:
        st.session_state["auto_scan_done"] = True

    if run_inline or auto_scan:
        symbols = crypto + stocks
        label = "Initialising — running first scan…" if auto_scan else f"Scanning {len(symbols)} symbols on {timeframe}…"
        with loading(label):
            t0 = time.time()
            try:
                fresh = scan_once(
                    symbols,
                    timeframes=[timeframe],
                    min_confidence=float(min_conf),
                )
                st.success(
                    f"Scan complete in {time.time() - t0:.1f}s — "
                    f"{len(fresh)} alert(s) above threshold."
                )
            except Exception as e:
                st.error(f"Scan error: {e}")

    rows = tail_alerts(limit=200)
    if not rows:
        st.warning(
            "**No data yet.** The provider (yfinance) returned nothing on this "
            "scan — usually a transient rate-limit on Replit. Click **Run scan "
            "now** in the sidebar to retry, or switch timeframe to 15m / 1h. "
            "Background worker: `python -m monte.alerts.engine`."
        )
        return

    actionable = [r for r in rows if r.get("confidence", 0) >= min_conf]
    holds = [r for r in rows if r.get("confidence", 0) < min_conf]
    actionable.sort(key=lambda r: (-r.get("confidence", 0), -r.get("ts", 0)))
    holds.sort(key=lambda r: -r.get("ts", 0))

    # Loud ACT_NOW banner — pick the most recent ACT_NOW signal in the last hour.
    act_now_rows = [
        r for r in actionable
        if r.get("tier") == "ACT_NOW" and r.get("ts", 0) >= time.time() - 3600
    ]
    if act_now_rows:
        act_now_banner(act_now_rows[0])

    st.subheader(f"Actionable ({len(actionable)})")
    if not actionable:
        st.markdown(
            status_pill(
                "no actionable patterns above threshold — pipeline still flowing",
                "muted",
            ),
            unsafe_allow_html=True,
        )
    for r in actionable[:30]:
        _render_alert_row(r)

    if show_holds and holds:
        st.subheader(f"Recent HOLD / below-threshold ({len(holds)})")
        st.caption("These show data is flowing even when nothing is firing.")
        for r in holds[:15]:
            _render_alert_row(r)


if __name__ == "__main__":
    main()
