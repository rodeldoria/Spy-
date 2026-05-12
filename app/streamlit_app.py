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
    action_pill,
    freshness_pill,
    inject_global_css,
    loading,
    status_pill,
)
from monte.alerts.engine import scan_once, tail_alerts
from monte.config import settings


def _render_alert_row(r: dict) -> None:
    with st.container(border=True):
        cols = st.columns([2.6, 1, 1, 1, 3])
        action = r.get("action", "HOLD")
        cols[0].markdown(
            f"### {r.get('symbol')} "
            f"<span class='spy-meta'>{r.get('timeframe')}</span><br/>"
            f"{action_pill(action, r.get('confidence', 0))} "
            f"{freshness_pill(r.get('ts'))}",
            unsafe_allow_html=True,
        )
        cols[1].metric("Confidence", f"{r.get('confidence', 0):.0f}%")
        cols[2].metric("Spot", f"${r.get('spot', 0):,.2f}")
        cols[3].metric("R:R", f"{r.get('rr', 0):.2f}")
        with cols[4]:
            st.caption(
                f"Entry **${r.get('entry', 0):,.2f}** · "
                f"Stop **${r.get('stop', 0):,.2f}** · "
                f"Target **${r.get('target', 0):,.2f}**"
            )
            contribs = r.get("contributions", [])
            if contribs:
                chips = " · ".join(
                    f"{c.get('name')} {c.get('score', 0):+.2f}"
                    for c in contribs
                    if abs(c.get("score", 0)) > 0.05
                )
                st.caption(chips or "All contributions near zero.")


def main() -> None:
    setup_page("Live Signals", icon="🎯")
    inject_global_css()

    st_autorefresh(interval=30_000, key="signals_refresh")
    crypto, stocks = sidebar_watchlists()

    with st.sidebar:
        st.subheader("Scan")
        timeframe = st.selectbox("Timeframe", ["1h", "15m", "5m", "1d"], index=0)
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

    st.markdown(
        f"{status_pill('auto-refresh 30s', 'info')} "
        f"{status_pill(f'watching {len(crypto) + len(stocks)} symbols', 'muted')} "
        f"{status_pill(f'min conf {min_conf}%', 'muted')}",
        unsafe_allow_html=True,
    )
    st.caption(
        "Mean-reversion dip/pump detector triangulating RSI / MACD / Bollinger /"
        " regime / Monte Carlo zone / vector-pattern similarity. **No automated"
        " execution — signals only.**"
    )

    if run_inline:
        symbols = crypto + stocks
        with loading(f"Scanning {len(symbols)} symbols on {timeframe}…"):
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
        st.info(
            "No alerts logged yet. Click **Run scan now** in the sidebar, or "
            "start the scanner via `python -m monte.alerts.engine` "
            "(background worker)."
        )
        return

    actionable = [r for r in rows if r.get("confidence", 0) >= min_conf]
    holds = [r for r in rows if r.get("confidence", 0) < min_conf]
    actionable.sort(key=lambda r: (-r.get("confidence", 0), -r.get("ts", 0)))
    holds.sort(key=lambda r: -r.get("ts", 0))

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
