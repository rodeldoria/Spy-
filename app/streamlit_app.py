"""Spy- home — live signal feed.

Tails `~/.monte/alerts.jsonl` (the file written by `monte.alerts.engine`) and
optionally runs a one-shot scan in-process so you can use the dashboard without
a separate worker.
"""

from __future__ import annotations

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from app._shared import action_color, setup_page, sidebar_watchlists
from monte.alerts.engine import scan_once, tail_alerts
from monte.config import settings


def main() -> None:
    setup_page("Live Signals", icon="🎯")

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

    st.caption(
        "Mean-reversion dip/pump detector triangulating RSI / MACD / Bollinger /"
        " regime / Monte Carlo zone / vector-pattern similarity. No automated"
        " execution — signals only."
    )

    if run_inline:
        with st.spinner(f"Scanning {len(crypto) + len(stocks)} symbols on {timeframe}..."):
            try:
                fresh = scan_once(
                    crypto + stocks,
                    timeframes=[timeframe],
                    min_confidence=float(min_conf),
                )
                st.success(f"Scan complete — {len(fresh)} alerts above threshold.")
            except Exception as e:
                st.error(f"Scan error: {e}")

    rows = tail_alerts(limit=50)
    if not rows:
        st.info(
            "No alerts yet. Click **Run scan now** in the sidebar, or start the"
            " scanner via `python -m monte.alerts.engine` (background worker)."
        )
        return

    rows = [r for r in rows if r.get("confidence", 0) >= min_conf]
    rows.sort(key=lambda r: (-r.get("confidence", 0), -r.get("ts", 0)))

    for r in rows[:30]:
        color = action_color(r.get("action", ""))
        with st.container(border=True):
            cols = st.columns([2, 1, 1, 1, 3])
            cols[0].markdown(
                f"### <span style='color:{color}'>{r.get('action')}</span> "
                f"&nbsp;{r.get('symbol')} {r.get('timeframe')}",
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


if __name__ == "__main__":
    main()
