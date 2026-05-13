"""Premortem — assume the trade already failed, then ask why.

Standalone page where you can paste any investment idea (a signal, a council
verdict, a thesis, an open position) and get back three ranked failure modes,
the biggest hidden assumption, a revised plan, and a 5-item pre-launch
checklist. Tuned for short investment windows (intraday → swing → position).

The engine prefers Claude (Haiku) when ANTHROPIC_API_KEY is set, and falls
back to a horizon-tuned heuristic when it isn't. Same output shape either way.
"""

from __future__ import annotations

import streamlit as st

from app._premortem_panel import render_premortem_panel
from app._shared import setup_page
from app._ui import inject_global_css


EXAMPLES: dict[str, dict[str, str]] = {
    "BTC swing breakout": {
        "horizon": "swing",
        "title": "Long BTC-USD on 4h breakout above 72k",
        "plan": (
            "Setup: BTC 4h closed above 72,000 resistance with above-average volume. "
            "RSI 58, MACD just crossed up. Regime classifier = TREND. "
            "Entry: market on confirmation candle close. "
            "Stop: 70,500 (1.2× ATR below entry). "
            "Target: 76,000 (next swing high). "
            "Size: 1% of bankroll risked to stop. "
            "Edge thesis: clean horizontal break with momentum after a 3-week base."
        ),
    },
    "Intraday SPY reversal": {
        "horizon": "intraday",
        "title": "Short SPY into resistance for a same-day mean-revert fade",
        "plan": (
            "SPY printing a wick at 580 prior-day high on the 5m chart. VWAP at 578.5. "
            "Entry: 579.80 short, stop 580.60, target 578.50 (VWAP). "
            "Risk: 0.4% of bankroll. Plan to hold ≤ 90 minutes. "
            "Edge: prior-day high + extended RSI on 5m + thin volume on the push."
        ),
    },
    "Kalshi BTC daily YES": {
        "horizon": "intraday",
        "title": "Kalshi YES — BTC closes above 71k tonight",
        "plan": (
            "Spot 71,420. Model probability 64%, book ask 52¢ → 12pp edge. "
            "Realised σ/min low. Triangulation: 4/5 agree. Liquidity OK. "
            "Plan: buy YES at 52¢, size 3% of bankroll Kelly, hold to settlement."
        ),
    },
    "Long-term ETH thesis": {
        "horizon": "long",
        "title": "Accumulate ETH for 12–18 months — staking + L2 narrative",
        "plan": (
            "Thesis: ETH cashflows from staking + restaking grow; L2 fees re-accrue; "
            "ETF flows compound. Plan: DCA 5% of bankroll/month for 12 months, hold. "
            "No explicit stop. Re-evaluate quarterly."
        ),
    },
}


def main() -> None:
    setup_page("Premortem", icon="🩺")
    inject_global_css()

    st.caption(
        "Klein's premortem applied to short-horizon investment decisions. "
        "Paste your plan, pick the window, and Claude returns the chain of events "
        "that would have caused this trade to lose — plus what to fix before you click buy."
    )

    with st.sidebar:
        st.subheader("Premortem")
        st.markdown(
            "**How to use**\n\n"
            "1. Paste the idea — signal, thesis, council verdict, open position.\n"
            "2. Pick the horizon (intraday → long).\n"
            "3. Hit *Premortem this*.\n\n"
            "Outputs:\n"
            "- 3 ranked failure modes\n"
            "- The biggest hidden assumption\n"
            "- A revised plan\n"
            "- A 5-item pre-launch checklist"
        )
        st.divider()
        example_key = st.selectbox(
            "Load example",
            ["—"] + list(EXAMPLES.keys()),
            index=0,
            help="Pre-fill the form with a worked example. The textarea remains editable.",
        )

    default_title = ""
    default_plan = ""
    default_horizon = "swing"
    if example_key != "—":
        ex = EXAMPLES[example_key]
        default_title = ex["title"]
        default_plan = ex["plan"]
        default_horizon = ex["horizon"]   # type: ignore[assignment]

    render_premortem_panel(
        key_prefix="premortem-page",
        default_title=default_title,
        default_plan=default_plan,
        default_horizon=default_horizon,   # type: ignore[arg-type]
        compact=False,
        expanded=True,
        intro=(
            "Be specific. Generic plans get generic premortems; the technique "
            "earns its keep when the analysis is concrete enough to be wrong."
        ),
    )


main()
