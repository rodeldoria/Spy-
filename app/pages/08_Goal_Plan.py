"""Goal plan — current pace vs target, projected curve, suggested risk %.

Tells you exactly what weekly/monthly return you need to hit your target
by the deadline, what risk per trade Monte Edge will suggest today given
your drawdown state, and what an idealised compounding curve looks like.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import plotly.graph_objects as go
import streamlit as st

from app._shared import live_price, setup_page
from app._ui import drawdown_gauge, inject_global_css, status_pill
from monte.broker.paper_book import PaperBook
from monte.config import settings
from monte.strategy.goal_tracker import (
    GoalConfig,
    on_pace,
    progress_pct,
    projected_equity_curve,
    required_monthly_return,
    required_weekly_return,
    suggested_risk_pct,
)


def main() -> None:
    setup_page("Goal Plan", icon="🎯")
    inject_global_css()

    cfg_default = GoalConfig.from_env()

    with st.sidebar:
        st.subheader("Goal")
        start_v = st.number_input(
            "Starting equity ($)",
            min_value=100.0,
            value=float(cfg_default.start_usd),
            step=500.0,
        )
        target_v = st.number_input(
            "Target equity ($)",
            min_value=float(start_v) + 1,
            value=float(cfg_default.target_usd),
            step=500.0,
        )
        deadline_v = st.date_input("Deadline", value=cfg_default.deadline)

    cfg = GoalConfig(
        start_usd=float(start_v),
        target_usd=float(target_v),
        deadline=deadline_v if isinstance(deadline_v, date) else cfg_default.deadline,
    )

    book = PaperBook(state_path=settings.paper_state_path)
    positions = book.positions()
    prices: dict[str, float] = {}
    for sym in positions:
        try:
            prices[sym], _ = live_price(sym)
        except Exception:
            prices[sym] = positions[sym].get("avg_cost", 1.0) if isinstance(positions[sym], dict) else 1.0
    eq = book.mark_to_market(prices)
    current_eq = eq.equity
    starting = book.starting_budget()

    status = on_pace(current_eq, cfg)

    st.subheader("Where you are")
    g = st.columns(4)
    g[0].metric("Equity", f"${current_eq:,.2f}", f"{(current_eq/starting - 1):+.2%}")
    g[1].metric(
        "Required / week",
        f"{required_weekly_return(current_eq, cfg.target_usd, cfg.deadline) * 100:.2f}%",
    )
    g[2].metric(
        "Required / month",
        f"{required_monthly_return(current_eq, cfg.target_usd, cfg.deadline) * 100:.2f}%",
    )
    g[3].metric("Days left", f"{status.days_remaining}")

    progress = progress_pct(current_eq, cfg)
    st.progress(
        progress,
        text=f"Progress · ${current_eq:,.2f} / ${cfg.target_usd:,.0f} target  "
        f"({progress * 100:.1f}%)",
    )
    pace_kind = "ok" if status.on_pace else "warn"
    st.markdown(
        status_pill(
            "on pace" if status.on_pace else "behind pace · push or extend deadline",
            pace_kind,
        ),
        unsafe_allow_html=True,
    )

    st.subheader("Risk allocation today")
    drawdown_gauge(book.current_drawdown(prices))
    risk_cols = st.columns(3)
    for label, conf, col in zip(
        ("Watch (60%)", "High conviction (80%)", "Maxed (95%)"),
        (60.0, 80.0, 95.0),
        risk_cols,
    ):
        r = suggested_risk_pct(current_eq, starting, conf, cfg)
        col.metric(label, f"{r * 100:.2f}%", f"${current_eq * r:,.2f}")
    st.caption(
        "Risk per trade scales 0.5%–1.5% with conviction. Halves at -5% drawdown, "
        "halts new entries at -10%. No exception, no revenge trades."
    )

    st.subheader("Projected curve")
    weekly_req = required_weekly_return(current_eq, cfg.target_usd, cfg.deadline)
    weeks_left = max(1, status.days_remaining // 7)
    curve = projected_equity_curve(current_eq, weekly_req, weeks_left)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[w for w, _ in curve],
        y=[e for _, e in curve],
        mode="lines",
        name="Required path",
        line=dict(width=3, color="#0a7d2a"),
    ))
    fig.add_hline(
        y=cfg.target_usd, line_dash="dash", line_color="#a16207",
        annotation_text=f"Target ${cfg.target_usd:,.0f}", annotation_position="top left",
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Weeks ahead",
        yaxis_title="Equity ($)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Framework rules")
    st.markdown(
        """
        - **Macro filter** — longs only when SPY > 200-day SMA; puts/shorts only below.
        - **Confluence ≥ 3 of 5** — RSI, MACD, BB %b, Trend, Regime must agree.
        - **Confidence-scaled risk** — 0.5% to 1.5% per trade, scaled by conviction.
        - **ATR stops** — 1.5× ATR initial stop, breakeven at 1R, trail by 2× ATR after 2R.
        - **Drawdown brake** — halve risk at -5%, halt new entries at -10% until equity makes a new high.
        """
    )
    st.caption(
        "This is a research and paper-trading framework. Not financial advice. "
        "Past performance — even in backtests — does not guarantee future results."
    )


main()
