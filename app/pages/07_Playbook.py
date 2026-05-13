"""Claude's playbook — replay every Monte Edge signal with reasoning.

This is the "track your own pattern buying so I can copy your tactics" view.
Every time the scanner fires a WATCH or ACT_NOW signal, a row is appended
to `~/.monte/playbook.jsonl` with the reasoning, indicator snapshot, and
(for SPY ACT_NOW) the suggested options ticket.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from app._shared import setup_page
from app._ui import inject_global_css, status_pill, tier_pill
from monte.strategy.playbook import list_playbook


def _ago(ts: float) -> str:
    if not ts:
        return ""
    age = max(0.0, time.time() - float(ts))
    if age < 60:
        return f"{int(age)}s ago"
    if age < 3600:
        return f"{int(age // 60)}m ago"
    if age < 86400:
        return f"{int(age // 3600)}h ago"
    return f"{int(age // 86400)}d ago"


def main() -> None:
    setup_page("Claude's Playbook", icon="📓")
    inject_global_css()

    st.markdown(
        "Every WATCH / ACT_NOW signal Claude generates is logged here with "
        "the **reasoning** behind it, the indicator snapshot, and the suggested "
        "options ticket (SPY only). Use this to learn the pattern."
    )

    with st.sidebar:
        st.subheader("Filter")
        symbol = st.text_input("Symbol (optional)", value="")
        tier_choice = st.selectbox("Tier", ["all", "ACT_NOW", "WATCH"], index=0)
        limit = st.slider("Rows", 10, 500, 100)

    rows = list_playbook(
        limit=limit,
        symbol=symbol.strip().upper() or None,
        tier=None if tier_choice == "all" else tier_choice,
    )

    if not rows:
        st.info(
            "No signals logged yet. Run a scan from the home page or trigger "
            "the worker (`python -m monte.alerts.engine`) so Claude has plays "
            "to record."
        )
        return

    # Tier breakdown chip-strip.
    by_tier: dict[str, int] = {}
    for r in rows:
        by_tier[r.tier] = by_tier.get(r.tier, 0) + 1
    chips = " ".join(
        tier_pill(t, None) + f" <span class='spy-meta'>×{by_tier[t]}</span>"
        for t in ("ACT_NOW", "WATCH")
        if t in by_tier
    )
    st.markdown(chips, unsafe_allow_html=True)

    for r in rows:
        with st.container(border=True):
            head_cols = st.columns([3, 1, 1, 1])
            ts_iso = datetime.fromtimestamp(r.ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            head_cols[0].markdown(
                f"### {r.symbol} <span class='spy-meta'>{r.timeframe} · {ts_iso} UTC · {_ago(r.ts)}</span>",
                unsafe_allow_html=True,
            )
            head_cols[1].markdown(tier_pill(r.tier, r.confidence), unsafe_allow_html=True)
            head_cols[2].metric("Score", f"{r.score:+.2f}")
            head_cols[3].metric("R:R", f"{r.rr:.2f}")

            st.markdown(f"**Action:** {r.action.replace('_',' ')} · **{r.horizon}**")
            st.info(f"💡 {r.reasoning}")
            st.caption(
                f"Entry **${r.entry:,.2f}** · Stop **${r.stop:,.2f}** · "
                f"Target **${r.target:,.2f}** · Confluence **{r.confluence}/5** · "
                f"Macro: {r.macro_note}"
            )
            snap = r.indicator_snapshot or {}
            if snap:
                cells = []
                for k in ("rsi", "bb_pctb", "macd_hist", "adx", "atr_pct"):
                    if k in snap:
                        cells.append(f"{k.upper()} {snap[k]:+.3f}")
                if cells:
                    st.caption(" · ".join(cells))
            if r.options_ticket:
                opt = r.options_ticket
                st.caption(
                    f"📈 Options: **{opt.get('side','')} ${opt.get('strike',0):.0f}** "
                    f"{opt.get('expiry','')} · premium ~${opt.get('premium',0):.2f} · "
                    f"breakeven ${opt.get('breakeven',0):.2f} · "
                    f"max risk ${opt.get('max_risk_per_contract',0):.0f}/contract"
                )

main()
