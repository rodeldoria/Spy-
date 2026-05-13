"""Compact Streamlit widget that renders a premortem on any investment idea.

Designed to slot into existing pages without taking over the layout: a single
expander → horizon picker → "Premortem this" button → tight result block
with failure modes, hidden assumption, revised plan, and pre-launch
checklist. All HTML is inline so the widget works inside another expander
or column without breaking layout.
"""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from monte.intel.premortem import (
    HORIZON_LABELS,
    FailureMode,
    Horizon,
    PremortemResult,
    premortem,
)

_HORIZON_KEYS: list[Horizon] = ["intraday", "swing", "position", "long"]

_SEV_COLOR = {
    "low": "#22c55e",
    "medium": "#eab308",
    "high": "#f97316",
    "critical": "#ef4444",
}


def render_premortem_panel(
    *,
    key_prefix: str,
    default_title: str = "",
    default_plan: str = "",
    default_horizon: Horizon = "swing",
    context: Optional[dict[str, Any]] = None,
    compact: bool = True,
    expanded: bool = False,
    intro: str = "",
) -> Optional[PremortemResult]:
    """Render a premortem widget. Returns the latest result (or None).

    Caller owns `key_prefix` — it must be unique per page-instance to keep
    Streamlit's session_state slots from colliding.
    """
    state_key = f"premortem::{key_prefix}"
    result: Optional[PremortemResult] = st.session_state.get(state_key)

    container = st.expander(
        "🩺 Premortem this idea — \"why will this trade have failed?\"",
        expanded=expanded,
    ) if compact else st.container(border=True)

    with container:
        if intro:
            st.caption(intro)
        else:
            st.caption(
                "Klein's premortem trick: assume the trade already lost, then ask "
                "Claude why. Returns 3 ranked failure modes, the hidden assumption, "
                "and a 5-item pre-launch checklist — tuned for short investment windows."
            )

        col_h, col_ai = st.columns([2, 1])
        with col_h:
            horizon = st.selectbox(
                "Horizon",
                _HORIZON_KEYS,
                index=_HORIZON_KEYS.index(default_horizon),
                format_func=lambda h: HORIZON_LABELS[h],
                key=f"{key_prefix}::horizon",
                help="Tunes the failure-mode lens. Intraday weights liquidity / news; "
                "swing weights regime; position weights thesis decay; long weights structural risk.",
            )
        with col_ai:
            enable_ai = st.toggle(
                "Use Claude",
                value=True,
                key=f"{key_prefix}::ai",
                help="When off (or no ANTHROPIC_API_KEY), falls back to a deterministic "
                "horizon-specific heuristic.",
            )

        title = st.text_input(
            "One-line title",
            value=default_title,
            placeholder="e.g. Long BTC on 4h breakout above 72k",
            key=f"{key_prefix}::title",
        )
        plan = st.text_area(
            "Plan / analysis",
            value=default_plan,
            height=160,
            placeholder=(
                "What you're trading, why, entry / stop / target, sizing, "
                "and the edge you think you have. The more specific, the better the premortem."
            ),
            key=f"{key_prefix}::plan",
        )

        col_btn, col_status = st.columns([1, 3])
        run_clicked = col_btn.button(
            "🩺 Premortem this",
            key=f"{key_prefix}::run",
            type="primary",
            use_container_width=True,
            disabled=not plan.strip(),
        )
        if run_clicked:
            with st.spinner("Asking the risk officer to imagine this trade already lost…"):
                result = premortem(
                    title=title,
                    plan=plan,
                    horizon=horizon,
                    context=context,
                    enable_ai=enable_ai,
                )
            st.session_state[state_key] = result

        if result is None:
            col_status.caption("Fill in the plan and click *Premortem this*.")
        elif not result.failure_modes:
            col_status.warning(
                f"No failure modes returned ({result.ai_error or 'unknown reason'})."
            )

        if result and result.failure_modes:
            _render_result(result)

    return result


def _render_result(result: PremortemResult) -> None:
    src = result.source
    badge_color, badge_text = (
        ("#8b5cf6", f"🤖 CLAUDE · {result.model or 'haiku'}")
        if src == "ai"
        else ("#94a3b8", "🧮 HEURISTIC (no AI)")
    )
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"margin:8px 0 4px;'>"
        f"<div style='color:#cbd5e1;font-size:0.85rem;'>"
        f"<strong>Premortem result</strong> · {HORIZON_LABELS[result.horizon]}"
        f"</div>"
        f"<span style='color:{badge_color};background:rgba(255,255,255,0.04);"
        f"font-size:0.7rem;padding:2px 8px;border-radius:8px;font-weight:700;'>"
        f"{badge_text} · {result.elapsed_ms} ms</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if result.ai_error and src == "heuristic":
        st.caption(f"AI fell back to heuristic: {result.ai_error}")

    # Biggest hidden assumption — the headline.
    st.markdown(
        f"<div style='padding:10px 12px;margin:6px 0 10px;background:#0b1220;"
        f"border-left:4px solid #f59e0b;border-radius:6px;'>"
        f"<div style='color:#fbbf24;font-size:0.72rem;font-weight:700;'>"
        f"🎯 BIGGEST HIDDEN ASSUMPTION</div>"
        f"<div style='color:#e2e8f0;font-size:0.88rem;margin-top:2px;'>"
        f"{result.biggest_hidden_assumption}</div></div>",
        unsafe_allow_html=True,
    )

    # Failure modes, ranked.
    ranked = sorted(result.failure_modes, key=lambda f: -f.risk_score)
    cells: list[str] = []
    for i, m in enumerate(ranked, 1):
        cells.append(_failure_mode_html(i, m))
    st.markdown(
        "<div style='display:flex;flex-direction:column;gap:6px;'>"
        + "".join(cells)
        + "</div>",
        unsafe_allow_html=True,
    )

    # Revised plan + checklist side-by-side on wide screens.
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "<div style='color:#94a3b8;font-size:0.72rem;font-weight:700;"
            "margin-top:10px;'>✏️ REVISED PLAN</div>",
            unsafe_allow_html=True,
        )
        for line in result.revised_plan:
            st.markdown(
                f"<div style='font-size:0.84rem;color:#e2e8f0;margin:3px 0;'>• {line}</div>",
                unsafe_allow_html=True,
            )
    with col_b:
        st.markdown(
            "<div style='color:#94a3b8;font-size:0.72rem;font-weight:700;"
            "margin-top:10px;'>✅ PRE-LAUNCH CHECKLIST</div>",
            unsafe_allow_html=True,
        )
        for i, item in enumerate(result.prelaunch_checklist, 1):
            st.markdown(
                f"<div style='font-size:0.84rem;color:#e2e8f0;margin:3px 0;'>"
                f"<span style='color:#94a3b8;'>{i}.</span> {item}</div>",
                unsafe_allow_html=True,
            )


def _failure_mode_html(rank: int, m: FailureMode) -> str:
    lh = _SEV_COLOR[m.likelihood]
    dg = _SEV_COLOR[m.danger]
    return (
        f"<div style='padding:8px 11px;background:#111827;border:1px solid #1f2937;"
        f"border-left:4px solid {dg};border-radius:6px;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<div style='color:#e2e8f0;font-size:0.86rem;font-weight:700;'>"
        f"#{rank} · {m.name}</div>"
        f"<div style='font-size:0.68rem;'>"
        f"<span style='color:{lh};font-weight:700;'>likelihood: {m.likelihood}</span>"
        f"&nbsp;·&nbsp;"
        f"<span style='color:{dg};font-weight:700;'>danger: {m.danger}</span>"
        f"</div></div>"
        f"<div style='color:#cbd5e1;font-size:0.8rem;margin-top:3px;line-height:1.35;'>"
        f"<strong>Chain:</strong> {m.chain}</div>"
        f"<div style='color:#cbd5e1;font-size:0.78rem;margin-top:3px;'>"
        f"<strong>Hidden assumption:</strong> {m.hidden_assumption}</div>"
        f"<div style='color:#cbd5e1;font-size:0.78rem;margin-top:3px;'>"
        f"<strong>Early warning:</strong> {m.early_warning}</div>"
        f"</div>"
    )


__all__ = ["render_premortem_panel"]
