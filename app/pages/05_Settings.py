"""Settings — read current config and triangulation weights. Editing is
session-only here (persisting requires writing to the .env file you control)."""

from __future__ import annotations

import streamlit as st

from app._shared import setup_page
from app._ui import inject_global_css, status_pill
from monte.config import settings


def main() -> None:
    setup_page("Settings", icon="⚙️")
    inject_global_css()

    st.markdown("Edits here apply for this Streamlit session only. Persist them by editing your `.env`.")

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Watchlists (current)")
        st.markdown("**Crypto**")
        st.code(", ".join(settings.crypto_watchlist) or "(empty)", language="text")
        st.markdown("**Stocks**")
        st.code(", ".join(settings.stock_watchlist) or "(empty)", language="text")
        st.caption("Edit `MONTE_CRYPTO_WATCHLIST` / `MONTE_STOCK_WATCHLIST` in `.env`.")
    with cols[1]:
        st.subheader("Paths")
        st.markdown("**Vector DB**")
        st.code(str(settings.vector_db_path), language="text")
        st.markdown("**Paper book state**")
        st.code(str(settings.paper_state_path), language="text")
        st.markdown("**Alerts log**")
        st.code(str(settings.alerts_log_path), language="text")

    st.subheader("Triangulation weights")
    weights = dict(settings.triangulation_weights)
    new = {}
    cols = st.columns(5)
    for i, (k, v) in enumerate(weights.items()):
        new[k] = cols[i].slider(k, 0.0, 1.0, float(v), 0.05)

    total = sum(new.values()) or 1.0
    st.caption(f"Sum = {total:.2f} (will be normalised on use)")
    st.session_state["triangulation_weights"] = {k: v / total for k, v in new.items()}

    st.subheader("Threshold + slippage")
    cols = st.columns(2)
    cols[0].slider(
        "Min confidence to alert",
        0,
        100,
        int(settings.min_confidence_alert),
        key="min_conf",
    )
    cols[1].slider(
        "Slippage (bps)",
        0,
        50,
        int(settings.slippage_bps),
        key="slippage_bps",
    )

    st.subheader("Integrations")
    import os as _os

    ntfy_configured = bool((_os.environ.get("MONTE_NTFY_TOPIC") or "").strip())
    integrations = [
        ("Anthropic (AI sentiment)", settings.anthropic_configured),
        ("Perplexity (live tape)", settings.perplexity_configured),
        ("Alpaca (SPY paper)", settings.alpaca_configured),
        ("ntfy push", ntfy_configured),
    ]
    pills = " ".join(
        status_pill(name + (" · connected" if ok else " · not configured"),
                    "ok" if ok else "muted")
        for name, ok in integrations
    )
    st.markdown(pills, unsafe_allow_html=True)

    st.subheader("Push notifications (ntfy.sh)")
    current_topic = _os.environ.get("MONTE_NTFY_TOPIC", "")
    st.text_input(
        "Topic name",
        value=current_topic,
        help="Pick something unguessable. Anyone with the topic can read your alerts.",
        key="ntfy_topic",
    )
    st.caption(
        "1. Install the **ntfy** app on your phone. 2. Subscribe to the same "
        "topic. 3. Save the topic into your `.env` as `MONTE_NTFY_TOPIC=...` "
        "and restart Streamlit."
    )
    if st.button("Send test push", disabled=not ntfy_configured):
        try:
            from monte.notify.ntfy import push as _push

            ok = _push("Test", "Monte Edge is live", priority="high")
            if ok:
                st.success("Test push sent — check your ntfy app.")
            else:
                st.warning("ntfy returned a non-OK response. Check the topic value.")
        except Exception as _err:
            st.error(f"Push failed: {_err}")
    if not ntfy_configured:
        st.caption("Set `MONTE_NTFY_TOPIC` in `.env` to enable this button.")

    st.subheader("Goal plan")
    cols = st.columns(3)
    cols[0].metric(
        "Start",
        f"${float(_os.environ.get('MONTE_GOAL_START_USD','10000')):,.0f}",
    )
    cols[1].metric(
        "Target",
        f"${float(_os.environ.get('MONTE_GOAL_TARGET_USD','15000')):,.0f}",
    )
    cols[2].metric("Deadline", _os.environ.get("MONTE_GOAL_DEADLINE", "2026-11-13"))
    st.caption("Edit `MONTE_GOAL_*` in your `.env` to change. The Goal Plan page reads these.")


main()
