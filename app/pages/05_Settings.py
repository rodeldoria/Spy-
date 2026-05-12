"""Settings — read current config and triangulation weights. Editing is
session-only here (persisting requires writing to the .env file you control)."""

from __future__ import annotations

import streamlit as st

from app._shared import setup_page
from monte.config import settings


def main() -> None:
    setup_page("Settings", icon="⚙️")

    st.markdown("Edits here apply for this Streamlit session only. Persist them by editing your `.env`.")

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Watchlists (current)")
        st.code(", ".join(settings.crypto_watchlist), language="text")
        st.code(", ".join(settings.stock_watchlist), language="text")
    with cols[1]:
        st.subheader("Paths")
        st.code(str(settings.vector_db_path))
        st.code(str(settings.paper_state_path))
        st.code(str(settings.alerts_log_path))

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

    with st.expander("Integrations"):
        st.write("Anthropic (AI sentiment):", "✅" if settings.anthropic_configured else "—")
        st.write("Perplexity (live tape):", "✅" if settings.perplexity_configured else "—")
        st.write("Alpaca (SPY paper):", "✅" if settings.alpaca_configured else "—")


main()
