"""Backtest Results — explore which setups have actually made money.

Reads from the SQLite database written by ``monte.backtest``. Five tabs:
1. Overview — engine-level KPIs (win rate, avg pnl, total trades)
2. Signal performance — drilldown by ``signal_buckets`` (RSI bands, etc.)
3. Trade ledger — filterable per-trade view
4. Runs — config history newest first
5. New run — form to kick off a fresh backtest in-process

The Trade ledger and KPIs respect the page-level filter chip selections.
"""

from __future__ import annotations

import threading
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from app._shared import setup_page
from app._ui import filter_chip_row, inject_global_css, status_pill
from monte.backtest import summaries
from monte.backtest.config import (
    BacktestConfig,
    DEFAULT_MIN_EDGE_PP,
    DEFAULT_MIN_EV_CENTS,
    DEFAULT_SEED,
    DEFAULT_TIMEOUT_BARS,
    EngineKind,
)
from monte.backtest.runner import run_engine_or_all

_ALL_ENGINES: tuple[EngineKind, ...] = ("dip_pump", "kalshi", "triangulation")


@st.cache_data(ttl=30, show_spinner=False)
def _runs(_cache_buster: int = 0) -> pd.DataFrame:
    return summaries.list_runs(limit=200)


@st.cache_data(ttl=30, show_spinner=False)
def _overview(run_ids: tuple[str, ...] | None) -> pd.DataFrame:
    return summaries.overview_kpis(run_ids=list(run_ids) if run_ids else None)


@st.cache_data(ttl=30, show_spinner=False)
def _signal_table(run_ids: tuple[str, ...] | None, engine: str | None) -> pd.DataFrame:
    return summaries.signal_table(run_ids=list(run_ids) if run_ids else None, engine=engine)


@st.cache_data(ttl=30, show_spinner=False)
def _ledger(run_ids: tuple[str, ...] | None, engine: str | None,
            symbol: str | None) -> pd.DataFrame:
    return summaries.trade_ledger(run_ids=list(run_ids) if run_ids else None,
                                  engine=engine, symbol=symbol)


def _select_run_ids(runs_df: pd.DataFrame, scope: str) -> tuple[str, ...] | None:
    """Translate the ``Run scope`` chip to a list of run_ids."""
    if runs_df.empty:
        return None
    if scope == "Latest run":
        return (runs_df.iloc[0]["run_id"],)
    if scope == "Last 7d":
        cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)).timestamp()
    elif scope == "Last 30d":
        cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)).timestamp()
    else:
        return None
    mask = runs_df["ts_started"] >= cutoff
    return tuple(runs_df.loc[mask, "run_id"].tolist()) or None


def _kick_off_run(*, engines: list[EngineKind], base: BacktestConfig) -> None:
    """Run engines in a background thread so the Streamlit page can keep
    responding. Results land in SQLite; cache invalidates on next refresh."""
    state = st.session_state.setdefault("_bt_state", {})
    if state.get("running"):
        return
    state["running"] = True
    state["log"] = "running…"

    def _target():
        try:
            results = run_engine_or_all(engines=engines, base_cfg=base)
            state["log"] = "\n".join(
                f"{r.engine}/{r.fixture_mode or '-'}: {r.status} ({r.n_trades} trades) {r.run_id}"
                for r in results
            )
            state["last_results"] = [r.__dict__ for r in results]
        except Exception as exc:
            state["log"] = f"FAILED: {exc}"
        finally:
            state["running"] = False

    threading.Thread(target=_target, daemon=True).start()


def main() -> None:
    setup_page("Backtest Results", icon="📊")
    inject_global_css()

    runs_df = _runs()
    if runs_df.empty:
        st.info(
            "No backtest runs yet. Use the **New run** tab below or run "
            "`python -m monte.backtest.run --engine all` from a shell."
        )

    # ── Page-level filter chips ───────────────────────────────────────────
    engine_chip = filter_chip_row(
        "Engine", ["dip_pump", "kalshi", "triangulation", "All"],
        state_key="bt_engine", mode="single", default="All",
    )
    fixture_chip = filter_chip_row(
        "Fixture", ["neutral", "seeded_random", "n/a"],
        state_key="bt_fixture", mode="single", default="n/a",
    )
    scope_chip = filter_chip_row(
        "Window", ["Latest run", "Last 7d", "Last 30d", "All time"],
        state_key="bt_scope", mode="single", default="All time",
    )

    selected_engine: str | None = None if engine_chip in ("All", "") else engine_chip
    selected_run_ids = _select_run_ids(runs_df, scope_chip)

    tab_over, tab_sig, tab_led, tab_runs, tab_new = st.tabs(
        ["Overview", "Signal performance", "Trade ledger", "Runs", "New run"]
    )

    with tab_over:
        df = _overview(selected_run_ids)
        if df.empty:
            st.caption("No trades match the selected scope yet.")
        else:
            if selected_engine:
                df = df[df["engine"] == selected_engine]
            if fixture_chip != "n/a":
                df = df[df["fixture_mode"].fillna("") == fixture_chip]
            if df.empty:
                st.caption("Empty after engine/fixture filters.")
            else:
                cols = st.columns(4)
                cols[0].metric("Trades", int(df["n_trades"].sum()))
                wins = int(df["wins"].sum())
                losses = int(df["losses"].sum())
                cols[1].metric("Wins / Losses", f"{wins} / {losses}")
                if (wins + losses) > 0:
                    cols[2].metric("Win rate", f"{100*wins/(wins+losses):.1f}%")
                avg_pnl = df["avg_pnl_pct"].dropna().mean() if not df["avg_pnl_pct"].dropna().empty else 0.0
                cols[3].metric("Avg pnl %", f"{avg_pnl:+.2f}%")
                st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_sig:
        df = _signal_table(selected_run_ids, selected_engine)
        if df.empty:
            st.caption("No signal-bucket rows for this scope yet.")
        else:
            st.caption(
                "Buckets are coarse on purpose — RSI bands of 10, "
                "categorical strike type, dominant vote, etc. — so each row has "
                "enough samples to mean something. Rows with **n ≥ 30 and "
                "win_rate ≥ 60** are the ones worth replicating."
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_led:
        df = _ledger(selected_run_ids, selected_engine, None)
        if df.empty:
            st.caption("No trades match the selected scope yet.")
        else:
            if fixture_chip != "n/a":
                df = df[df["fixture_mode"].fillna("") == fixture_chip]
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, file_name="backtest_trades.csv",
                               mime="text/csv")

    with tab_runs:
        if runs_df.empty:
            st.caption("No runs yet.")
        else:
            display = runs_df.copy()
            display["ts_started"] = pd.to_datetime(display["ts_started"], unit="s", utc=True)
            display["ts_finished"] = pd.to_datetime(display["ts_finished"], unit="s", utc=True)
            st.dataframe(display, use_container_width=True, hide_index=True)

    with tab_new:
        st.markdown("Kick off a fresh run. Triangulation always runs both fixture "
                    "modes (neutral + seeded_random) side by side.")
        with st.form("bt_new_run", clear_on_submit=False):
            engines_pick = st.multiselect(
                "Engines", list(_ALL_ENGINES), default=list(_ALL_ENGINES),
            )
            sym_str = st.text_input("Symbols (comma-separated)", "BTC,ETH,SOL")
            tf_str = st.text_input("Timeframes", "1h,1d")
            c1, c2, c3 = st.columns(3)
            start = c1.date_input("Start", value=date.today() - timedelta(days=180))
            end = c2.date_input("End", value=date.today())
            seed = c3.number_input("Seed", value=DEFAULT_SEED, step=1)
            c4, c5, c6 = st.columns(3)
            min_edge = c4.number_input("Min edge (pp)", value=DEFAULT_MIN_EDGE_PP, step=0.5)
            min_ev = c5.number_input("Min EV (¢)", value=DEFAULT_MIN_EV_CENTS, step=0.5)
            timeout = c6.number_input("Timeout (bars)", value=DEFAULT_TIMEOUT_BARS, step=4)
            submit = st.form_submit_button("Run backtest")

        state = st.session_state.setdefault("_bt_state", {})
        running = bool(state.get("running"))
        if submit and not running:
            if not engines_pick:
                st.warning("Pick at least one engine.")
            else:
                base = BacktestConfig(
                    engine="dip_pump",  # placeholder, replaced per engine
                    symbols=tuple(s.strip() for s in sym_str.split(",") if s.strip()),
                    timeframes=tuple(t.strip() for t in tf_str.split(",") if t.strip()),
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    seed=int(seed),
                    min_edge_pp=float(min_edge),
                    min_ev_cents=float(min_ev),
                    timeout_bars=int(timeout),
                )
                _kick_off_run(engines=engines_pick, base=base)
                st.success("Run started in background. Check the Runs tab in a few seconds.")
                st.rerun()

        if running:
            st.markdown(status_pill("running…", "info"), unsafe_allow_html=True)
        log = state.get("log")
        if log:
            st.code(log)


main()
