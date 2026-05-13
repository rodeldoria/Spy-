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
    signal_guide,
    status_pill,
    target_progress,
    tier_pill,
)
from monte.alerts.engine import scan_once, tail_alerts
from monte.broker.auto_trade import append_auto_trade_log, run_auto_trades
from monte.broker.ledger import build_summary, monthly_realised
from monte.broker.paper_book import PaperBook
from monte.config import settings


def _render_alert_row(r: dict) -> None:
    action = r.get("action", "HOLD")
    tier = r.get("tier", "")
    sym = r.get("symbol", "?")
    tf = r.get("timeframe", "")
    conf = r.get("confidence", 0)
    spot = r.get("spot", 0)
    entry = r.get("entry", 0)
    stop = r.get("stop", 0)
    target = r.get("target", 0)
    rr = r.get("rr", 0)

    if tier:
        pills = f"{tier_pill(tier, conf)} {action_pill(action)} {freshness_pill(r.get('ts'))}"
    else:
        pills = f"{action_pill(action, conf)} {freshness_pill(r.get('ts'))}"

    metrics_html = (
        f"<div class='spy-alert-metrics'>"
        f"<div class='spy-alert-metric'><div class='label'>Conf</div><div class='value'>{conf:.0f}%</div></div>"
        f"<div class='spy-alert-metric'><div class='label'>Spot</div><div class='value'>${spot:,.2f}</div></div>"
        f"<div class='spy-alert-metric'><div class='label'>Entry</div><div class='value'>${entry:,.2f}</div></div>"
        f"<div class='spy-alert-metric'><div class='label'>Stop</div><div class='value'>${stop:,.2f}</div></div>"
        f"<div class='spy-alert-metric'><div class='label'>Target</div><div class='value'>${target:,.2f}</div></div>"
        f"<div class='spy-alert-metric'><div class='label'>R:R</div><div class='value'>{rr:.2f}</div></div>"
        f"</div>"
    )

    _tier_tips = {
        "ACT_NOW":    ("🟢", "#0a7d2a", "rgba(10,125,42,0.10)", "BUY OPPORTUNITY — enter now, check stop &amp; target, go to Budget page to size your trade."),
        "WATCH":      ("🟡", "#b45309", "rgba(180,83,9,0.10)",  "Setup forming — prepare your order but wait for ACT_NOW confirmation before entering."),
        "STAND_DOWN": ("⚪", "#991b1b", "rgba(153,27,27,0.10)", "Conditions unfavourable — do not open new trades here. Wait for the tide to turn."),
    }
    if tier in _tier_tips:
        tip_icon, tip_col, tip_bg, tip_text = _tier_tips[tier]
        st.markdown(
            f"<div style='border-left:3px solid {tip_col};background:{tip_bg};"
            f"border-radius:6px;padding:6px 10px;font-size:0.82rem;margin-bottom:6px;'>"
            f"{tip_icon} {tip_text}</div>",
            unsafe_allow_html=True,
        )

    body_parts = []
    reasoning = r.get("reasoning")
    if reasoning:
        body_parts.append(f"💡 {reasoning}")
    contribs = r.get("contributions", [])
    if contribs:
        chips = " · ".join(
            f"{c.get('name')} {c.get('score', 0):+.2f}"
            for c in contribs
            if abs(c.get("score", 0)) > 0.05
        )
        if chips:
            body_parts.append(chips)
    options = r.get("options_ticket")
    if options and not options.get("is_crypto_note"):
        body_parts.append(
            f"📈 Options: {options.get('side','')} ${options.get('strike',0):.0f} "
            f"{options.get('expiry','')} · premium ~${options.get('premium',0):.2f} "
            f"(max risk ${options.get('max_risk_per_contract',0):.0f}/contract)"
        )
    elif options and options.get("is_crypto_note"):
        body_parts.append(f"💡 {options.get('rationale','')}")

    body_html = (
        f"<div class='spy-alert-body'>"
        + "<br/>".join(body_parts)
        + "</div>"
    ) if body_parts else ""

    st.markdown(
        f"<div class='spy-alert-card'>"
        f"<div class='spy-alert-top'>"
        f"<div class='spy-alert-symbol'>{sym} <span class='spy-meta'>{tf}</span></div>"
        f"<div class='spy-alert-pills'>{pills}</div>"
        f"</div>"
        f"{metrics_html}"
        f"{body_html}"
        f"</div>",
        unsafe_allow_html=True,
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
            index=2,
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
        st.divider()
        st.subheader("Auto-trade")
        auto_trade_enabled = st.toggle(
            "Auto-paper-trade ACT_NOW signals",
            value=st.session_state.get("auto_trade_enabled", False),
            help=(
                "When ON, every new ACT_NOW signal is automatically paper-bought "
                "at the suggested risk size. Trades go to your paper book — "
                "no real money ever moves."
            ),
        )
        st.session_state["auto_trade_enabled"] = auto_trade_enabled
        if auto_trade_enabled:
            st.markdown(
                "<div style='background:rgba(10,125,42,0.12);border-left:3px solid #0a7d2a;"
                "border-radius:6px;padding:8px 10px;font-size:0.82rem;'>"
                "🤖 <strong>Auto-trade ON</strong> — ACT_NOW signals will be "
                "paper-executed on every scan."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Auto-trade is off — use the Budget page to place orders manually.")

    auto_trade_badge = (
        status_pill("🤖 auto-trade ON", "ok") if auto_trade_enabled
        else status_pill("auto-trade off", "muted")
    )
    st.markdown(
        f"{status_pill(f'auto-refresh {refresh_secs}s', 'info')} "
        f"{status_pill(f'watching {len(crypto) + len(stocks)} symbols', 'muted')} "
        f"{status_pill(f'{timeframe} · min conf {min_conf}%', 'muted')} "
        f"{auto_trade_badge}",
        unsafe_allow_html=True,
    )
    st.caption(
        "7-factor signal engine: RSI · MACD · Bollinger · Trend · Regime · "
        "Volume Surge · Momentum ROC. Paper-trade simulation only — no real money."
    )

    signal_guide()

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

    # ── Auto-trade: execute any new ACT_NOW signals into the paper book ───────
    if auto_trade_enabled:
        try:
            book_for_auto = PaperBook(state_path=settings.paper_state_path)
            auto_results = run_auto_trades(rows, book=book_for_auto, max_per_run=3)
            if auto_results:
                append_auto_trade_log(auto_results)
                for res in auto_results:
                    if res.ok:
                        st.toast(
                            f"🤖 Auto-bought {res.qty:.4f} {res.symbol} @ ${res.price:,.2f} "
                            f"(conf {res.confidence:.0f}%)",
                            icon="✅",
                        )
                    else:
                        st.toast(f"Auto-trade skipped {res.symbol}: {res.error}", icon="⚠️")
        except Exception as _ae:
            pass  # never let auto-trade errors block the signal feed

    actionable = [r for r in rows if r.get("confidence", 0) >= min_conf]
    holds = [r for r in rows if r.get("confidence", 0) < min_conf]
    actionable.sort(key=lambda r: (-r.get("confidence", 0), -r.get("ts", 0)))
    holds.sort(key=lambda r: -r.get("ts", 0))

    # ── ACT NOW banner — loud, pulsing, impossible to miss ───────────────────
    act_now_rows = [
        r for r in actionable
        if r.get("tier") == "ACT_NOW" and r.get("ts", 0) >= time.time() - 3600
    ]
    if act_now_rows:
        act_now_banner(act_now_rows[0])
        st.markdown(
            "<div style='background:rgba(10,125,42,0.08);border:1px solid rgba(10,125,42,0.3);"
            "border-radius:8px;padding:8px 14px;font-size:0.85rem;margin-bottom:8px;'>"
            "🟢 <strong>This is a buy opportunity.</strong> "
            "The system has high conviction — check the entry, stop, and target above. "
            "Go to <strong>Budget &amp; Portfolio</strong> to size and execute a paper trade."
            "</div>",
            unsafe_allow_html=True,
        )

    st.subheader(f"Signal feed ({len(actionable)} actionable)")
    if not actionable:
        st.markdown(
            status_pill(
                "no actionable patterns above threshold — standing by, pipeline is flowing",
                "muted",
            ),
            unsafe_allow_html=True,
        )
    for r in actionable[:30]:
        _render_alert_row(r)

    if show_holds and holds:
        st.subheader(f"HOLD / STAND DOWN ({len(holds)})")
        st.caption(
            "These symbols have no clear edge right now — data is flowing normally. "
            "HOLD = stay flat if out; STAND DOWN = conditions unfavourable, wait."
        )
        for r in holds[:15]:
            _render_alert_row(r)


if __name__ == "__main__":
    main()
