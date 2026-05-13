"""Kalshi Crypto — decision dashboard.

Hybrid spot source: pulls Kalshi market odds automatically, while letting
the user override the spot price if Coinbase/Binance are blocked or stale.
For each market, surfaces:

- Implied probability (from the Kalshi book) AND model probability (from a
  driftless GBM with realised vol over time-to-close).
- Direction (YES / NO / PASS) with model confidence %.
- Expected value in cents per $1 for each side, plus capped Kelly fraction
  for sizing.

No automated execution. The user decides whether to bet; this just flags
where the book and the vol model disagree.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from app._shared import setup_page
from app._ui import inject_global_css, status_pill
from app.kalshi import (
    Decision,
    KalshiClient,
    KalshiMarket,
    get_spot_price,
    score_event,
)
from app.kalshi.spot import SpotQuote, default_sigma_per_min, manual_quote
from monte.learning import kalshi_calibration as kcal

SYMBOLS = ["BTC", "ETH", "SOL"]
HORIZONS = ["15min", "hourly", "daily", "weekly"]
HORIZON_LABELS = {
    "15min": "15-min Up/Down",
    "hourly": "Hourly settlement",
    "daily": "Daily settlement",
    "weekly": "Weekly (Friday 5pm EDT)",
}


@st.cache_resource(show_spinner=False)
def _client() -> KalshiClient:
    return KalshiClient()


@st.cache_data(ttl=8, show_spinner=False)
def _fetch_markets(symbol: str, horizons: tuple[str, ...]) -> dict[str, list[KalshiMarket]]:
    return _client().crypto_markets(symbol, horizons=horizons)


@st.cache_data(ttl=2, show_spinner=False)
def _fetch_spot(symbol: str) -> SpotQuote:
    return get_spot_price(symbol)


@st.cache_data(ttl=1, show_spinner=False)
def _fetch_markets_live(symbol: str, horizons: tuple[str, ...]) -> dict[str, list[KalshiMarket]]:
    """Sub-second TTL variant for when the user picks 1-3s autorefresh.

    Kalshi REST blocks direct browser CORS, so the Streamlit server has to
    proxy every poll. 1s TTL keeps repeated reruns from hammering the API
    while still feeling live.
    """
    return _client().crypto_markets(symbol, horizons=horizons)


@st.cache_data(ttl=15, show_spinner=False)
def _cached_calibration():
    """Reload calibration report at most every 15s."""
    return kcal.calibration_report()


def _maybe_settle(force: bool = False) -> None:
    """Run the settlement back-fill at most once every 5 minutes."""
    last = st.session_state.get("kalshi_last_settle_ts", 0)
    now = time.time()
    if not force and (now - last) < 300:
        return
    st.session_state["kalshi_last_settle_ts"] = now
    try:
        summary = kcal.settle_pending(_client(), max_lookups=10)
        st.session_state["kalshi_last_settle_summary"] = summary
        # Bust the calibration cache so the new outcome appears immediately
        _cached_calibration.clear()
    except Exception as e:
        st.session_state["kalshi_last_settle_summary"] = {"error": str(e)}


def _direction_pill(direction: str, confidence: float) -> str:
    palette = {
        "YES": ("#0a7d2a", "#e0f5e6", "🟢"),
        "NO": ("#a8261f", "#fbe9e7", "🔴"),
        "PASS": ("#5b6470", "#f1f3f5", "⚪"),
    }
    fg, bg, emoji = palette.get(direction, palette["PASS"])
    if direction == "PASS":
        # PASS confidence = how close book is to model. Don't show as a %
        # (people read it as "44% chance of YES"). Show as a qualitative label.
        if confidence >= 75:
            label = "book ≈ model"
        elif confidence >= 40:
            label = "no clear edge"
        else:
            label = "near threshold"
        suffix = f" · {label}"
    else:
        suffix = f" · {confidence:.0f}% conviction"
    return (
        f"<span class='spy-pill' style='color:{fg};background:{bg};"
        f"font-size:0.95rem;padding:4px 12px;font-weight:700;' "
        f"title='Conviction is a logistic of edge size: 4pp=50%, 8pp=~85%, 12pp=90%, 20pp≈99%.'>"
        f"{emoji} {direction}{suffix}</span>"
    )


def _kalshi_help() -> None:
    with st.expander("ℹ️ How to read these signals (EV · Kelly · Conviction)"):
        st.markdown(
            """
**What each market is asking** — Each row shows a binary contract: YES pays $1 if the
condition resolves true, NO pays $1 if it resolves false. The **Bet:** line under
each title spells out exactly what YES means in plain English (e.g. "BTC ≥ $80,000
at 12:00 UTC").

**The model** — A driftless GBM (geometric Brownian motion) fed with realised 1-minute
volatility. It computes the *fair* probability of YES given just spot price and vol.
This is a **benchmark, not a prediction** — its job is to flag when Kalshi's book
disagrees with vol-implied fair value.

**Edge** — `model_prob − implied_prob` in percentage points. 4pp is our default
floor; below that, model error swamps the signal.

**EV (¢ per $1)** — Expected return per $1 staked. `EV = model_prob × payout − 1`.
- **EV +5¢/$1 or higher** = strong; covers fees, slippage, and some model error
- **EV +2 to +5¢** = marginal; only take if Kelly is also small (1-3%)
- **EV ≤ +2¢** = skip — usually noise

**Kelly fraction** — Optimal % of bankroll to risk on this bet, capped at 25%
(full Kelly is too aggressive; the model has its own error bars).
- **Kelly < 1%** = skip even if EV is positive (model can't tell sides apart)
- **Kelly 2-8%** = normal scale-in size
- **Kelly 10-25%** = high conviction; still cap at 5-10% of bankroll in practice

**Conviction % scale (for YES/NO only)** — Logistic of edge size:
- **< 50%** = weak (edge ~4pp); maybe size at 0.25× Kelly
- **50–75%** = moderate (edge 4–8pp)
- **75–90%** = strong (edge 8–12pp)
- **> 90%** = high (edge ≥ 12pp); rare, often signals stale book

**PASS rows** — The book matches the model within tolerance, so there's no edge
worth taking. The qualifier ("book ≈ model" / "near threshold") tells you how close
to the edge floor — *not* a probability.
            """
        )


def _countdown(seconds: float) -> str:
    if seconds <= 0:
        return "closed"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h {(s % 3600) // 60:02d}m"
    return f"{s // 86400}d {(s % 86400) // 3600:02d}h"


def _render_decision_row(d: Decision, calib_report=None, stake: float = 10.0) -> None:
    with st.container(border=True):
        cols = st.columns([3.0, 1.4, 1.4, 1.6])

        # Calibration line: show what the model probability becomes after
        # learning from past settled markets, when we have enough data.
        calib_line = ""
        if calib_report and calib_report.n_settled >= 30:
            cal_yes, src = kcal.calibrate_prob(d.yes_side.model_prob)
            shift = (cal_yes - d.yes_side.model_prob) * 100
            shift_color = "#0a7d2a" if abs(shift) < 3 else "#a8261f"
            calib_line = (
                f"  \n<span style='color:{shift_color};font-size:0.78rem;'>"
                f"🧠 Calibrated YES prob: {cal_yes*100:.1f}% "
                f"(raw model {d.yes_side.model_prob*100:.1f}%, shift {shift:+.1f}pp · {src})"
                f"</span>"
            )

        # Determine which crypto symbol this market is for, so the live JS
        # spot ticker can subscribe to the right Coinbase channel.
        sym_guess = "BTC"
        for s in ("ETH", "SOL", "BTC"):
            if s in d.title.upper() or s in (d.bet_summary or "").upper():
                sym_guess = s
                break

        cols[0].markdown(
            f"**{d.title}**  \n"
            f"<span class='spy-meta'>Closes in <span class='kalshi-countdown' "
            f"data-close-ts='{d.close_time:.0f}'>{_countdown(d.horizon_seconds)}</span> · "
            f"Spot <span class='kalshi-spot-live' data-symbol='{sym_guess}'>"
            f"${d.spot_price:,.2f}</span> "
            f"({d.spot_source}) · σ {d.sigma_per_min*100:.3f}%/min</span>  \n"
            + (f"<span style='color:#2563eb;font-weight:600;font-size:0.85rem;'>"
               f"📌 Bet: {d.bet_summary}</span>" if d.bet_summary else "")
            + calib_line,
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            _direction_pill(d.direction, d.confidence_pct),
            unsafe_allow_html=True,
        )

        # YES side — concrete dollar payout calc
        yes = d.yes_side
        ev_color = "#0a7d2a" if yes.ev_per_dollar > 0 else "#a8261f"
        yes_profit = stake * (yes.payout - 1) if yes.payout > 1 else 0.0
        cols[2].markdown(
            f"**YES** @ {yes.ask_cents}¢ ({yes.payout:.2f}x)  \n"
            f"<span class='spy-meta'>book {yes.implied_prob*100:.1f}% · "
            f"model {yes.model_prob*100:.1f}% · edge {yes.edge*100:+.1f}pp</span>  \n"
            f"<span style='font-size:0.86rem;'>"
            f"💵 Bet <strong>${stake:,.0f}</strong> → "
            f"<span style='color:#0a7d2a;font-weight:700;'>+${yes_profit:,.2f}</span> if YES wins · "
            f"<span style='color:#a8261f;font-weight:700;'>-${stake:,.2f}</span> if NO"
            f"</span>  \n"
            f"<span style='color:{ev_color};font-weight:700;'>"
            f"EV {yes.ev_per_dollar*100:+.1f}¢/$1</span>"
            + (f" · Kelly {yes.kelly_fraction*100:.1f}%" if yes.kelly_fraction > 0 else ""),
            unsafe_allow_html=True,
        )

        # NO side — concrete dollar payout calc
        no = d.no_side
        ev_color = "#0a7d2a" if no.ev_per_dollar > 0 else "#a8261f"
        no_profit = stake * (no.payout - 1) if no.payout > 1 else 0.0
        cols[3].markdown(
            f"**NO** @ {no.ask_cents}¢ ({no.payout:.2f}x)  \n"
            f"<span class='spy-meta'>book {no.implied_prob*100:.1f}% · "
            f"model {no.model_prob*100:.1f}% · edge {no.edge*100:+.1f}pp</span>  \n"
            f"<span style='font-size:0.86rem;'>"
            f"💵 Bet <strong>${stake:,.0f}</strong> → "
            f"<span style='color:#0a7d2a;font-weight:700;'>+${no_profit:,.2f}</span> if NO wins · "
            f"<span style='color:#a8261f;font-weight:700;'>-${stake:,.2f}</span> if YES"
            f"</span>  \n"
            f"<span style='color:{ev_color};font-weight:700;'>"
            f"EV {no.ev_per_dollar*100:+.1f}¢/$1</span>"
            + (f" · Kelly {no.kelly_fraction*100:.1f}%" if no.kelly_fraction > 0 else ""),
            unsafe_allow_html=True,
        )

        st.caption(d.reasoning)
        for w in d.warnings:
            st.markdown(status_pill(w, "warn"), unsafe_allow_html=True)


def _resolve_spot(symbol: str) -> tuple[SpotQuote | None, str | None]:
    """Resolve spot using auto-fetch unless the user has overridden it."""
    override_key = f"kalshi_spot_override_{symbol}"
    override = st.session_state.get(override_key)
    if override and override.get("price"):
        sigma_override = override.get("sigma") or default_sigma_per_min(symbol)
        return manual_quote(symbol, float(override["price"]), float(sigma_override)), None
    try:
        return _fetch_spot(symbol), None
    except Exception as e:
        return None, str(e)


def _sort_decisions(decisions: list[Decision], sort_mode: str) -> list[Decision]:
    if sort_mode == "Closing soonest":
        return sorted(decisions, key=lambda d: d.horizon_seconds)
    if sort_mode == "Closing latest":
        return sorted(decisions, key=lambda d: -d.horizon_seconds)
    if sort_mode == "Strike low → high":
        # Use the YES side's implied prob as a proxy when no strike is parseable.
        return sorted(
            decisions,
            key=lambda d: (
                # Pull strike from bet_summary if it contains a $ amount.
                _strike_from_summary(d.bet_summary),
                d.horizon_seconds,
            ),
        )
    # Default: best edge first
    return sorted(decisions, key=lambda d: -max(d.yes_side.edge, d.no_side.edge))


def _inject_live_countdown_js() -> None:
    """Tick countdowns + stream live spot prices via Coinbase WebSocket.

    Two independent live loops, both running entirely in the user's browser:

    1. Countdown ticker (250ms) — decrements every `.kalshi-countdown` span
       from its `data-close-ts` epoch. Colours amber under 5min, red under 1min.

    2. Coinbase WebSocket — opens a single connection to
       `wss://ws-feed.exchange.coinbase.com`, subscribes to the `ticker`
       channel for every symbol present on the page (BTC-USD / ETH-USD /
       SOL-USD), and updates the text of every matching `.kalshi-spot-live`
       span on each tick. No flash, no pulse — silent text updates like a
       broker quote.

    Coinbase's public ticker feed is free, no auth, ~5-15 ticks/sec for
    BTC-USD. If the feed disconnects, we auto-reconnect with backoff.
    """
    import streamlit.components.v1 as components

    components.html(
        """
        <script>
        (function() {
          const root = window.parent.document;

          function fmt(secs) {
            if (secs <= 0) return "closed";
            const s = Math.floor(secs);
            if (s < 60) return s + "s";
            if (s < 3600) {
              const m = Math.floor(s / 60);
              const r = s % 60;
              return m + "m " + (r < 10 ? "0" : "") + r + "s";
            }
            if (s < 86400) {
              const h = Math.floor(s / 3600);
              const m = Math.floor((s % 3600) / 60);
              return h + "h " + (m < 10 ? "0" : "") + m + "m";
            }
            const d = Math.floor(s / 86400);
            const h = Math.floor((s % 86400) / 3600);
            return d + "d " + (h < 10 ? "0" : "") + h + "h";
          }

          function tick() {
            const now = Date.now() / 1000;
            const els = root.querySelectorAll(".kalshi-countdown");
            els.forEach(function(el) {
              const ts = parseFloat(el.dataset.closeTs);
              if (!ts) return;
              const remaining = ts - now;
              el.textContent = fmt(remaining);
              if (remaining > 0 && remaining < 60) {
                el.style.color = "#dc2626";
                el.style.fontWeight = "700";
              } else if (remaining > 0 && remaining < 300) {
                el.style.color = "#f59e0b";
                el.style.fontWeight = "600";
              } else {
                el.style.color = "";
                el.style.fontWeight = "";
              }
            });
          }

          // --- Coinbase WebSocket live spot ticker -----------------------
          function connectCoinbase() {
            const symbolsOnPage = new Set();
            root.querySelectorAll(".kalshi-spot-live").forEach(function(el) {
              const s = (el.dataset.symbol || "").toUpperCase();
              if (s) symbolsOnPage.add(s);
            });
            if (symbolsOnPage.size === 0) return null;

            const productIds = Array.from(symbolsOnPage).map(s => s + "-USD");
            const ws = new WebSocket("wss://ws-feed.exchange.coinbase.com");

            ws.onopen = function() {
              ws.send(JSON.stringify({
                type: "subscribe",
                product_ids: productIds,
                channels: ["ticker"]
              }));
            };

            ws.onmessage = function(ev) {
              let msg;
              try { msg = JSON.parse(ev.data); } catch(e) { return; }
              if (msg.type !== "ticker" || !msg.price) return;
              const sym = (msg.product_id || "").split("-")[0];
              const price = parseFloat(msg.price);
              if (!sym || !isFinite(price)) return;
              const formatted = "$" + price.toLocaleString("en-US", {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
              });
              root.querySelectorAll(
                ".kalshi-spot-live[data-symbol='" + sym + "']"
              ).forEach(function(el) {
                el.textContent = formatted;
              });
            };

            ws.onclose = function() {
              // Reconnect after 3s if the page is still open
              setTimeout(function() {
                if (window.parent.__kalshiWs === ws) {
                  window.parent.__kalshiWs = connectCoinbase();
                }
              }, 3000);
            };

            ws.onerror = function() { try { ws.close(); } catch(e){} };
            return ws;
          }

          // Cleanup any prior intervals/sockets from previous reruns
          if (window.parent.__kalshiTickInterval) {
            clearInterval(window.parent.__kalshiTickInterval);
          }
          if (window.parent.__kalshiWs) {
            try { window.parent.__kalshiWs.close(); } catch(e){}
          }

          window.parent.__kalshiTickInterval = setInterval(tick, 250);
          window.parent.__kalshiWs = connectCoinbase();
          tick();
        })();
        </script>
        """,
        height=0,
    )


def _strike_from_summary(summary: str) -> float:
    import re
    m = re.search(r"\$([\d,]+(?:\.\d+)?)", summary or "")
    if not m:
        return float("inf")
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return float("inf")


def _render_calibration_panel() -> None:
    rep = _cached_calibration()
    summary = st.session_state.get("kalshi_last_settle_summary") or {}
    settled_label = ""
    if summary and "settled" in summary:
        settled_label = (
            f" · last sweep: +{summary['settled']} settled, "
            f"{summary['still_pending']} pending"
        )

    if rep.n_settled == 0:
        st.info(
            "🧠 **Calibration learning** — no settled markets yet. "
            "Each time you load this page we snapshot the live decisions, "
            "and once any market closes we back-fill the outcome to learn "
            "from it. After ~30 settled markets, conviction scores become "
            f"data-driven instead of pure GBM. (Snapshots in log: {rep.n_snapshots}{settled_label})"
        )
        return

    with st.expander(
        f"🧠 Calibration: {rep.n_settled} settled · {rep.n_snapshots} snapshots{settled_label}",
        expanded=False,
    ):
        c1, c2, c3, c4 = st.columns(4)
        if rep.brier_model is not None:
            c1.metric(
                "Model Brier",
                f"{rep.brier_model:.3f}",
                help="Lower is better. 0 = perfect, 0.25 = coin flip.",
            )
        if rep.brier_book is not None:
            c2.metric(
                "Book Brier",
                f"{rep.brier_book:.3f}",
                help="Kalshi orderbook's own Brier score on these settled markets.",
            )
        if rep.edge_vs_book_brier is not None:
            beats = rep.beats_book
            delta_label = "model wins" if beats else "book wins"
            c3.metric(
                "Edge vs book",
                f"{rep.edge_vs_book_brier:+.3f}",
                delta=delta_label,
                help="Brier_book − Brier_model. Positive = model beats book.",
            )
        if rep.yes_recommend_hit_rate is not None or rep.no_recommend_hit_rate is not None:
            yes_hr = rep.yes_recommend_hit_rate or 0.0
            no_hr = rep.no_recommend_hit_rate or 0.0
            c4.metric(
                "Recommend hit rate",
                f"YES {yes_hr*100:.0f}% · NO {no_hr*100:.0f}%",
                help="When the dashboard said YES (or NO), did the bet win?",
            )

        if rep.decile_hit_rates:
            import pandas as pd
            df = pd.DataFrame(
                rep.decile_hit_rates,
                columns=["model_prob_bin", "actual_hit_rate", "n"],
            )
            df["perfect_calibration"] = df["model_prob_bin"]
            st.markdown(
                "**Calibration curve** — if the model is well-calibrated, "
                "the actual hit-rate column should track the model_prob column. "
                "Drift = bias the calibrator corrects."
            )
            st.dataframe(
                df.style.format(
                    {
                        "model_prob_bin": "{:.0%}",
                        "actual_hit_rate": "{:.1%}",
                        "perfect_calibration": "{:.0%}",
                        "n": "{:.0f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        if st.button("🔄 Force settlement sweep now", key="force_settle_btn"):
            _maybe_settle(force=True)
            st.rerun()


def _render_symbol(symbol: str, horizons: tuple[str, ...], min_edge: float, min_ev: float, sort_mode: str, stake: float = 10.0) -> None:
    st.subheader(f"{symbol}")
    spot, err = _resolve_spot(symbol)
    spot_col, override_col = st.columns([2.4, 1.6])
    if spot:
        age = max(0, int(time.time() - spot.ts))
        spot_col.markdown(
            f"**Spot ${spot.price:,.2f}**  "
            f"<span class='spy-meta'>source: {spot.source} · {age}s old · "
            f"σ {spot.sigma_per_min*100:.3f}%/min</span>",
            unsafe_allow_html=True,
        )
    else:
        spot_col.markdown(
            status_pill(f"spot fetch failed: {err}", "err"),
            unsafe_allow_html=True,
        )

    with override_col:
        with st.expander("Manual spot override", expanded=not spot):
            default_price = float(spot.price) if spot else 0.0
            new_price = st.number_input(
                f"{symbol} spot ($)",
                min_value=0.0,
                value=default_price,
                step=0.01,
                key=f"manual_price_{symbol}",
                format="%.2f",
            )
            new_sigma = st.number_input(
                f"σ per minute (%, optional)",
                min_value=0.0,
                value=float((spot.sigma_per_min if spot else default_sigma_per_min(symbol)) * 100),
                step=0.001,
                key=f"manual_sigma_{symbol}",
                format="%.4f",
            )
            c1, c2 = st.columns(2)
            if c1.button("Use override", key=f"use_override_{symbol}"):
                st.session_state[f"kalshi_spot_override_{symbol}"] = {
                    "price": new_price,
                    "sigma": new_sigma / 100.0,
                }
                st.rerun()
            if c2.button("Clear override", key=f"clear_override_{symbol}"):
                st.session_state.pop(f"kalshi_spot_override_{symbol}", None)
                st.rerun()

    if not spot:
        return

    # Pro-investor pattern strip: fetch a small candle history, run the
    # pattern engine, and render the top patterns with their tilt.
    try:
        from monte.data.crypto import get_candles
        from monte.signals.patterns import detect_patterns

        candles = get_candles(symbol, "15m", lookback_bars=120)
        if candles is not None and not candles.empty and "close" in candles.columns:
            bundle = detect_patterns(candles["close"], spot.price)
            if bundle.signals:
                consensus_color = {
                    "bullish": "#0a7d2a", "bearish": "#a8261f",
                    "mixed": "#6b7280", "quiet": "#6b7280",
                }[bundle.consensus]
                chips = []
                for s in bundle.top:
                    cc = {"bull": "#0a7d2a", "bear": "#a8261f", "neutral": "#6b7280"}[s.direction]
                    chips.append(
                        f"<span title=\"{s.note}\" "
                        f"style='display:inline-block;padding:3px 8px;margin:2px 4px 2px 0;"
                        f"border-radius:10px;background:rgba(107,114,128,0.10);"
                        f"font-size:0.78rem;color:{cc};font-weight:600;'>"
                        f"{s.emoji} {s.name} {s.bias_pp:+.1f}pp</span>"
                    )
                st.markdown(
                    f"<div style='margin:4px 0 8px 0;font-size:0.82rem;'>"
                    f"🧠 <strong>Patterns on {symbol}</strong> · "
                    f"<span style='color:{consensus_color};font-weight:700;'>"
                    f"{bundle.consensus.upper()}</span> "
                    f"<span style='color:#888;'>(net {bundle.net_bias_pp:+.1f}pp tilt to up-side)</span><br/>"
                    f"{''.join(chips)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    except Exception:
        pass

    try:
        markets_by_horizon = _fetch_markets(symbol, horizons)
    except Exception as e:
        st.error(f"Kalshi API error for {symbol}: {e}")
        return

    any_rendered = False
    for horizon in horizons:
        markets = [m for m in markets_by_horizon.get(horizon, []) if m.status == "active"]
        if not markets:
            continue
        any_rendered = True
        st.markdown(f"#### {HORIZON_LABELS[horizon]}")
        decisions = score_event(markets, spot, min_edge=min_edge, min_ev=min_ev)

        # Snapshot every active decision for the learning loop (deduped per
        # ticker per minute, so this is cheap on auto-refresh).
        try:
            kcal.snapshot_decisions(decisions, symbol)
        except Exception:
            pass

        calib = _cached_calibration()

        # Show actionable first, then top 8 passes for context. Sort within
        # each group by the user's chosen mode.
        actionable = _sort_decisions(
            [d for d in decisions if d.direction != "PASS"], sort_mode
        )
        passes = _sort_decisions(
            [d for d in decisions if d.direction == "PASS"], sort_mode
        )
        for d in actionable:
            _render_decision_row(d, calib_report=calib, stake=stake)
        if passes:
            with st.expander(f"PASS markets ({len(passes)}) — book in line with model"):
                for d in passes[:8]:
                    _render_decision_row(d, calib_report=calib, stake=stake)

    if not any_rendered:
        st.info(f"No active {symbol} markets in the selected horizons right now.")


def main() -> None:
    setup_page("Kalshi Crypto", icon="🪙")
    inject_global_css()

    with st.sidebar:
        st.subheader("Kalshi")
        symbols = st.multiselect(
            "Symbols",
            SYMBOLS,
            default=SYMBOLS,
            help="Crypto series to pull from Kalshi.",
        )
        horizons = st.multiselect(
            "Horizons",
            HORIZONS,
            default=HORIZONS,
            format_func=lambda h: HORIZON_LABELS[h],
        )
        min_edge_pp = st.slider(
            "Min edge (pp)",
            0,
            20,
            4,
            help="Minimum (model − book) probability gap, in percentage points, "
            "before we recommend YES or NO. Smaller edges are dominated by "
            "model error.",
        )
        min_ev_cents = st.slider(
            "Min EV (¢ per $1)",
            0,
            20,
            2,
            help="Minimum expected return in cents per $1 staked. Acts as a "
            "fee + slippage cushion.",
        )
        stake = st.number_input(
            "Stake size ($)",
            min_value=1.0,
            max_value=10000.0,
            value=10.0,
            step=1.0,
            help="Used to show concrete dollar payouts on each market row.",
        )
        sort_mode = st.selectbox(
            "Sort markets by",
            [
                "Best edge",
                "Closing soonest",
                "Closing latest",
                "Strike low → high",
            ],
            index=0,
            help="How to order markets within each horizon group.",
        )
        refresh_secs = st.select_slider(
            "Kalshi book refresh (sec)",
            options=[1, 2, 3, 5, 10, 30, 60, 120, 300],
            value=3,
            help="Kalshi REST is server-proxied (browser CORS is blocked), "
            "so this controls how often YES/NO ¢ refreshes. "
            "Countdown seconds tick every 250ms and Coinbase spot streams "
            "via WebSocket regardless of this slider.",
        )
        st.caption(
            f"📡 Live: Coinbase WS spot (sub-sec) · Kalshi book every {refresh_secs}s"
        )
        st_autorefresh(interval=refresh_secs * 1000, key="kalshi_refresh")

    st.caption(
        "Hybrid: Kalshi orderbook auto-pulled, spot price auto-pulled "
        "(Coinbase → Binance) with manual override. Direction is based on "
        "edge vs. a driftless GBM model using realised 1-min vol over "
        "time-to-close. **Advisory only — no orders are placed.**"
    )
    st.markdown(
        f"{status_pill(f'auto-refresh {refresh_secs}s', 'info')} "
        f"{status_pill(f'edge ≥ {min_edge_pp}pp · EV ≥ {min_ev_cents}¢', 'muted')} "
        f"{status_pill(datetime.now(timezone.utc).strftime('%H:%M:%S UTC'), 'muted')}",
        unsafe_allow_html=True,
    )

    if not symbols or not horizons:
        st.warning("Pick at least one symbol and one horizon in the sidebar.")
        return

    _kalshi_help()
    _maybe_settle()
    _render_calibration_panel()
    _inject_live_countdown_js()

    for symbol in symbols:
        _render_symbol(
            symbol,
            tuple(horizons),
            min_edge=min_edge_pp / 100.0,
            min_ev=min_ev_cents / 100.0,
            sort_mode=sort_mode,
            stake=float(stake),
        )


if __name__ == "__main__":
    main()
