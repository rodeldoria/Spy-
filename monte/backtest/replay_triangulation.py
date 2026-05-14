"""Replay the triangulation engine over synthetic Kalshi events.

The triangulation engine sums 5 votes (Crowd / Patterns / Influencers /
Session / News). Three of those are deterministic from data we DO have
(top market price, OHLCV patterns, timestamp); two — News (Perplexity)
and Influencers (X) — have no historical archive, so we inject fixture
votes via ``recommend_play(..., vote_overrides=...)``.

The runner ALWAYS schedules this engine twice — once with ``fixture_mode
= "neutral"`` (abstain) and once with ``fixture_mode = "seeded_random"``
— so the results page can A/B them and you can see how much the engine
depends on those votes.
"""

from __future__ import annotations

import math

import pandas as pd

from monte.backtest.config import BacktestConfig
from monte.backtest.data import load_ohlcv, realised_vol
from monte.backtest.fixtures import build_overrides
from monte.backtest.store import BacktestStore, TradeRow
from monte.signals.triangulation import recommend_play

_WARMUP_BARS = 60
_HORIZON_BARS = {"1h": 1, "1d": 1}
_STRIKE_PCTS = {"1h": 0.005, "1d": 0.02}


def _synth_event(symbol: str, spot: float, strike: float, sigma_total: float,
                 close_time: float) -> dict:
    """Build a minimal event dict matching what live Kalshi pulls produce.

    The fields used downstream by ``recommend_play``: ``markets`` (each a dict
    with ``status``, ``yes_bid_dollars``, ``yes_ask_dollars``, ``subtitle``)."""
    z = math.log(strike / spot) / sigma_total if sigma_total > 0 else 0.0
    prob_above = 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    yes_ask = max(0.02, min(0.98, prob_above))
    yes_bid = max(0.01, yes_ask - 0.02)
    return {
        "ticker": f"BTSYN-{symbol}",
        "title": f"{symbol} above ${strike:,.2f} at close",
        "close_time": close_time,
        "markets": [
            {
                "ticker": f"BTSYN-{symbol}-UP",
                "status": "active",
                "yes_bid_dollars": yes_bid,
                "yes_ask_dollars": yes_ask,
                "subtitle": f"{symbol} ≥ ${strike:,.2f}",
            },
            {
                "ticker": f"BTSYN-{symbol}-DN",
                "status": "active",
                "yes_bid_dollars": 1.0 - yes_ask - 0.02,
                "yes_ask_dollars": 1.0 - yes_bid,
                "subtitle": f"{symbol} < ${strike:,.2f}",
            },
        ],
    }


def _direction_action_to_side(action: str) -> str:
    if action in {"BUY", "STRONG_BUY"}:
        return "yes"
    if action in {"SELL", "STRONG_SELL", "AVOID"}:
        return "no"
    return "pass"


def run(cfg: BacktestConfig, store: BacktestStore, run_id: str) -> int:
    """Replay triangulation across each (symbol, timeframe) under
    ``cfg.fixture_mode``. Records one trade per synthetic event."""
    if cfg.fixture_mode is None:
        raise ValueError("triangulation replay requires cfg.fixture_mode")
    overrides = build_overrides(cfg.fixture_mode, cfg.seed)
    n = 0
    for symbol in cfg.symbols:
        for tf in cfg.timeframes:
            if tf not in _HORIZON_BARS:
                continue
            df = load_ohlcv(symbol, tf, cfg.start_date, cfg.end_date, cache_dir=cfg.cache_dir)
            if df.empty or len(df) <= _WARMUP_BARS + 1:
                continue
            horizon_bars = _HORIZON_BARS[tf]
            seconds_per_bar = 3600 if tf == "1h" else 86400
            sigma_lookback = 60 * 24 if tf == "1h" else 30
            strike_pct = _STRIKE_PCTS[tf]

            for i in range(_WARMUP_BARS, len(df) - horizon_bars):
                spot_price = float(df["Close"].iloc[i])
                ts_now = float(df.index[i].timestamp())
                close_time = ts_now + horizon_bars * seconds_per_bar
                sigma_min = realised_vol(df["Close"].iloc[: i + 1], lookback_bars=sigma_lookback)
                sigma_total = (sigma_min if sigma_min > 0 else 0.001) * math.sqrt(
                    horizon_bars * seconds_per_bar / 60.0
                )
                strike = spot_price * (1 + strike_pct)
                event = _synth_event(symbol, spot_price, strike, sigma_total, close_time)

                rec = recommend_play(
                    event=event,
                    market_type="range",
                    category="Crypto",
                    series_label=symbol,
                    stake=10.0,
                    vote_overrides=overrides,
                )
                if rec is None:
                    continue

                actual_close = float(df["Close"].iloc[i + horizon_bars])
                yes_settles = actual_close >= strike
                side = _direction_action_to_side(rec.action)
                if side == "pass":
                    pnl_pct = 0.0
                else:
                    book_prob = event["markets"][0]["yes_ask_dollars"] if side == "yes" \
                        else event["markets"][1]["yes_ask_dollars"]
                    ask = max(0.01, min(0.99, book_prob))
                    payout = 1.0 / ask
                    wins = yes_settles if side == "yes" else (not yes_settles)
                    pnl_pct = ((payout - 1.0) * 100.0) if wins else (-100.0)

                vote_summary = {
                    v.name: {"verdict": v.verdict, "confidence": v.confidence,
                             "source": "fixture" if v.name in overrides else "live"}
                    for v in rec.votes
                }
                store.record_trade(TradeRow(
                    run_id=run_id, engine="triangulation", fixture_mode=cfg.fixture_mode,
                    symbol=symbol, timeframe=tf, horizon=f"{tf}-pegged",
                    ts_entry=ts_now, ts_exit=close_time,
                    action=rec.action, direction=side,
                    entry_price=spot_price, exit_price=actual_close,
                    exit_reason="settle", pnl_pct=pnl_pct,
                    expected_outcome=rec.action,
                    actual_outcome="settled_yes" if yes_settles else "settled_no",
                    confidence=float(rec.confidence),
                    score=float(rec.confidence) - 0.5,
                    snapshot={"votes": vote_summary, "fixture_mode": cfg.fixture_mode,
                              "strike": strike, "spot": spot_price},
                ))
                n += 1
    return n
