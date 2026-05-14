"""Replay the Kalshi decision engine against synthetic asset-pegged markets.

Kalshi has no public historical settlement endpoint. We synthesise the same
shape of market the live page would have seen — a range market keyed off
the asset's spot price — and settle each one from the actual close at the
horizon end. This lets us measure the decision engine's calibration on the
markets we CAN reconstruct (BTC/ETH/SOL price ranges) without the macro
binaries (Fed, CPI, NBER) that have no oracle.

Strike grids (chosen to bracket realistic intraday/intraday-multi-bar
moves):

    1h    →  ±0.25%, ±0.5%, ±1.0%, ±2.0%
    1d    →  ±1%,    ±2%,   ±3%,   ±5%

Yes-ask is set from the GBM model probability plus seeded Gaussian noise
(σ = 3¢) so the book looks like the real one — sometimes mispriced.
"""

from __future__ import annotations

import math
import random
import time as _time_module
import uuid
from contextlib import contextmanager
from typing import Iterator

import pandas as pd

from app.kalshi.client import KalshiMarket
from app.kalshi.decisions import score_market
from app.kalshi.spot import SpotQuote
from monte.backtest.config import BacktestConfig
from monte.backtest.data import load_ohlcv, realised_vol
from monte.backtest.store import BacktestStore, TradeRow

_STRIKE_GRID_PCT = {
    "1h": (0.0025, 0.005, 0.01, 0.02),
    "1d": (0.01, 0.02, 0.03, 0.05),
}
_HORIZON_BARS = {"1h": 1, "1d": 1}
_NOISE_CENTS = 3.0
_WARMUP_BARS = 60


@contextmanager
def _fake_now(ts: float) -> Iterator[None]:
    """Patch ``time.time`` to a fixed value within the block, so frozen
    ``KalshiMarket`` instances report the right ``seconds_to_close`` when
    the decision engine asks. Restored on exit."""
    original = _time_module.time
    _time_module.time = lambda: ts  # type: ignore[assignment]
    try:
        yield
    finally:
        _time_module.time = original  # type: ignore[assignment]


def _make_market(
    *,
    symbol: str,
    spot_price: float,
    strike_type: str,
    floor: float | None,
    cap: float | None,
    yes_ask_cents: int,
    close_time: float,
) -> KalshiMarket:
    ticker = f"BT-{symbol}-{uuid.uuid4().hex[:6]}"
    if strike_type == "greater":
        title = f"{symbol} closes ≥ ${floor:,.2f}"
    elif strike_type == "less":
        title = f"{symbol} closes < ${floor:,.2f}"
    else:
        title = f"{symbol} closes between ${floor:,.2f} and ${cap:,.2f}"
    yes_ask = max(1, min(99, yes_ask_cents))
    yes_bid = max(0, yes_ask - 2)
    return KalshiMarket(
        ticker=ticker,
        event_ticker=ticker,
        title=title,
        subtitle=title,
        status="active",
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=max(0, 100 - yes_ask - 2),
        no_ask=max(1, 100 - yes_bid),
        last_price=yes_ask,
        volume=1000,
        open_interest=500,
        close_time=close_time,
        expiration_time=close_time,
        strike_type=strike_type,
        floor_strike=floor,
        cap_strike=cap,
        raw={},
    )


def _gbm_prob_yes(strike_type: str, floor: float | None, cap: float | None,
                  spot: float, sigma_total: float) -> float:
    """Inline driftless GBM probability — mirrors decisions.gbm_prob_above
    so we can pre-compute the book without circular imports."""
    if sigma_total <= 0 or spot <= 0:
        if strike_type == "greater":
            return 1.0 if spot >= (floor or 0) else 0.0
        if strike_type == "less":
            return 1.0 if spot < (floor or 0) else 0.0
        return 1.0 if (floor or 0) <= spot <= (cap or 0) else 0.0

    def _p_above(k: float) -> float:
        z = math.log(k / spot) / sigma_total
        return 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    if strike_type == "greater":
        return _p_above(floor or spot)
    if strike_type == "less":
        return 1.0 - _p_above(floor or spot)
    p_above_floor = _p_above(floor or spot)
    p_above_cap = _p_above(cap or spot)
    return max(0.0, p_above_floor - p_above_cap)


def _settle(strike_type: str, floor: float | None, cap: float | None,
            actual_close: float) -> float:
    if strike_type == "greater":
        return 1.0 if actual_close >= (floor or 0) else 0.0
    if strike_type == "less":
        return 1.0 if actual_close < (floor or 0) else 0.0
    return 1.0 if (floor or 0) <= actual_close <= (cap or 0) else 0.0


def run(cfg: BacktestConfig, store: BacktestStore, run_id: str) -> int:
    rng = random.Random(cfg.seed)
    n = 0
    for symbol in cfg.symbols:
        for tf in cfg.timeframes:
            if tf not in _STRIKE_GRID_PCT:
                continue
            df = load_ohlcv(symbol, tf, cfg.start_date, cfg.end_date, cache_dir=cfg.cache_dir)
            if df.empty or len(df) <= _WARMUP_BARS + 1:
                continue
            horizon_bars = _HORIZON_BARS[tf]
            sigma_lookback = 60 * 24 if tf == "1h" else 30
            seconds_per_bar = 3600 if tf == "1h" else 86400
            horizon_seconds = horizon_bars * seconds_per_bar

            for i in range(_WARMUP_BARS, len(df) - horizon_bars):
                spot_price = float(df["Close"].iloc[i])
                ts_now = float(df.index[i].timestamp())
                close_time = ts_now + horizon_seconds
                sigma_min = realised_vol(df["Close"].iloc[: i + 1], lookback_bars=sigma_lookback)
                sigma_per_min = sigma_min if sigma_min > 0 else 0.001
                sigma_total = sigma_per_min * math.sqrt(horizon_seconds / 60.0)
                actual_close = float(df["Close"].iloc[i + horizon_bars])

                for pct in _STRIKE_GRID_PCT[tf]:
                    for strike_type, floor, cap in (
                        ("greater", spot_price * (1 + pct), None),
                        ("less", spot_price * (1 - pct), None),
                    ):
                        model_prob = _gbm_prob_yes(strike_type, floor, cap, spot_price, sigma_total)
                        noise = rng.gauss(0.0, _NOISE_CENTS / 100.0)
                        book_prob = max(0.01, min(0.99, model_prob + noise))
                        yes_ask_cents = round(book_prob * 100)

                        market = _make_market(
                            symbol=symbol, spot_price=spot_price, strike_type=strike_type,
                            floor=floor, cap=cap, yes_ask_cents=yes_ask_cents,
                            close_time=close_time,
                        )
                        spot = SpotQuote(symbol=symbol, price=spot_price, ts=ts_now,
                                         source="backtest", sigma_per_min=sigma_per_min)
                        with _fake_now(ts_now):
                            dec = score_market(market, spot,
                                               min_edge=cfg.min_edge_pp / 100.0,
                                               min_ev=cfg.min_ev_cents / 100.0)

                        settlement = _settle(strike_type, floor, cap, actual_close)
                        if dec.direction == "PASS":
                            store.record_trade(TradeRow(
                                run_id=run_id, engine="kalshi", fixture_mode=cfg.fixture_mode,
                                symbol=symbol, timeframe=tf, horizon=f"{tf}-pegged",
                                ts_entry=ts_now, ts_exit=close_time,
                                action="PASS", direction="pass",
                                entry_price=spot_price, exit_price=actual_close,
                                exit_reason="settle", pnl_pct=0.0,
                                expected_outcome=f"model_yes={model_prob:.3f}",
                                actual_outcome="settled_yes" if settlement else "settled_no",
                                confidence=float(dec.confidence_pct), score=0.0,
                                snapshot={
                                    "model_prob_yes": model_prob, "book_prob_yes": book_prob,
                                    "strike_type": strike_type, "floor": floor, "cap": cap,
                                    "sigma_per_min": sigma_per_min,
                                    "edge_yes": dec.yes_side.edge,
                                    "edge_no": dec.no_side.edge,
                                },
                            ))
                            n += 1
                            continue

                        side = dec.yes_side if dec.direction == "YES" else dec.no_side
                        ask = side.ask_cents / 100.0
                        wins_yes = settlement == 1.0
                        wins_for_bet = wins_yes if dec.direction == "YES" else (not wins_yes)
                        payout = 1.0 / ask if ask > 0 else 0.0
                        pnl_pct = ((payout - 1.0) * 100.0) if wins_for_bet else (-100.0)
                        store.record_trade(TradeRow(
                            run_id=run_id, engine="kalshi", fixture_mode=cfg.fixture_mode,
                            symbol=symbol, timeframe=tf, horizon=f"{tf}-pegged",
                            ts_entry=ts_now, ts_exit=close_time,
                            action=dec.direction, direction=dec.direction.lower(),
                            entry_price=spot_price, exit_price=actual_close,
                            stop_price=None, target_price=None,
                            exit_reason="settle", pnl_pct=pnl_pct,
                            expected_outcome=f"model_yes={model_prob:.3f}",
                            actual_outcome="settled_yes" if settlement else "settled_no",
                            confidence=float(dec.confidence_pct),
                            score=float(side.edge),
                            snapshot={
                                "model_prob_yes": model_prob, "book_prob_yes": book_prob,
                                "ask_cents": side.ask_cents, "edge": side.edge,
                                "ev_per_dollar": side.ev_per_dollar,
                                "kelly_fraction": side.kelly_fraction,
                                "strike_type": strike_type, "floor": floor, "cap": cap,
                                "sigma_per_min": sigma_per_min,
                            },
                        ))
                        n += 1
    return n
