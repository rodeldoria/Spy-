"""Replay the dip_pump 7-factor detector over historical OHLCV bars.

At each bar i (after a 200-bar warmup so RSI/MACD/ADX have data), we call
``detect`` with ONLY bars 0..i (no future leakage). If the alert is BUY-ish
or SELL-ish, we simulate forward bar-by-bar: stop hit first, target hit
first, or timeout at i+timeout_bars.

After a trade closes, the cursor jumps past the exit bar so signals never
overlap — each trade is independent.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from monte.backtest.config import BacktestConfig
from monte.backtest.data import load_ohlcv
from monte.backtest.store import BacktestStore, TradeRow
from monte.signals.dip_pump import detect
from monte.strategies.signals import Action

WARMUP_BARS = 200
LONG_ACTIONS = {Action.BUY, Action.STRONG_BUY}
SHORT_ACTIONS = {Action.SELL, Action.STRONG_SELL}


def _simulate_exit(
    df: pd.DataFrame,
    *,
    entry_idx: int,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    timeout_bars: int,
) -> tuple[int, float, str]:
    """Walk forward bars looking for stop or target. Return (exit_idx,
    exit_price, exit_reason)."""
    end_idx = min(entry_idx + timeout_bars, len(df) - 1)
    for j in range(entry_idx + 1, end_idx + 1):
        low = float(df["Low"].iloc[j])
        high = float(df["High"].iloc[j])
        if direction == "long":
            if low <= stop:
                return j, stop, "stop"
            if high >= target:
                return j, target, "target"
        else:  # short
            if high >= stop:
                return j, stop, "stop"
            if low <= target:
                return j, target, "target"
    return end_idx, float(df["Close"].iloc[end_idx]), "timeout"


def _bars(symbols: Iterable[str], timeframes: Iterable[str],
          start_iso: str, end_iso: str, cache_dir) -> Iterable[tuple[str, str, pd.DataFrame]]:
    for sym in symbols:
        for tf in timeframes:
            df = load_ohlcv(sym, tf, start_iso, end_iso, cache_dir=cache_dir)
            if df.empty or len(df) <= WARMUP_BARS:
                continue
            yield sym, tf, df


def run(cfg: BacktestConfig, store: BacktestStore, run_id: str) -> int:
    """Execute the dip_pump replay for ``cfg`` and write trades to ``store``.

    Returns the number of trades recorded."""
    n = 0
    for symbol, timeframe, df in _bars(cfg.symbols, cfg.timeframes,
                                       cfg.start_date, cfg.end_date, cfg.cache_dir):
        i = WARMUP_BARS
        while i < len(df) - 1:
            window = df.iloc[: i + 1]
            alert = detect(window, symbol=symbol, timeframe=timeframe)
            act = alert.action
            if act not in LONG_ACTIONS and act not in SHORT_ACTIONS:
                i += 1
                continue
            direction = "long" if act in LONG_ACTIONS else "short"
            entry_price = float(window["Close"].iloc[-1])
            exit_idx, exit_price, exit_reason = _simulate_exit(
                df,
                entry_idx=i,
                direction=direction,
                entry=entry_price,
                stop=alert.stop,
                target=alert.target,
                timeout_bars=cfg.timeout_bars,
            )
            sign = 1 if direction == "long" else -1
            pnl_pct = sign * (exit_price - entry_price) / entry_price * 100.0 if entry_price else 0.0
            snapshot = {
                "indicators": dict(alert.indicator_snapshot),
                "contributions": list(alert.contributions),
                "regime": alert.regime,
                "rr": alert.rr,
            }
            store.record_trade(TradeRow(
                run_id=run_id,
                engine="dip_pump",
                fixture_mode=cfg.fixture_mode,
                symbol=symbol,
                timeframe=timeframe,
                horizon=str(alert.horizon.value if hasattr(alert.horizon, "value") else alert.horizon),
                ts_entry=float(df.index[i].timestamp()),
                ts_exit=float(df.index[exit_idx].timestamp()),
                action=str(act.value if hasattr(act, "value") else act),
                direction=direction,
                entry_price=entry_price,
                stop_price=float(alert.stop),
                target_price=float(alert.target),
                exit_price=float(exit_price),
                exit_reason=exit_reason,
                pnl_pct=pnl_pct,
                confidence=float(alert.confidence),
                score=float(alert.score),
                snapshot=snapshot,
            ))
            n += 1
            i = exit_idx + 1
    return n
