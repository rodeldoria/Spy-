"""Stub dip/pump detector signal."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from monte.strategies.signals import Action, Signal, action_from_score


@dataclass
class Alert:
    symbol: str
    timeframe: str
    action: Action
    confidence: float
    score: float
    entry: float
    stop: float
    target: float
    rr: float
    contributions: list[dict[str, Any]] = field(default_factory=list)


def detect(df: pd.DataFrame, symbol: str = "", timeframe: str = "") -> Alert:
    """Stub detector — computes a basic RSI-based score."""
    try:
        close = df["Close"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi_series = (100 - 100 / (1 + rs)).fillna(50.0)
        last_rsi = float(rsi_series.iloc[-1])

        if last_rsi <= 30:
            score = max(0.4, (30 - last_rsi) / 30)
        elif last_rsi >= 70:
            score = -max(0.4, (last_rsi - 70) / 30)
        else:
            score = 0.0

        spot = float(close.iloc[-1])
        atr = float(
            pd.concat(
                [(df.get("High", close) - df.get("Low", close)),
                 (df.get("High", close) - close.shift()).abs(),
                 (df.get("Low", close) - close.shift()).abs()],
                axis=1,
            ).max(axis=1).ewm(span=14, adjust=False).mean().iloc[-1]
        )
        atr = max(atr, spot * 0.005)

        if score > 0:
            stop = spot - 1.5 * atr
            target = spot + 2.5 * atr
        elif score < 0:
            stop = spot + 1.5 * atr
            target = spot - 2.5 * atr
        else:
            stop = spot - atr
            target = spot + atr

        risk = abs(spot - stop)
        reward = abs(target - spot)
        rr = reward / max(risk, 1e-9)

        confidence = min(95.0, abs(score) * 60 + 20)
        action = action_from_score(score)

        return Alert(
            symbol=symbol,
            timeframe=timeframe,
            action=action,
            confidence=confidence,
            score=score,
            entry=spot,
            stop=stop,
            target=target,
            rr=rr,
            contributions=[{"name": "RSI", "score": score}],
        )
    except Exception:
        spot = float(df["Close"].iloc[-1]) if not df.empty else 0.0
        return Alert(
            symbol=symbol,
            timeframe=timeframe,
            action=Action.HOLD,
            confidence=0.0,
            score=0.0,
            entry=spot,
            stop=spot * 0.99,
            target=spot * 1.01,
            rr=1.0,
        )
