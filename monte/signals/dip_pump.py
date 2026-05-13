"""Multi-factor dip/pump detector.

The original detector was RSI-only and almost always returned HOLD. This
version triangulates RSI, MACD histogram, Bollinger %b, regime and trend
slope so that the watchlist can surface a clear BUY / SELL conviction
when the factors agree.

The output is intentionally compatible with the previous `Alert` dataclass
so the rest of the dashboard does not need to change shape — but it now
also carries a `horizon` (day trade / swing / long hold) and a list of
per-factor contributions for the UI to display.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from monte.indicators.regime import RegimeLabel, classify_regime
from monte.indicators.technical import bollinger, macd, rsi
from monte.signals.horizon import Horizon, HorizonCall, classify_horizon
from monte.strategies.signals import Action, action_from_score


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
    horizon: Horizon = Horizon.SWING
    horizon_rationale: str = ""
    regime: str = ""
    contributions: list[dict[str, Any]] = field(default_factory=list)
    indicator_snapshot: dict[str, float] = field(default_factory=dict)

    def reasoning(self) -> str:
        """One-sentence "why this works" from contributions + snapshot.

        Stitches the top 1-2 contributors into a human-readable bullet so the
        user can read it on a phone notification.
        """
        if self.action is Action.HOLD or not self.contributions:
            return ""
        top = sorted(
            self.contributions, key=lambda c: abs(c.get("score", 0)), reverse=True
        )[:2]
        snap = self.indicator_snapshot or {}
        rsi_v = snap.get("rsi", 50.0)
        bb_v = snap.get("bb_pctb", 0.5)
        adx_v = snap.get("adx", 0.0)
        direction = "long" if self.score > 0 else "short"
        bits = [f"{direction} setup at {self.confidence:.0f}% confidence"]
        names = {c["name"] for c in top}
        if "RSI" in names and (rsi_v < 35 or rsi_v > 65):
            bits.append(
                "RSI extreme inside a trend → mean-reversion edge"
                if rsi_v < 35
                else "RSI hot but momentum confirms continuation"
            )
        if "BB %b" in names and (bb_v < 0.15 or bb_v > 0.85):
            bits.append("price at the Bollinger band edge → breakout/reversion zone")
        if "Regime" in names and adx_v >= 25:
            bits.append("ADX > 25 confirms a directional regime, not chop")
        return "; ".join(bits) + "."


def _atr(df: pd.DataFrame, close: pd.Series, period: int = 14) -> float:
    high = df.get("High", close)
    low = df.get("Low", close)
    tr = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


def _score_rsi(value: float, adx: float = 0.0) -> float:
    # Bullish below 30, bearish above 70, linear ramp.
    # In strong trends the extremes mean continuation, not reversion, so we
    # invert the sign and damp the magnitude — chasing a 0-RSI on a falling
    # knife is exactly the trade we don't want to flag as BUY.
    if value <= 30:
        base = min(1.0, (30 - value) / 30)
        return -base * 0.6 if adx >= 30 else base
    if value >= 70:
        base = -min(1.0, (value - 70) / 30)
        return -base * 0.6 if adx >= 30 else base
    if value <= 45:
        return (45 - value) / 60.0
    if value >= 55:
        return -(value - 55) / 60.0
    return 0.0


def _score_macd(hist: float, spot: float) -> float:
    # Normalise histogram by price so BTC and SPY are on the same scale.
    norm = hist / max(spot, 1e-9)
    return max(-1.0, min(1.0, norm * 400))


def _score_bb(pctb: float, adx: float = 0.0) -> float:
    # %b < 0 = below lower band (bullish reversion), > 1 = above upper (bearish).
    # Mean-reversion logic stands down in strong trends — riding the upper
    # band is continuation, not a fade signal.
    if adx >= 30:
        return 0.0
    damp = 1.0 if adx < 18 else max(0.0, (30 - adx) / 12)
    if pctb <= 0.1:
        return min(1.0, (0.2 - pctb) * 4) * damp
    if pctb >= 0.9:
        return -min(1.0, (pctb - 0.8) * 4) * damp
    return 0.0


def _score_trend(close: pd.Series) -> float:
    if len(close) < 50:
        return 0.0
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    last = float(close.iloc[-1])
    if sma50 <= 0:
        return 0.0
    spread = (sma20 - sma50) / sma50
    above = 0.3 if last > sma20 else -0.3
    return max(-1.0, min(1.0, spread * 20 + above))


def _score_regime(regime: RegimeLabel, adx: float) -> float:
    if regime is RegimeLabel.TRENDING_UP:
        return min(1.0, adx / 40)
    if regime is RegimeLabel.TRENDING_DOWN:
        return -min(1.0, adx / 40)
    return 0.0


def _score_volume(df: pd.DataFrame, close: pd.Series, window: int = 20) -> float:
    """Volume surge confirmation: high volume in the direction of price move.

    Institutional accumulation/distribution tends to arrive on above-average
    volume. A 2× surge with a positive price move = bullish confirmation.
    Returns 0.0 if no Volume column is available (e.g., Coinbase feed).
    """
    if "Volume" not in df.columns or len(df) < window + 2:
        return 0.0
    try:
        vol = df["Volume"].astype(float)
        avg_vol = float(vol.iloc[-window - 1:-1].mean())
        last_vol = float(vol.iloc[-1])
        if avg_vol <= 0 or last_vol <= 0:
            return 0.0
        ratio = last_vol / avg_vol
        price_move = float(close.iloc[-1]) - float(close.iloc[-2])
        direction = 1.0 if price_move > 0 else (-1.0 if price_move < 0 else 0.0)
        surge = min(1.0, max(0.0, (ratio - 0.8) / 1.7))
        return direction * surge
    except Exception:
        return 0.0


def _score_momentum(close: pd.Series, period: int = 10) -> float:
    """Rate-of-change momentum: price % change over `period` bars, normalised.

    Captures medium-term momentum that RSI and MACD can lag. A 5% move over
    10 bars maps to ±1.0. Works for both crypto (high vol) and equities.
    """
    if len(close) < period + 2:
        return 0.0
    try:
        roc = float(close.iloc[-1]) / max(float(close.iloc[-(period + 1)]), 1e-9) - 1.0
        return max(-1.0, min(1.0, roc * 18))
    except Exception:
        return 0.0


def detect(df: pd.DataFrame, symbol: str = "", timeframe: str = "") -> Alert:
    """Triangulate RSI / MACD / BB / Trend / Regime / Volume / Momentum into a
    directional alert.

    Seven-factor model. Volume and Momentum extend the original five-factor
    framework with institutional-flow and price-persistence signals from the
    discretionary-investor playbooks (O'Neil CANSLIM volume confirmation;
    Livermore/Druckenmiller momentum continuation). All factors are normalised
    to [-1, 1] before weighting so no single indicator dominates.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return _hold_fallback(symbol, timeframe, 0.0)

    try:
        close = df["Close"]
        spot = float(close.iloc[-1])

        last_rsi = float(rsi(close).iloc[-1])
        bb_row = bollinger(close).iloc[-1]
        pctb = float(bb_row["bb_pctb"])
        macd_hist = float(macd(close)["hist"].iloc[-1])
        regime_result = classify_regime(df)

        vol_score = _score_volume(df, close)
        mom_score = _score_momentum(close)

        contributions = [
            {"name": "RSI",      "score": _score_rsi(last_rsi, regime_result.adx), "weight": 0.22},
            {"name": "MACD",     "score": _score_macd(macd_hist, spot),             "weight": 0.18},
            {"name": "BB %b",    "score": _score_bb(pctb, regime_result.adx),       "weight": 0.17},
            {"name": "Trend",    "score": _score_trend(close),                      "weight": 0.18},
            {"name": "Regime",   "score": _score_regime(regime_result.regime, regime_result.adx), "weight": 0.12},
            {"name": "Volume",   "score": vol_score,                                "weight": 0.08},
            {"name": "Momentum", "score": mom_score,                                "weight": 0.05},
        ]
        score = sum(c["score"] * c["weight"] for c in contributions)
        score = max(-1.0, min(1.0, score))

        # Agreement boost: when ≥3 contributors point the same direction the
        # signal earns extra confidence even if magnitudes are modest.
        directional = [c["score"] for c in contributions if abs(c["score"]) > 0.05]
        if directional:
            agree_pos = sum(1 for v in directional if v > 0)
            agree_neg = sum(1 for v in directional if v < 0)
            agreement = max(agree_pos, agree_neg) / max(len(directional), 1)
        else:
            agreement = 0.0

        confidence = min(95.0, abs(score) * 55 + agreement * 40)

        atr_value = max(_atr(df, close), spot * 0.005)
        if score > 0:
            stop, target = spot - 1.5 * atr_value, spot + 2.5 * atr_value
        elif score < 0:
            stop, target = spot + 1.5 * atr_value, spot - 2.5 * atr_value
        else:
            stop, target = spot - atr_value, spot + atr_value
        rr = abs(target - spot) / max(abs(spot - stop), 1e-9)

        action = action_from_score(score)
        horizon_call: HorizonCall = classify_horizon(
            timeframe, regime_result.regime, regime_result.adx, score
        )

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
            horizon=horizon_call.horizon,
            horizon_rationale=horizon_call.rationale,
            regime=regime_result.regime.value,
            contributions=contributions,
            indicator_snapshot={
                "rsi": last_rsi,
                "bb_pctb": pctb,
                "macd_hist": macd_hist,
                "adx": regime_result.adx,
                "atr_pct": atr_value / max(spot, 1e-9),
                "volume_score": vol_score,
                "momentum_roc": mom_score,
            },
        )
    except Exception:
        spot = float(df["Close"].iloc[-1]) if not df.empty else 0.0
        return _hold_fallback(symbol, timeframe, spot)


def _hold_fallback(symbol: str, timeframe: str, spot: float) -> Alert:
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
        horizon=Horizon.SWING,
        horizon_rationale="insufficient data",
        regime="",
        contributions=[],
        indicator_snapshot={},
    )
