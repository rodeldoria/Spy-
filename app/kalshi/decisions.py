"""Decision engine for Kalshi crypto markets.

Given a market (with Kalshi's implied YES/NO prices) and a spot quote,
compute:

1. A model probability for YES using a Geometric-Brownian-Motion (GBM) /
   log-normal approximation over the time-to-close. This is a benchmark
   probability assuming a driftless random walk with realised vol — not
   a forecast.
2. The "edge" of each side: model probability minus implied probability.
3. Expected value per $1 staked on each side, using the orderbook ask.
4. A direction recommendation (YES / NO / PASS) and a model-confidence
   percentage based on how strongly the model disagrees with the market.

The output is advisory. The point isn't to predict where price goes — it's
to flag when the Kalshi book is mispriced relative to a benchmark vol
model. If the model agrees with the book (no edge), the recommendation is
PASS, regardless of how "confident" the implied probability looks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from app.kalshi.client import KalshiMarket
from app.kalshi.spot import SpotQuote, default_sigma_per_min


# Minimum edge (model_p - implied_p) below which we recommend PASS, even if
# the side has a positive EV. Below this threshold, the model uncertainty
# swamps the supposed edge.
DEFAULT_MIN_EDGE = 0.04   # 4 percentage points
DEFAULT_MIN_EV = 0.02     # 2 cents per $1 — covers fees + slippage cushion


@dataclass(frozen=True)
class SideAssessment:
    """Per-side (YES or NO) scoring."""

    side: str                # "YES" or "NO"
    implied_prob: float      # from Kalshi book (0-1)
    model_prob: float        # from GBM model (0-1)
    ask_cents: int           # what you'd pay to buy
    payout: float            # multiplier if it resolves true (1 / ask)
    edge: float              # model_prob - implied_prob
    ev_per_dollar: float     # expected $ return per $1 staked
    kelly_fraction: float    # optimal bankroll fraction (capped to [0, 0.25])


@dataclass(frozen=True)
class Decision:
    """Full assessment for a single Kalshi market."""

    market_ticker: str
    title: str
    horizon_seconds: float
    spot_price: float
    spot_source: str
    sigma_per_min: float
    yes_side: SideAssessment
    no_side: SideAssessment
    direction: str           # "YES", "NO", or "PASS"
    confidence_pct: float    # 0-100, model conviction in chosen direction
    reasoning: str
    bet_summary: str = ""    # plain-English: "YES = BTC ≥ $80,000 at 12:00 UTC"
    close_time: float = 0.0  # epoch seconds — for sorting by date
    warnings: list[str] = field(default_factory=list)

    @property
    def chosen(self) -> SideAssessment | None:
        if self.direction == "YES":
            return self.yes_side
        if self.direction == "NO":
            return self.no_side
        return None


# ---------------------------------------------------------------------------
# Probability model
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gbm_prob_above(spot: float, strike: float, sigma_total: float) -> float:
    """P(S_T >= K) under driftless GBM with total log-return stdev sigma_total.

    Driftless is deliberate: at minute-to-hour horizons, drift is a rounding
    error vs realised vol on crypto. Adding a guessed drift inflates false
    confidence without improving calibration.
    """
    if sigma_total <= 0 or spot <= 0 or strike <= 0:
        return 1.0 if spot >= strike else 0.0
    z = math.log(strike / spot) / sigma_total
    # P(S_T >= K) = 1 - Phi(z), where Z = log(S_T/S_0)/sigma is standard normal.
    return 1.0 - _norm_cdf(z)


def gbm_prob_between(
    spot: float, floor: float, cap: float, sigma_total: float
) -> float:
    """P(floor <= S_T <= cap) under driftless GBM."""
    if sigma_total <= 0 or spot <= 0:
        return 1.0 if (floor <= spot <= cap) else 0.0
    if cap <= floor:
        return 0.0
    p_above_floor = gbm_prob_above(spot, floor, sigma_total)
    p_above_cap = gbm_prob_above(spot, cap, sigma_total)
    return max(0.0, p_above_floor - p_above_cap)


def model_prob_yes(market: KalshiMarket, spot: SpotQuote) -> tuple[float, str]:
    """Compute P(YES resolves true) under the benchmark model.

    Returns (probability, explanation). Handles:
    - 15-min Up/Down markets (strike encoded in title / floor_strike).
    - Range markets ("$X or above" / "$X to $Y") via strike_type and
      floor/cap fields.
    - Falls back to a "no model" sentinel (50%) when the market shape is
      not recognised; the caller should suppress the recommendation in
      that case via the `warnings` field.
    """
    sigma_min = spot.sigma_per_min or default_sigma_per_min(spot.symbol)
    seconds = max(1.0, market.seconds_to_close)
    minutes = seconds / 60.0
    sigma_total = sigma_min * math.sqrt(minutes)

    strike_type = (market.strike_type or "").lower()
    floor = market.floor_strike
    cap = market.cap_strike

    # Range markets: prefer the structured strike fields when present.
    if strike_type == "greater" and floor is not None:
        p = gbm_prob_above(spot.price, floor, sigma_total)
        return p, f"P(price ≥ ${floor:,.2f}) over {minutes:.1f} min at σ={sigma_min*100:.3f}%/min"
    if strike_type == "less" and floor is not None:
        p = 1.0 - gbm_prob_above(spot.price, floor, sigma_total)
        return p, f"P(price < ${floor:,.2f}) over {minutes:.1f} min at σ={sigma_min*100:.3f}%/min"
    if strike_type == "between" and floor is not None and cap is not None:
        p = gbm_prob_between(spot.price, floor, cap, sigma_total)
        return p, f"P(${floor:,.2f} ≤ price ≤ ${cap:,.2f}) over {minutes:.1f} min"

    # 15-min Up/Down — Kalshi tags these as the strike == event target price,
    # and "yes" usually means "price > strike at close" but the side label
    # varies. We use the market title to detect direction.
    title = (market.title + " " + market.subtitle).lower()
    if floor is not None and ("up" in title or "above" in title or "higher" in title):
        p = gbm_prob_above(spot.price, floor, sigma_total)
        return p, f"P(Up: price ≥ ${floor:,.2f}) over {minutes:.1f} min"
    if floor is not None and ("down" in title or "below" in title or "lower" in title):
        p = 1.0 - gbm_prob_above(spot.price, floor, sigma_total)
        return p, f"P(Down: price < ${floor:,.2f}) over {minutes:.1f} min"

    # Unknown shape — return 50% and let the caller suppress.
    return 0.5, "unrecognised market shape; suppressing model recommendation"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _kelly(prob: float, payout: float) -> float:
    """Kelly fraction for a binary bet at multiplier `payout`.

    f* = (p * b - q) / b, where b = payout - 1 (net odds), q = 1 - p.
    Capped to [0, 0.25] — full Kelly is too aggressive in practice and the
    model probability has its own error bars.
    """
    b = payout - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - prob
    f = (prob * b - q) / b
    return max(0.0, min(0.25, f))


def _assess_side(
    side: str,
    implied: float,
    model: float,
    ask_cents: int,
) -> SideAssessment:
    ask = max(1, ask_cents)  # avoid div-by-zero; 1 cent floor
    payout = 100.0 / ask
    edge = model - implied
    # EV per $1: model_p * payout - 1, assuming you pay `ask/100` per share
    # and receive $1 if YES resolves true.
    ev = model * payout - 1.0
    return SideAssessment(
        side=side,
        implied_prob=implied,
        model_prob=model,
        ask_cents=ask,
        payout=payout,
        edge=edge,
        ev_per_dollar=ev,
        kelly_fraction=_kelly(model, payout),
    )


def _bet_summary(market: KalshiMarket, symbol: str) -> str:
    """Plain-English description of what YES means and when it settles."""
    import time as _time
    from datetime import datetime, timezone

    when = ""
    try:
        if market.close_time:
            dt = datetime.fromtimestamp(market.close_time, tz=timezone.utc)
            now = _time.time()
            secs = market.close_time - now
            if secs < 24 * 3600:
                when = f" at {dt.strftime('%H:%M UTC')}"
            else:
                when = f" by {dt.strftime('%a %b %d %H:%M UTC')}"
    except Exception:
        pass

    strike_type = (market.strike_type or "").lower()
    floor = market.floor_strike
    cap = market.cap_strike

    if strike_type == "greater" and floor is not None:
        return f"YES = {symbol} closes ≥ ${floor:,.2f}{when}"
    if strike_type == "less" and floor is not None:
        return f"YES = {symbol} closes < ${floor:,.2f}{when}"
    if strike_type == "between" and floor is not None and cap is not None:
        return f"YES = {symbol} closes between ${floor:,.2f} and ${cap:,.2f}{when}"

    title = (market.title + " " + market.subtitle).lower()
    if floor is not None and ("up" in title or "above" in title or "higher" in title):
        return f"YES = {symbol} closes ≥ ${floor:,.2f}{when} (Up market)"
    if floor is not None and ("down" in title or "below" in title or "lower" in title):
        return f"YES = {symbol} closes < ${floor:,.2f}{when} (Down market)"
    return f"YES{when} — see full Kalshi rules (market shape not auto-parsed)"


def score_market(
    market: KalshiMarket,
    spot: SpotQuote,
    *,
    min_edge: float = DEFAULT_MIN_EDGE,
    min_ev: float = DEFAULT_MIN_EV,
) -> Decision:
    """Score a single Kalshi market against the spot quote.

    Returns a Decision with the recommended direction (YES, NO, or PASS)
    plus a model-confidence percentage. PASS means: the book's pricing is
    inside the model's noise band, so there's no edge worth taking.
    """
    warnings: list[str] = []

    model_yes, why = model_prob_yes(market, spot)
    if "unrecognised" in why:
        warnings.append(why)
    model_no = 1.0 - model_yes

    yes = _assess_side("YES", market.implied_prob_yes, model_yes, market.yes_ask)
    no = _assess_side("NO", market.implied_prob_no, model_no, market.no_ask)

    # Choose direction by the larger positive edge that also meets minimum EV.
    cand = []
    if yes.edge >= min_edge and yes.ev_per_dollar >= min_ev:
        cand.append(("YES", yes))
    if no.edge >= min_edge and no.ev_per_dollar >= min_ev:
        cand.append(("NO", no))

    bet_summary = _bet_summary(market, spot.symbol)

    if not cand:
        direction = "PASS"
        # Confidence in PASS = how close the book is to model. Smaller |edge|
        # → higher confidence we should stand down.
        smaller_edge = min(abs(yes.edge), abs(no.edge))
        confidence = max(0.0, min(100.0, (1.0 - smaller_edge / max(min_edge, 1e-6)) * 50.0))
        reasoning = (
            f"No edge ≥ {min_edge*100:.0f}%. Book at {market.yes_mid:.0f}¢ YES / "
            f"{market.no_mid:.0f}¢ NO; model {model_yes*100:.1f}% YES."
        )
        return Decision(
            market_ticker=market.ticker,
            title=market.title or market.subtitle,
            horizon_seconds=market.seconds_to_close,
            spot_price=spot.price,
            spot_source=spot.source,
            sigma_per_min=spot.sigma_per_min or default_sigma_per_min(spot.symbol),
            yes_side=yes,
            no_side=no,
            direction=direction,
            confidence_pct=confidence,
            reasoning=reasoning,
            bet_summary=bet_summary,
            close_time=market.close_time,
            warnings=warnings,
        )

    # Pick the larger edge.
    cand.sort(key=lambda x: -x[1].edge)
    direction, chosen = cand[0]
    # Confidence: map edge size to 0-100. 4pp edge → 50%, 12pp edge → 90%,
    # 20pp edge → ~99%. Logistic so we don't overstate huge edges.
    confidence = 100.0 * (1.0 / (1.0 + math.exp(-(chosen.edge - 0.08) * 25.0)))
    confidence = max(0.0, min(99.0, confidence))

    reasoning = (
        f"{direction} edge {chosen.edge*100:+.1f}pp "
        f"(model {chosen.model_prob*100:.1f}% vs book {chosen.implied_prob*100:.1f}%). "
        f"Buy at {chosen.ask_cents}¢ → {chosen.payout:.2f}x payout, "
        f"EV {chosen.ev_per_dollar*100:+.1f}¢/$1. {why}"
    )

    # Liquidity sanity check — wide spreads make the implied probability
    # noisy. Surface as a warning but don't suppress.
    spread = market.yes_ask - market.yes_bid
    if spread >= 8:
        warnings.append(f"Wide spread ({spread}¢) — implied prob is noisy.")
    if market.volume == 0:
        warnings.append("Zero volume so far — book may be stale.")
    if market.seconds_to_close < 60:
        warnings.append("Less than a minute to close — vol model unreliable.")

    return Decision(
        market_ticker=market.ticker,
        title=market.title or market.subtitle,
        horizon_seconds=market.seconds_to_close,
        spot_price=spot.price,
        spot_source=spot.source,
        sigma_per_min=spot.sigma_per_min or default_sigma_per_min(spot.symbol),
        yes_side=yes,
        no_side=no,
        direction=direction,
        confidence_pct=confidence,
        reasoning=reasoning,
        bet_summary=bet_summary,
        close_time=market.close_time,
        warnings=warnings,
    )


def score_event(
    markets: Iterable[KalshiMarket],
    spot: SpotQuote,
    *,
    min_edge: float = DEFAULT_MIN_EDGE,
    min_ev: float = DEFAULT_MIN_EV,
) -> list[Decision]:
    """Score every market in an event. Returns decisions sorted by best edge."""
    decisions = [
        score_market(m, spot, min_edge=min_edge, min_ev=min_ev) for m in markets
    ]
    decisions.sort(key=lambda d: -max(d.yes_side.edge, d.no_side.edge))
    return decisions
