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
class ResolutionCondition:
    """Plain-English description of what a chosen side actually pays out on.

    The point is to remove the ambiguity between the Kalshi ticket label
    (YES / NO) and the underlying directional bet (price closes above /
    below some level). The card UI pairs ``side`` with ``relation`` so the
    user reads "CONFIRMED YES → BTC closes ABOVE $80,000" instead of
    having to infer which side of the strike YES corresponds to.
    """

    side: str                # "YES" or "NO" — the Kalshi ticket label
    relation: str            # "above" / "below" / "between" / "outside" / ""
    threshold: str           # formatted strike, e.g. "$80,000" or "$79,500–$79,750"
    close_phrase: str        # "at 12:00 UTC" / "by Fri Nov 15 17:00 UTC" / ""
    summary: str             # full sentence — "BTC closes above $80,000 at 12:00 UTC"
    symbol: str = ""         # "BTC" / "ETH" / "SOL" — convenience for the UI


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
    # Plain-English win condition for the chosen direction (None on PASS).
    resolution: ResolutionCondition | None = None

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


def _close_phrase(close_time: float) -> str:
    """Format the close time as "at HH:MM UTC" (same-day) or
    "by Day Mon DD HH:MM UTC" (further out). Empty string if unknown."""
    import time as _time
    from datetime import datetime, timezone

    if not close_time:
        return ""
    try:
        dt = datetime.fromtimestamp(close_time, tz=timezone.utc)
        secs = close_time - _time.time()
        if secs < 24 * 3600:
            return f"at {dt.strftime('%H:%M UTC')}"
        return f"by {dt.strftime('%a %b %d %H:%M UTC')}"
    except Exception:
        return ""


def _yes_relation_and_threshold(
    market: KalshiMarket,
) -> tuple[str, str]:
    """Map a market's strike/title to ('above'|'below'|'between'|'outside'|'',
    formatted threshold) for the YES side. NO is the inverse: above↔below,
    between↔outside. Empty relation means the market shape wasn't recognised."""
    strike_type = (market.strike_type or "").lower()
    floor = market.floor_strike
    cap = market.cap_strike

    if strike_type == "greater" and floor is not None:
        return "above", f"${floor:,.2f}"
    if strike_type == "less" and floor is not None:
        return "below", f"${floor:,.2f}"
    if strike_type == "between" and floor is not None and cap is not None:
        return "between", f"${floor:,.2f}–${cap:,.2f}"

    title = (market.title + " " + market.subtitle).lower()
    if floor is not None and ("up" in title or "above" in title or "higher" in title):
        return "above", f"${floor:,.2f}"
    if floor is not None and ("down" in title or "below" in title or "lower" in title):
        return "below", f"${floor:,.2f}"
    return "", ""


def _flip_relation(rel: str) -> str:
    return {
        "above": "below",
        "below": "above",
        "between": "outside",
        "outside": "between",
    }.get(rel, rel)


def resolution_for_side(
    market: KalshiMarket, symbol: str, side: str,
) -> ResolutionCondition:
    """Plain-English win condition for buying ``side`` on this market.

    Always returns a populated ``ResolutionCondition``. If the market shape
    isn't parseable, ``relation`` is empty and ``summary`` falls back to a
    "see full Kalshi rules" disclaimer so the UI can still display a
    coherent badge.
    """
    side = side.upper()
    close_phrase = _close_phrase(market.close_time)
    yes_rel, threshold = _yes_relation_and_threshold(market)

    if not yes_rel:
        summary = (
            f"{side} on this market — see full Kalshi rules "
            "(market shape not auto-parsed)"
            + (f" {close_phrase}" if close_phrase else "")
        )
        return ResolutionCondition(
            side=side, relation="", threshold="",
            close_phrase=close_phrase, summary=summary, symbol=symbol,
        )

    relation = yes_rel if side == "YES" else _flip_relation(yes_rel)

    when = f" {close_phrase}" if close_phrase else ""
    if relation in ("above", "below"):
        summary = f"{symbol} closes {relation} {threshold}{when}"
    elif relation == "between":
        summary = f"{symbol} closes between {threshold}{when}"
    elif relation == "outside":
        summary = f"{symbol} closes outside {threshold}{when}"
    else:
        summary = f"{symbol} {threshold}{when}"

    return ResolutionCondition(
        side=side, relation=relation, threshold=threshold,
        close_phrase=close_phrase, summary=summary, symbol=symbol,
    )


def _bet_summary(market: KalshiMarket, symbol: str) -> str:
    """Plain-English description of what YES means and when it settles.

    Kept as a thin wrapper over ``resolution_for_side`` so the legacy
    "YES = …" sentence (rendered in the card title and logged to the
    pattern tracker) stays unchanged.
    """
    yes = resolution_for_side(market, symbol, "YES")
    if not yes.relation:
        return yes.summary
    when = f" {yes.close_phrase}" if yes.close_phrase else ""
    if yes.relation == "above":
        return f"YES = {symbol} closes ≥ {yes.threshold}{when}"
    if yes.relation == "below":
        return f"YES = {symbol} closes < {yes.threshold}{when}"
    if yes.relation == "between":
        return f"YES = {symbol} closes between {yes.threshold}{when}"
    if yes.relation == "outside":
        return f"YES = {symbol} closes outside {yes.threshold}{when}"
    return f"YES{when}"


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
            resolution=None,
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
        resolution=resolution_for_side(market, spot.symbol, direction),
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


# ---------------------------------------------------------------------------
# Earning-potential helpers — bridge "edge per $1" → "expected $ on my bet"
# ---------------------------------------------------------------------------

def _effective_ask_cents(ask_cents: int) -> int:
    """Floor ask at 1¢ so price/payout/fee math all share one basis.

    A 0¢ ask is non-tradable on Kalshi (no offers); treating it as 1¢
    keeps every helper internally consistent (fee, payout, breakeven
    all derived from the same effective price) and avoids the bug
    where ev_per_dollar inflates because fee was computed at p=0.
    """
    return max(1, int(ask_cents))


def kalshi_fee_per_contract(ask_cents: int) -> float:
    """Kalshi trading fee per contract, in dollars. Uses the published
    formula: round_up(0.07 × price × (1 - price) × 100) / 100, applied to
    the winning leg only.

    For ask = 10¢ → ~1¢, ask = 50¢ → ~2¢, ask = 1¢ → ~1¢ floor.
    """
    eff = _effective_ask_cents(ask_cents)
    p = eff / 100.0
    raw_dollars = 0.07 * p * (1.0 - p)
    if raw_dollars <= 0:
        return 0.0
    # Round up to the nearest cent
    return math.ceil(raw_dollars * 100.0) / 100.0


def breakeven_prob(ask_cents: int, fee_aware: bool = True) -> float:
    """Probability you'd need to clear breakeven at the given ask.

    With fee_aware=True (default), accounts for the Kalshi fee paid on
    a winning settlement, so net-EV ≥ 0 lines up with the displayed
    breakeven. With fee_aware=False, returns the raw price (gross
    breakeven), useful for showing the basis traders are used to.

    Derivation (YES, ask = p, fee per $1 staked on win = f_d):
      EV_net = m*((1-p)/p - f_d) - (1-m)
      EV_net ≥ 0  ⟺  m ≥ (1 + f_d) / (1/p + f_d) = p*(1+f_d)/(1+p*f_d)
    """
    eff = _effective_ask_cents(ask_cents)
    p = eff / 100.0
    if not fee_aware:
        return p
    fee = kalshi_fee_per_contract(ask_cents)
    f_d = fee / p  # fee per $1 staked, on a winning outcome
    return max(0.0, min(1.0, p * (1.0 + f_d) / (1.0 + p * f_d)))


def net_ev_per_dollar(model_prob: float, ask_cents: int) -> tuple[float, float]:
    """Expected net P&L per $1 staked, after Kalshi fees. Returns
    (net_ev_per_dollar, fee_drag_per_dollar_on_win)."""
    eff = _effective_ask_cents(ask_cents)
    p = eff / 100.0
    fee = kalshi_fee_per_contract(ask_cents)
    # If you stake $1 you own 1/p contracts. On a win each contract pays
    # $1 gross, so gross win profit = (1-p)/p per $1. Fees are charged
    # only on settlement of winning contracts.
    fee_per_dollar_on_win = fee / p
    net_win_per_dollar = (1.0 - p) / p - fee_per_dollar_on_win
    net_ev = model_prob * net_win_per_dollar - (1.0 - model_prob)
    return net_ev, fee_per_dollar_on_win


def expected_dollars_at_stake(
    model_prob: float, ask_cents: int, stake_dollars: float
) -> dict:
    """Translate edge per $1 into the dollar numbers a human cares about.

    Returns:
      gross_win_$ — gross profit if the bet hits (no fees)
      fees_$ — Kalshi fee paid on a winning settlement
      net_win_$ — net profit if the bet hits (after fees)
      max_loss_$ — what you lose if the bet misses (= stake)
      net_expected_$ — probability-weighted net dollars
      contracts — number of contracts purchased
    """
    eff = _effective_ask_cents(ask_cents)
    p = eff / 100.0
    contracts = stake_dollars / p if p > 0 else 0.0
    fee_per = kalshi_fee_per_contract(ask_cents)
    gross_win = contracts * (1.0 - p)
    fees = contracts * fee_per
    net_win = gross_win - fees
    max_loss = stake_dollars
    net_expected = model_prob * net_win - (1.0 - model_prob) * max_loss
    return {
        "contracts": contracts,
        "gross_win": gross_win,
        "fees": fees,
        "net_win": net_win,
        "max_loss": max_loss,
        "net_expected": net_expected,
    }


def annualized_roi(net_ev_per_dollar: float, horizon_seconds: float) -> float:
    """Convert "X cents per $1 over T seconds" into an annualized return
    (decimal, e.g. 0.42 = 42%/yr). Capped at ±100×/yr because tiny-horizon
    edges otherwise produce nonsense like 50,000%/yr.

    Uses simple, not compound, scaling — the user is comparing a one-shot
    bet to other one-shot bets, not reinvesting at every bar.
    """
    if horizon_seconds <= 0:
        return 0.0
    seconds_per_year = 365.25 * 24 * 3600
    scale = seconds_per_year / horizon_seconds
    return max(-100.0, min(100.0, net_ev_per_dollar * scale))


def model_quality_factor(warnings: list[str]) -> tuple[float, str]:
    """Discount factor (0..1) on EV when the model is operating on a
    fallback prior, with a human-readable label. Returns (factor, label).

    1.0 = full confidence; 0.5 = half discount; 0.1 = "this number is
    largely cosmetic, treat with skepticism".
    """
    text = " ".join(warnings).lower()
    if "unrecognised" in text or "unrecognized" in text:
        return 0.10, "model fallback (50% prior) — ignore EV magnitude"
    if "zero volume" in text or "stale" in text:
        return 0.50, "stale book — EV may not be fillable at displayed price"
    if "less than a minute" in text:
        return 0.40, "≤ 1 min to close — vol model unreliable"
    if "wide spread" in text:
        return 0.70, "wide spread — implied prob is noisy"
    return 1.0, "model OK"
