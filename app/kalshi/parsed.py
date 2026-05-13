"""Bridge from vision-parsed markets into the decision engine.

A `ParsedMarket` lists multiple sides (Up/Down, or one strike row per side
of a range market). Each side becomes a synthetic `KalshiMarket` so the
existing scoring code runs unchanged: implied prob from the side's prob_pct,
ask price derived from the payout multiplier, strike fields from the side
shape (greater / less / between / up / down).
"""

from __future__ import annotations

import time

from app.kalshi.client import KalshiMarket
from app.kalshi.decisions import Decision, score_market
from app.kalshi.spot import SpotQuote
from app.kalshi.vision import ParsedMarket, ParsedSide


def market_from_parsed(
    parsed: ParsedMarket,
    side: ParsedSide,
    *,
    default_close_in_seconds: int = 15 * 60,
) -> KalshiMarket:
    """Build a `KalshiMarket` for a single side of a parsed event.

    The side's `prob_pct` becomes the implied YES probability for "this side
    resolves true". The payout multiplier (e.g. 2.01x) sets the YES ask
    price: ask_cents ≈ round(100 / payout). NO side fields are derived from
    the same data so the engine can still compute a 2-sided assessment for
    consistency, though only one side per parsed entry is meaningful.
    """
    prob = max(0.0, min(1.0, side.prob_pct / 100.0))
    yes_ask = max(1, min(99, round(100.0 / side.payout))) if side.payout else int(round(prob * 100))
    yes_bid = max(0, min(99, yes_ask - 1))
    no_ask = max(1, min(99, 100 - yes_bid))
    no_bid = max(0, 100 - yes_ask)

    close_secs = parsed.time_remaining_seconds if parsed.time_remaining_seconds is not None else default_close_in_seconds
    close_time = time.time() + max(1, close_secs)

    # Map the side shape to KalshiMarket strike fields.
    strike_type = side.strike_type
    floor = side.strike
    cap = None

    if strike_type == "up":
        strike_type_mapped = "greater"
        floor = side.strike if side.strike is not None else parsed.target_price
    elif strike_type == "down":
        strike_type_mapped = "less"
        floor = side.strike if side.strike is not None else parsed.target_price
    else:
        strike_type_mapped = strike_type

    title = f"{parsed.title} — {side.label}"

    return KalshiMarket(
        ticker=f"PARSED-{parsed.symbol}-{strike_type_mapped}-{floor or 0:.2f}",
        event_ticker=f"PARSED-{parsed.symbol}",
        title=title,
        subtitle=side.label,
        status="active",
        yes_bid=int(yes_bid),
        yes_ask=int(yes_ask),
        no_bid=int(no_bid),
        no_ask=int(no_ask),
        last_price=int(yes_ask),
        volume=int(parsed.volume_usd or 0),
        open_interest=0,
        close_time=close_time,
        expiration_time=close_time,
        strike_type=strike_type_mapped,
        floor_strike=float(floor) if floor is not None else None,
        cap_strike=float(cap) if cap is not None else None,
        raw=parsed.raw,
    )


def score_parsed_markets(
    parsed: list[ParsedMarket],
    spot_by_symbol: dict[str, SpotQuote],
    *,
    min_edge: float = 0.04,
    min_ev: float = 0.02,
) -> list[tuple[ParsedMarket, list[Decision]]]:
    """Score every side of every parsed market against the symbol's spot.

    Returns one (parsed_market, decisions) pair per parsed market. Markets
    whose symbol lacks a spot quote yield an empty decisions list.
    """
    out: list[tuple[ParsedMarket, list[Decision]]] = []
    for pm in parsed:
        spot = spot_by_symbol.get(pm.symbol)
        if not spot:
            out.append((pm, []))
            continue
        decisions = [
            score_market(market_from_parsed(pm, s), spot, min_edge=min_edge, min_ev=min_ev)
            for s in pm.sides
        ]
        out.append((pm, decisions))
    return out
