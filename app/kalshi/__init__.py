"""Kalshi crypto integration — API client + decision engine.

Surfaces Kalshi crypto event markets (15-min Up/Down, hourly/daily/weekly
range markets) alongside the dashboard's spot price feed and computes:

- Implied probability per side (from the orderbook's best yes/no).
- Model probability per side (from spot drift, time-to-close, vol).
- Direction + model confidence (how strongly the model agrees with one side).
- Expected value in cents-per-$1 per side, given the Kalshi yes-price.

No automated execution. Outputs are advisory; the user places the bet.
"""

from app.kalshi.ai_opinion import Opinion, get_opinion
from app.kalshi.client import KalshiClient, KalshiMarket, KalshiOrderbook
from app.kalshi.council import CouncilResult, evaluate as evaluate_council
from app.kalshi.decisions import Decision, score_market, score_event
from app.kalshi.spot import SpotQuote, get_spot_price

__all__ = [
    "CouncilResult",
    "Decision",
    "KalshiClient",
    "KalshiMarket",
    "KalshiOrderbook",
    "Opinion",
    "SpotQuote",
    "evaluate_council",
    "get_opinion",
    "get_spot_price",
    "score_event",
    "score_market",
]
