"""Confidence triangulation engine for Kalshi market plays.

For any Kalshi event we ask: "is this actually a good play, and what
exactly should we do?" — we triangulate across independent signals so
the confidence reflects how many of them agree:

  1. CROWD       — what the Kalshi price itself says (the "wisdom of
                   crowds" prior).
  2. PATTERNS    — technical pattern engine (crypto only — Bollinger
                   squeeze, EMA stack, mean-reversion, round-number
                   magnets, etc).
  3. INFLUENCERS — Perplexity-summarised commentary from the
                   market-moving accounts (crypto only).
  4. SESSION     — temporal calibration: are we currently in a session
                   the model historically wins in, or one it loses in?
  5. NEWS        — Perplexity headline brief (only fetched when the
                   user opts in, since it's a paid call per asset).

Each signal votes BULL / BEAR / NEUTRAL with a 0..1 confidence. The
recommender:
  - resolves the favoured market bucket (most likely outcome),
  - figures out whether external signals AGREE with the crowd
    (=ride the consensus, higher confidence) or DISAGREE
    (=potential contrarian play, lower confidence),
  - emits a concrete action: BUY YES / BUY NO / PASS,
  - computes the dollar math (stake → max payout / max loss),
  - suggests a horizon (hold to Kalshi close vs. buy options on the
    underlying for longer-dated convexity).

Weights are controllable so the user can tune them based on their own
returns. We expose `DEFAULT_WEIGHTS` and accept any subset override.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

DEFAULT_WEIGHTS: dict[str, float] = {
    "crowd": 1.0,
    "patterns": 0.9,
    "influencers": 0.6,
    "session": 0.4,
    "news": 0.7,
}


@dataclass
class SignalVote:
    name: str
    verdict: str            # "BULL" | "BEAR" | "NEUTRAL" | "N/A"
    confidence: float       # 0..1
    detail: str             # plain-English explanation (≤180 chars)


@dataclass
class PlayRecommendation:
    # Core verdict
    action: str             # "STRONG_BUY" | "BUY" | "WATCH" | "PASS" | "AVOID"
    side: str               # "YES on 0:: 0 bps", "NO", "BUY YES @ $0.69" etc.
    confidence: float       # 0..1 composite
    confidence_label: str   # "HIGH" | "MED" | "LOW"

    # Money math
    contract_price: float   # cost per contract ($0..$1)
    contracts: int          # contracts you can buy with your stake
    stake: float            # dollars in
    max_payout: float       # gross dollars back if you win
    net_win: float          # max_payout - stake
    roi_pct: float          # net_win / stake * 100
    max_loss: float         # = stake (binary contracts)

    # 3-step confirmation
    votes: list[SignalVote] = field(default_factory=list)
    n_bull: int = 0
    n_bear: int = 0
    n_neutral: int = 0

    # Plain-English advice
    horizon_advice: str = ""   # "Hold to Kalshi close in 6h" / "Consider options"
    why: str = ""              # one-paragraph reasoning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seconds_to_close(event: dict) -> int | None:
    ct = event.get("close_time") or event.get("end_date") or ""
    if not ct:
        return None
    try:
        dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        return int((dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


def _is_crypto_symbol(category: str, series_label: str) -> str | None:
    """Map an event to a crypto symbol if applicable, else None."""
    sl = (series_label or "").lower()
    if category != "Crypto":
        return None
    if "btc" in sl or "bitcoin" in sl or "₿" in sl:
        return "BTC-USD"
    if "eth" in sl or "ethereum" in sl or "⬨" in sl:
        return "ETH-USD"
    if "sol" in sl:
        return "SOL-USD"
    return None


def _crowd_vote(top_p: float) -> SignalVote:
    """Translate the favourite's mid-price into a directional vote."""
    if top_p >= 0.7:
        return SignalVote(
            "Crowd", "BULL", min(1.0, (top_p - 0.5) * 2),
            f"Heavy consensus on the favourite (${top_p:.2f}). Crowd is "
            f"strongly leaning this way.",
        )
    if top_p <= 0.3:
        return SignalVote(
            "Crowd", "BEAR", min(1.0, (0.5 - top_p) * 2),
            f"Heavy consensus AGAINST the favourite (${top_p:.2f}). Most "
            f"money is on alternative outcomes.",
        )
    if top_p >= 0.55:
        return SignalVote(
            "Crowd", "BULL", (top_p - 0.5) * 2,
            f"Mild lean to favourite (${top_p:.2f}). No strong consensus.",
        )
    if top_p <= 0.45:
        return SignalVote(
            "Crowd", "BEAR", (0.5 - top_p) * 2,
            f"Mild lean against (${top_p:.2f}).",
        )
    return SignalVote(
        "Crowd", "NEUTRAL", 1.0 - abs(top_p - 0.5) * 2,
        f"Toss-up at ${top_p:.2f} — crowd is genuinely uncertain.",
    )


def _patterns_vote(symbol: str | None) -> SignalVote:
    if not symbol:
        return SignalVote("Patterns", "N/A", 0.0, "Not a crypto market.")
    try:
        from monte.data.crypto import get_candles
        from monte.signals.patterns import detect_patterns
        candles = get_candles(symbol, "1h", lookback_bars=200)
        if candles is None or candles.empty:
            return SignalVote("Patterns", "N/A", 0.0, "No candles available.")
        # `get_candles()` normalises columns to capitalised OHLCV.
        close_col = "Close" if "Close" in candles.columns else "close"
        spot = float(candles[close_col].iloc[-1])
        bundle = detect_patterns(candles[close_col], spot)
    except Exception as e:
        return SignalVote("Patterns", "N/A", 0.0, f"Engine error: {str(e)[:80]}")

    bias = bundle.net_bias_pp  # signed, capped ±10
    consensus = bundle.consensus.lower() if bundle.consensus else "mixed"
    top_three = ", ".join(p.name for p in bundle.top) or "no active patterns"
    if consensus == "bullish":
        return SignalVote("Patterns", "BULL", min(1.0, abs(bias) / 5),
                          f"Pattern engine bullish ({bias:+.1f}pp): {top_three}.")
    if consensus == "bearish":
        return SignalVote("Patterns", "BEAR", min(1.0, abs(bias) / 5),
                          f"Pattern engine bearish ({bias:+.1f}pp): {top_three}.")
    return SignalVote("Patterns", "NEUTRAL", 0.3,
                      f"Pattern engine mixed/quiet: {top_three}.")


def _influencers_vote(symbol: str | None) -> SignalVote:
    if not symbol:
        return SignalVote("Influencers", "N/A", 0.0, "Not a crypto market.")
    try:
        from monte.intel.influencers import fetch_influencer_pulse
        pulse = fetch_influencer_pulse(symbol)
    except Exception as e:
        return SignalVote("Influencers", "N/A", 0.0, f"Pulse error: {str(e)[:60]}")

    if not pulse.configured:
        return SignalVote("Influencers", "N/A", 0.0, "Perplexity key not set.")
    if pulse.overall_sentiment == "quiet":
        return SignalVote("Influencers", "NEUTRAL", 0.2,
                          "Quiet on big crypto Twitter in last 24h.")
    if pulse.overall_sentiment == "bullish":
        return SignalVote("Influencers", "BULL", min(1.0, abs(pulse.net_bias)),
                          f"Big-name accounts net bullish (bias {pulse.net_bias:+.2f}, "
                          f"{len(pulse.voices)} voices).")
    if pulse.overall_sentiment == "bearish":
        return SignalVote("Influencers", "BEAR", min(1.0, abs(pulse.net_bias)),
                          f"Big-name accounts net bearish (bias {pulse.net_bias:+.2f}).")
    return SignalVote("Influencers", "NEUTRAL", 0.3, "Mixed influencer narrative.")


_SESSION_CACHE: dict[str, tuple[float, object]] = {}
_SESSION_TTL = 120.0   # seconds — re-read calibration files at most every 2m


def _cached_report():
    """Memoise the temporal report so we don't re-read calibration files
    once per Kalshi event on every page refresh."""
    from monte.learning.temporal_report import build_report
    now = time.time()
    cached = _SESSION_CACHE.get("rep")
    if cached and (now - cached[0]) < _SESSION_TTL:
        return cached[1]
    rep = build_report()
    _SESSION_CACHE["rep"] = (now, rep)
    return rep


def _session_vote() -> SignalVote:
    """Use the temporal report to flag if we're in a historically strong
    session (model has won here before) vs. weak."""
    try:
        from monte.learning.temporal_report import _to_pacific
    except Exception:
        return SignalVote("Session", "N/A", 0.0, "Temporal report unavailable.")
    try:
        rep = _cached_report()
        pac = _to_pacific(time.time())
        # Determine current session label
        from monte.learning.temporal_report import _session_for_pst_hour
        h = pac.hour + pac.minute / 60.0
        sess = _session_for_pst_hour(h)
        b = rep.by_session.get(sess)
        if b is None or b.n < 3 or b.hit_rate is None:
            return SignalVote("Session", "NEUTRAL", 0.1,
                              f"In {sess} (PT) — not enough history yet.")
        hr = b.hit_rate
        if hr >= 0.6:
            return SignalVote("Session", "BULL", min(1.0, (hr - 0.5) * 2),
                              f"In {sess} — model historically wins {hr*100:.0f}% here. "
                              f"Favourable session.")
        if hr <= 0.4:
            return SignalVote("Session", "BEAR", min(1.0, (0.5 - hr) * 2),
                              f"In {sess} — model historically only wins {hr*100:.0f}% "
                              f"here. Be cautious.")
        return SignalVote("Session", "NEUTRAL", 0.2,
                          f"In {sess} — historical hit rate near coin flip ({hr*100:.0f}%).")
    except Exception as e:
        return SignalVote("Session", "N/A", 0.0, f"Error: {str(e)[:60]}")


def _news_vote(symbol: str | None, enabled: bool) -> SignalVote:
    if not enabled:
        return SignalVote("News", "N/A", 0.0, "News check disabled in sidebar.")
    if not symbol:
        return SignalVote("News", "N/A", 0.0, "No mappable underlying for news.")
    try:
        from monte.intel.perplexity import fetch_news
        brief = fetch_news(symbol, action="BUY")
    except Exception as e:
        return SignalVote("News", "N/A", 0.0, f"News error: {str(e)[:60]}")
    if not brief.configured:
        return SignalVote("News", "N/A", 0.0, "PERPLEXITY_API_KEY not set.")
    if brief.error:
        return SignalVote("News", "N/A", 0.0, brief.error[:80])
    if brief.sentiment == "bullish":
        return SignalVote("News", "BULL", 0.7, brief.summary[:160])
    if brief.sentiment == "bearish":
        return SignalVote("News", "BEAR", 0.7, brief.summary[:160])
    return SignalVote("News", "NEUTRAL", 0.3, brief.summary[:160] or "Neutral coverage.")


# ---------------------------------------------------------------------------
# Main recommender
# ---------------------------------------------------------------------------

def recommend_play(
    *,
    event: dict,
    market_type: str,
    category: str,
    series_label: str,
    stake: float = 10.0,
    weights: Optional[dict[str, float]] = None,
    enable_news: bool = False,
    vote_overrides: dict[str, SignalVote] | None = None,
) -> Optional[PlayRecommendation]:
    """Build a triangulated play recommendation for a single Kalshi event.

    ``vote_overrides`` keys are vote names (``Crowd`` | ``Patterns`` |
    ``Influencers`` | ``Session`` | ``News``); any matching vote is taken
    from the override map instead of being computed live. The backtester
    uses this to inject fixture-based News/Influencer votes (which have
    no historical record) while leaving the deterministic Crowd / Patterns
    / Session votes computed normally.

    Returns None if no actionable market is found.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    markets = event.get("markets") or []
    active = [m for m in markets if m.get("status") == "active"] or markets
    if not active:
        return None

    # Resolve the favourite (the YES outcome with the highest mid-price)
    def _mid(m):
        bid = float(m.get("yes_bid_dollars") or 0)
        ask = float(m.get("yes_ask_dollars") or 0)
        if ask > 0 and bid > 0:
            return (bid + ask) / 2.0
        return bid or ask

    probs = sorted([(m, _mid(m)) for m in active], key=lambda x: x[1], reverse=True)
    top_m, top_p = probs[0]
    top_label = top_m.get("subtitle") or top_m.get("yes_sub_title") or "?"

    # Map event → underlying crypto symbol if applicable
    symbol = _is_crypto_symbol(category, series_label)

    # Collect signal votes (allow per-name override for backtest replay)
    overrides = vote_overrides or {}

    def _vote(name: str, fallback: SignalVote) -> SignalVote:
        return overrides.get(name, fallback)

    votes = [
        _vote("Crowd", _crowd_vote(top_p)),
        _vote("Patterns", _patterns_vote(symbol)),
        _vote("Influencers", _influencers_vote(symbol)),
        _vote("Session", _session_vote()),
        _vote("News", _news_vote(symbol, enable_news)),
    ]

    # Weighted aggregation: BULL = ride the favourite, BEAR = fade it.
    wmap = {
        "Crowd": weights["crowd"],
        "Patterns": weights["patterns"],
        "Influencers": weights["influencers"],
        "Session": weights["session"],
        "News": weights["news"],
    }
    bull, bear, neut = 0.0, 0.0, 0.0
    n_bull = n_bear = n_neut = 0
    for v in votes:
        w = wmap.get(v.name, 0.5) * v.confidence
        if v.verdict == "BULL":
            bull += w
            n_bull += 1
        elif v.verdict == "BEAR":
            bear += w
            n_bear += 1
        elif v.verdict == "NEUTRAL":
            neut += w * 0.5
            n_neut += 1

    total = bull + bear + neut
    if total == 0:
        confidence = 0.0
        side_dir = "PASS"
    else:
        # Confidence = how dominant the leading direction is, scaled by
        # how many independent signals contributed.
        leading = max(bull, bear)
        dominance = leading / total
        n_voted = n_bull + n_bear
        coverage = min(1.0, n_voted / 3.0)   # need ≥3 directional signals to max out
        confidence = dominance * coverage
        side_dir = "BULL" if bull > bear else ("BEAR" if bear > bull else "PASS")

    # Decide ACTION + which side of the Kalshi market to take
    if side_dir == "BULL":
        # External signals agree with the favourite → buy YES on the favourite
        side = f"BUY YES on '{top_label}' @ ${top_p:.2f}"
        if confidence >= 0.55 and top_p < 0.85:
            action = "STRONG_BUY" if confidence >= 0.7 else "BUY"
        elif confidence >= 0.35:
            action = "WATCH"
        else:
            action = "PASS"
        contract_price = top_p
    elif side_dir == "BEAR":
        # External signals disagree with the favourite → buy NO on the favourite
        no_p = 1.0 - top_p
        side = f"BUY NO on '{top_label}' @ ${no_p:.2f}  (contrarian)"
        if confidence >= 0.55 and no_p < 0.85:
            action = "BUY" if confidence < 0.7 else "STRONG_BUY"
        elif confidence >= 0.35:
            action = "WATCH"
        else:
            action = "PASS"
        contract_price = no_p
    else:
        side = "PASS — no clear directional edge"
        action = "PASS"
        contract_price = top_p   # for math display only

    # Money math (Kalshi binary: pay $price per contract, win = $1 each)
    contract_price = max(0.01, min(0.99, contract_price))
    contracts = int(stake // contract_price) if contract_price > 0 else 0
    max_payout = contracts * 1.0
    actual_stake = contracts * contract_price
    net_win = max_payout - actual_stake
    roi_pct = (net_win / actual_stake * 100) if actual_stake > 0 else 0.0

    # Confidence label
    if confidence >= 0.65:
        clabel = "HIGH"
    elif confidence >= 0.40:
        clabel = "MED"
    else:
        clabel = "LOW"

    # Horizon advice — Kalshi binary vs. options on the underlying
    secs = _seconds_to_close(event) or 0
    if action == "PASS":
        horizon = "Don't take this trade — signals don't agree."
    elif secs <= 0:
        horizon = "Market is closed."
    elif secs < 3600 * 6:
        horizon = (
            f"Buy on Kalshi and **hold to close** ({secs // 3600}h "
            f"{(secs % 3600) // 60}m). Short horizon — Kalshi binary is the right "
            f"vehicle, options would be too theta-heavy."
        )
    elif secs < 86400 * 3:
        horizon = (
            f"Buy on Kalshi and hold ~{secs // 86400}d. If you want larger "
            f"upside than the binary cap, also consider a directional **call/put "
            f"option** on the underlying for the same horizon."
        )
    else:
        days = secs // 86400
        if symbol:
            horizon = (
                f"Long horizon (~{days}d). Kalshi binary is fine but capital is "
                f"locked. For more capital efficiency consider **buying spot/options "
                f"on {symbol}** in parallel — same directional thesis, more flexibility."
            )
        else:
            horizon = (
                f"Long horizon (~{days}d). Kalshi binary is fine but capital is "
                f"locked the whole time. Check that the ROI justifies the wait."
            )

    # Why summary
    agreeing = [v.name for v in votes if v.verdict in ("BULL", "BEAR")
                and ((side_dir == "BULL" and v.verdict == "BULL")
                     or (side_dir == "BEAR" and v.verdict == "BEAR"))]
    conflicting = [v.name for v in votes if v.verdict in ("BULL", "BEAR")
                   and v.name not in agreeing]
    if action == "PASS":
        why = (
            f"{n_bull} bullish vs. {n_bear} bearish vs. {n_neut} neutral signals. "
            "Confidence too low for a directional bet."
        )
    else:
        why = (
            f"{', '.join(agreeing) or '—'} agree with the action. "
            f"{', '.join(conflicting) or 'No conflicts.'} "
            f"{n_bull + n_bear + n_neut} signals weighed in total."
        )

    return PlayRecommendation(
        action=action,
        side=side,
        confidence=confidence,
        confidence_label=clabel,
        contract_price=contract_price,
        contracts=contracts,
        stake=actual_stake,
        max_payout=max_payout,
        net_win=net_win,
        roi_pct=roi_pct,
        max_loss=actual_stake,
        votes=votes,
        n_bull=n_bull,
        n_bear=n_bear,
        n_neutral=n_neut,
        horizon_advice=horizon,
        why=why,
    )


__all__ = ["recommend_play", "PlayRecommendation", "SignalVote", "DEFAULT_WEIGHTS"]
