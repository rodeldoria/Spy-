"""Vote fixtures for engines whose live inputs have no historical record.

The triangulation engine consults News (Perplexity) and Influencers (X/Twitter)
votes that we cannot replay faithfully — there's no archive in the repo.
Fixture providers fill that gap with either a neutral abstention or a
seeded-random draw, so the backtester can still report what the OTHER three
votes (Crowd / Patterns / Session) would have produced.

Backtest runs always execute BOTH fixture modes side by side so the results
page can A/B them (see ``runner.run_one``).
"""

from __future__ import annotations

import random

from monte.signals.triangulation import SignalVote

NEUTRAL_DETAIL = "fixture: no historical record — abstaining"
RANDOM_DETAIL = "fixture: seeded random draw"


def neutral_vote(name: str) -> SignalVote:
    return SignalVote(name=name, verdict="NEUTRAL", confidence=0.0, detail=NEUTRAL_DETAIL)


def seeded_random_vote(name: str, rng: random.Random) -> SignalVote:
    verdict = rng.choices(["BULL", "BEAR", "NEUTRAL"], weights=[0.35, 0.35, 0.30], k=1)[0]
    confidence = round(rng.random(), 3)
    return SignalVote(name=name, verdict=verdict, confidence=confidence, detail=RANDOM_DETAIL)


def build_overrides(mode: str, seed: int, *, names: tuple[str, ...] = ("Influencers", "News")
                    ) -> dict[str, SignalVote]:
    if mode == "neutral":
        return {n: neutral_vote(n) for n in names}
    if mode == "seeded_random":
        rng = random.Random(seed)
        return {n: seeded_random_vote(n, rng) for n in names}
    raise ValueError(f"unknown fixture mode: {mode!r}")
