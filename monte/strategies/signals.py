"""Stub Signal types for the monte engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Action(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


def action_from_score(score: float) -> Action:
    if score >= 0.8:
        return Action.STRONG_BUY
    elif score >= 0.2:
        return Action.BUY
    elif score <= -0.8:
        return Action.STRONG_SELL
    elif score <= -0.2:
        return Action.SELL
    else:
        return Action.HOLD


@dataclass
class Signal:
    name: str
    score: float
    rationale: str = ""
    timeframe: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
