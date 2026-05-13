"""Claude's playbook — append-only log of every EdgeSignal we generate.

The user can flip to the Playbook page and see exactly *why* a call was
made, replay the reasoning, and learn the pattern.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from monte.strategy.monte_edge import EdgeSignal


DEFAULT_PATH = Path(os.path.expanduser("~/.monte/playbook.jsonl"))


@dataclass
class PlaybookRow:
    id: str
    ts: float
    symbol: str
    timeframe: str
    action: str
    tier: str
    confidence: float
    score: float
    confluence: int
    macro_aligned: bool | None
    spot: float
    entry: float
    stop: float
    target: float
    rr: float
    horizon: str
    regime: str
    reasoning: str
    reasoning_codes: list[str] = field(default_factory=list)
    macro_note: str = ""
    indicator_snapshot: dict[str, float] = field(default_factory=dict)
    options_ticket: dict[str, Any] | None = None

    @classmethod
    def from_signal(
        cls,
        sig: EdgeSignal,
        *,
        options_ticket: dict[str, Any] | None = None,
    ) -> "PlaybookRow":
        return cls(
            id=uuid.uuid4().hex[:12],
            ts=time.time(),
            symbol=sig.symbol,
            timeframe=sig.timeframe,
            action=sig.action,
            tier=sig.tier.value if hasattr(sig.tier, "value") else str(sig.tier),
            confidence=sig.confidence,
            score=sig.score,
            confluence=sig.confluence,
            macro_aligned=sig.macro_aligned,
            spot=sig.spot,
            entry=sig.entry,
            stop=sig.stop,
            target=sig.target,
            rr=sig.rr,
            horizon=sig.horizon,
            regime=sig.regime,
            reasoning=sig.reasoning,
            reasoning_codes=list(sig.reasoning_codes),
            macro_note=sig.macro_note,
            indicator_snapshot=dict(sig.indicator_snapshot),
            options_ticket=options_ticket,
        )


def _path() -> Path:
    p = Path(os.environ.get("MONTE_PLAYBOOK_PATH", str(DEFAULT_PATH)))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def record_signal(
    sig: EdgeSignal,
    *,
    options_ticket: dict[str, Any] | None = None,
) -> PlaybookRow:
    row = PlaybookRow.from_signal(sig, options_ticket=options_ticket)
    with _path().open("a") as f:
        f.write(json.dumps(asdict(row), default=str) + "\n")
    return row


def list_playbook(
    *,
    limit: int = 200,
    symbol: str | None = None,
    tier: str | None = None,
) -> list[PlaybookRow]:
    path = _path()
    if not path.exists():
        return []
    rows: list[PlaybookRow] = []
    for line in reversed(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if symbol and payload.get("symbol") != symbol:
            continue
        if tier and payload.get("tier") != tier:
            continue
        try:
            rows.append(PlaybookRow(**payload))
        except TypeError:
            continue
        if len(rows) >= limit:
            break
    return rows


__all__ = ["PlaybookRow", "record_signal", "list_playbook"]
