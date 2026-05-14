"""Parallel orchestration of catalyst feeds for the chat widget.

`gather(idea)` fans out to perplexity, FRED, the economic calendar, and
the on-chain bundle in a single thread pool with a 6-second overall
deadline. Per-source failures land in `EventBundle.errors` rather than
raising; the rest of the bundle still renders.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Optional

from monte.data.econ_calendar import CalendarEvent, next_hours
from monte.data.fred import FredSnapshot, snapshot as fred_snapshot
from monte.data.onchain import OnChainSnapshot, snapshot as onchain_snapshot
from monte.intel.perplexity import NewsBrief, fetch_news


@dataclass
class IdeaContext:
    symbol: str
    direction: str = "long"           # "long" | "short"
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    horizon_hours: float = 168.0
    note: str = ""
    is_crypto: bool = False


@dataclass
class EventBundle:
    idea: IdeaContext
    news: Optional[NewsBrief] = None
    fred: Optional[FredSnapshot] = None
    calendar: list[CalendarEvent] = field(default_factory=list)
    onchain: Optional[OnChainSnapshot] = None
    errors: dict[str, str] = field(default_factory=dict)
    elapsed_ms: int = 0

    def tier_1_in_window(self) -> list[CalendarEvent]:
        horizon = self.idea.horizon_hours
        return [e for e in self.calendar if e.is_tier_1 and 0 <= e.hours_from_now <= horizon]

    def unlock_in_window(self) -> Optional[float]:
        """Return hours-to-unlock if a token unlock falls inside the hold window."""
        if not self.onchain or self.onchain.next_unlock_at_utc is None:
            return None
        import time as _t
        hours = (self.onchain.next_unlock_at_utc - _t.time()) / 3600.0
        if 0 <= hours <= self.idea.horizon_hours:
            return hours
        return None


def gather(idea: IdeaContext, *, deadline_seconds: float = 6.0) -> EventBundle:
    """Run all catalyst feeds in parallel and return an `EventBundle`."""
    import time as _t
    started = _t.time()
    bundle = EventBundle(idea=idea)

    with ThreadPoolExecutor(max_workers=4) as pool:
        f_news = pool.submit(fetch_news, idea.symbol, "BUY" if idea.direction == "long" else "SELL")
        f_fred = pool.submit(fred_snapshot)
        f_cal = pool.submit(next_hours, max(24, int(idea.horizon_hours)))
        f_chain = pool.submit(onchain_snapshot, idea.symbol, hold_window_hours=idea.horizon_hours) \
            if idea.is_crypto else None

        for label, fut, applier in [
            ("news", f_news, _apply_news),
            ("fred", f_fred, _apply_fred),
            ("calendar", f_cal, _apply_calendar),
            ("onchain", f_chain, _apply_onchain),
        ]:
            if fut is None:
                continue
            try:
                v = fut.result(timeout=max(0.1, deadline_seconds - (_t.time() - started)))
            except FutureTimeout:
                bundle.errors[label] = "timed out"
                continue
            except Exception as e:  # noqa: BLE001
                bundle.errors[label] = f"{type(e).__name__}: {str(e)[:120]}"
                continue
            applier(bundle, v)

    bundle.elapsed_ms = int((_t.time() - started) * 1000)
    return bundle


def _apply_news(bundle: EventBundle, v: NewsBrief) -> None:
    bundle.news = v


def _apply_fred(bundle: EventBundle, v: FredSnapshot) -> None:
    bundle.fred = v


def _apply_calendar(bundle: EventBundle, v: list[CalendarEvent]) -> None:
    bundle.calendar = v or []


def _apply_onchain(bundle: EventBundle, v: OnChainSnapshot) -> None:
    bundle.onchain = v


__all__ = ["IdeaContext", "EventBundle", "gather"]
