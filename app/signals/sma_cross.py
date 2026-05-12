"""SMA fast/slow crossover signal — pure function over a close series.

No I/O, no caching, no Streamlit. Returns a `monte.strategies.signals.Signal`
so the score plugs into the existing composite combiner.
"""

from __future__ import annotations

import pandas as pd

from monte.strategies.signals import Signal


def sma_crossover(
    closes: pd.Series,
    *,
    fast: int = 20,
    slow: int = 50,
    timeframe: str = "",
) -> Signal:
    """Score a fast/slow SMA crossover on the most recent bar.

    Score rules (evaluated on the last bar of `closes`):
      cross_up   (fast was <= slow, now > slow):  +1.0
      cross_down (fast was >= slow, now < slow):  -1.0
      no cross, fast > slow:                       +0.4
      no cross, fast < slow:                       -0.4
      fast == slow:                                 0.0

    Raises ValueError when `fast >= slow` or `len(closes) < slow + 1`
    (a cross needs two consecutive SMA pairs).
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    if len(closes) < slow + 1:
        raise ValueError(
            f"need at least slow+1={slow + 1} closes, got {len(closes)}"
        )

    fast_sma = closes.rolling(fast).mean()
    slow_sma = closes.rolling(slow).mean()

    f_now, f_prev = float(fast_sma.iloc[-1]), float(fast_sma.iloc[-2])
    s_now, s_prev = float(slow_sma.iloc[-1]), float(slow_sma.iloc[-2])

    crossed_up = f_prev <= s_prev and f_now > s_now
    crossed_down = f_prev >= s_prev and f_now < s_now

    if crossed_up:
        score, rationale = 1.0, f"fast({fast}) crossed above slow({slow})"
    elif crossed_down:
        score, rationale = -1.0, f"fast({fast}) crossed below slow({slow})"
    elif f_now > s_now:
        score, rationale = 0.4, f"fast({fast}) above slow({slow})"
    elif f_now < s_now:
        score, rationale = -0.4, f"fast({fast}) below slow({slow})"
    else:
        score, rationale = 0.0, "fast == slow"

    return Signal(
        name="SMA cross",
        score=score,
        rationale=rationale,
        timeframe=timeframe,
        meta={"fast": fast, "slow": slow, "fast_sma": f_now, "slow_sma": s_now},
    )
