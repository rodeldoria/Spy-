"""VWAP-reversion signal — pure function over close & volume series.

Mean-reversion bias: when the last close sits meaningfully above VWAP we
fade upward (SELL); when below, we fade downward (BUY). Distance is
measured as a fraction of VWAP itself.

No I/O, no caching, no Streamlit. Returns `monte.strategies.signals.Signal`.
"""

from __future__ import annotations

import pandas as pd

from monte.strategies.signals import Signal


def vwap_reversion(
    closes: pd.Series,
    volumes: pd.Series,
    *,
    weak_threshold: float = 0.005,
    strong_threshold: float = 0.02,
    timeframe: str = "",
) -> Signal:
    """Score the last close's % deviation from VWAP.

    Score rules (deviation = (close - vwap) / vwap):
      deviation >=  strong_threshold:  -1.0  (strong fade)
      deviation >=  weak_threshold:    -0.4
      deviation <= -strong_threshold:  +1.0  (strong fade)
      deviation <= -weak_threshold:    +0.4
      otherwise:                        0.0

    Raises ValueError on length mismatch, empty input, negative volume,
    zero total volume, or non-positive / out-of-order thresholds.
    """
    if len(closes) != len(volumes):
        raise ValueError(
            f"closes ({len(closes)}) and volumes ({len(volumes)}) "
            "must be the same length"
        )
    if len(closes) == 0:
        raise ValueError("need at least 1 bar, got empty series")
    if (volumes < 0).any():
        raise ValueError("volumes must be non-negative")
    if not (0 < weak_threshold < strong_threshold):
        raise ValueError(
            "thresholds must satisfy 0 < weak < strong, "
            f"got weak={weak_threshold}, strong={strong_threshold}"
        )

    total_volume = float(volumes.sum())
    if total_volume == 0.0:
        raise ValueError("total volume is zero; VWAP is undefined")

    vwap = float((closes * volumes).sum()) / total_volume
    close = float(closes.iloc[-1])
    deviation = (close - vwap) / vwap

    if deviation >= strong_threshold:
        score = -1.0
        rationale = f"close {deviation:+.2%} above VWAP (strong fade)"
    elif deviation >= weak_threshold:
        score = -0.4
        rationale = f"close {deviation:+.2%} above VWAP"
    elif deviation <= -strong_threshold:
        score = 1.0
        rationale = f"close {deviation:+.2%} below VWAP (strong fade)"
    elif deviation <= -weak_threshold:
        score = 0.4
        rationale = f"close {deviation:+.2%} below VWAP"
    else:
        score = 0.0
        rationale = f"close {deviation:+.2%} from VWAP (within band)"

    return Signal(
        name="VWAP reversion",
        score=score,
        rationale=rationale,
        timeframe=timeframe,
        meta={
            "vwap": vwap,
            "close": close,
            "deviation": deviation,
        },
    )
