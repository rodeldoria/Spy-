"""2-state Gaussian HMM regime classifier.

Uses `hmmlearn` when installed; falls back to a deterministic 200-EMA +
slope rule when the dep is missing so the gate keeps working in the
default install. Either path returns the same `HMMResult` shape so callers
don't branch on `source`.

The fitted model is keyed only by the length and head-tail-hash of the
input series, which is stable enough to avoid refitting on every Streamlit
rerun while still picking up new data.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

_FIT_CACHE: dict[str, "HMMResult"] = {}


@dataclass
class HMMResult:
    label: str               # human-readable: "bull-quiet", "bull-volatile", "bear", ...
    bull_prob: float         # P(bull state) ∈ [0, 1]
    state: int               # raw HMM state index
    source: str              # "hmmlearn" | "fallback"
    note: str = ""


def hmm_state(closes: pd.Series, log_rets: pd.Series) -> HMMResult:
    """Return the most likely current state for the given price series.

    Raises only on truly unrecoverable input (empty series). Network /
    optional-dep failures degrade to the deterministic fallback.
    """
    closes = closes.dropna()
    log_rets = log_rets.dropna()
    if closes.empty or len(log_rets) < 30:
        raise ValueError("hmm_state: need ≥30 returns")

    cache_key = _series_key(log_rets)
    cached = _FIT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        result = _hmmlearn_fit(closes, log_rets)
    except Exception:  # noqa: BLE001 — fall back rather than fail the panel
        result = _fallback_state(closes, log_rets)

    _FIT_CACHE[cache_key] = result
    return result


def clear_cache() -> None:
    _FIT_CACHE.clear()


# ---------------------------------------------------------------------------
# Fitted-HMM path
# ---------------------------------------------------------------------------

def _hmmlearn_fit(closes: pd.Series, log_rets: pd.Series) -> HMMResult:
    from hmmlearn.hmm import GaussianHMM

    rv = log_rets.rolling(10, min_periods=5).std().bfill().fillna(0.0)
    X = np.column_stack(
        [
            log_rets.to_numpy(dtype=float),
            rv.to_numpy(dtype=float),
        ]
    )
    if not np.isfinite(X).all() or len(X) < 30:
        return _fallback_state(closes, log_rets)

    model = GaussianHMM(
        n_components=2,
        covariance_type="diag",
        n_iter=50,
        tol=1e-3,
        random_state=7,
    )
    model.fit(X)
    posteriors = model.predict_proba(X)
    state_now = int(model.predict(X[-50:])[-1])

    means = model.means_[:, 0]   # mean log-return per state
    bull_idx = int(np.argmax(means))
    bull_prob = float(posteriors[-1, bull_idx])

    label = _label_from_state(
        is_bull=(state_now == bull_idx),
        vol=float(model.means_[state_now, 1]),
    )
    return HMMResult(
        label=label,
        bull_prob=bull_prob,
        state=state_now,
        source="hmmlearn",
        note=(
            f"state means: bull μ={means[bull_idx]:+.4f}, bear μ={means[1 - bull_idx]:+.4f}"
        ),
    )


def _label_from_state(*, is_bull: bool, vol: float) -> str:
    side = "bull" if is_bull else "bear"
    # Vol means here are realized-σ averages; use a coarse bucket for the label.
    if vol > 0.025:
        return f"{side}-volatile"
    return f"{side}-quiet"


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

def _fallback_state(closes: pd.Series, log_rets: pd.Series) -> HMMResult:
    """200-EMA + slope rule. Bull when price > EMA200 AND 20d slope > 0."""
    if len(closes) >= 200:
        ema = closes.ewm(span=200, adjust=False).mean()
        spot = float(closes.iloc[-1])
        ema_now = float(ema.iloc[-1])
        slope_20 = float(closes.iloc[-1] - closes.iloc[-21]) if len(closes) >= 21 else 0.0
        is_bull = (spot > ema_now) and (slope_20 > 0)
    else:
        slope = float(closes.iloc[-1] - closes.iloc[0])
        is_bull = slope > 0

    rv = log_rets.tail(20).std()
    rv = float(rv) if rv is not None and not np.isnan(rv) else 0.0
    bull_prob = 0.7 if is_bull else 0.3
    label = _label_from_state(is_bull=is_bull, vol=rv)
    return HMMResult(
        label=label,
        bull_prob=bull_prob,
        state=1 if is_bull else 0,
        source="fallback",
        note="hmmlearn not installed; using 200-EMA + slope rule",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _series_key(s: pd.Series) -> str:
    arr = s.to_numpy(dtype=float)
    h = hashlib.sha1()
    h.update(str(len(arr)).encode())
    if len(arr) > 0:
        h.update(arr[:5].tobytes())
        h.update(arr[-5:].tobytes())
    return h.hexdigest()


__all__ = ["HMMResult", "hmm_state", "clear_cache"]
