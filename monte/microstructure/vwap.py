"""Session VWAP + 1σ band on a 1m/5m OHLCV frame.

Equity sessions reset at 13:30 UTC (US cash open). Crypto sessions reset
at UTC midnight. Returns the VWAP and the per-bar standard deviation of
typical price minus VWAP, so the gate can talk about "extension in σ".
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def session_vwap(
    df: pd.DataFrame,
    *,
    asset_class: str = "equity",
) -> tuple[Optional[float], Optional[float]]:
    """Return (vwap_now, sigma_now) for the most recent session, or (None, None)."""
    if df is None or df.empty:
        return None, None

    needed = {"High", "Low", "Close", "Volume"}
    missing = needed - set(df.columns)
    cols = {c.lower(): c for c in df.columns}
    if missing:
        # Try lowercase mapping.
        if not needed.issubset({c.title() for c in df.columns}) and not all(c.lower() in cols for c in needed):
            return None, None

    high = df.get("High", df.get("high"))
    low = df.get("Low", df.get("low"))
    close = df.get("Close", df.get("close"))
    vol = df.get("Volume", df.get("volume"))
    if high is None or low is None or close is None or vol is None:
        return None, None

    typical = (high.astype(float) + low.astype(float) + close.astype(float)) / 3.0
    volume = vol.astype(float).clip(lower=0)

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(idx, utc=True)
        except (TypeError, ValueError):
            return _flat_vwap(typical, volume)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    if asset_class == "crypto":
        # Reset on UTC midnight.
        session_id = idx.normalize().asi8
    else:
        # Reset on US cash open ~13:30 UTC.
        shifted = idx - pd.Timedelta(hours=13, minutes=30)
        session_id = shifted.normalize().asi8

    last_sid = session_id[-1]
    mask = session_id == last_sid
    tp = typical[mask].to_numpy(dtype=float)
    v = volume[mask].to_numpy(dtype=float)
    if len(tp) == 0 or v.sum() <= 0:
        return _flat_vwap(typical, volume)

    cumv = v.cumsum()
    cum_pv = (tp * v).cumsum()
    vwap_curve = np.divide(cum_pv, cumv, out=np.zeros_like(cum_pv), where=cumv > 0)
    diff = tp - vwap_curve
    weights = v / max(v.sum(), 1e-12)
    sigma = float(np.sqrt(np.sum(weights * diff * diff)))
    return float(vwap_curve[-1]), max(sigma, 1e-9)


def _flat_vwap(typical: pd.Series, volume: pd.Series) -> tuple[Optional[float], Optional[float]]:
    tp = typical.dropna().to_numpy(dtype=float)
    v = volume.dropna().to_numpy(dtype=float)
    n = min(len(tp), len(v))
    if n == 0 or v[-n:].sum() <= 0:
        return None, None
    tp = tp[-n:]
    v = v[-n:]
    vwap = float((tp * v).sum() / v.sum())
    sigma = float(np.sqrt(((tp - vwap) ** 2 * v).sum() / v.sum()))
    return vwap, max(sigma, 1e-9)


__all__ = ["session_vwap"]
