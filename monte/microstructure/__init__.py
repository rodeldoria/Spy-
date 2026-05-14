"""Intraday microstructure assessment from OHLCV-only data.

Returns a `MicrostructureReport` that the gate scores. Each component
runs cheaply enough to call on every chat submit. When richer data is
available (real L2 book, tick prints) you can swap the proxies in
`cvd_proxy.py` and `imbalance_proxy.py` without touching this layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from monte.microstructure.cvd_proxy import cvd_series
from monte.microstructure.imbalance_proxy import imbalance_score
from monte.microstructure.realized_vol import realized_vol_zscore
from monte.microstructure.vwap import session_vwap


@dataclass
class MicrostructureReport:
    spot: float
    vwap: Optional[float]
    vwap_band_sigma: Optional[float]   # ± multiples of σ from VWAP
    cvd_now: Optional[float]
    cvd_divergence: int                 # +1 = bullish divergence, -1 = bearish, 0 = none
    imbalance_score: Optional[float]    # signed [-1, +1]
    rv_zscore: Optional[float]          # realized-vol z-score vs trailing 20 sessions
    note: str = ""

    def vwap_relation(self) -> str:
        if self.vwap is None or self.vwap_band_sigma is None:
            return "—"
        if self.vwap_band_sigma >= 2.0:
            return f"{self.vwap_band_sigma:+.1f}σ above VWAP — extended"
        if self.vwap_band_sigma <= -2.0:
            return f"{self.vwap_band_sigma:+.1f}σ below VWAP — extended"
        if abs(self.vwap_band_sigma) <= 0.3:
            return "at VWAP"
        return f"{self.vwap_band_sigma:+.1f}σ from VWAP"


def assess(df_intraday: pd.DataFrame, *, asset_class: str = "equity") -> MicrostructureReport:
    """Assess intraday microstructure from a 1m or 5m OHLCV frame.

    `asset_class`: "equity" | "crypto". Equity uses the regular session
    VWAP reset (13:30 UTC); crypto resets daily UTC midnight.
    """
    if df_intraday is None or df_intraday.empty:
        return MicrostructureReport(
            spot=0.0,
            vwap=None,
            vwap_band_sigma=None,
            cvd_now=None,
            cvd_divergence=0,
            imbalance_score=None,
            rv_zscore=None,
            note="no intraday data",
        )

    closes = _col(df_intraday, "Close", "close")
    if closes is None or closes.empty:
        return MicrostructureReport(0.0, None, None, None, 0, None, None, "no Close column")
    spot = float(closes.iloc[-1])

    vwap, sigma = session_vwap(df_intraday, asset_class=asset_class)
    band: Optional[float] = None
    if vwap is not None and sigma is not None and sigma > 0:
        band = (spot - vwap) / sigma

    cvd, divergence = cvd_series(df_intraday)
    imb = imbalance_score(df_intraday)
    rv_z = realized_vol_zscore(df_intraday)

    note = (
        f"VWAP: {vwap:.4f} (±{sigma:.4f}σ); CVD: {cvd[-1] if cvd is not None and len(cvd) else 0:+,.0f}; "
        f"divergence={divergence:+d}"
    ) if vwap is not None else "VWAP unavailable"
    return MicrostructureReport(
        spot=spot,
        vwap=vwap,
        vwap_band_sigma=band,
        cvd_now=float(cvd[-1]) if cvd is not None and len(cvd) else None,
        cvd_divergence=divergence,
        imbalance_score=imb,
        rv_zscore=rv_z,
        note=note,
    )


def _col(df: pd.DataFrame, *names: str) -> Optional[pd.Series]:
    for n in names:
        if n in df.columns:
            return df[n].astype(float).dropna()
    return None


__all__ = ["MicrostructureReport", "assess"]
