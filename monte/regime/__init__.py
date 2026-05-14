"""Regime classification for the event-aware chat widget.

Composes six independent views of "what kind of market are we in?":

  - **HMM**             — 2-state Gaussian HMM on (log-return, realized-vol)
                          via `hmmlearn`, with a deterministic 200-EMA + slope
                          fallback when the optional dep isn't installed.
  - **BOCPD**           — Adams & MacKay 2007 Bayesian Online Change-Point
                          Detection over daily returns; surfaces P(regime
                          shifted within the last few bars).
  - **Hurst**           — R/S Hurst exponent on a 256-bar rolling window;
                          tags the market as trending / mean-reverting / random.
  - **Variance ratio**  — Lo-MacKinlay VR test with heteroskedasticity
                          correction; reports the p-value of the random-walk
                          null.
  - **Wyckoff**         — phase classifier (accumulation / markup /
                          distribution / markdown) from price-vs-200MA, vol
                          percentile, and trend slope.
  - **Macro quadrant**  — Dalio quadrant from a `FredSnapshot` (growth ↑↓ ×
                          inflation ↑↓), with a one-line bias note for SPY
                          and BTC.

`assess(symbol, df_daily, fred_snapshot=None)` runs each view in parallel,
captures per-view errors instead of raising, and returns a single
`RegimeReport` the gate can score.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from monte.regime.bocpd import bocpd_posterior
from monte.regime.hmm import HMMResult, hmm_state
from monte.regime.hurst import hurst_exponent
from monte.regime.macro_quadrant import MacroQuadrant, classify_quadrant
from monte.regime.variance_ratio import variance_ratio_test
from monte.regime.wyckoff import WyckoffPhase, wyckoff_phase


@dataclass
class RegimeReport:
    symbol: str

    hmm: Optional[HMMResult] = None
    bocpd_changepoint_prob: Optional[float] = None
    hurst: Optional[float] = None
    hurst_label: str = ""
    vr_pvalue: Optional[float] = None
    vr_label: str = ""
    wyckoff: Optional[WyckoffPhase] = None
    macro: Optional[MacroQuadrant] = None

    errors: dict[str, str] = field(default_factory=dict)

    def directional_bias(self) -> str:
        """Composite bias: bull / bear / neutral, ignoring missing axes."""
        votes: list[str] = []
        if self.hmm is not None:
            votes.append("bull" if self.hmm.bull_prob >= 0.55 else
                          ("bear" if self.hmm.bull_prob <= 0.45 else "neutral"))
        if self.hurst is not None:
            if self.hurst >= 0.55:
                votes.append("bull")  # trend-friendly — assumes price slope is up
            elif self.hurst <= 0.45:
                votes.append("neutral")  # mean-reverting — direction-agnostic
        if self.wyckoff is not None:
            votes.append(_wyckoff_bias(self.wyckoff.phase))
        if self.macro is not None:
            votes.append(self.macro.equity_bias)

        if not votes:
            return "neutral"
        score = sum(1 if v == "bull" else (-1 if v == "bear" else 0) for v in votes)
        if score >= 1:
            return "bull"
        if score <= -1:
            return "bear"
        return "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "hmm": None if self.hmm is None else {
                "label": self.hmm.label,
                "bull_prob": self.hmm.bull_prob,
                "source": self.hmm.source,
            },
            "bocpd_changepoint_prob": self.bocpd_changepoint_prob,
            "hurst": self.hurst,
            "hurst_label": self.hurst_label,
            "vr_pvalue": self.vr_pvalue,
            "vr_label": self.vr_label,
            "wyckoff": None if self.wyckoff is None else {
                "phase": self.wyckoff.phase,
                "confidence": self.wyckoff.confidence,
                "note": self.wyckoff.note,
            },
            "macro": None if self.macro is None else {
                "quadrant": self.macro.quadrant,
                "equity_bias": self.macro.equity_bias,
                "crypto_bias": self.macro.crypto_bias,
                "note": self.macro.note,
            },
            "errors": dict(self.errors),
        }


def assess(
    symbol: str,
    df_daily: pd.DataFrame,
    *,
    fred_snapshot: Optional[Any] = None,
) -> RegimeReport:
    """Run all regime axes in parallel against `df_daily` (capitalised OHLCV).

    `fred_snapshot` is a `monte.data.fred.FredSnapshot`; pass `None` to skip
    the macro-quadrant axis entirely. Returns a fully-populated
    `RegimeReport` even when individual axes fail (the failure goes in
    `report.errors`).
    """
    report = RegimeReport(symbol=symbol)
    if df_daily is None or df_daily.empty:
        report.errors["df"] = "empty daily frame"
        return report

    closes = _close_series(df_daily)
    if closes is None or len(closes) < 30:
        report.errors["df"] = f"need ≥30 daily closes, got {0 if closes is None else len(closes)}"
        # Still try the macro axis since it doesn't depend on price.
        if fred_snapshot is not None:
            try:
                report.macro = classify_quadrant(fred_snapshot)
            except Exception as e:  # noqa: BLE001
                report.errors["macro"] = f"{type(e).__name__}: {str(e)[:80]}"
        return report

    log_rets = closes.pct_change().dropna()

    def _hmm():
        return hmm_state(closes, log_rets)

    def _bocpd():
        return bocpd_posterior(log_rets.tail(180).to_numpy())

    def _hurst():
        return hurst_exponent(closes.tail(256).to_numpy())

    def _vr():
        return variance_ratio_test(log_rets.tail(180).to_numpy(), q=4)

    def _wy():
        return wyckoff_phase(df_daily)

    def _mq():
        return classify_quadrant(fred_snapshot) if fred_snapshot is not None else None

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {
            "hmm": pool.submit(_hmm),
            "bocpd": pool.submit(_bocpd),
            "hurst": pool.submit(_hurst),
            "vr": pool.submit(_vr),
            "wyckoff": pool.submit(_wy),
            "macro": pool.submit(_mq),
        }
        for name, fut in futs.items():
            try:
                v = fut.result(timeout=10)
            except Exception as e:  # noqa: BLE001
                report.errors[name] = f"{type(e).__name__}: {str(e)[:120]}"
                continue
            if name == "hmm":
                report.hmm = v
            elif name == "bocpd":
                report.bocpd_changepoint_prob = v
            elif name == "hurst":
                if v is None:
                    report.errors["hurst"] = "insufficient data"
                else:
                    report.hurst = v
                    report.hurst_label = _hurst_label(v)
            elif name == "vr":
                if v is None:
                    report.errors["vr"] = "insufficient data"
                else:
                    vr, pval = v
                    report.vr_pvalue = pval
                    report.vr_label = _vr_label(vr, pval)
            elif name == "wyckoff":
                report.wyckoff = v
            elif name == "macro":
                if v is not None:
                    report.macro = v
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _close_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if "Close" in df.columns:
        return df["Close"].astype(float).dropna()
    if "close" in df.columns:
        return df["close"].astype(float).dropna()
    return None


def _hurst_label(h: float) -> str:
    if h >= 0.55:
        return f"trending (H={h:.2f})"
    if h <= 0.45:
        return f"mean-reverting (H={h:.2f})"
    return f"random walk (H={h:.2f})"


def _vr_label(vr: float, pval: float) -> str:
    if pval < 0.05 and vr > 1.0:
        return f"trending (VR={vr:.2f}, p={pval:.2f})"
    if pval < 0.05 and vr < 1.0:
        return f"mean-reverting (VR={vr:.2f}, p={pval:.2f})"
    return f"random walk (VR={vr:.2f}, p={pval:.2f})"


def _wyckoff_bias(phase: str) -> str:
    return {
        "accumulation": "bull",
        "markup": "bull",
        "distribution": "bear",
        "markdown": "bear",
    }.get(phase, "neutral")


__all__ = ["RegimeReport", "assess"]
