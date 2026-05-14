"""Dalio-style macro quadrant classifier on a FRED snapshot.

Two axes:

  - **Growth**     proxied by the YoY change in `INDPRO` (industrial
                   production) and the level/30d-Δ of the Conference
                   Board's `USSLIND` leading indicator.
  - **Inflation**  proxied by `CPIAUCSL` YoY.

Mapping:

                       Inflation ↑           Inflation ↓
        Growth ↑    "Reflation"           "Goldilocks"
        Growth ↓    "Stagflation"         "Deflationary bust"

Each quadrant carries a one-line bias note for SPY and BTC drawn from
post-2008 history. The gate consumes `equity_bias` and `crypto_bias` as
"+1 / 0 / -1" votes layered on top of the technical view.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Bias = Literal["bull", "neutral", "bear"]


@dataclass
class MacroQuadrant:
    quadrant: str            # "Reflation" | "Goldilocks" | "Stagflation" | "Deflationary bust" | "Mixed"
    growth_label: str        # "↑" | "↓" | "flat"
    inflation_label: str     # "↑" | "↓" | "flat"
    equity_bias: Bias
    crypto_bias: Bias
    note: str


def classify_quadrant(snapshot) -> Optional[MacroQuadrant]:
    if snapshot is None or not getattr(snapshot, "available", False):
        return None

    indpro = snapshot.by_id("INDPRO")
    cpi = snapshot.by_id("CPIAUCSL")
    leading = snapshot.by_id("USSLIND")

    growth_score = 0.0
    if indpro is not None and indpro.yoy_pct is not None:
        growth_score += indpro.yoy_pct
    if leading is not None:
        growth_score += leading.value * 5  # USSLIND prints in units of pct points

    inflation_yoy = cpi.yoy_pct if (cpi is not None and cpi.yoy_pct is not None) else None

    growth_up = growth_score > 1.0
    growth_dn = growth_score < -1.0
    if inflation_yoy is None:
        inflation_up = False
        inflation_dn = False
    else:
        inflation_up = inflation_yoy >= 3.0
        inflation_dn = inflation_yoy <= 2.0   # below the Fed's 2% target → disinflation

    growth_label = "↑" if growth_up else ("↓" if growth_dn else "flat")
    inflation_label = "↑" if inflation_up else ("↓" if inflation_dn else "flat")

    if growth_up and inflation_up:
        quad, eq, cr = "Reflation", "bull", "bull"
        note = (
            "Growth + inflation both rising — equities OK but volatile; commodities/crypto "
            "tend to outperform real assets that hedge inflation."
        )
    elif growth_up and inflation_dn:
        quad, eq, cr = "Goldilocks", "bull", "bull"
        note = (
            "Growth up, inflation cooling — best equity backdrop. Risk-on tends to compound; "
            "BTC historically follows equities here."
        )
    elif growth_dn and inflation_up:
        quad, eq, cr = "Stagflation", "bear", "neutral"
        note = (
            "Growth slowing into rising inflation — equities struggle (P/E compression + "
            "earnings hit). BTC mixed; gold tends to win."
        )
    elif growth_dn and inflation_dn:
        quad, eq, cr = "Deflationary bust", "bear", "bear"
        note = (
            "Growth and inflation falling together — risk assets de-rate broadly. Cash and "
            "long-duration treasuries usually outperform until policy responds."
        )
    else:
        quad, eq, cr = "Mixed", "neutral", "neutral"
        note = "Macro mixed — no decisive growth/inflation signal. Trade the chart, not the macro."

    return MacroQuadrant(
        quadrant=quad,
        growth_label=growth_label,
        inflation_label=inflation_label,
        equity_bias=eq,
        crypto_bias=cr,
        note=note,
    )


__all__ = ["MacroQuadrant", "classify_quadrant", "Bias"]
