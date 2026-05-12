"""Pure-function signal layer local to the Spy- app.

Each signal takes a price series and returns a `monte.strategies.signals.Signal`.
No I/O, no caching, no Streamlit — those layers live in `app/_shared.py` and
the pages. This keeps signals testable in isolation.
"""

from app.signals.rsi import rsi
from app.signals.sma_cross import sma_crossover

__all__ = ["rsi", "sma_crossover"]
