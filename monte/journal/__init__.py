"""Pattern journal — remembers what setups led to successful trades.

Exports `record_entry`, `record_exit`, `similar_history`, `summary` so the
Streamlit pages can log a paper trade and then ask "have I seen this setup
before, and how did it work out?"
"""
from monte.journal.store import (
    JournalEntry,
    SimilarHistory,
    record_entry,
    record_exit,
    similar_history,
    summary,
)

__all__ = [
    "JournalEntry",
    "SimilarHistory",
    "record_entry",
    "record_exit",
    "similar_history",
    "summary",
]
