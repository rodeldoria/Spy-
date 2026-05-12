"""Stub vector store (no Chroma dependency)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryResult:
    metadatas: list[dict[str, Any]] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)


class PatternStore:
    def __init__(self) -> None:
        self._collections: dict[str, list[dict[str, Any]]] = {}

    def _key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}_{timeframe}"

    def count(self, symbol: str, timeframe: str) -> int:
        return len(self._collections.get(self._key(symbol, timeframe), []))

    def query(
        self,
        symbol: str,
        timeframe: str,
        vector: list[float],
        k: int = 20,
    ) -> QueryResult:
        return QueryResult(metadatas=[], distances=[])

    def add(
        self,
        symbol: str,
        timeframe: str,
        vector: list[float],
        metadata: dict[str, Any],
        doc_id: str | None = None,
    ) -> None:
        key = self._key(symbol, timeframe)
        self._collections.setdefault(key, []).append({"vector": vector, "meta": metadata})
