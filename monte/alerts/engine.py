"""Stub alerts engine for the Monte engine."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from monte.config import settings


def tail_alerts(limit: int = 50) -> list[dict[str, Any]]:
    """Read the last `limit` alerts from the alerts log file."""
    path = settings.alerts_log_path
    if not path.exists():
        return []
    try:
        lines = path.read_text().strip().splitlines()
        rows = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) >= limit:
                break
        return rows
    except Exception:
        return []


def scan_once(
    symbols: list[str],
    timeframes: list[str] | None = None,
    min_confidence: float = 65.0,
) -> list[dict[str, Any]]:
    """Stub scan — returns empty list (no live data without API keys)."""
    return []
