"""Stub config/settings for the Monte engine."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class Settings:
    def __init__(self) -> None:
        self.crypto_watchlist: list[str] = json.loads(
            os.environ.get("MONTE_CRYPTO_WATCHLIST", '["BTC-USD","ETH-USD","SOL-USD"]')
        )
        self.stock_watchlist: list[str] = json.loads(
            os.environ.get("MONTE_STOCK_WATCHLIST", '["SPY"]')
        )
        # $500 default — sized for a real-money paper-to-live ramp. Override
        # with MONTE_BUDGET_USD for backtesting at a larger scale.
        self.budget_usd: float = float(os.environ.get("MONTE_BUDGET_USD", "500"))
        # Monthly profit target surfaced on the dashboard. Informational only.
        self.monthly_target_usd: float = float(
            os.environ.get("MONTE_MONTHLY_TARGET_USD", "4000")
        )
        self.vector_db_path: Path = Path(
            os.environ.get("MONTE_VECTOR_DB_PATH", str(Path.home() / ".monte" / "chroma"))
        )
        self.paper_state_path: Path = Path(
            os.environ.get("MONTE_PAPER_STATE_PATH", str(Path.home() / ".monte" / "paper"))
        )
        self.alerts_log_path: Path = Path(
            os.environ.get("MONTE_ALERTS_LOG_PATH", str(Path.home() / ".monte" / "alerts.jsonl"))
        )
        self.min_confidence_alert: float = float(
            os.environ.get("MONTE_MIN_CONFIDENCE", "65")
        )
        self.slippage_bps: int = int(os.environ.get("MONTE_SLIPPAGE_BPS", "5"))
        self.pattern_window: int = int(os.environ.get("MONTE_PATTERN_WINDOW", "60"))
        self.allow_live: bool = os.environ.get("MONTE_ALLOW_LIVE", "0") == "1"
        self.risk_per_trade: float = 0.01

        self.triangulation_weights: dict[str, float] = {
            "technical": 0.30,
            "monte_carlo": 0.20,
            "sentiment": 0.15,
            "regime": 0.20,
            "pattern": 0.15,
        }

        self.anthropic_configured: bool = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.perplexity_configured: bool = bool(os.environ.get("PERPLEXITY_API_KEY"))
        self.alpaca_configured: bool = bool(
            os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")
        )


settings = Settings()
