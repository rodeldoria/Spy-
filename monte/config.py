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

        # Confluence threshold — how many of the gate's axes must vote in the
        # idea's direction before the verdict can be GO. Mirrored from
        # MONTE_CONFLUENCE_MIN so the existing engine and the new chat widget
        # share one knob.
        self.confluence_min: int = int(os.environ.get("MONTE_CONFLUENCE_MIN", "3"))

        # Likelihood-gate weights for the event-aware chat widget. These layer
        # on top of `triangulation_weights` (technical / monte_carlo / sentiment
        # / regime / pattern) — the new keys cover the regime, macro and
        # microstructure axes that the chat surface adds.
        self.gate_weights: dict[str, float] = {
            "regime": float(os.environ.get("MONTE_REGIME_WEIGHT", "0.7")),
            "macro": float(os.environ.get("MONTE_MACRO_WEIGHT", "0.5")),
            "microstructure": float(os.environ.get("MONTE_MICROSTRUCTURE_WEIGHT", "0.4")),
        }

        self.anthropic_configured: bool = bool(
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY")
        )
        self.perplexity_configured: bool = bool(os.environ.get("PERPLEXITY_API_KEY"))
        self.alpaca_configured: bool = bool(
            os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")
        )

        # Optional feeds for the event-aware chat widget. Each provider is
        # graceful when its key is missing — the panel renders an "unavailable"
        # chip instead of crashing.
        self.fred_api_key: str = os.environ.get("FRED_API_KEY", "").strip()
        self.te_api_key: str = os.environ.get("TE_API_KEY", "").strip()
        self.coinglass_api_key: str = os.environ.get("COINGLASS_API_KEY", "").strip()
        self.calendar_provider: str = (
            os.environ.get("MONTE_CALENDAR_PROVIDER", "forexfactory").strip().lower()
        )

        self.fred_configured: bool = bool(self.fred_api_key)
        self.te_configured: bool = bool(self.te_api_key)


settings = Settings()
