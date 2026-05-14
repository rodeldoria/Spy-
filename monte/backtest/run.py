"""CLI entrypoint:

    python -m monte.backtest.run --engine all --symbols BTC,ETH --timeframes 1h \\
        --start 2024-01-01 --end 2025-05-01 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from monte.backtest.config import (
    BacktestConfig,
    DEFAULT_MIN_EDGE_PP,
    DEFAULT_MIN_EV_CENTS,
    DEFAULT_SEED,
    DEFAULT_TIMEOUT_BARS,
    EngineKind,
)
from monte.backtest.runner import run_engine_or_all

_ENGINES: tuple[EngineKind, ...] = ("dip_pump", "kalshi", "triangulation")


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m monte.backtest.run")
    p.add_argument("--engine", default="all",
                   choices=[*_ENGINES, "all"], help="engine to replay (default: all)")
    p.add_argument("--symbols", default="BTC,ETH,SOL", type=_csv)
    p.add_argument("--timeframes", default="1h,1d", type=_csv)
    p.add_argument("--start", default="2024-01-01", help="ISO start date")
    p.add_argument("--end", default=date.today().isoformat(), help="ISO end date")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--min-edge-pp", type=float, default=DEFAULT_MIN_EDGE_PP)
    p.add_argument("--min-ev-cents", type=float, default=DEFAULT_MIN_EV_CENTS)
    p.add_argument("--timeout-bars", type=int, default=DEFAULT_TIMEOUT_BARS)
    args = p.parse_args(argv)

    base = BacktestConfig(
        engine="dip_pump",
        symbols=tuple(args.symbols),
        timeframes=tuple(args.timeframes),
        start_date=args.start,
        end_date=args.end,
        seed=args.seed,
        min_edge_pp=args.min_edge_pp,
        min_ev_cents=args.min_ev_cents,
        timeout_bars=args.timeout_bars,
    )
    engines: list[EngineKind] = list(_ENGINES) if args.engine == "all" else [args.engine]
    results = run_engine_or_all(engines=engines, base_cfg=base)

    print(f"{'engine':<16} {'fixture':<14} {'status':<8} {'n_trades':>8}  run_id")
    for r in results:
        print(f"{r.engine:<16} {(r.fixture_mode or '-'):<14} {r.status:<8} "
              f"{r.n_trades:>8}  {r.run_id}")
        if r.error:
            print(f"  error: {r.error}")
    bad = [r for r in results if r.status != "ok"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
