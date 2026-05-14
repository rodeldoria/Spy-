"""In-process driver that the CLI and Streamlit form both invoke.

A single ``run_one(cfg)`` call:
1. Opens (or creates) the SQLite DB and JSONL mirror dir.
2. Inserts a ``runs`` row with ``status='running'``.
3. Dispatches to the engine-specific ``replay_*.run`` function.
4. Aggregates ``signal_buckets`` for that run.
5. Marks the ``runs`` row ``status='ok'`` or ``status='error'``.

Triangulation runs are expanded to two configs (one per fixture mode) by
``run_engine_or_all`` so callers never have to remember the dual-mode
rule.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, replace
from typing import Callable

from monte.backtest import replay_dip_pump, replay_kalshi, replay_triangulation
from monte.backtest.config import BacktestConfig, EngineKind
from monte.backtest.scoring import aggregate_buckets
from monte.backtest.store import BacktestStore

_ENGINE_FN: dict[str, Callable] = {
    "dip_pump": replay_dip_pump.run,
    "kalshi": replay_kalshi.run,
    "triangulation": replay_triangulation.run,
}


@dataclass
class RunResult:
    run_id: str
    engine: str
    fixture_mode: str | None
    n_trades: int
    status: str
    error: str | None = None


def run_one(cfg: BacktestConfig) -> RunResult:
    """Execute exactly one engine run for ``cfg``. For triangulation, the
    caller is expected to pass a config with ``fixture_mode`` set; use
    ``run_engine_or_all`` to schedule both fixtures automatically."""
    with BacktestStore.open(cfg) as store:
        run_id = store.start_run(cfg)
        try:
            fn = _ENGINE_FN[cfg.engine]
            n = fn(cfg, store, run_id)
            aggregate_buckets(store, run_id, cfg.engine)
            store.finish_run(run_id, n_trades=n, status="ok")
            return RunResult(run_id=run_id, engine=cfg.engine,
                             fixture_mode=cfg.fixture_mode, n_trades=n, status="ok")
        except Exception as exc:
            err = traceback.format_exc()
            store.finish_run(run_id, n_trades=0, status="error", error=err)
            return RunResult(run_id=run_id, engine=cfg.engine,
                             fixture_mode=cfg.fixture_mode, n_trades=0,
                             status="error", error=str(exc))


def run_engine_or_all(cfg: BacktestConfig | None = None,
                      *, engines: list[EngineKind] | None = None,
                      base_cfg: BacktestConfig | None = None
                      ) -> list[RunResult]:
    """Schedule one or more runs. If ``engines`` is passed, dispatch one
    config per engine (with triangulation expanded to both fixture modes).
    Otherwise run the single ``cfg`` as-is.
    """
    if engines is None and cfg is None:
        raise ValueError("must pass either cfg or engines + base_cfg")
    if engines is not None:
        if base_cfg is None:
            raise ValueError("engines requires base_cfg")
        results: list[RunResult] = []
        for eng in engines:
            if eng == "triangulation":
                for mode in ("neutral", "seeded_random"):
                    results.append(run_one(replace(base_cfg, engine=eng, fixture_mode=mode)))
            else:
                results.append(run_one(replace(base_cfg, engine=eng, fixture_mode=None)))
        return results
    return [run_one(cfg)]  # type: ignore[arg-type]
