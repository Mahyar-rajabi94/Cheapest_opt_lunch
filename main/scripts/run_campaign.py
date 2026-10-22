"""Run a single optimization campaign: one task, one optimizer, one seed.

Writes two files per campaign:
    results/<task_name>/<optimizer_name>/seed_<seed>/trace.csv
    results/<task_name>/<optimizer_name>/seed_<seed>/meta.json

trace.csv has one row per evaluation (eval_idx, batch_idx, optimizer,
x0..x{dim-1}, y, y_best_so_far, eval_seconds, cumulative_seconds).
Constrained tasks add merit, violation and feasible after y, and
y_best_so_far is then the best *feasible* y so far (blank until the first
feasible point).
Noisy tasks add y_true (the noise-free value, never shown to the
optimizer) after y, and y_best_so_far is then the y_true of the point with
the lowest *observed* y so far, i.e. what "return the best observed point"
would deliver; meta.json gains task_noisy.
meta.json records the campaign-level configuration and provenance.

No optimizer hyperparameters are exposed here by design: each optimizer
uses its own standard defaults (see src/optimizers/).

Usage
-----
    python scripts/run_campaign.py --task rosenbrock10 --optimizer random_search --seed 0 --budget 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimizers import OptimizerExhausted
from src.optimizers.bobyqa import BOBYQA
from src.optimizers.cma_es import CMAES
from src.optimizers.differential_evolution import DifferentialEvolution
from src.optimizers.gp_bo import GPBO
from src.optimizers.lbfgsb import LBFGSB
from src.optimizers.nelder_mead import NelderMead
from src.optimizers.random_search import RandomSearch
from src.optimizers.sobol_search import SobolSearch
from src.optimizers.turbo import TuRBO
from src.tasks.registry import TASKS

# Optimizer registry: add a new optimizer here; nothing else in this file
# needs to change. Tasks are registered in src/tasks/registry.py.
OPTIMIZERS = {
    "random_search": RandomSearch,
    "sobol": SobolSearch,
    "cma_es": CMAES,
    "de": DifferentialEvolution,
    "gp_bo": GPBO,
    "turbo": TuRBO,
    "bobyqa": BOBYQA,
    "nelder_mead": NelderMead,
    "lbfgsb": LBFGSB,
}

# scipy's Sobol sampler warns whenever the *cumulative* number of points
# drawn so far is not a power of 2. Asking one point at a time (below)
# triggers this on most calls; it is harmless, just noisy, so it is
# silenced here rather than in src/optimizers/sobol_search.py.
warnings.filterwarnings("ignore", message=".*balance properties.*")


def _merit(ev) -> float:
    """Scalar fed to the optimizers: y if feasible, else a large penalty that
    ranks every infeasible point above every feasible one, by violation first."""
    if ev.feasible:
        return ev.y
    return 1e12 + 1e6 * ev.violation + float(np.tanh(ev.y))


def run_campaign(
    task_name: str,
    optimizer_name: str,
    seed: int,
    budget: int,
    outdir: str = "results",
) -> Path:
    """Run one (task, optimizer, seed) campaign for up to `budget` evaluations.

    Drives the optimizer one point at a time (ask(1) -> evaluate -> tell),
    so the same loop works for both the non-adaptive baselines here and
    any adaptive optimizer added later. Returns the campaign's output
    directory.

    Some optimizers (closed-loop local solvers such as BOBYQA or
    NelderMead) can converge and have nothing further to propose before
    `budget` evaluations are used; `ask` signals this by raising
    OptimizerExhausted. When that happens, the campaign stops early with
    whatever evaluations were actually completed, rather than crashing;
    meta.json records this with `stopped_early` and `stop_reason`.
    """
    task = TASKS[task_name]()
    task.reseed(seed)
    optimizer = OPTIMIZERS[optimizer_name](bounds=task.bounds, seed=seed)

    campaign_dir = Path(outdir) / task.name / optimizer_name / f"seed_{seed}"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    constrained = task.n_constraints > 0
    rows = []
    best_so_far = None
    best_observed = None
    cumulative_seconds = 0.0
    start_wall = time.perf_counter()
    start_time_utc = datetime.now(timezone.utc).isoformat()
    stopped_early = False
    stop_reason = None

    for i in range(budget):
        try:
            X = optimizer.ask(n=1)
        except OptimizerExhausted as e:
            stopped_early = True
            stop_reason = str(e)
            break
        x = X[0]
        ev = task.evaluate(x)
        y, elapsed = ev.y, ev.elapsed
        merit = _merit(ev)
        optimizer.tell(X, [merit])

        cumulative_seconds += elapsed
        if ev.feasible and (best_observed is None or y < best_observed):
            best_observed = y
            best_so_far = y if ev.y_true is None else ev.y_true

        row = {"eval_idx": i + 1, "batch_idx": i, "optimizer": optimizer_name}
        for d in range(task.dim):
            row[f"x{d}"] = x[d]
        row["y"] = y
        if task.noisy:
            row["y_true"] = ev.y_true
        if constrained:
            row["merit"] = merit
            row["violation"] = ev.violation
            row["feasible"] = ev.feasible
        row["y_best_so_far"] = best_so_far
        row["eval_seconds"] = elapsed
        row["cumulative_seconds"] = cumulative_seconds
        rows.append(row)

    end_time_utc = datetime.now(timezone.utc).isoformat()
    total_wall_seconds = time.perf_counter() - start_wall

    trace_path = campaign_dir / "trace.csv"
    if rows:
        with open(trace_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        trace_path.write_text("")  # optimizer was exhausted before any evaluation

    meta = {
        "task_name": task.name,
        "task_dim": task.dim,
        "task_direction": task.direction,
        "task_f_opt": task.f_opt,
        **({"task_n_constraints": task.n_constraints} if constrained else {}),
        **({"task_noisy": True} if task.noisy else {}),
        "optimizer_name": optimizer_name,
        "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
        "seed": seed,
        "requested_budget": budget,
        "evaluations_used": len(rows),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "start_time_utc": start_time_utc,
        "end_time_utc": end_time_utc,
        "total_wall_seconds": total_wall_seconds,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    with open(campaign_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return campaign_dir


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--optimizer", required=True, choices=sorted(OPTIMIZERS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--outdir", default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out_dir = run_campaign(args.task, args.optimizer, args.seed, args.budget, args.outdir)
    print(f"Wrote {out_dir / 'trace.csv'} and {out_dir / 'meta.json'}")
