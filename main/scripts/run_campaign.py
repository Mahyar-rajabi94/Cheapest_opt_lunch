"""Run a single optimization campaign: one task, one optimizer, one seed.

Writes two files per campaign:
    results/<task_name>/<optimizer_name>/seed_<seed>/trace.csv
    results/<task_name>/<optimizer_name>/seed_<seed>/meta.json

trace.csv has one row per evaluation (eval_idx, batch_idx, optimizer,
x0..x{dim-1}, y, y_best_so_far, eval_seconds, cumulative_seconds).
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
from src.optimizers.rule_based import RuleBased
from src.optimizers.sobol_search import SobolSearch
from src.optimizers.turbo import TuRBO
from src.tasks.bbob import BBOBTask
from src.tasks.bent_cigar import BentCigarTask
from src.tasks.cec2017 import CEC2017Task
from src.tasks.coco_constrained import COCOConstrainedTask
from src.tasks.coco_noisy import COCONoisyTask
from src.tasks.nevergrad_noisy import NevergradNoisyTask
from src.tasks.pymoo_g import PymooGTask
from src.tasks.rastrigin import RastriginTask
from src.tasks.rosenbrock import RosenbrockTask
from src.tasks.sphere import SphereTask
from src.tasks.yahpo import YAHPOTask


class _TaskCompat:
    """Adapts the newer Task interface (evaluate(x) -> Evaluation) to this
    script's older convention (evaluate(x) -> (y, elapsed))."""

    def __init__(self, t):
        self.name = t.name
        self.dim = t.dim
        self.bounds = t.bounds
        self.direction = t.direction
        self.f_opt = t.f_opt
        self._t = t

    def evaluate(self, x):
        ev = self._t.evaluate(x)
        return ev.y, ev.elapsed


class _BBOBCompat(_TaskCompat):
    def __init__(self, name: str, function_id: int, dim: int):
        super().__init__(BBOBTask(name, function_id, dim=dim, instance=1))


class _CEC2017Compat(_TaskCompat):
    def __init__(self, function_id: int, dim: int, half_width: float = 100.0):
        super().__init__(CEC2017Task(function_id, dim=dim, half_width=half_width))


class _CocoNoisyCompat(_TaskCompat):
    def __init__(self, function_id: int, dim: int, instance: int = 1):
        super().__init__(COCONoisyTask(function_id, dim=dim, instance=instance))


class _NevergradNoisyCompat(_TaskCompat):
    def __init__(self, function_name: str, dim: int):
        super().__init__(NevergradNoisyTask(function_name, dim=dim))


class _ConstrainedCompat(_TaskCompat):
    """Exposes a constrained task as a single scalar: a penalized merit
    (y + penalty_lambda * total constraint violation), since this project's
    optimizers and harness only understand one float per point. The merit's
    minimum coincides with f_opt because the true constrained optimum has
    zero violation. penalty_lambda is tuned per task so violation dominates
    the objective's own scale, to push search toward feasibility."""

    def __init__(self, t, penalty_lambda: float):
        super().__init__(t)
        self._lambda = penalty_lambda

    def evaluate(self, x):
        ev = self._t.evaluate(x)
        merit = ev.y + self._lambda * ev.violation
        return merit, ev.elapsed


class _PymooCompat(_ConstrainedCompat):
    def __init__(self, problem_name: str, penalty_lambda: float):
        super().__init__(PymooGTask(problem_name), penalty_lambda)


class _CocoConstrainedCompat(_ConstrainedCompat):
    def __init__(self, function_id: int, dim: int = 10, instance: int = 1, penalty_lambda: float = 1.0):
        super().__init__(COCOConstrainedTask(function_id, dim=dim, instance=instance), penalty_lambda)


class _YAHPOCompat(_TaskCompat):
    def __init__(self, scenario: str, instance: str = "40981"):
        super().__init__(YAHPOTask(scenario, instance=instance))


# Registries: add a new task or optimizer here; nothing else in this file
# needs to change.
TASKS = {
    "rosenbrock10": lambda: RosenbrockTask(dim=10, instance=1, rotated=False),
    "rastrigin10": lambda: RastriginTask(dim=10, instance=1),
    "bent_cigar10": lambda: BentCigarTask(dim=10, instance=1),
    "sphere10": lambda: SphereTask(dim=10, instance=1),
    "rosenbrock20": lambda: _BBOBCompat("rosenbrock", 8, 20),
    "rosenbrock40": lambda: _BBOBCompat("rosenbrock", 8, 40),
    "step_ellipsoid10": lambda: _BBOBCompat("step_ellipsoid", 7, 10),
    "sharp_ridge10": lambda: _BBOBCompat("sharp_ridge", 13, 10),
    "weierstrass10": lambda: _BBOBCompat("weierstrass", 16, 10),
    "cec_f5_d30": lambda: _CEC2017Compat(5, 30),
    "cec_f6_d30_b5": lambda: _CEC2017Compat(6, 30, half_width=5.0),
    "cec_f6_d50_b5": lambda: _CEC2017Compat(6, 50, half_width=5.0),
    "cec_f8_d30": lambda: _CEC2017Compat(8, 30),
    "noisy_f110_d5": lambda: _CocoNoisyCompat(110, 5),
    "noisy_f111_d5": lambda: _CocoNoisyCompat(111, 5),
    "noisy_f112_d5": lambda: _CocoNoisyCompat(112, 5),
    "ng_noisy_rosenbrock20": lambda: _NevergradNoisyCompat("rosenbrock", 20),
    "ng_noisy_cigar20": lambda: _NevergradNoisyCompat("cigar", 20),
    "pymoo_g04": lambda: _PymooCompat("g4", penalty_lambda=5000.0),
    "pymoo_g07": lambda: _PymooCompat("g7", penalty_lambda=10.0),
    "constrained_f20": lambda: _CocoConstrainedCompat(20, dim=10, penalty_lambda=1.0),
    "constrained_f38": lambda: _CocoConstrainedCompat(38, dim=10, penalty_lambda=1.0),
    "constrained_f44": lambda: _CocoConstrainedCompat(44, dim=10, penalty_lambda=1.0),
    "yahpo_svm": lambda: _YAHPOCompat("rbv2_svm"),
    "yahpo_xgboost": lambda: _YAHPOCompat("rbv2_xgboost"),
}

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
    "rule_based": RuleBased,
}

# scipy's Sobol sampler warns whenever the *cumulative* number of points
# drawn so far is not a power of 2. Asking one point at a time (below)
# triggers this on most calls; it is harmless, just noisy, so it is
# silenced here rather than in src/optimizers/sobol_search.py.
warnings.filterwarnings("ignore", message=".*balance properties.*")


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
    optimizer = OPTIMIZERS[optimizer_name](bounds=task.bounds, seed=seed)

    campaign_dir = Path(outdir) / task.name / optimizer_name / f"seed_{seed}"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best_so_far = None
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
        y, elapsed = task.evaluate(x)
        optimizer.tell(X, [y])

        cumulative_seconds += elapsed
        if best_so_far is None or y < best_so_far:
            best_so_far = y

        # Some optimizers (e.g. rule_based) internally route between several
        # sub-methods; such an optimizer exposes a `tags_` list (one entry
        # per ask() call, same order) so the trace can record which
        # sub-method actually produced each point, instead of one flat label.
        tags = getattr(optimizer, "tags_", None)
        tag = tags[i] if tags is not None and i < len(tags) else optimizer_name
        row = {"eval_idx": i + 1, "batch_idx": i, "optimizer": tag}
        for d in range(task.dim):
            row[f"x{d}"] = x[d]
        row["y"] = y
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
