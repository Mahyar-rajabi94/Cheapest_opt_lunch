"""Run one fixed-optimizer CO2 uptake baseline: one optimizer, one seed.

Writes the same unconstrained trace columns and campaign provenance into:
    <outdir>/co2_uptake_d188_i1/<optimizer>/seed_<seed>/trace.csv
    <outdir>/co2_uptake_d188_i1/<optimizer>/seed_<seed>/meta.json

The default outdir is results_linux/ on Linux and results/ elsewhere.
An explicit --outdir overrides this default.

The objective is negative predicted uptake, so lower values are better. The
saved model and its ordered descriptor bounds come from CO2UptakeTask. There
is no supplied reference optimum, so metadata does not claim one or compute
regret.

Optimizers receive only the nonconstant descriptor coordinates, with their
original bounds and default settings. Every evaluation and trace row uses
the complete descriptor vector, including unchanged fixed coordinates.
Only the selected optimizer is imported; optional dependencies for other
methods are not needed to run random search.

Usage from the MOF sandbox root:
    .venv/bin/python scripts/run_campaign.py --task co2_uptake \
        --optimizer random_search --seed 0 --budget 20 --outdir /tmp/smoke
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import shlex
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import scipy

SANDBOX = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX))

from src.agent_harness.co2_uptake import CO2UptakeTask
from src.optimizers import OptimizerExhausted

DEFAULT_OUTDIR = "results_linux" if sys.platform.startswith("linux") else "results"

# Keep the task local: this sandbox does not depend on main/src/tasks/.
TASKS = {"co2_uptake": CO2UptakeTask}

# Match the main runner's names; load just the requested backend.
OPTIMIZERS = {
    "random_search": ("src.optimizers.random_search", "RandomSearch"),
    "sobol": ("src.optimizers.sobol_search", "SobolSearch"),
    "cma_es": ("src.optimizers.cma_es", "CMAES"),
    "de": ("src.optimizers.differential_evolution", "DifferentialEvolution"),
    "gp_bo": ("src.optimizers.gp_bo", "GPBO"),
    "turbo": ("src.optimizers.turbo", "TuRBO"),
    "bobyqa": ("src.optimizers.bobyqa", "BOBYQA"),
    "nelder_mead": ("src.optimizers.nelder_mead", "NelderMead"),
    "lbfgsb": ("src.optimizers.lbfgsb", "LBFGSB"),
}

warnings.filterwarnings("ignore", message=".*balance properties.*")


def _load_optimizer(name: str):
    module_name, class_name = OPTIMIZERS[name]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        install_command = shlex.join(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        raise ImportError(
            f"Cannot import optimizer {name!r} using {sys.executable!r}: {exc}. "
            f"Install this sandbox's requirements with {install_command}"
        ) from exc
    return getattr(module, class_name)


def run_campaign(
    task_name: str,
    optimizer_name: str,
    seed: int,
    budget: int,
    outdir: str = DEFAULT_OUTDIR,
) -> Path:
    """Run up to ``budget`` real evaluations and return the campaign folder.

    Like the main runner, drive every optimizer with ask(1)/evaluate/tell.
    OptimizerExhausted ends a local method early and is recorded in metadata.
    Existing campaigns are preserved; select another output root or seed to
    run again. Each call creates an independent fixed-optimizer baseline.
    """
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name!r}")
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {optimizer_name!r}")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    optimizer_class = _load_optimizer(optimizer_name)
    task = TASKS[task_name]()
    bounds = np.asarray(task.bounds, dtype=float)
    free = bounds[:, 0] < bounds[:, 1]
    if not free.any():
        raise ValueError("The task has no free coordinates to optimize")

    campaign_dir = Path(outdir) / task.name / optimizer_name / f"seed_{seed}"
    if campaign_dir.exists() and (
        not campaign_dir.is_dir() or any(campaign_dir.iterdir())
    ):
        raise FileExistsError(
            f"Campaign output already exists: {campaign_dir}. "
            "Choose a fresh --outdir or --seed."
        )

    if optimizer_name in {"gp_bo", "turbo"}:
        # TuRBO uses torch's RNG as well as its task-local numpy generator.
        import torch

        torch.manual_seed(seed)
    optimizer = optimizer_class(bounds=bounds[free], seed=seed)
    campaign_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best_so_far = None
    cumulative_seconds = 0.0
    start_wall = time.perf_counter()
    start_time_utc = datetime.now(timezone.utc).isoformat()
    stop_reason = None
    status = "completed"

    try:
        for i in range(budget):
            try:
                X = optimizer.ask(n=1)
            except OptimizerExhausted as exc:
                stop_reason = str(exc)
                break

            # Preserve the original coordinate order and fixed values.
            x = bounds[:, 0].copy()
            x[free] = X[0]
            y, elapsed = task.evaluate(x)
            y, elapsed = float(y), float(elapsed)

            cumulative_seconds += elapsed
            if best_so_far is None or y < best_so_far:
                best_so_far = y

            row = {"eval_idx": i + 1, "batch_idx": i, "optimizer": optimizer_name}
            for d in range(task.dim):
                row[f"x{d}"] = float(x[d])
            row["y"] = y
            row["y_best_so_far"] = best_so_far
            row["eval_seconds"] = elapsed
            row["cumulative_seconds"] = cumulative_seconds
            rows.append(row)

            # The optimizer sees its free-coordinate candidate and raw loss.
            optimizer.tell(X, [y])
    except BaseException as exc:
        # Preserve completed real evaluations on interruption or solver error.
        stop_reason = f"{type(exc).__name__}: {exc}"
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        raise
    finally:
        end_time_utc = datetime.now(timezone.utc).isoformat()
        total_wall_seconds = time.perf_counter() - start_wall

        trace_path = campaign_dir / "trace.csv"
        if rows:
            with trace_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        else:
            trace_path.write_text("")

        meta = {
            "task_name": task.name,
            "task_dim": task.dim,
            "task_direction": task.direction,
            "task_free_dim": int(free.sum()),
            "optimizer_name": optimizer_name,
            "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
            "seed": seed,
            "requested_budget": budget,
            "evaluations_used": len(rows),
            "stopped_early": len(rows) < budget,
            "stop_reason": stop_reason,
            "status": status,
            "start_time_utc": start_time_utc,
            "end_time_utc": end_time_utc,
            "total_wall_seconds": total_wall_seconds,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "xgboost_version": version("xgboost"),
        }
        with (campaign_dir / "meta.json").open("w") as stream:
            json.dump(meta, stream, indent=2)

    return campaign_dir


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--optimizer", required=True, choices=sorted(OPTIMIZERS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    return parser, parser.parse_args()


if __name__ == "__main__":
    parser, args = _parse_args()
    try:
        out_dir = run_campaign(
            args.task, args.optimizer, args.seed, args.budget, args.outdir
        )
    except (ValueError, FileExistsError, ImportError) as exc:
        parser.error(str(exc))
    print(f"Wrote {out_dir / 'trace.csv'} and {out_dir / 'meta.json'}")
