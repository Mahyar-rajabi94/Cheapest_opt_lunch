"""Run CO2 uptake baselines across optimizers and seeds.

Calls this package's run_campaign,
which minimizes negative predicted uptake and preserves all descriptor
coordinates in each trace. Outputs are written to:
    <outdir>/co2_uptake_d188_i1/<optimizer>/seed_<seed>/{trace.csv,meta.json}

Defaults are budget 2000, seeds 0..9, and results_linux/ on Linux or results/
elsewhere. An explicit --outdir overrides the output default.
GP-BO, TuRBO, and BOBYQA are capped at 100 evaluations. BOBYQA, Nelder-Mead,
and L-BFGS-B run seed 0 only. Completed campaigns with matching budgets are
skipped. Conflicting, failed, or incomplete outputs require a fresh outdir;
the sweep never overwrites them or resets a partially used campaign.

Usage from the MOF sandbox root:
    .venv/bin/python scripts/run_sweep.py --task co2_uptake --workers 4

Small smoke sweep using two inexpensive methods:
    .venv/bin/python scripts/run_sweep.py --task co2_uptake \
        --optimizers random_search sobol --budget 20 --seeds 2 \
        --workers 2 --outdir /tmp/co2_sweep_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_campaign import DEFAULT_OUTDIR, OPTIMIZERS, TASKS, run_campaign

SLOWEST_FIRST = ["gp_bo", "bobyqa", "turbo"]
BUDGET_CAPS = {"gp_bo": 100, "turbo": 100, "bobyqa": 100}

# The local registry stores lazy (module, class) tuples, not class objects.
DETERMINISTIC_OPTIMIZERS = {"bobyqa", "nelder_mead", "lbfgsb"}

# Older campaigns lack an explicit status. Recognize only the known clean
# optimizer-exhaustion messages; ambiguous saved errors must not be skipped.
_LEGACY_EXHAUSTION_PREFIXES = {
    "bobyqa": "BOBYQA's solver thread has already returned",
    "nelder_mead": "Nelder-Mead has already converged",
    "lbfgsb": "L-BFGS-B has already converged",
}
_thread_limiter = None


def _init_worker():
    """Keep worker BLAS threads bounded without requiring optional torch."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    global _thread_limiter
    _thread_limiter = threadpool_limits(limits=1)


def _timed_campaign(task_name, optimizer_name, seed, budget, outdir):
    start = time.perf_counter()
    if optimizer_name in {"gp_bo", "turbo"}:
        import torch

        torch.set_num_threads(1)
    run_campaign(task_name, optimizer_name, seed, budget, outdir)
    return time.perf_counter() - start


def _completion_problem(campaign_dir, task_name, optimizer_name, seed, budget):
    """Return None for a completed campaign, otherwise its conflict reason."""
    meta_path = campaign_dir / "meta.json"
    trace_path = campaign_dir / "trace.csv"
    if not meta_path.is_file() or not trace_path.is_file():
        return "incomplete output (trace.csv and meta.json are both required)"
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, UnicodeError, ValueError) as exc:
        return f"unreadable metadata: {exc}"
    if not isinstance(meta, dict):
        return "metadata must be a JSON object"
    if meta.get("requested_budget") != budget:
        return f"different budget: saved {meta.get('requested_budget')!r}, requested {budget}"
    if (
        meta.get("task_name") != task_name
        or meta.get("optimizer_name") != optimizer_name
        or meta.get("seed") != seed
    ):
        return "saved task, optimizer, or seed does not match the campaign path"

    used = meta.get("evaluations_used")
    if type(used) is not int or not 0 <= used <= budget:
        return "invalid evaluations_used"
    early = used < budget
    if meta.get("stopped_early") is not early:
        return "inconsistent stopped_early flag"

    reason = meta.get("stop_reason")
    status = meta.get("status")
    if status is not None and status != "completed":
        return f"campaign status is {status!r}"
    if early:
        if not isinstance(reason, str) or not reason:
            return "early campaign has no completion reason"
        if status is None:
            prefix = _LEGACY_EXHAUSTION_PREFIXES.get(optimizer_name)
            if prefix is None or not reason.startswith(prefix):
                return "legacy campaign has no unambiguous completion status"
    elif reason is not None:
        return "full-budget campaign has a saved error or interruption"

    try:
        with trace_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                return None if used == 0 else "trace is empty"
            if not {"eval_idx", "optimizer", "y", "y_best_so_far"} <= set(reader.fieldnames):
                return "trace is missing required columns"
            count = sum(1 for _ in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        return f"unreadable trace: {exc}"
    if count != used:
        return f"incomplete trace: {count} rows, metadata records {used}"
    return None


def _sweep(task_name, budget, seeds, outdir, workers, optimizers=None):
    """Return (completed campaign directories, failed (optimizer, seed) pairs)."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name!r}")
    for name, value in (("budget", budget), ("seeds", seeds), ("workers", workers)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if optimizers is None:
        names = list(OPTIMIZERS)
    else:
        if isinstance(optimizers, str) or not optimizers:
            raise ValueError("optimizers must be a nonempty sequence of optimizer names")
        if any(name not in OPTIMIZERS for name in optimizers):
            raise ValueError(f"Unknown optimizer in selection: {optimizers!r}")
        names = list(dict.fromkeys(optimizers))
    names.sort(
        key=lambda name: SLOWEST_FIRST.index(name)
        if name in SLOWEST_FIRST else len(SLOWEST_FIRST)
    )

    output_root = Path(outdir)
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"Output root is not a directory: {output_root}")
    task_dir = output_root / TASKS[task_name]().name
    jobs = []
    for name in names:
        n_seeds = 1 if name in DETERMINISTIC_OPTIMIZERS else seeds
        job_budget = min(budget, BUDGET_CAPS.get(name, budget))
        jobs.extend((name, seed, job_budget) for seed in range(n_seeds))

    todo, done, conflicts = [], [], []
    for name, seed, job_budget in jobs:
        campaign_dir = task_dir / name / f"seed_{seed}"
        if not campaign_dir.exists():
            todo.append((name, seed, job_budget))
        elif campaign_dir.is_dir() and not any(campaign_dir.iterdir()):
            todo.append((name, seed, job_budget))
        else:
            problem = _completion_problem(
                campaign_dir, task_dir.name, name, seed, job_budget
            )
            if problem is None:
                done.append((name, seed))
            else:
                conflicts.append(f"{campaign_dir}: {problem}")
    if conflicts:
        listing = "\n  ".join(conflicts)
        raise ValueError(
            f"Existing campaigns conflict with this sweep:\n  {listing}\n"
            "Use a fresh --outdir; existing results have been preserved."
        )

    total = len(jobs)
    count = 0
    failed = []
    campaign_dirs = [task_dir / name / f"seed_{seed}" for name, seed in done]
    for name, seed in done:
        count += 1
        print(f"[{count}/{total}] {name} seed={seed} skipped", flush=True)

    def finished(name, seed, seconds=None, error=None):
        nonlocal count
        count += 1
        if error is None:
            campaign_dirs.append(task_dir / name / f"seed_{seed}")
            print(f"[{count}/{total}] {name} seed={seed} {seconds:.1f}s", flush=True)
        else:
            failed.append((name, seed))
            print(f"[{count}/{total}] {name} seed={seed} FAILED: {error!r}", flush=True)

    if workers == 1:
        for name, seed, job_budget in todo:
            try:
                seconds = _timed_campaign(task_name, name, seed, job_budget, str(output_root))
            except Exception as exc:
                finished(name, seed, error=exc)
            else:
                finished(name, seed, seconds)
    elif todo:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            futures = {
                pool.submit(
                    _timed_campaign, task_name, name, seed, job_budget, str(output_root)
                ): (name, seed)
                for name, seed, job_budget in todo
            }
            for future in as_completed(futures):
                name, seed = futures[future]
                try:
                    seconds = future.result()
                except Exception as exc:
                    finished(name, seed, error=exc)
                else:
                    finished(name, seed, seconds)

    print(f"ran {len(todo) - len(failed)}, skipped {len(done)}, failed {len(failed)}", flush=True)
    return campaign_dirs, failed


def run_sweep(
    task_name: str,
    budget: int = 2000,
    seeds: int = 10,
    outdir: str = DEFAULT_OUTDIR,
    workers: int = 1,
    optimizers: list[str] | None = None,
) -> list[Path]:
    """Run a resumable sweep and return newly completed or skipped campaigns.

    Budget/output conflicts raise before any evaluation. Individual optimizer
    failures are logged and omitted from the returned paths, like main's
    runner. The command-line interface exits nonzero if any job fails.
    """
    return _sweep(task_name, budget, seeds, outdir, workers, optimizers)[0]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--optimizers", nargs="+", choices=sorted(OPTIMIZERS))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        _, failed = _sweep(
            args.task, args.budget, args.seeds, args.outdir, args.workers, args.optimizers
        )
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    sys.exit(1 if failed else 0)
