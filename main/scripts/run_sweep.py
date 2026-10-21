"""Run every optimizer over many seeds on one task, via run_campaign().

Writes the same per-campaign files as run_campaign.py:
    <outdir>/<task_name>/<optimizer_name>/seed_<seed>/{trace.csv,meta.json}

Seeds 0..seeds-1 are run for each optimizer, except deterministic ones
(class attribute `deterministic = True`), which run seed 0 only. Optimizers
in BUDGET_CAPS run at min(budget, cap).

Reruns are safe: a campaign whose meta.json already exists with the same
requested_budget is skipped. If an existing campaign used a different
budget, the sweep errors before running anything; use a new --outdir.

Usage
-----
    python scripts/run_sweep.py --task rosenbrock10 --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_campaign import OPTIMIZERS, TASKS, run_campaign

# Slowest first, so a parallel sweep does not end waiting on one long job.
# Optimizers not listed here go after these.
SLOWEST_FIRST = ["gp_bo", "bobyqa", "turbo"]

# Expensive methods run at a small budget: GP-BO refits an exact GP every
# step (over an hour per seed at 1000 on CPU), TuRBO refits every batch, and
# the threaded Py-BOBYQA wrapper slows superlinearly with evaluations.
BUDGET_CAPS = {"gp_bo": 100, "turbo": 100, "bobyqa": 100}


def _init_worker():
    import torch

    torch.set_num_threads(1)


def _timed_campaign(task_name, optimizer_name, seed, budget, outdir):
    start = time.perf_counter()
    run_campaign(task_name, optimizer_name, seed, budget, outdir)
    return time.perf_counter() - start


def _sweep(task_name, budget, seeds, outdir, workers):
    """Return (completed campaign dirs, failed (optimizer, seed) pairs)."""
    task_dir = Path(outdir) / TASKS[task_name]().name

    names = sorted(
        OPTIMIZERS,
        key=lambda n: SLOWEST_FIRST.index(n) if n in SLOWEST_FIRST else len(SLOWEST_FIRST),
    )
    jobs = []
    for name in names:
        n_seeds = 1 if getattr(OPTIMIZERS[name], "deterministic", False) else seeds
        job_budget = min(budget, BUDGET_CAPS.get(name, budget))
        jobs += [(name, seed, job_budget) for seed in range(n_seeds)]

    todo, done, conflicts = [], [], []
    for name, seed, job_budget in jobs:
        meta_path = task_dir / name / f"seed_{seed}" / "meta.json"
        if not meta_path.exists():
            todo.append((name, seed, job_budget))
        elif json.loads(meta_path.read_text())["requested_budget"] == job_budget:
            done.append((name, seed))
        else:
            conflicts.append(meta_path.parent)
    if conflicts:
        listing = "\n  ".join(str(p) for p in conflicts)
        raise ValueError(
            f"existing results use a different budget than this sweep's:\n  {listing}\n"
            "Pass a different --outdir."
        )

    total = len(jobs)
    count = 0
    failed = []
    campaign_dirs = [task_dir / name / f"seed_{seed}" for name, seed in done]
    for name, seed in done:
        count += 1
        print(f"[{count}/{total}] {name} seed={seed} skipped")

    def finished(name, seed, seconds=None, error=None):
        nonlocal count
        count += 1
        if error is None:
            campaign_dirs.append(task_dir / name / f"seed_{seed}")
            print(f"[{count}/{total}] {name} seed={seed} {seconds:.1f}s")
        else:
            failed.append((name, seed))
            print(f"[{count}/{total}] {name} seed={seed} FAILED: {error!r}")

    if workers == 1:
        for name, seed, job_budget in todo:
            try:
                seconds = _timed_campaign(task_name, name, seed, job_budget, outdir)
            except Exception as e:
                finished(name, seed, error=e)
            else:
                finished(name, seed, seconds)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            futures = {
                pool.submit(_timed_campaign, task_name, name, seed, job_budget, outdir): (name, seed)
                for name, seed, job_budget in todo
            }
            for fut in as_completed(futures):
                name, seed = futures[fut]
                try:
                    seconds = fut.result()
                except Exception as e:
                    finished(name, seed, error=e)
                else:
                    finished(name, seed, seconds)

    print(f"ran {len(todo) - len(failed)}, skipped {len(done)}, failed {len(failed)}")
    return campaign_dirs, failed


def run_sweep(
    task_name: str,
    budget: int = 2000,
    seeds: int = 10,
    outdir: str = "results_linux",
    workers: int = 1,
) -> list[Path]:
    """Run the full optimizer x seed sweep; return the campaign directories
    that completed (newly run or already present).

    Raises ValueError, before running any job, if an existing campaign in
    `outdir` used a different requested_budget. Failed jobs are logged and
    skipped.
    """
    return _sweep(task_name, budget, seeds, outdir, workers)[0]


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--outdir", default="results_linux")
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        _, failed = _sweep(args.task, args.budget, args.seeds, args.outdir, args.workers)
    except ValueError as e:
        sys.exit(f"error: {e}")
    sys.exit(1 if failed else 0)
