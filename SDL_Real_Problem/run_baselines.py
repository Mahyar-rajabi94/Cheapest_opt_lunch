"""Run the master-branch optimizers on one SDL problem, then plot and summarize.

Each problem folder holds a `task.py` exposing `make_task()`, which returns a
`src.tasks.base.Task` (main/src/tasks/base.py). This script registers that
factory in main's task registry under the key "sdl_<folder>" and drives the
unchanged `run_campaign()` loop from main/scripts/run_campaign.py, so every
optimizer runs exactly as it does for the benchmark tasks.

Sweep settings follow main/scripts/run_sweep.py: budget 2000; gp_bo, turbo and
bobyqa capped at 100 (BUDGET_CAPS); deterministic optimizers (bobyqa,
nelder_mead, lbfgsb) run seed 0 only. Reruns skip finished campaigns.

Writes
    <problem>/results/<task.name>/<optimizer>/seed_<n>/{trace.csv,meta.json}
    <problem>/results/<task.name>/plot1_counterfactual_bar.png, plot2_regret_vs_eval.png
    <problem>/results/summary.md

Usage (from the repository root)
    python SDL_Real_Problem/run_baselines.py 01_nonnewtonian_blend --seeds 10 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MAIN = ROOT.parent / "main"
sys.path.insert(0, str(MAIN))

from scripts import run_campaign as rc  # noqa: E402
from scripts.run_sweep import BUDGET_CAPS, SLOWEST_FIRST  # noqa: E402

warnings.filterwarnings("ignore")


def register(problem: str) -> str:
    """Load <problem>/task.py and add its factory to main's TASKS registry."""
    path = ROOT / problem / "task.py"
    spec = importlib.util.spec_from_file_location(f"sdl_{problem}_task", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    key = f"sdl_{problem}"
    rc.TASKS[key] = module.make_task
    return key


def _init_worker(problem):
    import torch

    torch.set_num_threads(1)
    register(problem)


def _job(problem, key, optimizer, seed, budget, outdir):
    start = time.perf_counter()
    rc.run_campaign(key, optimizer, seed, budget, outdir)
    return time.perf_counter() - start


def summarize(task_dir: Path) -> str:
    """Markdown table of final best value and regret per optimizer (median over seeds)."""
    rows = []
    for opt_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
        finals, f_opt, budget = [], None, None
        for meta_path in sorted(opt_dir.glob("seed_*/meta.json")):
            meta = json.loads(meta_path.read_text())
            f_opt, budget = meta["task_f_opt"], meta["requested_budget"]
            with open(meta_path.parent / "trace.csv") as f:
                trace = list(csv.DictReader(f))
            if trace and trace[-1]["y_best_so_far"]:
                finals.append(float(trace[-1]["y_best_so_far"]))
        if finals:
            med = float(np.median(finals))
            rows.append((med - f_opt, opt_dir.name, med, min(finals), max(finals), len(finals), budget))
    rows.sort()
    lines = [
        f"f_opt (best-known reference) = {f_opt:.6g}",
        "",
        "| rank | optimizer | budget | seeds | median best y | median regret | best seed | worst seed |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, (reg, name, med, lo, hi, n, budget) in enumerate(rows, 1):
        lines.append(f"| {i} | {name} | {budget} | {n} | {med:.5g} | {reg:.3g} | {lo:.5g} | {hi:.5g} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("problem", help="problem folder name, e.g. 01_nonnewtonian_blend")
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--optimizers", nargs="*", default=None, help="subset of optimizers (default: all)")
    args = p.parse_args()

    key = register(args.problem)
    outdir = str(ROOT / args.problem / "results")
    task_name = rc.TASKS[key]().name
    task_dir = Path(outdir) / task_name

    names = args.optimizers or sorted(
        rc.OPTIMIZERS, key=lambda n: SLOWEST_FIRST.index(n) if n in SLOWEST_FIRST else len(SLOWEST_FIRST)
    )
    jobs = []
    for name in names:
        n_seeds = 1 if getattr(rc.OPTIMIZERS[name], "deterministic", False) else args.seeds
        budget = min(args.budget, BUDGET_CAPS.get(name, args.budget))
        for seed in range(n_seeds):
            if not (task_dir / name / f"seed_{seed}" / "meta.json").exists():
                jobs.append((name, seed, budget))
    print(f"{task_name}: {len(jobs)} campaigns to run")

    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(args.problem,)) as pool:
        futures = {pool.submit(_job, args.problem, key, n, s, b, outdir): (n, s) for n, s, b in jobs}
        for fut in as_completed(futures):
            n, s = futures[fut]
            try:
                print(f"  {n} seed {s}: {fut.result():.1f}s", flush=True)
            except Exception as e:  # keep the sweep going; report at the end
                print(f"  {n} seed {s}: FAILED {e!r}", flush=True)

    subprocess.run(
        [sys.executable, str(MAIN / "scripts" / "make_comparison_plots.py"), "--results", outdir, "--task", task_name],
        cwd=MAIN,
        check=True,
    )
    table = summarize(task_dir)
    (Path(outdir) / "summary.md").write_text(f"# Baseline results: {task_name}\n\n{table}\n")
    print(table)


if __name__ == "__main__":
    main()
