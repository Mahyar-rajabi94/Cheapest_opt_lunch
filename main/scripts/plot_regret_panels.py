"""Regret-vs-evaluation plot, one panel per task, in the clean reference
style (white background, thick flat-colored lines, no gridlines, spines
only left/bottom, legend centered below). One line per classical baseline
optimizer; if multiple seeds exist for an optimizer, average them.

Usage
-----
    python scripts/plot_regret_panels.py --task rosenbrock_d10_i1 --task sphere_d10_i1 ...
    python scripts/plot_regret_panels.py --all-in slide_figures/blockbox_functions
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

OPTIMIZERS = [
    "random_search", "sobol", "cma_es", "de", "turbo", "gp_bo", "bobyqa", "nelder_mead", "lbfgsb",
]
DISPLAY = {
    "random_search": "Random search", "sobol": "Sobol", "cma_es": "CMA-ES",
    "de": "Differential evolution", "turbo": "TuRBO", "gp_bo": "GP-BO",
    "bobyqa": "BOBYQA", "nelder_mead": "Nelder-Mead", "lbfgsb": "L-BFGS-B",
}
# One representative optimizer per classical "family", for the reduced view.
# gp_bo_2000 is GP-BO rerun at the full 2000-evaluation budget (10 seeds),
# pulled from origin/master:main/results_full -- this project's own gp_bo/
# directory is capped at 100 evals by convention, which would just be a
# short stub next to the other families' full-budget curves.
FAMILY_DISPLAY = {
    "gp_bo_2000": "Bayesian (GP-BO)",
    "lbfgsb": "Derivative-based (L-BFGS-B)",
    "cma_es": "Evolutionary (CMA-ES)",
    "random_search": "Random search",
}
COLORS = {
    "random_search": "#9b9b9b", "sobol": "#49b6a8", "cma_es": "#e0393e",
    "de": "#f2a93b", "turbo": "#e87ba4", "gp_bo": "#2a6fdb", "gp_bo_2000": "#2a6fdb",
    "bobyqa": "#1fab54", "nelder_mead": "#6a3fd6", "lbfgsb": "#5c4033",
}
LINEWIDTH = 2.6
BUDGET = 2000

# Friendly titles, from the astra_max report's own task names.
FRIENDLY_TITLE = {
    "rosenbrock_d10_i1": "Rosenbrock (d=10)",
    "rosenbrock_d20_i1": "Rosenbrock (d=20)",
    "rosenbrock_d40_i1": "Rosenbrock (d=40)",
    "rastrigin_d10_i1": "Rastrigin (d=10)",
    "sep_rastrigin_d10_i1": "Separable Rastrigin (d=10)",
    "bent_cigar_d10_i1": "Bent Cigar (d=10)",
    "sphere_d10_i1": "Sphere (d=10)",
    "step_ellipsoid_d10_i1": "Step Ellipsoid (d=10)",
    "sharp_ridge_d10_i1": "Sharp Ridge (d=10)",
    "weierstrass_d10_i1": "Weierstrass (d=10)",
    "schwefel_d10_i1": "Schwefel (d=10)",
    "gallagher101_d10_i1": "Gallagher 101 Peaks (d=10)",
    "gallagher21_d10_i1": "Gallagher 21 Peaks (d=10)",
    "cec2017_f5_d30": "CEC-2017 Rastrigin (d=30)",
    "cec2017_f6_d30_b5": "CEC-2017 Schaffer (d=30)",
    "cec2017_f6_d50_b5": "CEC-2017 Schaffer (d=50)",
    "cec2017_f8_d30": "CEC-2017 Non-continuous Rastrigin (d=30)",
    "pymoo_g04": "Pymoo G04 (constrained)",
    "pymoo_g07": "Pymoo G07 (constrained)",
    "bbob_constrained_f20_d10_i1": "Constrained Ellipsoid (d=10)",
    "bbob_constrained_f38_d10_i1": "Constrained Different Powers (d=10)",
    "bbob_constrained_f44_d10_i1": "Constrained Rastrigin (d=10)",
    "bbob_noisy_f110_d5_i1": "Noisy Rosenbrock, Gaussian (d=5)",
    "bbob_noisy_f111_d5_i1": "Noisy Rosenbrock, Uniform (d=5)",
    "bbob_noisy_f112_d5_i1": "Noisy Rosenbrock, Cauchy (d=5)",
    "nevergrad_noisy_rosenbrock_d20_n0.1": "Noisy Rosenbrock (d=20)",
    "nevergrad_noisy_cigar_d20_n0.1": "Noisy Cigar (d=20)",
    "bbob_mixint_f8_d10_i1": "Mixed-Integer Rosenbrock (d=10)",
    "bbob_mixint_f15_d10_i1": "Mixed-Integer Rastrigin (d=10)",
    "bbob_mixint_f17_d10_i1": "Mixed-Integer Schaffer (d=10)",
}


def load_rows(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def mean_regret_curve(task_dir: Path, optimizer: str, f_opt: float):
    opt_dir = task_dir / optimizer
    if not opt_dir.exists():
        return None
    seed_dirs = sorted(opt_dir.glob("seed_*"), key=lambda p: int(p.name.split("_")[1]))
    curves = []
    for seed_dir in seed_dirs:
        trace = seed_dir / "trace.csv"
        if not trace.exists():
            continue
        rows = load_rows(trace)
        if not rows:
            continue
        # Some pre-existing constrained-task baselines separate y/merit/
        # violation and leave y_best_so_far blank until the first feasible
        # point (sometimes never); recompute the running best ourselves from
        # 'merit' (or 'y' where there's no such column) instead of trusting it.
        col = "merit" if "merit" in rows[0] else "y"
        vals = np.array([float(r[col]) for r in rows])
        y = np.minimum.accumulate(vals) - f_opt
        curves.append(np.maximum(y, 1e-12))
    if not curves:
        return None
    # A method that stops before `BUDGET` (declared its own convergence, or
    # was capped by this project's budget convention) made no further
    # progress; carry its last incumbent forward rather than cutting the
    # line short, so every curve is shown over the full shared x-axis.
    full_curves = []
    for c in curves:
        if len(c) < BUDGET:
            c = np.concatenate([c, np.full(BUDGET - len(c), c[-1])])
        full_curves.append(c[:BUDGET])
    mean_curve = np.mean(np.stack(full_curves), axis=0)
    return np.arange(1, BUDGET + 1), mean_curve


def agent_regret_curve(trace_path: Path, f_opt: float):
    if not trace_path.exists():
        return None
    rows = load_rows(trace_path)
    if not rows:
        return None
    # Constrained tasks: regret is undefined before the first feasible point
    # (the astra_max trace leaves y_best_so_far blank there); only count
    # feasible evaluations and start the line at the first one.
    if "feasible" in rows[0]:
        rows = [r for r in rows if r["feasible"] == "True"]
        if not rows:
            return None
    eval_idx = np.array([int(r["eval_idx"]) for r in rows])
    y = np.minimum.accumulate(np.array([float(r["y"]) for r in rows])) - f_opt
    return eval_idx, np.maximum(y, 1e-12)


def plot_task(task: str, out_path: Path, agent_traces_dir: Path | None = None,
              optimizers: list[str] | None = None, display: dict[str, str] | None = None):
    optimizers = optimizers or OPTIMIZERS
    display = display or DISPLAY
    task_dir = ROOT / "results" / task
    meta_files = list(task_dir.glob("*/seed_0/meta.json"))
    f_opt = json.load(open(meta_files[0]))["task_f_opt"]

    fig, ax = plt.subplots(figsize=(7, 5.2), facecolor="white")
    ax.set_facecolor("white")
    for opt in optimizers:
        curve = mean_regret_curve(task_dir, opt, f_opt)
        if curve is None:
            continue
        x, y = curve
        ax.plot(x, y, color=COLORS[opt], linewidth=LINEWIDTH, label=display[opt])

    if agent_traces_dir is not None:
        curve = agent_regret_curve(agent_traces_dir / task / "trace.csv", f_opt)
        if curve is not None:
            x, y = curve
            ax.plot(x, y, color="#111111", linewidth=LINEWIDTH + 0.4, label="Astra max agent (1 run)")

    ax.set_yscale("log")
    ax.set_title(FRIENDLY_TITLE.get(task, task), fontsize=18, pad=14)
    ax.set_xlabel("Evaluation", fontsize=13)
    ax.set_ylabel("Regret", fontsize=13)
    ax.tick_params(labelsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 3), frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", action="append", default=[], dest="tasks")
    p.add_argument("--all-in", help="Directory whose subfolder names are task keys")
    p.add_argument("--outdir", default=str(ROOT.parent / "slide_figures" / "blockbox_functions"))
    p.add_argument("--agent-traces", help="Directory of <task>/trace.csv agent runs to overlay (same budget/axis)")
    p.add_argument("--flat", action="store_true", help="Write <outdir>/<task>.png instead of <outdir>/<task>/regret_vs_eval.png")
    p.add_argument("--families", action="store_true",
                    help="Only 4 representative baselines (Bayesian/derivative/evolutionary/random) instead of all 9")
    args = p.parse_args()

    tasks = list(args.tasks)
    if args.all_in:
        base = Path(args.all_in)
        if not base.is_absolute():
            base = ROOT.parent / base
        tasks += sorted(d.name for d in base.iterdir() if d.is_dir())

    outdir = Path(args.outdir)
    agent_traces_dir = Path(args.agent_traces) if args.agent_traces else None
    optimizers = list(FAMILY_DISPLAY) if args.families else None
    display = FAMILY_DISPLAY if args.families else None
    for task in tasks:
        out_path = outdir / f"{task}.png" if args.flat else outdir / task / "regret_vs_eval.png"
        plot_task(task, out_path, agent_traces_dir, optimizers, display)


if __name__ == "__main__":
    main()
