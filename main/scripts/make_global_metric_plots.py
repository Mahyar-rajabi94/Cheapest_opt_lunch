"""Global metric: every method's regret relative to random search, averaged over tasks.

For task i at evaluation budget B,
    r_{a,i}(B) = (f_{a,i}(B) - f*_i) / (f_{random,i}(B) - f*_i + EPS)
where f(B) is the best value found within B evaluations, mean over seeds. Then
    R_a(B) = mean_i r_{a,i}(B)
Lower is better: 0 = optimal, 1 = matches random search under the same budget.

A method is scored only up to its requested budget (capped GP-BO/TuRBO/BOBYQA
stop at 100). A run that stopped early on its own (e.g. L-BFGS-B converged)
keeps its final best value up to its requested budget. Seed spread: R is also
computed per seed index (denominator = the seed-mean random search), so the
mean of the per-seed R equals R_a; methods run with one seed get no band.

Constrained tasks: f(B) is undefined until a run finds a feasible point. The
random-search denominator is the mean over its seeds that have one; a seed of
any method with no feasible point yet scores r = 1 (no better than random
search). A task where random search never finds a feasible point has no
denominator and is left out of R (named in the output). Until every remaining
task has a feasible random-search point, R(B) is undefined and not drawn.
Noisy tasks use the observed best (y_best_so_far), as the per-task plots do.
Results of superseded task versions (SUPERSEDED) stay on disk but are left out.

Agents ran on only a few tasks, so their R covers just those tasks (counted in
the printed table) and they are left off the aggregate curve.

Writes <results>/global_metric_vs_budget.png (default results_linux/), R_a(B)
vs. evaluations for every baseline, and prints R_a at B = 100 and 2000.

Usage (from main/):
    python scripts/make_global_metric_plots.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullLocator

from make_comparison_plots import (
    AGENT_RUNS,
    AXIS,
    GRIDLINE,
    MAIN_ROOT,
    SANDBOX_ROOT,
    TEXT_MUTED,
    TEXT_PRIMARY,
    baseline_display_names,
    baseline_style,
    load_rows,
    regret_trajectory,
)

EPS = 1e-12
BASELINE = "random_search"
CHECKPOINTS = (100, 2000)
# Old task versions that separated no methods (see TASKS.md): Schaffer on CEC's
# [-100, 100] box, nevergrad noise level 10 with x* near the box center.
SUPERSEDED = {
    "cec2017_f6_d30",
    "cec2017_f6_d50",
    "nevergrad_noisy_rosenbrock_d20",
    "nevergrad_noisy_cigar_d20",
}


def padded(regret, budget):
    """Carry the final best value forward when a run stopped before its budget."""
    return np.concatenate([regret, np.full(budget - len(regret), regret[-1])])


def load_task_regrets(task_dir: Path):
    """method -> (n_seeds, budget) regret array; agents keyed by their label."""
    metas = sorted(task_dir.glob("*/seed_*/meta.json"))
    f_opt = json.loads(metas[0].read_text())["task_f_opt"]
    out = {}
    for name in baseline_style:
        paths = sorted((task_dir / name).glob("seed_*/meta.json"))
        if not paths:
            continue
        budget = json.loads(paths[0].read_text())["requested_budget"]
        out[name] = np.stack(
            [padded(regret_trajectory(load_rows(p.parent / "trace.csv"), f_opt), budget) for p in paths]
        )
    for label, rel in AGENT_RUNS.get(task_dir.name, {}).items():
        run_dir = SANDBOX_ROOT / rel
        if (run_dir / "trace.csv").exists():
            budget = json.loads((run_dir / "meta.json").read_text())["requested_budget"]
            out[label] = padded(regret_trajectory(load_rows(run_dir / "trace.csv"), f_opt), budget)[None, :]
    return out


def relative_regrets(results_root: Path):
    """(task -> method -> (n_seeds, budget) array of r_{a,i,s}(B), tasks left out)."""
    rel, skipped = {}, []
    for task_dir in sorted(p for p in results_root.iterdir() if (p / BASELINE).is_dir() and p.name not in SUPERSEDED):
        regrets = load_task_regrets(task_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column: no feasible seed yet
            denom = np.nanmean(regrets[BASELINE], axis=0) + EPS
        if np.isnan(denom).all():
            skipped.append(task_dir.name)
            continue
        rel[task_dir.name] = {}
        for m, r in regrets.items():
            d = denom[: r.shape[1]]
            # NaN under a defined denominator = this seed has no feasible point yet.
            rel[task_dir.name][m] = np.where(np.isnan(r) & np.isfinite(d), 1.0, r / d)
    return rel, skipped


def aggregate(rel, method):
    """R_a(B) per seed index, over the tasks the method ran on: (n_seeds, budget), n_tasks."""
    runs = [t[method] for t in rel.values() if method in t]
    n_seeds = min(r.shape[0] for r in runs)
    budget = min(r.shape[1] for r in runs)
    return np.mean([r[:n_seeds, :budget] for r in runs], axis=0), len(runs)


def methods_in(rel):
    seen = [m for t in rel.values() for m in t]
    baselines = [m for m in baseline_style if m in seen]
    agents = list(dict.fromkeys(m for m in seen if m not in baseline_style))
    return baselines, agents


def print_table(rel, skipped):
    baselines, agents = methods_in(rel)
    header = f"{'method':<24}{'tasks':>6}" + "".join(f"{f'R@{b}':>22}" for b in CHECKPOINTS)
    print(header)
    print("-" * len(header))
    for m in baselines + agents:
        per_seed, n_tasks = aggregate(rel, m)
        cells = []
        for b in CHECKPOINTS:
            if per_seed.shape[1] < b:
                cells.append(f"{'--':>22}")
                continue
            v = per_seed[:, b - 1]
            spread = f" ± {v.std():.3g}" if len(v) > 1 else ""
            cells.append(f"{f'{v.mean():.3g}{spread}':>22}")
        print(f"{baseline_display_names.get(m, m):<24}{n_tasks:>6}" + "".join(cells))
    print(f"\nMean ± std over seed indices. {len(rel)} tasks; left out (random search never feasible): "
          + (", ".join(skipped) or "none"))


def plot_vs_budget(results_root, rel, skipped):
    baselines, _ = methods_in(rel)
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    for m in baselines:
        per_seed, _ = aggregate(rel, m)
        x = np.arange(1, per_seed.shape[1] + 1)
        mean = per_seed.mean(axis=0)
        style = baseline_style[m]
        ax.plot(x, mean, linewidth=1.6, alpha=0.9, label=baseline_display_names[m], zorder=2, **style)
        if per_seed.shape[0] > 1:
            sd = per_seed.std(axis=0)
            ax.fill_between(x, np.clip(mean - sd, EPS, None), mean + sd, color=style["color"],
                            alpha=0.12, linewidth=0, zorder=1)

    ax.set_yscale("log")
    # Plain decimals: R spans well under a decade, so 10^k labels would be mostly empty.
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 3, 4, 5, 6, 8)))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(1, max(r.shape[1] for t in rel.values() for r in t.values()))
    ax.set_xlabel("Evaluations", fontsize=10)
    ax.set_ylabel("Relative regret R", fontsize=10)
    ax.grid(True, which="major", axis="both", color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_title(f"Regret relative to random search, mean over {len(rel)} tasks",
                 fontsize=12, color=TEXT_PRIMARY, pad=14, loc="left")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9.5,
              labelcolor=TEXT_PRIMARY)
    note = "Lower is better: 1 = random search, 0 = optimal. Bands: ±1 std over seeds. Infeasible runs count as 1."
    if skipped:
        note += " Left out (random search never feasible): " + ", ".join(skipped) + "."
    fig.text(0.0, -0.02, note, fontsize=8, color=TEXT_MUTED)
    fig.tight_layout()
    fig.savefig(results_root / "global_metric_vs_budget.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results_linux", help="results folder under main/")
    args = parser.parse_args()
    results_root = MAIN_ROOT / args.results
    rel, skipped = relative_regrets(results_root)
    print_table(rel, skipped)
    plot_vs_budget(results_root, rel, skipped)


if __name__ == "__main__":
    main()
