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

Agents ran on only a few tasks, so their R covers just those tasks (counted in
the printed table) and they are left off the aggregate curve.

Produces in <results>/ (default results_linux/):
  global_metric_vs_budget.png  -- R_a(B) vs. evaluations, every baseline
  global_metric_heatmap.png    -- r_{a,i} per task x method at B=100 and B=2000,
                                  with R_a as the bottom row
and prints R_a at those budgets.

Usage (from main/):
    python scripts/make_global_metric_plots.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from make_comparison_plots import (
    AGENT_RUNS,
    AXIS,
    GRIDLINE,
    MAIN_ROOT,
    SANDBOX_ROOT,
    SURFACE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    baseline_display_names,
    baseline_style,
    load_rows,
    regret_trajectory,
)

EPS = 1e-12
BASELINE = "random_search"
CHECKPOINTS = (100, 2000)
# Diverging pair from the dataviz palette: blue = beats random, red = worse,
# neutral gray midpoint at r = 1.
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])
MAX_LOG_RATIO = 6  # color scale clipped at 1e-6 .. 1e6


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
    """task -> method -> (n_seeds, budget) array of r_{a,i,s}(B)."""
    rel = {}
    for task_dir in sorted(p for p in results_root.iterdir() if (p / BASELINE).is_dir()):
        regrets = load_task_regrets(task_dir)
        denom = regrets[BASELINE].mean(axis=0) + EPS
        rel[task_dir.name] = {m: r / denom[: r.shape[1]] for m, r in regrets.items()}
    return rel


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


def print_table(rel):
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
    print(f"\nMean ± std over seed indices; random search = 1 by construction. {len(rel)} tasks total.")


def plot_vs_budget(results_root, rel):
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
    ax.set_xlabel("Evaluation budget B", fontsize=10)
    ax.set_ylabel("R(B): mean over tasks of regret / random-search regret (log scale)", fontsize=10)
    ax.grid(True, which="major", axis="both", color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_title(f"Regret relative to random search, averaged over {len(rel)} tasks (lower is better)",
                 fontsize=12, color=TEXT_PRIMARY, pad=14, loc="left")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9.5,
              labelcolor=TEXT_PRIMARY, title="Paired hue = same family\n(solid vs. dashed)", title_fontsize=8.5)
    fig.text(0.0, -0.03,
             "R = 1: matches random search at the same budget; R = 0: optimal. "
             "Bands: ±1 std over seed indices (single-seed methods have none). "
             "Capped methods end at their budget. Tasks: " + ", ".join(rel) + ".",
             fontsize=8, color=TEXT_MUTED, wrap=True)
    fig.tight_layout()
    fig.savefig(results_root / "global_metric_vs_budget.png", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(results_root, rel):
    baselines, agents = methods_in(rel)
    methods = baselines + agents
    tasks = list(rel)
    norm = TwoSlopeNorm(vmin=-MAX_LOG_RATIO, vcenter=0, vmax=MAX_LOG_RATIO)

    fig, axes = plt.subplots(1, len(CHECKPOINTS), figsize=(7 * len(CHECKPOINTS), 0.5 * len(tasks) + 3.2),
                             dpi=150, sharey=True)
    for ax, b in zip(axes, CHECKPOINTS):
        grid = np.full((len(tasks) + 1, len(methods)), np.nan)
        for j, m in enumerate(methods):
            for i, t in enumerate(tasks):
                r = rel[t].get(m)
                if r is not None and r.shape[1] >= b:
                    grid[i, j] = r[:, b - 1].mean()
            # R_a only over every task, so a method missing a task gets no mean cell.
            if not np.isnan(grid[: len(tasks), j]).any():
                grid[-1, j] = grid[: len(tasks), j].mean()

        log_r = np.log10(np.clip(grid, 10.0**-MAX_LOG_RATIO, 10.0**MAX_LOG_RATIO))
        ax.imshow(np.ma.masked_invalid(log_r), cmap=DIVERGING, norm=norm, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                text = "--" if np.isnan(v) else f"{v:.2g}"
                ink = "white" if not np.isnan(v) and abs(log_r[i, j]) > 0.6 * MAX_LOG_RATIO else TEXT_PRIMARY
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color=TEXT_MUTED if np.isnan(v) else ink,
                        fontweight="bold" if i == len(tasks) else "normal")

        ax.axhline(len(tasks) - 0.5, color=SURFACE, linewidth=4)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([baseline_display_names.get(m, m) for m in methods], rotation=45,
                           ha="right", fontsize=9, color=TEXT_PRIMARY)
        ax.set_yticks(range(len(tasks) + 1))
        ax.set_yticklabels(tasks + ["R (mean over tasks)"], fontsize=9, color=TEXT_PRIMARY)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"Budget B = {b}", fontsize=11, color=TEXT_SECONDARY, loc="left")

    fig.suptitle("Regret relative to random search, r = regret / random-search regret (lower is better)",
                 fontsize=12, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.text(0.01, -0.02,
             "Blue: beats random search; gray: r = 1; red: worse. Color is log-scaled, clipped at "
             f"1e±{MAX_LOG_RATIO}. Cells: mean over seeds. '--': not run at that budget or on that task "
             "(capped baselines stop at 100; agents ran on a subset of tasks).",
             fontsize=8, color=TEXT_MUTED)
    fig.tight_layout()
    fig.savefig(results_root / "global_metric_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results_linux", help="results folder under main/")
    args = parser.parse_args()
    results_root = MAIN_ROOT / args.results
    rel = relative_regrets(results_root)
    print_table(rel)
    plot_vs_budget(results_root, rel)
    plot_heatmap(results_root, rel)


if __name__ == "__main__":
    main()
