"""Average rank vs. budget: the agent against four baselines, over the 30 agent tasks.

Methods (legend label -> solver): Agent -> GPT-6 Astra agent (one run per
task); Bayesian -> GP-BO; Derivative-based -> L-BFGS-B; Evolutionary -> CMA-ES;
Random search -> uniform random search. Colors are set per method in METHODS
(chosen for this figure, overriding fs.SERIES_COLORS).

For task i and budget B, the five methods are ranked by g_{a,i}(B), the geometric
mean over seeds of the best regret within B evaluations (same scoring as
make_sbs_vbs_plot.py: floor 1e-8, infeasibility penalty, early stops held).
Rank 1 = lowest regret; tied methods share the average of their ranks (e.g.
several methods all at the floor). The plotted value is the mean rank over tasks,
from 1 (best on every task) to 5.

The plotted curves are smoothed: each point is the mean of the average-rank curve
over a centered window of +/- SMOOTH_FRAC * B evaluations (at least +/- 1), so
early budgets, where ranks change fast, are smoothed less. The printed table is
unsmoothed.

Follows main/FIGURES.md via figure_style.

Usage (from main/):
    ../.venv/bin/python scripts/make_avg_rank_plot.py
Writes <agent>/figures/avg_rank_vs_budget.png and prints mean ranks at checkpoints.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import rankdata

from make_comparison_plots import MAIN_ROOT
from make_sbs_vbs_plot import BUDGET, CHECKPOINTS, distances

import figure_style as fs  # after make_comparison_plots, which sets its own rcParams
import matplotlib.pyplot as plt

TIE_TOL = 1e-9  # log10 units; scores closer than this count as tied
# method -> (legend label, color), in legend order.
METHODS = {
    "agent": ("Agent", "#C93A5E"),
    "gp_bo": ("Bayesian", "#2761B4"),
    "lbfgsb": ("Derivative-based", "#87D3E1"),
    "cma_es": ("Evolutionary", "#3EACF1"),
    "random_search": ("Random search", "#A9A9A9"),
}
AGENT_LINE_WIDTH = 4.5  # thicker than fs.LINE_WIDTH so the agent stands out
FIGSIZE = (9, 5.5)
AXIS_LABEL_SIZE = 16  # larger than fs.LABEL_SIZE for this single-panel figure
SMOOTH_FRAC = 0.1


def smooth(y, frac=SMOOTH_FRAC):
    """Centered moving average whose half-width grows with budget: +/- max(1, frac * B)."""
    c = np.concatenate([[0.0], np.cumsum(y)])
    b = np.arange(1, len(y) + 1)
    half = np.maximum(1, np.round(frac * b)).astype(int)
    lo, hi = np.maximum(b - 1 - half, 0), np.minimum(b + half, len(y))
    return (c[hi] - c[lo]) / (hi - lo)


def ranks(dist):
    """method -> (n_tasks, budget) rank among METHODS on that task and budget."""
    methods = list(METHODS)
    # Distances to the VBS differ from log regret by a per-task constant, so they rank identically.
    scores = np.round(np.stack([dist[m] for m in methods]) / TIE_TOL) * TIE_TOL
    r = rankdata(scores, method="average", axis=0)
    return {m: r[k] for k, m in enumerate(methods)}


def print_table(rk, tasks):
    print(f"{len(tasks)} tasks, {len(rk)} methods. Mean rank over tasks; 1 = best.\n")
    header = f"{'method':<28}" + "".join(f"{f'B={b}':>10}" for b in CHECKPOINTS)
    print(header + "\n" + "-" * len(header))
    for m, r in rk.items():
        print(f"{METHODS[m][0]:<28}" + "".join(f"{r.mean(axis=0)[b - 1]:>10.2f}" for b in CHECKPOINTS))
    a = rk["agent"][:, -1]
    print(f"\nAt B={rk['agent'].shape[1]}: agent ranked (or tied for) first on {(a < 2).sum()} of {len(tasks)} tasks.")


def plot(rk, out):
    budget = rk["agent"].shape[1]
    x = np.arange(1, budget + 1)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # Agent drawn last so it sits on top; legend keeps METHODS order.
    for m, (label, color) in reversed(METHODS.items()):
        ax.plot(x, smooth(rk[m].mean(axis=0)), color=color, label=label,
                linewidth=AGENT_LINE_WIDTH if m == "agent" else fs.LINE_WIDTH)
    handles, labels = ax.get_legend_handles_labels()

    ax.set_xlim(0, budget)
    ax.set_ylim(len(rk) + 0.1, 0.9)  # inverted: rank 1 (best) at the top
    ax.set_yticks(range(1, len(rk) + 1))
    ax.set_xlabel("Budget", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Average Rank", fontsize=AXIS_LABEL_SIZE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(fs.AXIS)
        ax.spines[side].set_linewidth(fs.SPINE_WIDTH)
    ax.tick_params(which="both", color=fs.AXIS, labelsize=fs.TICK_SIZE)

    fig.legend(handles[::-1], labels[::-1], loc="lower center", ncol=len(labels), frameon=False,
               fontsize=fs.LEGEND_SIZE, bbox_to_anchor=(0.5, -0.04), columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=fs.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results_full", help="baseline results folder under main/")
    parser.add_argument("--agent", default="agent_runs/astra_max_2000_20261022", help="agent cohort folder under main/")
    parser.add_argument("--budget", type=int, default=BUDGET)
    args = parser.parse_args()
    agent_root = MAIN_ROOT / args.agent
    dist, tasks = distances(MAIN_ROOT / args.results, agent_root, args.budget,
                           portfolio=[m for m in METHODS if m != "agent"])
    rk = ranks(dist)
    print_table(rk, tasks)
    plot(rk, agent_root / "figures" / "avg_rank_vs_budget.png")


if __name__ == "__main__":
    main()
