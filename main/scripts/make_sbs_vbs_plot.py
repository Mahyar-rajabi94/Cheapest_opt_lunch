"""Single best solver (SBS) vs. virtual best solver (VBS): agent against the baseline portfolio.

For task i, method a and budget B, g_{a,i}(B) is the geometric mean over seeds of
the regret of the best point found within B evaluations, floored at FLOOR. The
VBS picks, per task and per budget, the best baseline:
    g_{VBS,i}(B) = min_a g_{a,i}(B)            (baselines only; the agent is excluded)
Every method is scored by its mean log-distance to the VBS:
    D_a(B) = mean_i log10( g_{a,i}(B) / g_{VBS,i}(B) )
so D_VBS = 0 and D = 1 means one order of magnitude worse regret than the
per-task oracle choice, on average. The ratio removes each task's regret scale,
so tasks with huge regrets do not dominate the mean. The SBS at budget B is the
one baseline with the lowest D(B) over all tasks. The agent (one run per task)
can go below 0 by beating every baseline on a task.

Gap closed = (D_SBS - D_agent) / (D_SBS - D_VBS) = 1 - D_agent / D_SBS:
0 = no better than always running the SBS, 1 = as good as the per-task oracle.

Details:
- Portfolio: the baselines in PORTFOLIO (Sobol, DE and GP-BO are excluded).
- Tasks: those in both --results (baselines) and --agent (agent analysis/) folders.
- Regret uses the noise-free score of the incumbent (y_best_so_far) for both.
- Runs that stopped early are held at their final value up to the budget.
- Constrained tasks: before a run's first feasible point its regret is the worst
  feasible regret any method reached on that task (a penalty, not a measurement).
- Regret <= FLOOR (incl. tiny negatives from the stored optimum) counts as solved.

Usage (from main/):
    ../.venv/bin/python scripts/make_sbs_vbs_plot.py
Writes <agent>/figures/sbs_vbs_vs_budget.png and prints D and gap closed at checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from make_comparison_plots import (
    AXIS,
    MAIN_ROOT,
    TEXT_MUTED,
    TEXT_PRIMARY,
    baseline_display_names,
    baseline_style,
    load_rows,
    regret_trajectory,
)

FLOOR = 1e-8
BUDGET = 2000
CHECKPOINTS = (50, 100, 250, 500, 1000, 2000)
AGENT_COLOR = "#2a78d6"
# Left out of the portfolio entirely: not plotted, and not candidates for the SBS or VBS.
EXCLUDED = ("sobol", "de", "gp_bo")
PORTFOLIO = [m for m in baseline_style if m not in EXCLUDED]


def held(regret, budget):
    """Truncate to budget; carry the last value forward if the run stopped early."""
    regret = regret[:budget]
    return np.concatenate([regret, np.full(budget - len(regret), regret[-1])])


def load_task(task_dir: Path, agent_dir: Path, budget: int, portfolio=PORTFOLIO):
    """method -> (n_seeds, budget) raw regret (NaN = no feasible point yet); agent keyed 'agent'."""
    f_opt = json.loads(next(task_dir.glob("*/seed_*/meta.json")).read_text())["task_f_opt"]
    runs = {}
    for name in portfolio:
        traces = sorted((task_dir / name).glob("seed_*/trace.csv"))
        if traces:
            runs[name] = np.stack([held(regret_trajectory(load_rows(t), f_opt), budget) for t in traces])
    runs["agent"] = held(regret_trajectory(load_rows(agent_dir / "trace.csv"), f_opt), budget)[None, :]
    return runs


def log_geomean(runs):
    """method -> (budget,) log10 geometric-mean regret, with the infeasibility penalty applied."""
    finite = np.concatenate([r[np.isfinite(r)] for r in runs.values()])
    penalty = max(finite.max(), FLOOR)
    return {m: np.log10(np.clip(np.where(np.isnan(r), penalty, r), FLOOR, None)).mean(axis=0)
            for m, r in runs.items()}


def distances(results_root: Path, agent_root: Path, budget: int, portfolio=PORTFOLIO):
    """(method -> (n_tasks, budget) log10 distance to the VBS, task names)."""
    tasks = sorted(p.name for p in (agent_root / "analysis").iterdir()
                   if (p / "trace.csv").exists() and (results_root / p.name).is_dir())
    per_task = [log_geomean(load_task(results_root / t, agent_root / "analysis" / t, budget, portfolio)) for t in tasks]
    methods = [m for m in portfolio if all(m in g for g in per_task)] + ["agent"]
    dist = {m: [] for m in methods}
    for g in per_task:
        vbs = np.min([g[m] for m in methods if m != "agent"], axis=0)
        for m in methods:
            dist[m].append(g[m] - vbs)
    return {m: np.array(d) for m, d in dist.items()}, tasks


def sbs_curve(dist):
    """Per budget: the lowest mean distance among baselines, and which baseline it is."""
    names = [m for m in dist if m != "agent"]
    means = np.array([dist[m].mean(axis=0) for m in names])
    return means.min(axis=0), [names[i] for i in means.argmin(axis=0)]


def print_table(dist, tasks):
    sbs, sbs_names = sbs_curve(dist)
    agent = dist["agent"].mean(axis=0)
    print(f"{len(tasks)} tasks. D = mean log10(regret / VBS regret); lower is better, VBS = 0.\n")
    header = f"{'method':<24}" + "".join(f"{f'B={b}':>10}" for b in CHECKPOINTS)
    print(header + "\n" + "-" * len(header))
    for m, d in dist.items():
        label = "Agent" if m == "agent" else baseline_display_names[m]
        print(f"{label:<24}" + "".join(f"{d.mean(axis=0)[b - 1]:>10.3f}" for b in CHECKPOINTS))
    print(f"{'SBS':<24}" + "".join(f"{sbs[b - 1]:>10.3f}" for b in CHECKPOINTS))
    print(f"{'  SBS method':<24}" + "".join(f"{baseline_display_names[sbs_names[b - 1]]:>10.9}" for b in CHECKPOINTS))
    print(f"{'Gap closed (agent)':<24}" + "".join(f"{1 - agent[b - 1] / sbs[b - 1]:>10.2f}" for b in CHECKPOINTS))
    a = dist["agent"][:, -1]
    print(f"\nAt B={dist['agent'].shape[1]}: agent beats the VBS on {(a < -1e-9).sum()}, "
          f"ties on {(np.abs(a) <= 1e-9).sum()}, loses on {(a > 1e-9).sum()} of {len(tasks)} tasks.")


def plot(dist, tasks, out: Path):
    budget = dist["agent"].shape[1]
    x = np.arange(1, budget + 1)
    sbs, _ = sbs_curve(dist)
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    for m, d in dist.items():
        if m == "agent":
            continue
        ax.plot(x, d.mean(axis=0), linewidth=1.3, alpha=0.55, label=baseline_display_names[m],
                **baseline_style[m])
    ax.plot(x, sbs, color="#000000", linestyle="--", linewidth=2.4, label="Single best solver (SBS)")
    ax.axhline(0, color="#000000", linewidth=2.4, label="Virtual best solver (VBS)")
    ax.plot(x, dist["agent"].mean(axis=0), color=AGENT_COLOR, linewidth=3, label="Agent")

    ax.set_xscale("log")
    ax.set_xlim(1, budget)
    ax.set_xlabel("Budget (objective evaluations)", fontsize=13)
    ax.set_ylabel("Mean log$_{10}$(regret / VBS regret)", fontsize=13)
    ax.tick_params(labelsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.set_title(f"Agent vs. single and virtual best baseline, mean over {len(tasks)} tasks",
                 fontsize=14, color=TEXT_PRIMARY, pad=12, loc="left")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10,
              labelcolor=TEXT_PRIMARY)
    fig.text(0.0, -0.03,
             "Lower is better. VBS = best baseline per task and budget; SBS = best single baseline on average "
             f"at each budget. Regret: geometric mean over seeds, floored at {FLOOR:g}; agent = one run per task.",
             fontsize=8, color=TEXT_MUTED)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"\nWrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results_full", help="baseline results folder under main/")
    parser.add_argument("--agent", default="agent_runs/astra_max_2000_20261022", help="agent cohort folder under main/")
    parser.add_argument("--budget", type=int, default=BUDGET)
    args = parser.parse_args()
    agent_root = MAIN_ROOT / args.agent
    dist, tasks = distances(MAIN_ROOT / args.results, agent_root, args.budget)
    print_table(dist, tasks)
    plot(dist, tasks, agent_root / "figures" / "sbs_vbs_vs_budget.png")


if __name__ == "__main__":
    main()
