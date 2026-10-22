"""Side-by-side 'optimizer family' comparison panels: a single deterministic
local (gradient-family) solver vs. the population-based (evolutionary-family)
solver's mean +/- spread across seeds, one panel per task category.

Matches the reference figure style: clean white background, thick lines,
log-scale regret, no gridlines, legend centered below.

Usage
-----
    python scripts/plot_family_comparison.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

GRADIENT_COLOR = "#2a6fdb"
EVO_COLOR = "#e0393e"
EVO_BAND_ALPHA = 0.18

# One representative task per category, and which baseline represents each
# "family" for that task (gradient = single deterministic local solver;
# evolutionary = population-based method, aggregated over its 5 seeds).
PANELS = [
    dict(title="Smooth", task="rosenbrock_d10_i1", gradient="bobyqa", evolutionary="cma_es"),
    dict(title="Mixed Integer", task="bbob_mixint_f8_d10_i1", gradient="bobyqa", evolutionary="cma_es"),
]

BUDGET = 2000


def load_rows(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def regret_curve(task: str, optimizer: str, seed: int, f_opt: float, budget: int = BUDGET) -> np.ndarray:
    rows = load_rows(ROOT / "results" / task / optimizer / f"seed_{seed}" / "trace.csv")
    y = np.array([float(r["y_best_so_far"]) for r in rows]) - f_opt
    y = np.maximum(y, 1e-12)
    if len(y) < budget:
        # Carry the incumbent forward: an early stop (convergence, or a
        # budget cap well short of `budget`) means no further progress was
        # recorded, not that the method would do better given more budget.
        y = np.concatenate([y, np.full(budget - len(y), y[-1])])
    return y[:budget]


def f_opt_of(task: str) -> float:
    meta = json.load(open(ROOT / "results" / task / "cma_es" / "seed_0" / "meta.json"))
    return float(meta["task_f_opt"])


def main():
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6 * len(PANELS), 5.2), facecolor="white")
    if len(PANELS) == 1:
        axes = [axes]

    for ax, panel in zip(axes, PANELS):
        f_opt = f_opt_of(panel["task"])
        grad = regret_curve(panel["task"], panel["gradient"], 0, f_opt)
        evo_seeds = np.stack([regret_curve(panel["task"], panel["evolutionary"], s, f_opt) for s in range(5)])
        evo_mean = evo_seeds.mean(axis=0)
        evo_lo = evo_seeds.min(axis=0)
        evo_hi = evo_seeds.max(axis=0)

        x = np.arange(1, BUDGET + 1)
        ax.set_facecolor("white")
        ax.fill_between(x, evo_lo, evo_hi, color=EVO_COLOR, alpha=EVO_BAND_ALPHA, linewidth=0)
        ax.plot(x, evo_mean, color=EVO_COLOR, linewidth=2.6, label="Evolutionary Based")
        ax.plot(x, grad, color=GRADIENT_COLOR, linewidth=2.6, label="Gradient Based")

        ax.set_yscale("log")
        ax.set_title(panel["title"], fontsize=20, pad=14)
        ax.set_xlabel("Budget", fontsize=13)
        if ax is axes[0]:
            ax.set_ylabel("Regret", fontsize=13)
        ax.set_xlim(0, BUDGET)
        ax.tick_params(labelsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")
        ax.grid(False)

    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index("Gradient Based"), labels.index("Evolutionary Based")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=2, frameon=False, fontsize=13, bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_dir = ROOT.parent / "slide_figures" / "blackbox_functions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "regret_by_family_smooth_vs_mixedint.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
