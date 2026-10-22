"""No-free-lunch crossover figure: the same two baselines on two tasks, each
winning one and failing the other.

  rosenbrock_d10_i1      smooth curved valley -> finite-difference L-BFGS-B
                         reaches ~1e-7 regret; every CMA-ES seed stalls at ~1e0.
  bbob_mixint_f8_d10_i1  the same Rosenbrock valley with 8 of 10 inputs on an
                         integer lattice -> L-BFGS-B stops after 121
                         evaluations at ~1e3; CMA-ES reaches ~1e-6 (geometric
                         mean of 10 seeds).

Legend: "Gradient Based" is L-BFGS-B (deterministic, one seed), "Evolutionary
Based" is CMA-ES (10 seeds). Style follows FIGURES.md via figure_style.py.

Usage (from main/):
    python scripts/make_crossover_plot.py    # writes results_linux/crossover_rosenbrock_vs_mixint.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_comparison_plots import MAIN_ROOT, load_rows, regret_trajectory  # noqa: E402
import figure_style as fs  # noqa: E402  (after make_comparison_plots: its rcParams must not override these)
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = MAIN_ROOT / "results_linux"
OUT = RESULTS / "crossover_rosenbrock_vs_mixint.png"
METHODS = [("lbfgsb", "Gradient Based"), ("cma_es", "Evolutionary Based")]
PANELS = [  # (task, title)
    ("rosenbrock_d10_i1", "Smooth"),
    ("bbob_mixint_f8_d10_i1", "Mixed Integer"),
]


def seed_trajectories(task, method):
    metas = sorted((RESULTS / task / method).glob("seed_*/meta.json"))
    meta = json.loads(metas[0].read_text())
    return [regret_trajectory(load_rows(m.parent / "trace.csv"), meta["task_f_opt"]) for m in metas], meta["requested_budget"]


def main():
    fig, axes = plt.subplots(1, len(PANELS), figsize=fs.FIGSIZE_1x2)
    for i, (ax, (task, title)) in enumerate(zip(axes, PANELS)):
        for (method, label), color in zip(METHODS, fs.SERIES_COLORS):
            trajs, budget = seed_trajectories(task, method)
            fs.plot_series(ax, trajs, budget, color, label)
        fs.style_axes(ax, title, budget, ylabel=i == 0)
    fs.finish(fig, axes, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
