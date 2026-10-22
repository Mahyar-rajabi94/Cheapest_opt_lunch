"""Animated regret-vs-budget GIF for a MOF task: the agent vs. the best
standard optimizer, revealing the data collection path evaluation by
evaluation. Same visual style as the DIONYSUS gif (plot_dionysus_gif.py):
white background, thick flat-colored lines, no gridlines, spines only
left/bottom, "Budget" x-axis, a side panel (outside the plot) with one
individually-boxed entry per method the agent used, the active one
highlighted.

Data sources
------------
- Regret-per-evaluation for every method (already computed, wide format):
  slide_figures/MOF_process/plot2_regret_vs_eval_data.csv
- The agent's own per-evaluation method tag (not in the file above):
  main/data/mof/task_060_trace.csv (pulled from origin/mof's
  task_060_sandbox_openai/results/agent_task_060/trace.csv)

Usage
-----
    python scripts/plot_mof_gif.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

AGENT_COLOR = "#C93A5E"
AGENT_LINEWIDTH = 4.5
BASELINE_COLOR = "#2761b4"
BASELINE_LINEWIDTH = 2.6


def load_rows(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def column_curve(rows, column: str):
    """Evaluation index + value for a column, dropping blank cells (a
    baseline shorter than the full run leaves the rest of its column
    blank) and carrying the last real value forward to the full length."""
    evals, vals = [], []
    for r in rows:
        if r[column].strip() == "":
            continue
        evals.append(int(r["evaluation"]))
        vals.append(float(r[column]))
    evals = np.array(evals)
    vals = np.maximum(np.array(vals), 1e-6)
    return evals, vals


def agent_method_segments(trace_path: Path):
    rows = load_rows(trace_path)
    segments, cur = [], None
    for r in rows:
        if r["optimizer"] != cur:
            segments.append((int(r["eval_idx"]), r["optimizer"]))
            cur = r["optimizer"]
    return segments


_FRIENDLY_OVERRIDES = {
    "lbfgsb": "L-BFGS-B",
    "cma_es": "CMA-ES",
    "sobol_search": "Sobol search",
    "bobyqa": "BOBYQA",
    "gp_bo": "GP-BO",
    "nelder_mead": "Nelder-Mead",
    "random_search": "Random search",
    "differential_evolution": "Differential evolution",
    "turbo": "TuRBO",
}


def friendly_method_name(name: str) -> str:
    return _FRIENDLY_OVERRIDES.get(name, name.replace("_", " ").capitalize())


def main():
    data_path = ROOT.parent / "slide_figures" / "MOF_process" / "plot2_regret_vs_eval_data.csv"
    trace_path = ROOT / "data" / "mof" / "task_060_trace.csv"
    agent_label = "Agent"
    baseline_label = "SOTA (feasibility BO, mean of 10 seeds)"
    max_budget = 100
    step = 1
    fps = 8
    outdir = ROOT.parent / "slide_figures" / "MOF_process"

    rows = load_rows(data_path)
    agent_x, agent_y = column_curve(rows, "Agent regret")
    base_x, base_y = column_curve(rows, "SOTA (feasibility BO, mean of 10 seeds) regret")
    method_segments = agent_method_segments(trace_path)

    n_frames_total = max(agent_x[-1], base_x[-1])
    n_frames_total = min(n_frames_total, max_budget)
    a_keep = agent_x <= n_frames_total
    b_keep = base_x <= n_frames_total
    agent_x, agent_y = agent_x[a_keep], agent_y[a_keep]
    base_x, base_y = base_x[b_keep], base_y[b_keep]
    method_segments = [(s, m) for s, m in method_segments if s <= n_frames_total]
    frame_budgets = list(range(1, n_frames_total + 1, step))
    if frame_budgets[-1] != n_frames_total:
        frame_budgets.append(n_frames_total)

    unique_methods = []
    for _, m in method_segments:
        if m not in unique_methods:
            unique_methods.append(m)
    n_methods = len(unique_methods)

    fig = plt.figure(figsize=(10.5, 5.2), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.2], wspace=0.06)
    ax = fig.add_subplot(gs[0])
    side_ax = fig.add_subplot(gs[1])
    side_ax.set_facecolor("white")
    side_ax.axis("off")
    side_ax.set_xlim(0, 1)
    side_ax.set_ylim(0, 1)

    ax.set_facecolor("white")
    ax.set_yscale("log")
    ax.set_xlim(0, n_frames_total + n_frames_total * 0.02)
    y_all = np.concatenate([agent_y, base_y])
    ax.set_ylim(y_all.min() * 0.5, y_all.max() * 2)
    ax.set_xlabel("Budget", fontsize=13)
    ax.set_ylabel("Regret", fontsize=13)
    ax.tick_params(labelsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.grid(False)

    (base_line,) = ax.plot([], [], color=BASELINE_COLOR, linewidth=BASELINE_LINEWIDTH)
    (agent_line,) = ax.plot([], [], color=AGENT_COLOR, linewidth=AGENT_LINEWIDTH)
    (base_dot,) = ax.plot([], [], "o", color=BASELINE_COLOR, markersize=7)
    (agent_dot,) = ax.plot([], [], "o", color=AGENT_COLOR, markersize=8)

    gap = 0.05
    box_h = (1.0 - gap * (n_methods + 1)) / n_methods
    method_boxes, method_texts = [], []
    for i, m in enumerate(unique_methods):
        y0 = 1.0 - gap * (i + 1) - box_h * (i + 1)
        box = plt.Rectangle((0.04, y0), 0.92, box_h, transform=side_ax.transAxes,
                             facecolor="white", edgecolor="#cccccc", linewidth=1.4, zorder=5)
        side_ax.add_patch(box)
        t = side_ax.text(0.5, y0 + box_h / 2, friendly_method_name(m), transform=side_ax.transAxes,
                          ha="center", va="center", fontsize=13, color="#999999", fontweight="normal",
                          wrap=True, zorder=6)
        method_boxes.append(box)
        method_texts.append(t)

    def active_method_index(budget):
        idx = 0
        for i, (start, _) in enumerate(method_segments):
            if start <= budget:
                idx = i
        return unique_methods.index(method_segments[idx][1])

    def update(budget):
        a_mask = agent_x <= budget
        b_mask = base_x <= budget
        agent_line.set_data(agent_x[a_mask], agent_y[a_mask])
        base_line.set_data(base_x[b_mask], base_y[b_mask])
        if a_mask.any():
            agent_dot.set_data([agent_x[a_mask][-1]], [agent_y[a_mask][-1]])
        if b_mask.any():
            base_dot.set_data([base_x[b_mask][-1]], [base_y[b_mask][-1]])
        active_idx = active_method_index(budget)
        for i, (box, t) in enumerate(zip(method_boxes, method_texts)):
            if i == active_idx:
                t.set_color(AGENT_COLOR)
                t.set_fontweight("bold")
                box.set_edgecolor(AGENT_COLOR)
                box.set_facecolor("#fbe9ee")
                box.set_linewidth(2.2)
            else:
                t.set_color("#999999")
                t.set_fontweight("normal")
                box.set_edgecolor("#cccccc")
                box.set_facecolor("white")
                box.set_linewidth(1.4)
        return base_line, agent_line, base_dot, agent_dot, *method_texts, *method_boxes

    anim = FuncAnimation(fig, update, frames=frame_budgets, blit=False)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "co2_uptake_regret_vs_budget.gif"
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Wrote {out_path}")

    legend_fig = plt.figure(figsize=(3.4, 0.9), facecolor="white")
    handles = [
        plt.Line2D([0], [0], color=BASELINE_COLOR, linewidth=BASELINE_LINEWIDTH),
        plt.Line2D([0], [0], color=AGENT_COLOR, linewidth=AGENT_LINEWIDTH),
    ]
    legend_fig.legend(handles, [baseline_label, agent_label], loc="center", frameon=False,
                       fontsize=12, labelspacing=0.3, handlelength=1.8, handletextpad=0.6,
                       borderpad=0, borderaxespad=0)
    legend_path = outdir / "co2_uptake_legend.png"
    legend_fig.savefig(legend_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(legend_fig)
    print(f"Wrote {legend_path}")


if __name__ == "__main__":
    main()
