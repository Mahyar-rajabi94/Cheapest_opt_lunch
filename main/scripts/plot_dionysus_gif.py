"""Animated regret-vs-budget GIF for a DIONYSUS task: the best agent variant
vs. the best standard (classical ML-surrogate) baseline, revealing the data
collection path evaluation by evaluation. Same visual style as the
agent_wins panels: white background, thick flat-colored lines, no
gridlines, spines only left/bottom, "Budget" x-axis, no in-frame legend
(a separate legend.png is written alongside the gif).

Usage
-----
    python scripts/plot_dionysus_gif.py --task freesolv --f-opt -23.62 \
        --agent claude_agent_run_hptuning_tweak --agent-label "Agent (hp-tuning tweak)" \
        --baseline-model gp --baseline-feature mordred --baseline-label "GP (mordred)"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from plot_dionysus_vs_agents import baseline_trajectory  # noqa: E402

AGENT_COLOR = "#C93A5E"
AGENT_LINEWIDTH = 4.5
BASELINE_COLOR = "#2761b4"
BASELINE_LINEWIDTH = 2.6


def load_rows(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def agent_regret_curve(task: str, agent_dir: str, f_opt: float):
    rows = load_rows(ROOT / "results" / f"dionysus_{task}" / agent_dir / "trace.csv")
    y = np.array([float(r["y_best_so_far"]) for r in rows]) - f_opt
    eval_idx = np.array([int(r["eval_idx"]) for r in rows])
    return eval_idx, np.maximum(y, 1e-6)


def agent_method_segments(task: str, agent_dir: str):
    """[(start_eval, method_name), ...] in order, one entry per method switch."""
    rows = load_rows(ROOT / "results" / f"dionysus_{task}" / agent_dir / "trace.csv")
    segments, cur = [], None
    for r in rows:
        if r["optimizer"] != cur:
            segments.append((int(r["eval_idx"]), r["optimizer"]))
            cur = r["optimizer"]
    return segments


def friendly_method_name(name: str) -> str:
    return name.replace("_", " ").capitalize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="dionysus task name, e.g. freesolv")
    p.add_argument("--f-opt", type=float, required=True)
    p.add_argument("--agent", required=True, help="agent result subdir, e.g. claude_agent_run_hptuning_tweak")
    p.add_argument("--agent-label", default="Agent")
    p.add_argument("--baseline-model", required=True)
    p.add_argument("--baseline-feature", required=True)
    p.add_argument("--baseline-label", default="Baseline")
    p.add_argument("--step", type=int, default=1, help="evaluations advanced per frame")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--max-budget", type=int, default=None, help="cut the animation off at this evaluation")
    p.add_argument("--outdir", default=str(ROOT.parent / "slide_figures" / "dionysus"))
    args = p.parse_args()

    data_dir = ROOT / "data" / "dionysus" / args.task
    agent_x, agent_y = agent_regret_curve(args.task, args.agent, args.f_opt)
    base_x, base_y = baseline_trajectory(data_dir, args.baseline_model, args.baseline_feature, args.f_opt, "minimize")
    base_y = np.maximum(base_y, 1e-6)
    method_segments = agent_method_segments(args.task, args.agent)

    n_frames_total = max(agent_x[-1], base_x[-1])
    if args.max_budget is not None:
        n_frames_total = min(n_frames_total, args.max_budget)
        a_keep = agent_x <= n_frames_total
        b_keep = base_x <= n_frames_total
        agent_x, agent_y = agent_x[a_keep], agent_y[a_keep]
        base_x, base_y = base_x[b_keep], base_y[b_keep]
        method_segments = [(s, m) for s, m in method_segments if s <= n_frames_total]
    frame_budgets = list(range(1, n_frames_total + 1, args.step))
    if frame_budgets[-1] != n_frames_total:
        frame_budgets.append(n_frames_total)

    fig, ax = plt.subplots(figsize=(7, 5.2), facecolor="white")
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
    fig.tight_layout()

    (base_line,) = ax.plot([], [], color=BASELINE_COLOR, linewidth=BASELINE_LINEWIDTH)
    (agent_line,) = ax.plot([], [], color=AGENT_COLOR, linewidth=AGENT_LINEWIDTH)
    (base_dot,) = ax.plot([], [], "o", color=BASELINE_COLOR, markersize=7)
    (agent_dot,) = ax.plot([], [], "o", color=AGENT_COLOR, markersize=8)

    # Box listing every method the agent used (up to the cutoff); whichever
    # one is active at the current budget is highlighted.
    unique_methods = []
    for _, m in method_segments:
        if m not in unique_methods:
            unique_methods.append(m)
    n_methods = len(unique_methods)
    line_h = 0.065
    top = 0.97
    box_pad = 0.025
    box_x0, box_x1 = 0.46, 0.99
    box_y0 = top - n_methods * line_h - box_pad
    box_y1 = top + box_pad
    ax.add_patch(plt.Rectangle((box_x0, box_y0), box_x1 - box_x0, box_y1 - box_y0,
                                transform=ax.transAxes, facecolor="white", edgecolor="#cccccc",
                                linewidth=1.2, zorder=5))
    method_texts = []
    for i, m in enumerate(unique_methods):
        t = ax.text(box_x1 - 0.02, top - i * line_h, friendly_method_name(m), transform=ax.transAxes,
                    ha="right", va="top", fontsize=9.5, color="#999999", fontweight="normal", zorder=6)
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
        for i, t in enumerate(method_texts):
            if i == active_idx:
                t.set_color(AGENT_COLOR)
                t.set_fontweight("bold")
            else:
                t.set_color("#999999")
                t.set_fontweight("normal")
        return base_line, agent_line, base_dot, agent_dot, *method_texts

    anim = FuncAnimation(fig, update, frames=frame_budgets, blit=False)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{args.task}_regret_vs_budget.gif"
    anim.save(out_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Standalone legend, same convention as the agent_wins panels.
    legend_fig = plt.figure(figsize=(3.0, 0.9), facecolor="white")
    handles = [
        plt.Line2D([0], [0], color=BASELINE_COLOR, linewidth=BASELINE_LINEWIDTH),
        plt.Line2D([0], [0], color=AGENT_COLOR, linewidth=AGENT_LINEWIDTH),
    ]
    legend_fig.legend(handles, [args.baseline_label, args.agent_label], loc="center", frameon=False,
                       fontsize=12, labelspacing=0.3, handlelength=1.8, handletextpad=0.6,
                       borderpad=0, borderaxespad=0)
    legend_path = outdir / f"{args.task}_legend.png"
    legend_fig.savefig(legend_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(legend_fig)
    print(f"Wrote {legend_path}")


if __name__ == "__main__":
    main()
