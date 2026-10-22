"""Comparison plots: Claude agent run(s) vs. the DIONYSUS benchmark's own
precomputed BO baselines, for a lookup-table (fixed finite candidate pool)
molecular property task.

DIONYSUS (the-matter-lab/dionysus) baselines are precomputed over 30
restarts per (model, feature) combination, stored as
main/data/dionysus/<task>/bo_traces/bo_traces_<model>_<feature>.csv with one
row per evaluation: columns include 'trace' (running best, already
cummin/cummax'd per their own convention) and 'run' (restart index) and
'eval' (0-indexed evaluation number within that run). This script averages
'trace' across the 30 restarts at each eval index to get one mean curve per
baseline, exactly mirroring this project's SEED5 convention for its own
classical baselines elsewhere.

Usage
-----
    python scripts/plot_dionysus_vs_agents.py --task freesolv --f-opt -23.62 --title "FreeSolv" \\
        --agent claude_agent_run="Claude agent" \\
        --agent claude_agent_run_hptuning_tweak="Claude agent (hp-tuning tweak)"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
)
GREEN, VIOLET, RED, BROWN = "#008300", "#6250d6", "#e34948", "#8a5a2b"
MUTED_GRAY = "#898781"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.edgecolor": AXIS,
    "axes.labelcolor": TEXT_SECONDARY,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# (model, feature) -> (display label, color)
BASELINES = [
    ("random", "mfp", "Random", MUTED_GRAY),
    ("nn", "mfp", "1-NN heuristic", GREEN),
    ("gp", "mfp", "GP (fingerprint)", BLUE),
    ("gp", "mordred", "GP (mordred)", AQUA),
    ("bnn", "mfp", "BNN (fingerprint)", ORANGE),
    ("ngboost", "mfp", "NGBoost (fingerprint)", YELLOW),
    ("sngp", "mfp", "SNGP (fingerprint)", MAGENTA),
    ("gnngp", "graphnet", "GNN-GP (graph)", VIOLET),
]

AGENT_COLORS = [RED, BROWN, "#1b5e6e", "#9c4f96"]


def load_rows(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def phases_for(rows):
    phases, cur, start, prev_idx = [], None, None, None
    for r in rows:
        if r["optimizer"] != cur:
            if cur is not None:
                phases.append((cur, int(start), int(prev_idx)))
            cur, start = r["optimizer"], r["eval_idx"]
        prev_idx = r["eval_idx"]
    phases.append((cur, int(start), int(prev_idx)))
    return phases


PHASE_PALETTE = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED, BROWN, MUTED_GRAY]


def parse_agent_spec(spec: str):
    directory, _, label = spec.partition("=")
    return directory, (label or directory)


def baseline_trajectory(task_dir: Path, model: str, feature: str, f_opt: float, goal: str):
    path = task_dir / "bo_traces" / f"bo_traces_{model}_{feature}.csv"
    if not path.exists():
        return None
    rows = load_rows(path)
    by_run: dict[int, list[tuple[int, float]]] = {}
    for r in rows:
        by_run.setdefault(int(r["run"]), []).append((int(r["eval"]), float(r["trace"])))
    min_len = min(len(v) for v in by_run.values())
    curves = []
    for run_rows in by_run.values():
        run_rows.sort(key=lambda t: t[0])
        curves.append([v for _, v in run_rows[:min_len]])
    mean_curve = np.mean(np.array(curves), axis=0)
    evals = np.arange(1, min_len + 1)  # 1-indexed, matches our own trace.csv convention
    regret = (mean_curve - f_opt) if goal == "minimize" else (f_opt - mean_curve)
    return evals, regret


def agent_trajectory(task_dir: Path, agent_dir: str, f_opt: float, goal: str):
    rows = load_rows(task_dir / agent_dir / "trace.csv")
    y = np.array([float(r["y_best_so_far"]) for r in rows])
    eval_idx = np.array([int(r["eval_idx"]) for r in rows])
    regret = (y - f_opt) if goal == "minimize" else (f_opt - y)
    switches, cur = [], None
    for r in rows:
        if r["optimizer"] != cur:
            switches.append(int(r["eval_idx"]))
            cur = r["optimizer"]
    return eval_idx, regret, switches, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="dionysus task name, e.g. freesolv")
    p.add_argument("--f-opt", type=float, required=True)
    p.add_argument("--goal", default="minimize", choices=["minimize", "maximize"])
    p.add_argument("--title", required=True)
    p.add_argument("--agent", action="append", required=True, dest="agents",
                    help="repeatable: 'result_subdir=Display label'")
    args = p.parse_args()

    data_dir = ROOT / "data" / "dionysus" / args.task
    results_dir = ROOT / "results" / f"dionysus_{args.task}"
    figures_dir = ROOT / "figures" / f"dionysus_{args.task}"
    figures_dir.mkdir(parents=True, exist_ok=True)
    f_opt = args.f_opt
    goal = args.goal

    agents = [parse_agent_spec(s) for s in args.agents]

    baseline_traj = {}
    for model, feature, label, color in BASELINES:
        traj = baseline_trajectory(data_dir, model, feature, f_opt, goal)
        if traj is not None:
            baseline_traj[(model, feature)] = (label, color, *traj)

    agent_data = []
    for i, (adir, label) in enumerate(agents):
        eval_idx, regret, switches, rows = agent_trajectory(results_dir, adir, f_opt, goal)
        agent_data.append(dict(
            dir=adir, label=label, color=AGENT_COLORS[i % len(AGENT_COLORS)],
            eval_idx=eval_idx, regret=regret, switches=switches, rows=rows,
        ))

    # ==================== PLOT 1: counterfactual bar chart ====================
    final_regret = {key: v[3][-1] for key, v in baseline_traj.items()}
    for a in agent_data:
        final_regret[a["dir"]] = a["regret"][-1]

    order = [a["dir"] for a in agent_data] + list(baseline_traj.keys())
    labels = {a["dir"]: f"{a['label']} (used {len(a['rows'])})" for a in agent_data}
    labels.update({key: f"{v[0]} (mean of 30 restarts)" for key, v in baseline_traj.items()})
    colors = {a["dir"]: a["color"] for a in agent_data}
    colors.update({key: v[1] for key, v in baseline_traj.items()})

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(order) + 1.2))
    values = [final_regret[k] for k in order]
    bar_colors = [colors[k] for k in order]
    bar_labels = [labels[k] for k in order]
    y_pos = np.arange(len(order))
    ax.barh(y_pos, values, color=bar_colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Final regret (lower is better)")
    ax.set_xscale("log")
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.suptitle(f"Final regret after full budget -- {args.title}", fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figures_dir / "plot1_counterfactual_bar.png", dpi=150)
    plt.close(fig)

    # ==================== PLOT 2: regret vs evaluation ====================
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, (label, color, evals, regret) in baseline_traj.items():
        ax.plot(evals, np.maximum(regret, 1e-6), color=color, linewidth=1.6,
                 label=f"{label} (mean of 30)", alpha=0.85)
    for a in agent_data:
        ax.plot(a["eval_idx"], np.maximum(a["regret"], 1e-6), color=a["color"], linewidth=2.4,
                 label=a["label"])
        ax.scatter(a["switches"], np.maximum(a["regret"][[i - 1 for i in a["switches"]]], 1e-6),
                    color=a["color"], s=18, zorder=5)
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Regret (log scale)")
    ax.set_yscale("log")
    ax.grid(True, color=GRIDLINE, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"Regret vs. evaluation -- {args.title}", fontsize=13, x=0.02, ha="left")
    fig.text(0.01, 0.01, "Dots mark each agent's method-tag switches. DIONYSUS baselines are "
                          "mean-of-30-restart curves from their own precomputed traces.",
              fontsize=8, color=TEXT_MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(figures_dir / "plot2_regret_vs_eval.png", dpi=150)
    plt.close(fig)

    # ==================== PLOT 3: which method each agent was running ====================
    NARROW_THRESHOLD_EVALS = max(15, max(a["eval_idx"].max() for a in agent_data) // 15)
    n_lanes = len(agent_data)
    requested_budget = max(a["eval_idx"].max() for a in agent_data)
    fig, ax = plt.subplots(figsize=(11, 1.6 * n_lanes + 1.0), dpi=150)

    method_color: dict[str, str] = {}

    def color_for(label: str) -> str:
        if label not in method_color:
            method_color[label] = PHASE_PALETTE[len(method_color) % len(PHASE_PALETTE)]
        return method_color[label]

    for lane, a in enumerate(agent_data):
        y0 = n_lanes - 1 - lane
        phases = phases_for(a["rows"])
        for raw_label, start, end in phases:
            color = color_for(raw_label)
            width = end - start + 1
            ax.broken_barh([(start, width)], (y0 - 0.32, 0.64), facecolors=color,
                            edgecolors=SURFACE, linewidth=1.5, zorder=2)
            center = start + width / 2
            if width >= NARROW_THRESHOLD_EVALS:
                ax.text(center, y0, raw_label, ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="medium", zorder=3)
            else:
                label_y = y0 + 0.62
                ax.plot([center, center], [y0 + 0.32, label_y - 0.05], color=TEXT_MUTED, linewidth=1, zorder=2)
                ax.text(center, label_y, raw_label, ha="center", va="bottom", fontsize=8,
                        color=TEXT_PRIMARY, zorder=3)

    ax.set_xlim(0, requested_budget)
    ax.set_ylim(-0.85, n_lanes - 1 + 0.85)
    ax.set_yticks(list(range(n_lanes)))
    ax.set_yticklabels([a["label"] for a in reversed(agent_data)], fontsize=10, color=TEXT_PRIMARY)
    ax.set_xlabel("Evaluation", fontsize=10)
    ax.grid(True, which="major", axis="x", color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"Which method each agent was running, by evaluation ({args.title})",
                 fontsize=12, color=TEXT_PRIMARY, pad=14, loc="left")
    fig.tight_layout()
    fig.savefig(figures_dir / "plot3_agent_method_timeline.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {figures_dir / 'plot1_counterfactual_bar.png'}")
    print(f"Wrote {figures_dir / 'plot2_regret_vs_eval.png'}")
    print(f"Wrote {figures_dir / 'plot3_agent_method_timeline.png'}")


if __name__ == "__main__":
    main()
