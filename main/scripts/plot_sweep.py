"""Per-task figures for a run_sweep.py output directory.

Writes into each <sweep_dir>/<task_name>/:
    plot1_final_regret_bar.png      -- final regret: agents and every baseline
    plot2_regret_vs_eval.png        -- regret-vs-evaluation, baseline means
                                       plus agents (dots at phase switches)
    plot3_agent_method_timeline.png -- which method each agent ran, by
                                       evaluation (only if agent runs exist)

Baselines are the mean over seeds. Agent traces come from the sandbox
results folders listed in AGENTS. Palette, styles and names follow
make_comparison_plots.py so the figures read as one set.

Usage
-----
    python scripts/plot_sweep.py results_sweep_b2000
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREEN, VIOLET, RED = "#008300", "#6250d6", "#e34948"
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update(
    {
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
    }
)

# Paired hue = same family (solid vs. dashed), as in make_comparison_plots.py.
BASELINE_STYLE = {
    "random_search": dict(color=AQUA, linestyle="-"),
    "sobol": dict(color=AQUA, linestyle="--"),
    "cma_es": dict(color=YELLOW, linestyle="-"),
    "de": dict(color=YELLOW, linestyle="--"),
    "turbo": dict(color=MAGENTA, linestyle="-"),
    "gp_bo": dict(color=MAGENTA, linestyle="--"),
    "bobyqa": dict(color=GREEN, linestyle="-"),
    "nelder_mead": dict(color=VIOLET, linestyle="-"),
    "lbfgsb": dict(color=RED, linestyle="-"),
}
NAMES = {
    "random_search": "Random search",
    "sobol": "Sobol",
    "cma_es": "CMA-ES",
    "de": "Differential evolution",
    "turbo": "TuRBO",
    "gp_bo": "GP-BO",
    "bobyqa": "BOBYQA",
    "nelder_mead": "Nelder-Mead",
    "lbfgsb": "L-BFGS-B",
}

# Task results folder -> agent traces, relative to the repo root.
AGENTS = {
    "rosenbrock_d10_i1": {
        "Claude agent": "task_001_sandbox_claud/results/agent_task_001/trace.csv",
        "OpenAI agent": "task_001_sandbox_openai/results/agent_task_001/trace.csv",
    },
    "rastrigin_d10_i1": {"Agent": "task_002_sandbox/results/agent_task_002/trace.csv"},
    "bent_cigar_d10_i1": {"Agent": "task_003_sandbox/results/agent_task_003/trace.csv"},
    "sphere_d10_i1": {"Agent": "task_004_sandbox/results/agent_task_004/trace.csv"},
}
AGENT_COLORS = [BLUE, ORANGE]
FLOOR = 1e-12  # regret 0 is drawn at this floor on log axes
NARROW_THRESHOLD_EVALS = 180  # timeline segments narrower than this get a label above


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def best_so_far(rows):
    return np.array([float(r["y_best_so_far"]) for r in rows])


def canonical_method(raw_label):
    # Longest name first, so "de" cannot match inside "nelder_mead".
    raw = raw_label.lower()
    for key in sorted(BASELINE_STYLE, key=len, reverse=True):
        if key in raw:
            return key
    return raw_label


def load_baselines(task_dir, f_opt):
    means, budgets, n_seeds = {}, {}, {}
    for name in BASELINE_STYLE:
        metas = sorted((task_dir / name).glob("seed_*/meta.json"))
        trajs = [best_so_far(load_rows(m.parent / "trace.csv")) - f_opt for m in metas]
        n = min(len(t) for t in trajs)  # early-stopping seeds: mean over the common prefix
        means[name] = np.stack([t[:n] for t in trajs]).mean(axis=0)
        budgets[name] = json.loads(metas[0].read_text())["requested_budget"]
        n_seeds[name] = len(metas)
    return means, budgets, n_seeds


def load_agents(task_name, f_opt):
    agents = {}
    for label, rel in AGENTS.get(task_name, {}).items():
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        rows = load_rows(path)
        phases, cur = [], None
        for r in rows:
            if r["optimizer"] != cur:
                phases.append([r["optimizer"], int(r["eval_idx"]), int(r["eval_idx"])])
                cur = r["optimizer"]
            phases[-1][2] = int(r["eval_idx"])
        agents[label] = (best_so_far(rows) - f_opt, phases)
    return agents


def capped_note(budgets):
    capped = sorted(n for n in budgets if budgets[n] < max(budgets.values()))
    return "Capped: " + ", ".join(f"{NAMES[n]} at {budgets[n]}" for n in capped) + ". " if capped else ""


def plot_final_bar(task_dir, baselines, budgets, n_seeds, agents):
    entries = [
        (f"{label} (budget {len(regret)})", regret[-1], AGENT_COLORS[i])
        for i, (label, (regret, _)) in enumerate(agents.items())
    ]
    for name in sorted(baselines, key=lambda k: baselines[k][-1]):
        seeds = f", mean of {n_seeds[name]} seeds" if n_seeds[name] > 1 else ""
        entries.append(
            (f"{NAMES[name]} (budget {budgets[name]}{seeds})", baselines[name][-1], BASELINE_STYLE[name]["color"])
        )

    y_pos = np.arange(len(entries))[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(entries) + 1.6), dpi=150)
    ax.barh(y_pos, [max(v, FLOOR) for _, v, _ in entries], height=0.55, color=[c for _, _, c in entries], zorder=3)
    ax.set_xscale("log")
    ax.grid(True, which="major", axis="x", color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([label for label, _, _ in entries], color=TEXT_PRIMARY, fontsize=9.5)
    ax.set_xlabel("Final regret (log scale)", fontsize=10)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    for yi, (_, v, _) in zip(y_pos, entries):
        ax.text(
            max(v, FLOOR) * 1.3, yi, f"{v:.2e}" if v < 0.01 else f"{v:.2f}",
            va="center", ha="left", fontsize=9, color=TEXT_SECONDARY,
        )
    subject = "agents vs. every baseline" if agents else "every baseline"
    ax.set_title(f"{task_dir.name}: final regret, {subject}", fontsize=12, color=TEXT_PRIMARY, pad=14, loc="left")
    note = capped_note(budgets)
    if note:
        fig.text(0.0, -0.01, note + "Bars compare final values at each method's own budget.",
                 fontsize=8, color=TEXT_MUTED)
    fig.tight_layout()
    fig.savefig(task_dir / "plot1_final_regret_bar.png", bbox_inches="tight")
    plt.close(fig)


def plot_regret_vs_eval(task_dir, baselines, budgets, agents):
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    for name, y in baselines.items():
        ax.plot(np.arange(1, len(y) + 1), np.clip(y, FLOOR, None), linewidth=1.6, alpha=0.9,
                label=NAMES[name], zorder=2, **BASELINE_STYLE[name])
    for i, (label, (y, phases)) in enumerate(agents.items()):
        x, yc = np.arange(1, len(y) + 1), np.clip(y, FLOOR, None)
        ax.plot(x, yc, color=AGENT_COLORS[i], linewidth=2.6, label=label, zorder=3)
        for _, start, _ in phases:
            if 0 <= start - 1 < len(yc):
                ax.plot(x[start - 1], yc[start - 1], marker="o", markersize=7, color=AGENT_COLORS[i],
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("Evaluation", fontsize=10)
    ax.set_ylabel("Regret (log scale)", fontsize=10)
    ax.grid(True, which="major", axis="both", color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    suffix = " -- agents against every baseline" if agents else ""
    ax.set_title(f"{task_dir.name}: regret vs. evaluation{suffix}", fontsize=12, color=TEXT_PRIMARY, pad=14, loc="left")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9.5, labelcolor=TEXT_PRIMARY,
              title="Paired hue = same family\n(solid vs. dashed)", title_fontsize=8.5)
    note = "Baselines: mean over seeds. " + capped_note(budgets)
    if agents:
        note += "Dots mark each agent's phase switches."
    fig.text(0.0, -0.03, note, fontsize=8, color=TEXT_MUTED)
    fig.tight_layout()
    fig.savefig(task_dir / "plot2_regret_vs_eval.png", bbox_inches="tight")
    plt.close(fig)


def plot_agent_timeline(task_dir, budgets, agents):
    fig, ax = plt.subplots(figsize=(11, 1.0 + 1.1 * len(agents)), dpi=150)
    lanes = list(agents)[::-1]
    for y0, label in enumerate(lanes):
        for raw, start, end in agents[label][1]:
            canon = canonical_method(raw)
            color = BASELINE_STYLE.get(canon, {"color": TEXT_MUTED})["color"]
            display = NAMES.get(canon, raw)
            if "restart" in raw.lower():  # color already names the method
                display = "restart " + raw.rsplit("_", 1)[-1]
            width = end - start + 1
            ax.broken_barh([(start, width)], (y0 - 0.32, 0.64), facecolors=color,
                           edgecolors=SURFACE, linewidth=1.5, zorder=2)
            center = start + width / 2
            if width >= NARROW_THRESHOLD_EVALS:
                ax.text(center, y0, display, ha="center", va="center", fontsize=9,
                        color="white", fontweight="medium", zorder=3)
            else:
                ax.plot([center, center], [y0 + 0.32, y0 + 0.5], color=TEXT_MUTED, linewidth=1, zorder=2)
                ax.text(center, y0 + 0.52, display, ha="center", va="bottom", fontsize=8.5,
                        color=TEXT_PRIMARY, zorder=3)
    ax.set_xlim(0, max(budgets.values()))
    ax.set_ylim(-0.6, len(lanes) - 0.1)
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes, fontsize=10, color=TEXT_PRIMARY)
    ax.set_xlabel("Evaluation", fontsize=10)
    ax.grid(True, which="major", axis="x", color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{task_dir.name}: which method each agent was running, by evaluation",
                 fontsize=12, color=TEXT_PRIMARY, pad=14, loc="left")
    fig.text(0.0, -0.02, "Segment color matches that method's line in plot2_regret_vs_eval.png. "
             "Narrow segments are labeled above with a leader line.", fontsize=8, color=TEXT_MUTED)
    fig.tight_layout()
    fig.savefig(task_dir / "plot3_agent_method_timeline.png", bbox_inches="tight")
    plt.close(fig)


def plot_sweep(sweep_dir):
    for task_dir in sorted(p for p in Path(sweep_dir).iterdir() if (p / "cma_es").is_dir()):
        f_opt = json.loads(next(task_dir.glob("*/seed_0/meta.json")).read_text())["task_f_opt"]
        baselines, budgets, n_seeds = load_baselines(task_dir, f_opt)
        agents = load_agents(task_dir.name, f_opt)
        plot_final_bar(task_dir, baselines, budgets, n_seeds, agents)
        plot_regret_vs_eval(task_dir, baselines, budgets, agents)
        if agents:
            plot_agent_timeline(task_dir, budgets, agents)
        print(f"{task_dir.name}: agents={list(agents)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dir", help="run_sweep.py --outdir, e.g. results_sweep_b2000")
    plot_sweep(p.parse_args().sweep_dir)
