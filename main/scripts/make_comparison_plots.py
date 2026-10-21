"""Comparison plots: agent runs vs. the classical baselines, per task.

Produces three figures in each <results>/<task>/ (default results_linux/):
  plot1_counterfactual_bar.png  -- final regret: the agents vs. each
                                   single method they cascaded through
                                   (every baseline if the task has no
                                   agent run yet)
  plot2_regret_vs_eval.png      -- regret-vs-evaluation trajectories, all
                                   baselines (mean over seeds) plus the
                                   agents (highlighted, with markers at
                                   each phase switch)
  plot3_agent_method_timeline.png -- which method each agent was running,
                                   by evaluation, with the deviations it
                                   logged in changelog.md (only for tasks
                                   with an agent run)

f_opt, seeds and budgets are read from the baseline meta.json files; agent
traces come from AGENT_RUNS.

Constrained tasks: regret is undefined until the first feasible point, so
the seed mean skips seeds without one yet, and plot 1 labels how many seeds
found a feasible point ("no feasible point" if none did).

Colors are the dataviz skill's documented default categorical palette
(docs/.../dataviz skill, references/palette.md), used in its published fixed
order. The validator script (validate_palette.js) could not be run directly
in this environment (no Node.js installed); the slot order used here is a
prefix of the full 8-slot sequence the palette doc already publishes as
passing every adjacent-pair gate (bars, lines) in light mode, which is the
context both charts use it in.

Usage (from main/):
    python scripts/make_comparison_plots.py                  # every task
    python scripts/make_comparison_plots.py --task rastrigin_d10_i1
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAIN_ROOT = Path(__file__).resolve().parent.parent
SANDBOX_ROOT = MAIN_ROOT.parent

# Task results folder -> agent label -> sandbox results folder (trace.csv,
# changelog.md). Missing folders are skipped, so a task can be listed
# before its agent run exists.
AGENT_RUNS = {
    "rosenbrock_d10_i1": {
        "Claude agent": "task_001_sandbox_claud/results/agent_task_001",
        "OpenAI agent": "task_001_sandbox_openai/results/agent_task_001",
    },
    "rastrigin_d10_i1": {"Agent": "task_002_sandbox/results/agent_task_002"},
    "bent_cigar_d10_i1": {"Agent": "task_003_sandbox/results/agent_task_003"},
    "sphere_d10_i1": {"Agent": "task_004_sandbox/results/agent_task_004"},
}

# Validated default categorical palette, fixed order (see module docstring).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
)
GREEN, VIOLET, RED = "#008300", "#6250d6", "#e34948"
MUTED_GRAY = "#898781"
STATUS_WARNING = "#fab219"  # reserved status color; used only for the logged-deviation marker, always paired with a label
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

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

# 11 series > the 8 safe categorical hues, so baselines use composite
# encoding (hue + linestyle) past that limit, grouped by family: two
# members sharing a hue are distinguished by solid vs. dashed. The three
# standalone local refiners each get their own hue. Agents keep the same
# blue/orange identity in every plot (color follows the entity across charts).
baseline_style = {
    "random_search":   dict(color=AQUA,    linestyle="-"),
    "sobol":           dict(color=AQUA,    linestyle="--"),
    "cma_es":          dict(color=YELLOW,  linestyle="-"),
    "de":              dict(color=YELLOW,  linestyle="--"),
    "turbo":           dict(color=MAGENTA, linestyle="-"),
    "gp_bo":           dict(color=MAGENTA, linestyle="--"),
    "bobyqa":          dict(color=GREEN,   linestyle="-"),
    "nelder_mead":     dict(color=VIOLET,  linestyle="-"),
    "lbfgsb":          dict(color=RED,     linestyle="-"),
}
baseline_display_names = {
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
AGENT_COLORS = [BLUE, ORANGE]
FLOOR = 1e-12  # regret 0 is drawn at this floor on the log axes

# Longest-name-first, so a short name (e.g. "de") can't falsely match
# inside a longer one that happens to contain it as a substring (e.g.
# "de" is literally inside "nelder_mead").
CANONICAL_BY_LENGTH = sorted(baseline_style, key=len, reverse=True)


def canonical_method(raw_label: str) -> str:
    raw = raw_label.lower()
    for key in CANONICAL_BY_LENGTH:
        if key in raw:
            return key
    return raw_label  # unrecognized label: show as-is, no established color


def load_rows(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def regret_trajectory(rows, f_opt):
    # Blank until the first feasible point (constrained tasks): regret is undefined there.
    y = np.array([float(r["y_best_so_far"] or "nan") for r in rows])
    return y - f_opt


def phases_for(rows):
    phases, cur, start = [], None, None
    for r in rows:
        if r["optimizer"] != cur:
            if cur is not None:
                phases.append((cur, int(start), int(prev_idx)))
            cur, start = r["optimizer"], r["eval_idx"]
        prev_idx = r["eval_idx"]
    phases.append((cur, int(start), int(prev_idx)))
    return phases


def changelog_evals(path: Path):
    """Evaluation numbers of the logged roadmap deviations ("## Evaluation N")."""
    if not path.exists():
        return []
    return [int(m) for m in re.findall(r"^## Evaluation (\d+)", path.read_text(), re.MULTILINE)]


def load_task(task_dir: Path):
    metas = {
        name: sorted((task_dir / name).glob("seed_*/meta.json"))
        for name in baseline_style
        if (task_dir / name).is_dir()
    }
    f_opt = json.loads(next(iter(metas.values()))[0].read_text())["task_f_opt"]

    # ---- baseline trajectories (mean across seeds where there are several) ----
    baselines = {}  # name -> dict(x, regret, n_feasible, budget, n_seeds)
    for name, paths in metas.items():
        seed_ys = [regret_trajectory(load_rows(p.parent / "trace.csv"), f_opt) for p in paths]
        min_len = min(len(a) for a in seed_ys)
        stacked = np.stack([a[:min_len] for a in seed_ys])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column: no seed feasible yet
            mean_regret = np.nanmean(stacked, axis=0)  # seeds without a feasible point yet are skipped
        baselines[name] = dict(
            x=np.arange(1, min_len + 1),
            regret=mean_regret,
            n_feasible=int(np.isfinite(stacked[:, -1]).sum()),
            budget=json.loads(paths[0].read_text())["requested_budget"],
            n_seeds=len(paths),
        )

    # ---- agent trajectories + phases + logged deviations ----
    agents = {}  # label -> dict(x, regret, phases, deviations, color)
    for label, rel in AGENT_RUNS.get(task_dir.name, {}).items():
        run_dir = SANDBOX_ROOT / rel
        if not (run_dir / "trace.csv").exists():
            continue
        rows = load_rows(run_dir / "trace.csv")
        agents[label] = dict(
            x=np.array([int(r["eval_idx"]) for r in rows]),
            regret=regret_trajectory(rows, f_opt),
            phases=phases_for(rows),
            deviations=changelog_evals(run_dir / "changelog.md"),
            color=AGENT_COLORS[len(agents)],
        )
    return baselines, agents


def capped_note(baselines):
    full = max(b["budget"] for b in baselines.values())
    capped = [n for n, b in baselines.items() if b["budget"] < full]
    if not capped:
        return ""
    return "Capped baselines (see BUDGET_CAPS in scripts/run_sweep.py): " + ", ".join(
        f"{baseline_display_names[n]} at {baselines[n]['budget']}" for n in capped
    ) + "."


# ==================== PLOT 1: counterfactual bar chart ====================
def plot_counterfactual_bar(task_dir, baselines, agents):
    used = {canonical_method(p[0]) for a in agents.values() for p in a["phases"]}
    names = [n for n in baselines if n in used] if agents else list(baselines)
    names.sort(key=lambda n: (np.isnan(baselines[n]["regret"][-1]), baselines[n]["regret"][-1]))

    entries = [(f"{label} (budget {len(a['x'])})", a["regret"][-1], a["color"]) for label, a in agents.items()]
    for n in names:
        b = baselines[n]
        seeds = f", mean of {b['n_seeds']} seeds" if b["n_seeds"] > 1 else ""
        if b["n_feasible"] < b["n_seeds"]:
            seeds += f", {b['n_feasible']}/{b['n_seeds']} found a feasible point"
        alone = " alone" if agents else ""
        entries.append(
            (f"{baseline_display_names[n]}{alone} (budget {b['budget']}{seeds})", b["regret"][-1],
             baseline_style[n]["color"])
        )

    values = [FLOOR if np.isnan(v) else max(v, FLOOR) for _, v, _ in entries]  # floor only for the log axis
    y_pos = np.arange(len(entries))[::-1]

    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(entries) + 1.6), dpi=150)
    ax.barh(y_pos, values, height=0.5, color=[c for _, _, c in entries], zorder=3)
    ax.set_xscale("log")
    ax.grid(True, which="major", axis="x", color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([label for label, _, _ in entries], color=TEXT_PRIMARY, fontsize=10)
    ax.set_xlabel("Final regret (log scale)", fontsize=10)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    for yi, (_, v, _), plotted in zip(y_pos, entries, values):
        label = "no feasible point" if np.isnan(v) else f"{v:.2e}" if v < 0.01 else f"{v:.2f}"
        ax.text(plotted * 1.3, yi, label, va="center", ha="left", fontsize=9.5, color=TEXT_SECONDARY)

    title = (
        "The agents vs. the single-method counterfactuals they cascaded through"
        if agents
        else "Final regret of every baseline (no agent run yet)"
    )
    ax.set_title(f"{task_dir.name}: {title}", fontsize=12, color=TEXT_PRIMARY, pad=14, loc="left")
    note = capped_note({n: baselines[n] for n in names})
    if note:
        fig.text(0.0, -0.01, note, fontsize=8, color=TEXT_MUTED)

    fig.tight_layout()
    fig.savefig(task_dir / "plot1_counterfactual_bar.png", bbox_inches="tight")
    plt.close(fig)


# ==================== PLOT 2: regret vs. evaluation ====================
def plot_regret_vs_eval(task_dir, baselines, agents):
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)

    for name, b in baselines.items():
        ax.plot(
            b["x"],
            np.clip(b["regret"], FLOOR, None),
            linewidth=1.6,
            alpha=0.9,
            label=baseline_display_names[name],
            zorder=2,
            **baseline_style[name],
        )

    for label, a in agents.items():
        x, y_c = a["x"], np.clip(a["regret"], FLOOR, None)
        ax.plot(x, y_c, color=a["color"], linewidth=2.6, label=label, zorder=3)
        for _, start, _ in a["phases"]:
            idx = start - 1
            if 0 <= idx < len(y_c):
                ax.plot(
                    x[idx],
                    y_c[idx],
                    marker="o",
                    markersize=7,
                    color=a["color"],
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.5,
                    zorder=4,
                )

    ax.set_yscale("log")
    ax.set_xlabel("Evaluation", fontsize=10)
    ax.set_ylabel("Regret (log scale)", fontsize=10)
    ax.grid(True, which="major", axis="both", color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)

    subject = "the agents against every individual baseline" if agents else "every individual baseline"
    ax.set_title(
        f"{task_dir.name}: regret vs. evaluation -- {subject}",
        fontsize=12,
        color=TEXT_PRIMARY,
        pad=14,
        loc="left",
    )
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9.5,
        labelcolor=TEXT_PRIMARY,
        title="Paired hue = same family\n(solid vs. dashed)",
        title_fontsize=8.5,
    )
    note = "Baselines: mean over seeds; capped ones simply end early. " + capped_note(baselines)
    if agents:
        note = "Dots mark each agent's phase switches. " + note
    fig.text(0.0, -0.03, note, fontsize=8, color=TEXT_MUTED)

    fig.tight_layout()
    fig.savefig(task_dir / "plot2_regret_vs_eval.png", bbox_inches="tight")
    plt.close(fig)


# ==================== PLOT 3: which method was active, when ====================
# Reuses Plot 2's per-method colors so all three plots stay visually
# consistent (same entity -> same color everywhere).
NARROW_THRESHOLD_EVALS = 180  # below this width, route the label above with a leader


def plot_agent_timeline(task_dir, baselines, agents):
    method_color = {k: v["color"] for k, v in baseline_style.items()}
    lanes = list(agents)[::-1]  # first agent on top
    budget = max(b["budget"] for b in baselines.values())

    fig, ax = plt.subplots(figsize=(11, 1.2 + 1.5 * len(lanes)), dpi=150)
    for y0, label in enumerate(lanes):
        a = agents[label]
        for raw_label, start, end in a["phases"]:
            canon = canonical_method(raw_label)
            color = method_color.get(canon, MUTED_GRAY)
            display = baseline_display_names.get(canon, raw_label)
            if "restart" in raw_label.lower():  # segment color already names the method
                display = "restart " + raw_label.rsplit("_", 1)[-1]
            width = end - start + 1
            ax.broken_barh(
                [(start, width)],
                (y0 - 0.32, 0.64),
                facecolors=color,
                edgecolors=SURFACE,
                linewidth=1.5,
                zorder=2,
            )
            center = start + width / 2
            if width >= NARROW_THRESHOLD_EVALS:
                ax.text(center, y0, display, ha="center", va="center", fontsize=9,
                        color="white", fontweight="medium", zorder=3)
            else:
                label_y = y0 + 0.62
                ax.plot([center, center], [y0 + 0.32, label_y - 0.05], color=TEXT_MUTED, linewidth=1, zorder=2)
                ax.text(center, label_y, display, ha="center", va="bottom", fontsize=8.5,
                        color=TEXT_PRIMARY, zorder=3)

        # Logged deviations from the agent's pre-registered roadmap
        # (changelog.md), not just "which method ran".
        for ev in a["deviations"]:
            ax.plot([ev, ev], [y0 - 0.32, y0 + 0.95], color=STATUS_WARNING, linestyle="--",
                    linewidth=1.3, zorder=4)
            ax.plot(ev, y0 + 0.95, marker="*", markersize=11, color=STATUS_WARNING,
                    markeredgecolor=TEXT_PRIMARY, markeredgewidth=0.6, zorder=5)
            ax.text(ev + 12, y0 + 0.95, str(ev), ha="left", va="center",
                    fontsize=7.8, color=TEXT_PRIMARY, zorder=5)
        if not a["deviations"]:
            ax.text(budget - 2, y0 - 0.45, f"{label}: no changelog entries -- executed exactly as planned",
                    ha="right", va="top", fontsize=7.8, color=TEXT_MUTED, style="italic", zorder=3)

    ax.set_xlim(0, budget)
    ax.set_ylim(-0.85, len(lanes) - 1 + 1.25)
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes, fontsize=10, color=TEXT_PRIMARY)
    ax.set_xlabel("Evaluation", fontsize=10)
    ax.grid(True, which="major", axis="x", color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0)

    ax.set_title(
        f"{task_dir.name}: which method each agent was running, by evaluation",
        fontsize=12,
        color=TEXT_PRIMARY,
        pad=14,
        loc="left",
    )
    fig.text(
        0.0,
        -0.05,
        "Segment color matches that method's line in plot2_regret_vs_eval.png. "
        "Narrow segments are labeled above with a leader line. "
        "Stars flag logged deviations from the agent's pre-registered roadmap (changelog.md), "
        "labeled with the evaluation.",
        fontsize=8,
        color=TEXT_MUTED,
    )

    fig.tight_layout()
    fig.savefig(task_dir / "plot3_agent_method_timeline.png", bbox_inches="tight")
    plt.close(fig)


def make_plots(task_dir: Path):
    baselines, agents = load_task(task_dir)
    plot_counterfactual_bar(task_dir, baselines, agents)
    plot_regret_vs_eval(task_dir, baselines, agents)
    if agents:
        plot_agent_timeline(task_dir, baselines, agents)
    print(f"{task_dir.name}: {3 if agents else 2} plots, agents={list(agents)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(MAIN_ROOT / "results_linux"))
    p.add_argument("--task", help="one results/<task> folder name; default: every task")
    args = p.parse_args()
    results = Path(args.results)
    tasks = [results / args.task] if args.task else sorted(d for d in results.iterdir() if d.is_dir())
    for task_dir in tasks:
        make_plots(task_dir)
