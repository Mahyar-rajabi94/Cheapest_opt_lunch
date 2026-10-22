"""Write this task's baseline and agent plots to poly_results/, named so the
task and the baseline/agent split are clear outside the repo. Uses the
plotting functions of main/scripts/make_comparison_plots.py on the results in
main/results_linux/opv_pce_d11/. Run from task_005_baseline with the project venv:

    ../.venv/bin/python make_poly_results.py

Writes poly_results/task_005_opv_pce_d11/:
    baselines/      the 9 fixed optimizers only
    agent_claude/   the Claude agent run (task_005_sandbox) against them
"""
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main" / "scripts"))
import make_comparison_plots as plots  # noqa: E402

RESULTS = ROOT / "main" / "results_linux" / "opv_pce_d11"
NAME = "task_005_opv_pce_d11"  # plot-title prefix and file-name prefix
OUT = ROOT / "poly_results" / NAME

# A bar chart without agents is titled "(no agent run yet)" by the shared
# script; here the agent run exists and is plotted separately.
_set_title = Axes.set_title
Axes.set_title = lambda ax, label, *a, **k: _set_title(
    ax, label.replace("(no agent run yet)", "(baselines only)"), *a, **k)

# Agent labels that name a baseline method without containing its key.
METHOD_ALIASES = {"cma_local": "cma_es"}


def plot_timeline(path, agent):
    """Which method the agent ran, by evaluation. Unlike the shared plot3, each
    method label gets its own row (named on the axis, so no text sits on or
    above the bars), plan changes are ticks in their own row with the numbers
    in the caption, and the evaluation of the best point is marked."""
    order = list(dict.fromkeys(name for name, _, _ in agent["phases"]))  # first use on top
    row = {name: len(order) - i for i, name in enumerate(order)}
    change_y = len(order) + 1
    n_evals = int(agent["x"][-1])

    fig, ax = plt.subplots(figsize=(11, 0.36 * (len(order) + 2) + 1.2), dpi=150)
    for name, start, end in agent["phases"]:
        canon = METHOD_ALIASES.get(name, plots.canonical_method(name))
        color = plots.baseline_style.get(canon, {}).get("color", plots.MUTED_GRAY)
        ax.broken_barh([(start, end - start + 1)], (row[name] - 0.3, 0.6), facecolors=color,
                       edgecolors=plots.SURFACE, linewidth=1.0, zorder=2)

    ax.axhline(len(order) + 0.5, color=plots.GRIDLINE, linewidth=1.0, zorder=1)
    ax.plot(agent["deviations"], [change_y] * len(agent["deviations"]), linestyle="none", marker="|",
            markersize=13, markeredgewidth=1.6, color=plots.TEXT_SECONDARY, zorder=4)

    best_i = int(np.nanargmin(agent["regret"]))
    best_eval = int(agent["x"][best_i])
    reached = agent["regret"][best_i] <= 1e-9
    ax.plot([best_eval, best_eval], [0.5, change_y + 0.45], color=plots.TEXT_PRIMARY, linewidth=1.1,
            linestyle=(0, (4, 3)), zorder=3)
    ax.text(best_eval, change_y + 0.55,
            f"{'best-known optimum reached' if reached else 'best point found'}, evaluation {best_eval:,}",
            ha="center", va="bottom", fontsize=8.5, color=plots.TEXT_PRIMARY)

    ax.set_xlim(0, n_evals)
    ax.set_ylim(0.4, change_y + 1.3)
    ax.set_yticks([*row.values(), change_y])
    ax.set_yticklabels([*row, "plan change logged"], fontsize=9, color=plots.TEXT_PRIMARY)
    ax.set_xlabel("Evaluation", fontsize=10)
    ax.grid(True, which="major", axis="x", color=plots.GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(plots.AXIS)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{NAME}: methods the Claude agent ran, by evaluation", fontsize=12,
                 color=plots.TEXT_PRIMARY, pad=10, loc="left")
    caption = (
        "One row per method label the agent used, in order of first use. Colored rows are standard "
        "methods, in the same colors as the regret plot (Sobol, TuRBO, CMA-ES); gray rows are the "
        "agent's own routines. Ticks: the plan changes logged in changelog.md, at evaluations "
        + ", ".join(f"{e:,}" for e in agent["deviations"]) + "."
    )
    fig.text(0.0, -0.01, textwrap.fill(caption, 175), fontsize=8, color=plots.TEXT_MUTED, va="top")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


baselines, agents = plots.load_task(RESULTS)
if list(agents) != ["Claude agent"]:
    raise SystemExit(f"Expected one Claude agent run for opv_pce_d11, found {list(agents)}.")

# The shared functions save fixed file names into a folder whose name they put
# in the titles, so render into a temporary folder named NAME, then rename.
renders = {
    "baselines": (
        {},
        {"plot1_counterfactual_bar.png": "baselines_final_regret.png",
         "plot2_regret_vs_eval.png": "baselines_regret_vs_eval.png"},
    ),
    "agent_claude": (
        agents,
        {"plot1_counterfactual_bar.png": "agent_claude_vs_methods_used_final_regret.png",
         "plot2_regret_vs_eval.png": "agent_claude_vs_baselines_regret_vs_eval.png",
         # The shared plot3, kept beside the cleaner plot_timeline version.
         "plot3_agent_method_timeline.png": "agent_claude_method_timeline_original.png"},
    ),
}
for sub, (run_agents, names) in renders.items():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp) / NAME
        tmp_dir.mkdir()
        plots.plot_counterfactual_bar(tmp_dir, baselines, run_agents)
        plots.plot_regret_vs_eval(tmp_dir, baselines, run_agents)
        if run_agents:
            plots.plot_agent_timeline(tmp_dir, baselines, run_agents)
        (OUT / sub).mkdir(parents=True, exist_ok=True)
        for src, dst in names.items():
            shutil.copy(tmp_dir / src, OUT / sub / f"{NAME}_{dst}")
plot_timeline(OUT / "agent_claude" / f"{NAME}_agent_claude_method_timeline.png", agents["Claude agent"])
for path in sorted(OUT.rglob("*.png")):
    print(path.relative_to(ROOT))
