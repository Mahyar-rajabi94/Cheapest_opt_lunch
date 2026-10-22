"""Write this task's baseline and agent plots to poly_results/, named so the
task and the baseline/agent split are clear outside the repo. Uses the
plotting functions of main/scripts/make_comparison_plots.py on the results in
main/results_linux/opv_pce_d11/. Run from task_005_baseline with the project venv:

    ../.venv/bin/python make_poly_results.py

Writes poly_results/task_005_opv_pce_d11/:
    baselines/                the 9 fixed optimizers only
    agent_claude/             the Claude agent run (task_005_sandbox) against them
    baselines_budget100/      the same baselines cut at evaluation 100
    agent_claude_budget100/   the 100-evaluation Claude agent run
                              (task_005_sandbox_budget100) against them
"""
import json
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
RUN_100 = ROOT / "task_005_sandbox_budget100" / "results" / "agent_task_005"
F_OPT = json.loads(next(RESULTS.glob("*/seed_*/meta.json")).read_text())["task_f_opt"]

# A bar chart without agents is titled "(no agent run yet)" by the shared
# script; here the agent run exists and is plotted separately.
_set_title = Axes.set_title
Axes.set_title = lambda ax, label, *a, **k: _set_title(
    ax, label.replace("(no agent run yet)", "(baselines only)"), *a, **k)

# Agent labels that name a baseline method without containing its key.
METHOD_ALIASES = {"cma_local": "cma_es", "trbo_local": "turbo",
                  "gpbo_logei_lincon": "gp_bo", "logei_obs_global": "gp_bo"}
_shared_canonical = plots.canonical_method


def canonical(label):
    return METHOD_ALIASES.get(label, _shared_canonical(label))


def counterfactual_bar(task_dir, baselines, agents):
    """The shared plot1 with METHOD_ALIASES applied, so it includes the baselines
    the agent's own labels stand for. The shared plot3 stays unaliased: its
    output is kept as the original timeline."""
    plots.canonical_method = canonical
    try:
        plots.plot_counterfactual_bar(task_dir, baselines, agents)
    finally:
        plots.canonical_method = _shared_canonical


def load_agent(run_dir):
    """An agent run in the form plots.load_task returns, for a run not in AGENT_RUNS."""
    rows = plots.load_rows(run_dir / "trace.csv")
    return dict(x=np.array([int(r["eval_idx"]) for r in rows]), regret=plots.regret_trajectory(rows, F_OPT),
                phases=plots.phases_for(rows), deviations=plots.changelog_evals(run_dir / "changelog.md"),
                color=plots.AGENT_COLORS[0])


def cut(baselines, n):
    """The baselines as if run with budget n. The optimizers never see the
    budget, so a run's first n evaluations are what a budget-n run would do."""
    out = {}
    for name, b in baselines.items():
        last = [plots.regret_trajectory(plots.load_rows(p), F_OPT)[:n][-1]
                for p in sorted((RESULTS / name).glob("seed_*/trace.csv"))]
        out[name] = dict(b, x=b["x"][:n], regret=b["regret"][:n], budget=min(b["budget"], n),
                         n_feasible=int(np.isfinite(last).sum()))
    return out


def plot_timeline(path, agent, name):
    """Which method the agent ran, by evaluation. Unlike the shared plot3, each
    method label gets its own row (named on the axis, so no text sits on or
    above the bars), plan changes are ticks in their own row with the numbers
    in the caption, and the evaluation of the best point is marked."""
    order = list(dict.fromkeys(label for label, _, _ in agent["phases"]))  # first use on top
    row = {label: len(order) - i for i, label in enumerate(order)}
    change_y = len(order) + 1
    n_evals = int(agent["x"][-1])
    standard = list(dict.fromkeys(canonical(label) for label in order if canonical(label) in plots.baseline_style))

    fig, ax = plt.subplots(figsize=(11, 0.36 * (len(order) + 2) + 1.2), dpi=150)
    for label, start, end in agent["phases"]:
        color = plots.baseline_style.get(canonical(label), {}).get("color", plots.MUTED_GRAY)
        ax.broken_barh([(start, end - start + 1)], (row[label] - 0.3, 0.6), facecolors=color,
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
            ha="right" if best_eval > 0.8 * n_evals else "left" if best_eval < 0.2 * n_evals else "center",
            va="bottom", fontsize=8.5, color=plots.TEXT_PRIMARY)

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
    ax.set_title(f"{name}: methods the Claude agent ran, by evaluation", fontsize=12,
                 color=plots.TEXT_PRIMARY, pad=10, loc="left")
    caption = (
        "One row per method label the agent used, in order of first use. Colored rows are standard "
        "methods, in the same colors as the regret plot ("
        + ", ".join(plots.baseline_display_names[c] for c in standard) + "); gray rows are the "
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

baselines_100 = cut(baselines, 100)
agents_100 = {"Claude agent": load_agent(RUN_100)}

# The shared functions save fixed file names into a folder whose name they put
# in the titles, so render into a temporary folder with the title name, then rename.
BASELINE_FILES = {"plot1_counterfactual_bar.png": "baselines_final_regret.png",
                  "plot2_regret_vs_eval.png": "baselines_regret_vs_eval.png"}
AGENT_FILES = {"plot1_counterfactual_bar.png": "agent_claude_vs_methods_used_final_regret.png",
               "plot2_regret_vs_eval.png": "agent_claude_vs_baselines_regret_vs_eval.png"}
# The first run's plot1 without METHOD_ALIASES and its shared plot3, as first
# published, kept beside the current versions.
ORIGINAL_FILES = {"plot1_original.png": "agent_claude_vs_methods_used_final_regret_original.png",
                  "plot3_agent_method_timeline.png": "agent_claude_method_timeline_original.png"}
renders = {  # folder -> (title and file prefix, baselines, agents, files to keep)
    "baselines": (NAME, baselines, {}, BASELINE_FILES),
    "agent_claude": (NAME, baselines, agents, {**AGENT_FILES, **ORIGINAL_FILES}),
    "baselines_budget100": (f"{NAME}_budget100", baselines_100, {}, BASELINE_FILES),
    "agent_claude_budget100": (f"{NAME}_budget100", baselines_100, agents_100, AGENT_FILES),
}
for sub, (name, run_baselines, run_agents, files) in renders.items():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp) / name
        tmp_dir.mkdir()
        if ORIGINAL_FILES.keys() <= files.keys():
            plots.plot_counterfactual_bar(tmp_dir, run_baselines, run_agents)
            (tmp_dir / "plot1_counterfactual_bar.png").rename(tmp_dir / "plot1_original.png")
            plots.plot_agent_timeline(tmp_dir, run_baselines, run_agents)
        counterfactual_bar(tmp_dir, run_baselines, run_agents)
        plots.plot_regret_vs_eval(tmp_dir, run_baselines, run_agents)
        (OUT / sub).mkdir(parents=True, exist_ok=True)
        for src, dst in files.items():
            shutil.copy(tmp_dir / src, OUT / sub / f"{name}_{dst}")
    if run_agents:
        plot_timeline(OUT / sub / f"{name}_agent_claude_method_timeline.png", run_agents["Claude agent"], name)
for path in sorted(OUT.rglob("*.png")):
    print(path.relative_to(ROOT))
