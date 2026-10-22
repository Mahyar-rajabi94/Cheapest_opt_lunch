"""Write this task's baseline and agent plots to poly_results/, named so the
task and the baseline/agent split are clear outside the repo. Uses the data
loading and styles of main/scripts/make_comparison_plots.py on the results in
main/results_linux/opv_pool_pce_d11/ and the agent run in task_006_sandbox/.
Run from task_006_baseline with the project venv:

    ../.venv/bin/python make_poly_results.py

Writes poly_results/task_006_opv_pool_pce_d11/:
    baselines/               the 10 fixed optimizers only (budget 100)
    agent_claude/            the first Claude agent run (task_006_sandbox) against them
    agent_claude_run2/       the second, independent run (task_006_sandbox_run2) against them
    agent_claude_both_runs/  both runs against them

Regret is the best PCE in the pool minus the best PCE found, in percentage
points. Unlike the shared plots it is drawn on a linear axis: many runs reach
exactly 0 (they found the best device), which a log axis cannot show.
"""
import sys
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main" / "scripts"))
import make_comparison_plots as plots  # noqa: E402

RESULTS = ROOT / "main" / "results_linux" / "opv_pool_pce_d11"
NAME = "task_006_opv_pool_pce_d11"  # plot-title prefix and file-name prefix
OUT = ROOT / "poly_results" / NAME
F_OPT = plots.json.loads(next(RESULTS.glob("*/seed_*/meta.json")).read_text())["task_f_opt"]
FOUND_TOL = 1e-9  # regret at or below this: the best device was found

# Agent labels that name a baseline method without containing its key.
METHOD_ALIASES = {"st_trbo_local": "turbo", "st_gp_pi_final": "gp_bo",  # run 1
                  "trbo_local": "turbo", "trbo_pm_local": "turbo", "gp_dnn_eiobs": "gp_bo"}  # run 2


def canonical(label):
    return METHOD_ALIASES.get(label, plots.canonical_method(label))


def seeds_found(name):
    """Seeds of baseline `name` that found the best device within the budget."""
    last = [plots.regret_trajectory(plots.load_rows(p), F_OPT)[-1]
            for p in sorted((RESULTS / name).glob("seed_*/trace.csv"))]
    return sum(v <= FOUND_TOL for v in last)


def style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(plots.AXIS)
    ax.spines["bottom"].set_color(plots.AXIS)
    ax.set_axisbelow(True)


def final_regret_bar(path, baselines, agents, title):
    """Final regret per run, linear axis, with how many seeds found the best device.
    With agents, only the baselines they used are shown, each run alone."""
    used = {canonical(label) for a in agents.values() for label, _, _ in a["phases"]}
    names = sorted((n for n in baselines if n in used or not agents), key=lambda n: baselines[n]["regret"][-1])

    entries = []  # (label, final regret, color, note)
    for label, a in agents.items():
        found = np.flatnonzero(a["regret"] <= FOUND_TOL)
        note = f"found the best device at evaluation {a['x'][found[0]]}" if len(found) else ""
        entries.append((f"{label} (budget {len(a['x'])})", a["regret"][-1], a["color"], note))
    for n in names:
        b = baselines[n]
        k = seeds_found(n)
        seeds = f", mean of {b['n_seeds']} seeds" if b["n_seeds"] > 1 else ""
        note = (f"{k}/{b['n_seeds']} seeds found the best device" if b["n_seeds"] > 1
                else "found the best device" if k else "")
        entries.append((f"{plots.baseline_display_names[n]}{' alone' if agents else ''} (budget {b['budget']}{seeds})",
                        b["regret"][-1], plots.baseline_style[n]["color"], note))

    y_pos = np.arange(len(entries))[::-1]
    x_max = max(max(v for _, v, _, _ in entries), 0.1)
    fig, ax = plt.subplots(figsize=(9.5, 0.45 * len(entries) + 1.8), dpi=150)
    ax.barh(y_pos, [v for _, v, _, _ in entries], height=0.5, color=[c for _, _, c, _ in entries], zorder=3)
    for yi, (_, v, _, note) in zip(y_pos, entries):
        ax.text(v + 0.02 * x_max, yi, f"{v:.2f}" + (f"   {note}" if note else ""), va="center", ha="left",
                fontsize=9.5, color=plots.TEXT_SECONDARY)
    ax.set_xlim(0, x_max * 1.75)  # room for the notes
    ax.set_yticks(y_pos)
    ax.set_yticklabels([label for label, _, _, _ in entries], color=plots.TEXT_PRIMARY, fontsize=10)
    ax.set_xlabel("Final regret (PCE percentage points below the best device)", fontsize=10)
    ax.grid(True, which="major", axis="x", color=plots.GRIDLINE, linewidth=1.0, zorder=0)
    style_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{NAME}: {title}", fontsize=12, color=plots.TEXT_PRIMARY, pad=14, loc="left")
    fig.text(0.0, -0.01, f"Regret 0: the run found the best device in the pool ({-F_OPT:.2f}% PCE).",
             fontsize=8, color=plots.TEXT_MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def regret_vs_eval(path, baselines, agents, subject):
    """Mean regret over seeds by evaluation, linear axis; the agents on top, dots at their phase switches."""
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    for name, b in baselines.items():
        ax.plot(b["x"], b["regret"], linewidth=1.6, alpha=0.9, label=plots.baseline_display_names[name], zorder=2,
                **plots.baseline_style[name])
    for label, a in agents.items():
        ax.plot(a["x"], a["regret"], color=a["color"], linewidth=2.6, label=label, zorder=3)
        starts = [start - 1 for _, start, _ in a["phases"]]
        ax.plot(a["x"][starts], a["regret"][starts], linestyle="none", marker="o", markersize=7, color=a["color"],
                markeredgecolor=plots.SURFACE, markeredgewidth=1.5, zorder=4)

    top = max(np.nanmax(b["regret"]) for b in [*baselines.values(), *agents.values()])
    ax.set_xlim(0, max(len(b["x"]) for b in baselines.values()))
    ax.set_ylim(-0.03 * top, top * 1.05)  # a line at regret 0 stays visible above the axis
    ax.set_xlabel("Evaluation", fontsize=10)
    ax.set_ylabel("Regret (PCE percentage points below the best device)", fontsize=10)
    ax.grid(True, which="major", axis="both", color=plots.GRIDLINE, linewidth=1.0, zorder=0)
    style_axes(ax)
    ax.set_title(f"{NAME}: regret vs. evaluation, {subject}", fontsize=12, color=plots.TEXT_PRIMARY, pad=14,
                 loc="left")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9.5,
              labelcolor=plots.TEXT_PRIMARY, title="Shared hue = same family\n(solid, dashed, dotted)",
              title_fontsize=8.5)
    note = "Baselines: mean over seeds. Regret 0: the best device in the pool was found."
    if agents:
        note = ("Dots mark each run's phase switches. " if len(agents) > 1
                else "Dots mark the agent's phase switches. ") + note
    fig.text(0.0, -0.03, note, fontsize=8, color=plots.TEXT_MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_timeline(path, agent, run=""):
    """Which method the agent ran, by evaluation: one row per method label (in
    order of first use), plan changes as ticks in their own row, and the
    evaluation of the best point marked. `run` (e.g. " in run 2") goes in the title."""
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
    reached = agent["regret"][best_i] <= FOUND_TOL
    ax.plot([best_eval, best_eval], [0.5, change_y + 0.45], color=plots.TEXT_PRIMARY, linewidth=1.1,
            linestyle=(0, (4, 3)), zorder=3)
    ax.text(best_eval, change_y + 0.55,
            f"{'best device in the pool found' if reached else 'best point found'}, evaluation {best_eval:,}",
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
    ax.set_title(f"{NAME}: methods the Claude agent ran{run}, by evaluation", fontsize=12,
                 color=plots.TEXT_PRIMARY, pad=10, loc="left")
    caption = (
        "One row per method label the agent used, in order of first use. Colored rows are standard "
        "methods, in the same colors as the regret plot ("
        + ", ".join(plots.baseline_display_names[c] for c in standard) + "); gray rows are the "
        "agent's own routines. Ticks: the plan changes logged in changelog.md, at evaluations "
        + ", ".join(f"{e:,}" + (f" ({n} entries)" if n > 1 else "")
                    for e, n in Counter(agent["deviations"]).items()) + "."
    )
    fig.text(0.0, -0.01, textwrap.fill(caption, 175), fontsize=8, color=plots.TEXT_MUTED, va="top")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


baselines, agents = plots.load_task(RESULTS)
if list(agents) != ["Claude agent", "Claude agent, run 2"]:
    raise SystemExit(f"Expected two Claude agent runs for opv_pool_pce_d11, found {list(agents)}.")
run1, run2 = agents.values()  # blue and orange in every plot

for sub in ("baselines", "agent_claude", "agent_claude_run2", "agent_claude_both_runs"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
final_regret_bar(OUT / "baselines" / f"{NAME}_baselines_final_regret.png", baselines, {},
                 "final regret of every baseline")
regret_vs_eval(OUT / "baselines" / f"{NAME}_baselines_regret_vs_eval.png", baselines, {}, "every baseline")

one = {"Claude agent": run1}
final_regret_bar(OUT / "agent_claude" / f"{NAME}_agent_claude_vs_methods_used_final_regret.png", baselines, one,
                 "the Claude agent vs. each method it used, run alone")
regret_vs_eval(OUT / "agent_claude" / f"{NAME}_agent_claude_vs_baselines_regret_vs_eval.png", baselines, one,
               "the Claude agent against every baseline")
plot_timeline(OUT / "agent_claude" / f"{NAME}_agent_claude_method_timeline.png", run1)

two = {"Claude agent, run 2": run2}
sub = OUT / "agent_claude_run2"
final_regret_bar(sub / f"{NAME}_agent_claude_run2_vs_methods_used_final_regret.png", baselines, two,
                 "Claude agent run 2 vs. each method it used, run alone")
regret_vs_eval(sub / f"{NAME}_agent_claude_run2_vs_baselines_regret_vs_eval.png", baselines, two,
               "Claude agent run 2 against every baseline")
plot_timeline(sub / f"{NAME}_agent_claude_run2_method_timeline.png", run2, " in run 2")

both = {"Claude agent, run 1": run1, "Claude agent, run 2": run2}
sub = OUT / "agent_claude_both_runs"
final_regret_bar(sub / f"{NAME}_agent_claude_both_runs_vs_methods_used_final_regret.png", baselines, both,
                 "both Claude agent runs vs. each method they used, run alone")
regret_vs_eval(sub / f"{NAME}_agent_claude_both_runs_vs_baselines_regret_vs_eval.png", baselines, both,
               "both Claude agent runs against every baseline")
for path in sorted(OUT.rglob("*.png")):
    print(path.relative_to(ROOT))
