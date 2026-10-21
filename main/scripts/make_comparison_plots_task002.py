"""Comparison plots: 1 agent run (Claude) vs. the 9 classical baselines,
Rastrigin task (task_002, BBOB f15).

Same structure/conventions as make_comparison_plots.py (task_001 /
Rosenbrock), adapted for a single agent run and this task's specifics:
  - F_OPT = 1000.0 (rastrigin_d10_i1's true optimum)
  - Only one agent (no OpenAI run yet for this task)
  - The agent's run stopped early at 1898/2000 evaluations (local-polish
    plateau, a pre-registered stopping criterion -- not a crash) and logged
    FOUR real method-switch deviations (not just one, as in task_001):
    two CMA-ES stagnation restarts, abandoning CMA-ES after the restart
    cap, and BOBYQA converging mid-polish and falling back to L-BFGS-B.

Colors reuse the same validated default categorical palette and the same
per-method color assignment as make_comparison_plots.py, so a method's
color means the same thing across every task's figures.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results" / "rastrigin_d10_i1"
SANDBOX_ROOT = Path(__file__).resolve().parent.parent.parent
F_OPT = 1000.0

# Validated default categorical palette, fixed order (same as task_001's script).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
)
GREEN, VIOLET, RED = "#008300", "#6250d6", "#e34948"
MUTED_GRAY = "#898781"
STATUS_WARNING = "#fab219"
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


def load_rows(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def regret_trajectory(rows):
    y = np.array([float(r["y_best_so_far"]) for r in rows])
    return y - F_OPT


# ---- baseline trajectories (mean across 5 seeds where available) ----
SEED5 = ["random_search", "sobol", "cma_es", "de", "turbo", "gp_bo"]
SINGLE = ["bobyqa", "nelder_mead", "lbfgsb"]

baseline_traj = {}
for name in SEED5:
    seed_ys = []
    for seed in range(5):
        rows = load_rows(RESULTS / name / f"seed_{seed}" / "trace.csv")
        seed_ys.append(regret_trajectory(rows))
    min_len = min(len(a) for a in seed_ys)
    mean_regret = np.stack([a[:min_len] for a in seed_ys]).mean(axis=0)
    baseline_traj[name] = (np.arange(1, min_len + 1), mean_regret)

for name in SINGLE:
    rows = load_rows(RESULTS / name / "seed_0" / "trace.csv")
    regret = regret_trajectory(rows)
    baseline_traj[name] = (np.arange(1, len(rows) + 1), regret)

# ---- agent trajectory + phase-switch points (one agent: Claude) ----
AGENT_PATHS = {
    "claude_agent": SANDBOX_ROOT / "task_002_sandbox" / "results" / "agent_task_002" / "trace.csv",
}

agent_traj = {}
agent_switches = {}
for key, path in AGENT_PATHS.items():
    rows = load_rows(path)
    regret = regret_trajectory(rows)
    eval_idx = np.array([int(r["eval_idx"]) for r in rows])
    agent_traj[key] = (eval_idx, regret)
    switches, cur = [], None
    for r in rows:
        if r["optimizer"] != cur:
            switches.append(int(r["eval_idx"]))
            cur = r["optimizer"]
    agent_switches[key] = switches

final_regret = {
    "claude_agent": agent_traj["claude_agent"][1][-1],
    "cma_es": baseline_traj["cma_es"][1][-1],
    "lbfgsb": baseline_traj["lbfgsb"][1][-1],
    "bobyqa": baseline_traj["bobyqa"][1][-1],
}

# ==================== PLOT 1: counterfactual bar chart ====================
order = ["claude_agent", "cma_es", "lbfgsb", "bobyqa"]
labels = {
    "claude_agent": "Claude agent (budget 2000, stopped early at 1898)",
    "cma_es": "CMA-ES alone (budget 2000, mean of 5 seeds)",
    "lbfgsb": "L-BFGS-B alone (budget 2000)",
    "bobyqa": "BOBYQA alone (budget 500*)",
}
colors = {
    "claude_agent": BLUE,
    "cma_es": AQUA,
    "lbfgsb": YELLOW,
    "bobyqa": MAGENTA,
}

values = [max(final_regret[k], 1e-12) for k in order]
y_pos = np.arange(len(order))[::-1]

fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=150)
ax.barh(y_pos, values, height=0.5, color=[colors[k] for k in order], zorder=3)
ax.set_xscale("log")
ax.grid(True, which="major", axis="x", color=GRIDLINE, linewidth=1.0, zorder=0)
ax.set_axisbelow(True)
ax.set_yticks(y_pos)
ax.set_yticklabels([labels[k] for k in order], color=TEXT_PRIMARY, fontsize=10)
ax.set_xlabel("Final regret (log scale)", fontsize=10)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)

for yi, k in zip(y_pos, order):
    v = final_regret[k]
    label = f"{v:.2e}" if v < 0.01 else f"{v:.2f}"
    ax.text(
        max(v, 1e-12) * 1.3,
        yi,
        label,
        va="center",
        ha="left",
        fontsize=9.5,
        color=TEXT_SECONDARY,
    )

ax.set_title(
    "Claude agent vs. the single-method counterfactuals it cascaded through (task 2: Rastrigin)",
    fontsize=11.5,
    color=TEXT_PRIMARY,
    pad=14,
    loc="left",
)
ax.text(
    0.0,
    -0.2,
    "*BOBYQA's own baseline uses budget 500, not 2000 -- it was found to stall past that point (see src/optimizers/bobyqa.py).",
    transform=ax.transAxes,
    fontsize=8,
    color=TEXT_MUTED,
)

fig.tight_layout()
out1 = RESULTS / "plot1_counterfactual_bar.png"
fig.savefig(out1, bbox_inches="tight")
plt.close(fig)

# ==================== PLOT 2: regret vs. evaluation ====================
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

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)

for name, (x, y) in baseline_traj.items():
    y_c = np.clip(y, 1e-12, None)
    style = baseline_style[name]
    ax.plot(
        x, y_c, linewidth=1.6, alpha=0.9, label=baseline_display_names[name],
        zorder=2, **style,
    )

x, y = agent_traj["claude_agent"]
y_c = np.clip(y, 1e-12, None)
ax.plot(x, y_c, color=BLUE, linewidth=2.6, label="Claude agent", zorder=3)
for s in agent_switches["claude_agent"]:
    idx = s - 1
    if 0 <= idx < len(y_c):
        ax.plot(
            x[idx], y_c[idx], marker="o", markersize=7, color=BLUE,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
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

ax.set_title(
    "Regret vs. evaluation -- Claude agent against every individual baseline (task 2: Rastrigin)",
    fontsize=12,
    color=TEXT_PRIMARY,
    pad=14,
    loc="left",
)
handles, labels_ = ax.get_legend_handles_labels()
ax.legend(
    handles, labels_, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
    fontsize=9.5, labelcolor=TEXT_PRIMARY,
    title="Paired hue = same family\n(solid vs. dashed)", title_fontsize=8.5,
)
fig.text(
    0.0, -0.03,
    "Dots mark the agent's phase switches (sobol -> cma_es [x2 restarts] -> bobyqa -> lbfgsb). "
    "Agent stopped early at eval 1898/2000 (pre-registered plateau criterion). "
    "Baselines shorter than 2000 evals (gp_bo, bobyqa) simply end early.",
    fontsize=8, color=TEXT_MUTED,
)

fig.tight_layout()
out2 = RESULTS / "plot2_regret_vs_eval.png"
fig.savefig(out2, bbox_inches="tight")
plt.close(fig)

# ==================== PLOT 3: which method was active, when ====================
method_color = {k: v["color"] for k, v in baseline_style.items()}
method_display = baseline_display_names
CANONICAL_BY_LENGTH = sorted(method_color, key=len, reverse=True)


def canonical_method(raw_label: str) -> str:
    raw = raw_label.lower()
    for key in CANONICAL_BY_LENGTH:
        if key in raw:
            return key
    return raw_label


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


agent_phases = {key: phases_for(load_rows(path)) for key, path in AGENT_PATHS.items()}

NARROW_THRESHOLD_EVALS = 180

fig, ax = plt.subplots(figsize=(11, 2.4), dpi=150)
lane_y = {"claude_agent": 0}
lane_label = {"claude_agent": "Claude agent"}

display_override = {
    "sobol_init": "Sobol",
    "cma_es_restart_1": "Restart 1",
    "cma_es_restart_2": "Restart 2",
    "local_polish_bobyqa": "BOBYQA",
    "local_polish_lbfgsb": "L-BFGS-B",
}

for key, phases in agent_phases.items():
    y0 = lane_y[key]
    for raw_label, start, end in phases:
        canon = canonical_method(raw_label)
        color = method_color.get(canon, MUTED_GRAY)
        display = display_override.get(raw_label, method_display.get(canon, raw_label))
        width = end - start + 1
        ax.broken_barh(
            [(start, width)], (y0 - 0.32, 0.64),
            facecolors=color, edgecolors=SURFACE, linewidth=1.5, zorder=2,
        )
        center = start + width / 2
        if width >= NARROW_THRESHOLD_EVALS:
            ax.text(
                center, y0, display, ha="center", va="center", fontsize=8.5,
                color="white", fontweight="medium", zorder=3,
            )
        else:
            label_y = y0 + 0.62
            ax.plot([center, center], [y0 + 0.32, label_y - 0.05], color=TEXT_MUTED, linewidth=1, zorder=2)
            ax.text(
                center, label_y, display, ha="center", va="bottom", fontsize=8,
                color=TEXT_PRIMARY, zorder=3,
            )

# Logged deviations (changelog.md): 4 real method-switch events, each
# exactly at a phase boundary above, plus the early-stop event at the end
# (not a method switch -- the pre-registered plateau criterion firing).
# The two restarts share one callout (same trigger, 200 evals apart --
# separate callouts at this spacing would overlap).
y0 = lane_y["claude_agent"]

for ev in (737, 937):
    ax.plot([ev, ev], [y0 - 0.32, y0 + 1.5], color=STATUS_WARNING,
            linestyle="--", linewidth=1.3, zorder=4)
    ax.plot(ev, y0 + 1.5, marker="*", markersize=11, color=STATUS_WARNING,
            markeredgecolor=TEXT_PRIMARY, markeredgewidth=0.6, zorder=5)
ax.text(837, y0 + 1.72, "2 CMA-ES restarts (criterion 1)", ha="center", va="center",
        fontsize=7.6, color=TEXT_PRIMARY, zorder=5)
ax.text(837, y0 + 1.5, "stagnated at eval 737 & 937", ha="center", va="center",
        fontsize=7.3, color=TEXT_SECONDARY, zorder=5, fontstyle="italic")

for ev, mh, note in [
    (1137, 1.1, "restart cap reached ->\nlocal polish (criterion 1)"),
    (1447, 1.1, "BOBYQA converged ->\nL-BFGS-B (criterion 3)"),
]:
    ax.plot([ev, ev], [y0 - 0.32, y0 + mh], color=STATUS_WARNING,
            linestyle="--", linewidth=1.3, zorder=4)
    ax.plot(ev, y0 + mh, marker="*", markersize=11, color=STATUS_WARNING,
            markeredgecolor=TEXT_PRIMARY, markeredgewidth=0.6, zorder=5)
    ax.text(ev + 25, y0 + mh, note, ha="left", va="center",
            fontsize=7.3, color=TEXT_PRIMARY, zorder=5)

# Early stop: the agent used only 1898 of its 2000-evaluation budget.
ax.axvspan(1898, 2000, color=MUTED_GRAY, alpha=0.15, zorder=1)
ax.annotate(
    "stopped early: 1898/2000 evals used (criterion 4)",
    xy=(1898, y0 - 0.32), xytext=(1550, y0 - 0.95),
    ha="left", va="top", fontsize=7.6, color=TEXT_MUTED, style="italic",
    arrowprops=dict(arrowstyle="-", color=TEXT_MUTED, linewidth=1),
)

ax.set_xlim(0, 2000)
ax.set_ylim(-1.4, 2.25)
ax.set_yticks([0])
ax.set_yticklabels([lane_label["claude_agent"]], fontsize=10, color=TEXT_PRIMARY)
ax.set_xlabel("Evaluation", fontsize=10)
ax.grid(True, which="major", axis="x", color=GRIDLINE, linewidth=1.0, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.tick_params(axis="y", length=0)

ax.set_title(
    "Which method the Claude agent was running, by evaluation (task 2: Rastrigin)",
    fontsize=12, color=TEXT_PRIMARY, pad=14, loc="left",
)
fig.text(
    0.0, -0.08,
    "Segment color matches that method's line in plot2_regret_vs_eval.png. "
    "The star marker flags a logged deviation from the agent's pre-registered roadmap (changelog.md).",
    fontsize=8, color=TEXT_MUTED,
)

fig.tight_layout()
out3 = RESULTS / "plot3_agent_method_timeline.png"
fig.savefig(out3, bbox_inches="tight")
plt.close(fig)

print(f"Wrote {out1}")
print(f"Wrote {out2}")
print(f"Wrote {out3}")
