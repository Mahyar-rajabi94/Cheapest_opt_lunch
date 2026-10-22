"""Generic comparison plots: one or more Claude agent runs vs. the 9
classical baselines, for any task that follows this project's
results-directory convention (results/<task>_d10_i1/<optimizer>/seed_N/
trace.csv plus results/<task>_d10_i1/<agent_dir>/trace.csv per agent run).

All agent variants for a task are plotted together on the same regret-vs-
evaluation chart and the same method-timeline chart (one lane per agent),
rather than one file per variant.

Usage
-----
    python scripts/plot_agent_vs_baselines.py --task rastrigin_d10_i1 --f-opt 1000.0 --title "Rastrigin" \\
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
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Validated default categorical palette, fixed order (same as other tasks' scripts).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
)
GREEN, VIOLET, RED = "#008300", "#6250d6", "#e34948"
MUTED_GRAY = "#898781"
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

SEED5 = ["random_search", "sobol", "cma_es", "de", "turbo", "gp_bo", "rule_based"]
SINGLE = ["bobyqa", "nelder_mead", "lbfgsb"]

BROWN = "#8a5a2e"

# Agent-line colors, in the order agents are given on the command line.
AGENT_COLORS = [BLUE, ORANGE, VIOLET, "#1a8fa3"]

baseline_style = {
    "random_search": dict(color=AQUA, linestyle="-"),
    "sobol": dict(color=AQUA, linestyle="--"),
    "cma_es": dict(color=YELLOW, linestyle="-"),
    "de": dict(color=YELLOW, linestyle="--"),
    "turbo": dict(color=MAGENTA, linestyle="-"),
    "gp_bo": dict(color=MAGENTA, linestyle="--"),
    "bobyqa": dict(color=GREEN, linestyle="-"),
    "nelder_mead": dict(color=VIOLET, linestyle="-"),
    "lbfgsb": dict(color=RED, linestyle="-"),
    "rule_based": dict(color=BROWN, linestyle="-"),
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
    "rule_based": "Rule-based router (v2 scripted)",
}
CANONICAL_BY_LENGTH = sorted(baseline_style, key=len, reverse=True)


def canonical_method(raw_label: str) -> str:
    raw = raw_label.lower()
    for key in CANONICAL_BY_LENGTH:
        if key in raw:
            return key
    return raw_label


def load_rows(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def regret_trajectory(rows, f_opt):
    # Recompute the running best ourselves from the per-eval scalar (the
    # "merit" column for pre-existing constrained-task baselines that
    # separate raw y / merit / violation; otherwise "y", which already IS
    # the merit for every trace this project's own harness/compat layer
    # writes). This avoids relying on each source's own y_best_so_far
    # column, which some pre-existing constrained-task baselines leave
    # blank until their first feasible point (sometimes never, within a
    # capped window), and is equivalent to it wherever that column is
    # actually populated.
    col = "merit" if "merit" in rows[0] else "y"
    vals = np.array([float(r[col]) for r in rows])
    return np.minimum.accumulate(vals) - f_opt


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


def parse_agent_spec(spec: str):
    """'dir=Label' -> (dir, Label); 'dir' alone -> (dir, dir)."""
    if "=" in spec:
        d, label = spec.split("=", 1)
        return d, label
    return spec, spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="results dir name, e.g. rastrigin_d10_i1")
    p.add_argument("--f-opt", type=float, required=True)
    p.add_argument("--title", required=True, help="display name, e.g. Rastrigin")
    p.add_argument(
        "--agent", action="append", required=True, dest="agents",
        help="repeatable: 'result_subdir=Display label', e.g. claude_agent_run='Claude agent'",
    )
    p.add_argument(
        "--timeline-extra", action="append", default=[], dest="timeline_extra",
        help="repeatable: an additional lane for plot3 only (not plot1/plot2), "
             "e.g. a baseline's per-seed trace: 'rule_based/seed_0=Rule-based router'",
    )
    p.add_argument(
        "--cap-evals", type=int, default=None,
        help="truncate every baseline trajectory (and the plot1 final-regret bar) to at "
             "most this many evaluations, so baselines run at a larger budget (e.g. the "
             "project's 'cheap methods get budget 2000' convention) are shown on equal "
             "footing with agents run at a smaller budget. Does not affect agent traces "
             "themselves (already at their own budget) or methods that stopped early "
             "before reaching this many evals.",
    )
    args = p.parse_args()

    results = ROOT / "results" / args.task
    figures = ROOT / "figures" / args.task
    figures.mkdir(parents=True, exist_ok=True)
    f_opt = args.f_opt
    cap = args.cap_evals

    agents = [parse_agent_spec(s) for s in args.agents]

    baseline_traj = {}
    for name in SEED5:
        if not (results / name / "seed_0" / "trace.csv").exists():
            continue  # not every task has every baseline (e.g. rule_based skips mixed-integer tasks)
        seed_ys = []
        for seed in range(5):
            rows = load_rows(results / name / f"seed_{seed}" / "trace.csv")
            if cap:
                rows = rows[:cap]
            seed_ys.append(regret_trajectory(rows, f_opt))
        min_len = min(len(a) for a in seed_ys)
        mean_regret = np.stack([a[:min_len] for a in seed_ys]).mean(axis=0)
        baseline_traj[name] = (np.arange(1, min_len + 1), mean_regret)
    for name in SINGLE:
        if not (results / name / "seed_0" / "trace.csv").exists():
            continue
        rows = load_rows(results / name / "seed_0" / "trace.csv")
        if cap:
            rows = rows[:cap]
        regret = regret_trajectory(rows, f_opt)
        baseline_traj[name] = (np.arange(1, len(rows) + 1), regret)

    agent_data = []  # list of dicts: dir, label, color, rows, eval_idx, regret, switches
    for i, (adir, label) in enumerate(agents):
        rows = load_rows(results / adir / "trace.csv")
        regret = regret_trajectory(rows, f_opt)
        eval_idx = np.array([int(r["eval_idx"]) for r in rows])
        switches, cur = [], None
        for r in rows:
            if r["optimizer"] != cur:
                switches.append(int(r["eval_idx"]))
                cur = r["optimizer"]
        agent_data.append(dict(
            dir=adir, label=label, color=AGENT_COLORS[i % len(AGENT_COLORS)],
            rows=rows, eval_idx=eval_idx, regret=regret, switches=switches,
        ))

    requested_budget = max(a["eval_idx"].max() for a in agent_data)

    # ==================== PLOT 1: counterfactual bar chart ====================
    # Not every task has every baseline (e.g. rule_based skips mixed-integer
    # tasks), so only include bars for baselines actually present.
    _PLOT1_BASELINES = [
        ("cma_es", "CMA-ES alone", AQUA, True),
        ("lbfgsb", "L-BFGS-B alone", YELLOW, False),
        ("bobyqa", "BOBYQA alone", MAGENTA, False),
        ("rule_based", "Rule-based router", BROWN, True),
    ]
    final_regret = {
        name: baseline_traj[name][1][-1]
        for name, _, _, _ in _PLOT1_BASELINES
        if name in baseline_traj
    }
    for a in agent_data:
        final_regret[a["dir"]] = a["regret"][-1]

    def _baseline_budget(name: str) -> int:
        # Evals actually shown for this baseline (post-cap, if any), from its
        # own trajectory -- not assumed/hardcoded, since different tasks in
        # this project use different budgets per optimizer.
        return int(baseline_traj[name][0][-1])

    present_baselines = [name for name, _, _, _ in _PLOT1_BASELINES if name in baseline_traj]
    order = [a["dir"] for a in agent_data] + present_baselines
    labels = {a["dir"]: f"{a['label']} (used {len(a['rows'])}/{requested_budget})" for a in agent_data}
    labels.update({
        name: f"{disp} (budget {_baseline_budget(name)}" + (", mean of 5 seeds)" if seeded else ")")
        for name, disp, _, seeded in _PLOT1_BASELINES
        if name in baseline_traj
    })
    colors = {a["dir"]: a["color"] for a in agent_data}
    colors.update({name: color for name, _, color, _ in _PLOT1_BASELINES if name in baseline_traj})

    values = [max(final_regret[k], 1e-12) for k in order]
    y_pos = np.arange(len(order))[::-1]

    fig, ax = plt.subplots(figsize=(8.5, 3.2 + 0.4 * len(order)), dpi=150)
    ax.barh(y_pos, values, height=0.5, color=[colors[k] for k in order], zorder=3)
    ax.set_xscale("log")
    # With all values pinned near the 1e-12 floor (e.g. every method reaches
    # the true optimum), log-scale auto-ticking can generate dozens of
    # overlapping minor-decade labels. Capping numticks keeps it readable.
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
    ax.set_xlim(right=max(values) * 30)
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
        ax.text(max(v, 1e-12) * 1.3, yi, label, va="center", ha="left", fontsize=9.5, color=TEXT_SECONDARY)
    ax.set_title(
        f"Claude agent(s) vs. the single-method counterfactuals they cascaded through ({args.title})",
        fontsize=11.5, color=TEXT_PRIMARY, pad=14, loc="left",
    )
    fig.tight_layout()
    out1 = figures / "plot1_counterfactual_bar.png"
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)

    # ==================== PLOT 2: regret vs. evaluation (all agents together) ====================
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    for name, (x, y) in baseline_traj.items():
        y_c = np.clip(y, 1e-12, None)
        ax.plot(x, y_c, linewidth=1.6, alpha=0.9, label=baseline_display_names[name], zorder=2, **baseline_style[name])
    for a in agent_data:
        y_c = np.clip(a["regret"], 1e-12, None)
        ax.plot(a["eval_idx"], y_c, color=a["color"], linewidth=2.6, label=a["label"], zorder=3)
        for s in a["switches"]:
            idx = s - 1
            if 0 <= idx < len(y_c):
                ax.plot(a["eval_idx"][idx], y_c[idx], marker="o", markersize=6, color=a["color"],
                         markeredgecolor=SURFACE, markeredgewidth=1.3, zorder=4)
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
        f"Regret vs. evaluation -- all Claude agent variants against every baseline ({args.title})",
        fontsize=12, color=TEXT_PRIMARY, pad=14, loc="left",
    )
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles, labels_, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=9.5, labelcolor=TEXT_PRIMARY, title="Paired hue = same family\n(solid vs. dashed)", title_fontsize=8.5)
    fig.text(
        0.0, -0.03,
        "Dots mark each agent's method-tag switches. Baselines shorter than 2000 evals (gp_bo, bobyqa) simply end early.",
        fontsize=8, color=TEXT_MUTED,
    )
    fig.tight_layout()
    out2 = figures / "plot2_regret_vs_eval.png"
    fig.savefig(out2, bbox_inches="tight")
    plt.close(fig)

    # ==================== PLOT 3: which method each lane was running ====================
    timeline_lanes = list(agent_data)
    for spec in args.timeline_extra:
        path_part, label = parse_agent_spec(spec)
        rows = load_rows(results / path_part / "trace.csv")
        timeline_lanes.append(dict(label=label, rows=rows))

    NARROW_THRESHOLD_EVALS = max(120, requested_budget // 15)
    n_lanes = len(timeline_lanes)
    fig, ax = plt.subplots(figsize=(11, 1.6 * n_lanes + 1.0), dpi=150)

    for lane, a in enumerate(timeline_lanes):
        y0 = n_lanes - 1 - lane
        phases = phases_for(a["rows"])
        for raw_label, start, end in phases:
            canon = canonical_method(raw_label)
            color = baseline_style.get(canon, {}).get("color", MUTED_GRAY)
            display = baseline_display_names.get(canon, raw_label)
            width = end - start + 1
            ax.broken_barh([(start, width)], (y0 - 0.32, 0.64), facecolors=color, edgecolors=SURFACE, linewidth=1.5, zorder=2)
            center = start + width / 2
            if width >= NARROW_THRESHOLD_EVALS:
                ax.text(center, y0, display, ha="center", va="center", fontsize=8.5, color="white", fontweight="medium", zorder=3)
            else:
                label_y = y0 + 0.62
                ax.plot([center, center], [y0 + 0.32, label_y - 0.05], color=TEXT_MUTED, linewidth=1, zorder=2)
                ax.text(center, label_y, display, ha="center", va="bottom", fontsize=8, color=TEXT_PRIMARY, zorder=3)

    ax.set_xlim(0, requested_budget)
    ax.set_ylim(-0.85, n_lanes - 1 + 0.85)
    ax.set_yticks(list(range(n_lanes)))
    ax.set_yticklabels([a["label"] for a in reversed(timeline_lanes)], fontsize=10, color=TEXT_PRIMARY)
    ax.set_xlabel("Evaluation", fontsize=10)
    ax.grid(True, which="major", axis="x", color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"Which method each agent/policy was running, by evaluation ({args.title})",
                 fontsize=12, color=TEXT_PRIMARY, pad=14, loc="left")
    fig.text(0.0, -0.06 - 0.02 * n_lanes, "Segment color matches that method's line in plot2_regret_vs_eval.png.",
              fontsize=8, color=TEXT_MUTED)
    fig.tight_layout()
    out3 = figures / "plot3_agent_method_timeline.png"
    fig.savefig(out3, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out1}")
    print(f"Wrote {out2}")
    print(f"Wrote {out3}")


if __name__ == "__main__":
    main()
