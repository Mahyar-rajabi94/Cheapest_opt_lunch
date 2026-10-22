"""Plot saved high-pressure CH4 uptake results without running an evaluator.

Run from the workspace with:
    .venv/bin/python scripts/plot_ch4_uptake_results_linux.py

The runner documents y as negative predicted uptake. At each evaluation the
plotted value is the arithmetic mean of -y_best_so_far over recorded seeds.
Only evaluation indices shared by every seed of a method are plotted; runs
are never extrapolated. Single-run methods use their own trace. Plot the saved
baseline methods that are available; an optional Agent run is read from the
--agent-method directory. Metadata must identify the CH4 task.
Both nested ch4_uptake_d188_i1 results and the flat task_062 layout are supported.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mof-ch4-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "ch4_uptake_d188_i1"
TASK_IDS = {TASK_NAME, "task_062"}
# Paired hues and solid/dashed styles follow the supplied reference figure.
METHODS = (
    ("random_search", "Random search", "#1BBC8D", "-"),
    ("sobol", "Sobol", "#1BBC8D", "--"),
    ("cma_es", "CMA-ES", "#F59E0B", "-"),
    ("de", "Differential evolution", "#F59E0B", "--"),
    ("turbo", "TuRBO", "#F585B2", "-"),
    ("gp_bo", "GP-BO", "#F585B2", "--"),
    ("bobyqa", "BOBYQA", "#159623", "-"),
    ("nelder_mead", "Nelder-Mead", "#7052ED", "-"),
    ("lbfgsb", "L-BFGS-B", "#FC4C56", "-"),
)


def load_trace(path: Path) -> dict[int, float]:
    """Validate saved observations and return physical best uptake by index."""
    metadata = json.loads(path.with_name("meta.json").read_text())
    if metadata.get("task_name", metadata.get("task_id")) not in TASK_IDS:
        raise ValueError(f"Expected CH4 uptake metadata ({sorted(TASK_IDS)}): {path}")
    if metadata["task_direction"] != "min":
        raise ValueError(f"Expected the documented negative-uptake objective: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len(rows) != metadata["evaluations_used"]:
        raise ValueError(f"Trace length does not match metadata: {path}")

    indices = np.array([int(row["eval_idx"]) for row in rows])
    raw = np.array([float(row["y"]) for row in rows])
    best = np.array([float(row["y_best_so_far"]) for row in rows])
    if not np.array_equal(indices, np.arange(1, len(rows) + 1)):
        raise ValueError(f"Missing or duplicate evaluation indices: {path}")
    if not np.isfinite(raw).all() or not np.isfinite(best).all():
        raise ValueError(f"Non-finite observations: {path}")
    if not np.allclose(best, np.minimum.accumulate(raw), rtol=1e-10, atol=1e-10):
        raise ValueError(f"Saved running best is inconsistent with observations: {path}")
    return dict(zip(indices.tolist(), (-best).tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT.parent / "mof_results" / "task_062"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Figure destination (default: <results-dir>/figures)",
    )
    parser.add_argument("--methods", nargs="+", help="Plot only these method directories")
    parser.add_argument(
        "--agent-method", default="agent",
        help="Optional Agent directory name (also discovers agent_task_062)",
    )
    parser.add_argument("--agent-label", help="Agent legend label (task_062 uses the recorded GPT model)")
    args = parser.parse_args()
    if args.agent_method in {method[0] for method in METHODS}:
        parser.error("--agent-method must differ from baseline method names")
    task_dir = args.results_dir / TASK_NAME
    if not task_dir.is_dir():
        task_dir = args.results_dir
    output_dir = args.output_dir if args.output_dir is not None else args.results_dir / "figures"
    if not task_dir.is_dir():
        parser.error(f"No saved CH4 results directory: {task_dir}")

    traces = {}
    discovered = {
        method_dir.name
        for method_dir in task_dir.iterdir()
        if method_dir.is_dir()
        and next(method_dir.rglob("trace.csv"), None) is not None
    }
    agent_method = args.agent_method
    if agent_method == "agent" and "agent" not in discovered:
        agent_method = next(
            (name for name in ("agent_task_062",) if name in discovered),
            agent_method,
        )
    agent_label = args.agent_label or (
        "Agent (OpenAI, GPT-6.1 Sol Ultra)" if agent_method == "agent_task_062" else "Agent"
    )
    methods = (*METHODS, (agent_method, agent_label, "#FF6434", "-"))
    selected = set(args.methods) if args.methods is not None else discovered
    known = {method[0] for method in methods}
    if not selected:
        parser.error(f"No saved traces found in {task_dir}")
    if selected - known:
        parser.error(f"Unknown methods: {sorted(selected - known)}; use --agent-method for an Agent run")
    if selected - discovered:
        parser.error(f"No saved traces for methods: {sorted(selected - discovered)}")
    methods = tuple(method for method in methods if method[0] in selected)
    for key, _, _, _ in methods:
        paths = sorted((task_dir / key).rglob("trace.csv"))
        traces[key] = [load_trace(path) for path in paths]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(17, 8.7), facecolor="white")
    fig.subplots_adjust(left=0.075, right=0.69, top=0.86, bottom=0.17)
    handles = []
    for key, label, color, linestyle in methods:
        runs = traces[key]
        indices = sorted(set.intersection(*(set(run) for run in runs)))
        values = np.array([[run[index] for index in indices] for run in runs])
        mean = values.mean(axis=0)
        is_agent = key == agent_method
        width = 3.6 if is_agent else 2.1
        ax.step(
            indices, mean, where="post", color=color, linestyle=linestyle,
            linewidth=width, zorder=5 if is_agent else 3,
        )
        # End markers identify the short recorded traces without extending them.
        if indices[-1] < 2000:
            ax.scatter(indices[-1], mean[-1], s=22, color=color, zorder=6)
        count = len(runs)
        legend_label = label if is_agent else f"{label} ({count} {'seed' if count == 1 else 'seeds'})"
        handles.append(Line2D(
            [0], [0], color=color, linestyle=linestyle, linewidth=width,
            label=legend_label,
        ))
        print(f"{label}: {count} run(s), evaluations {indices[0]}–{indices[-1]}, mean best uptake {mean[-1]:.6f} mol/kg")

    ax.set_xlabel("Evaluation", labelpad=10)
    ax.set_ylabel("Best CH₄ uptake at HP (mol/kg)", labelpad=12)
    last_evaluation = max(max(run) for runs in traces.values() for run in runs)
    ax.set_xlim(0, last_evaluation)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(
        MultipleLocator(250) if last_evaluation >= 1000
        else MaxNLocator(nbins=8, integer=True)
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.tick_params(axis="both", length=4, pad=6)
    ax.set_axisbelow(True)
    ax.grid(True, color="#DDDDDD", linewidth=0.9)
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.025, 0.5),
        frameon=False, fontsize=12, handlelength=2.6, handletextpad=0.9,
        labelspacing=1.0, borderaxespad=0,
        title="Optimization method", title_fontsize=13,
        alignment="left",
    )
    fig.text(
        0.075, 0.935, "Best CH₄ uptake at HP vs. evaluation",
        fontsize=22, ha="left", va="center",
    )
    fig.text(
        0.075, 0.89, "Mean best-so-far uptake across seeds · higher is better",
        fontsize=13, color="#606060", ha="left", va="center",
    )
    curve_description = (
        "Each curve shows the mean of the recorded best-so-far uptake at each evaluation; the Agent has one run.\n"
        if agent_method in traces
        else "Each curve shows the mean of the recorded best-so-far uptake at each evaluation across saved runs.\n"
    )
    fig.text(
        0.075, 0.065,
        curve_description + "Curves end at their last recorded evaluation; evaluation budgets differ across methods.",
        fontsize=10.5, color="#737373", ha="left", va="center", linespacing=1.6,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        path = output_dir / f"best_ch4_uptake_vs_evaluation.{extension}"
        fig.savefig(path, dpi=240)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
