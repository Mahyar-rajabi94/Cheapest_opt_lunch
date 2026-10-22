"""Plot saved CO2 uptake results without importing or running any evaluator.

Run from the workspace with:
    .venv/bin/python scripts/plot_co2_uptake_results_linux.py

The runner documents y as negative predicted uptake. At each evaluation the
plotted value is the arithmetic mean of -y_best_so_far over recorded seeds.
Only evaluation indices shared by every seed of a method are plotted; runs
are never extrapolated. Single-run methods and the optional Agent use their own trace.
The agent_task_060_codex_free run is explicitly excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mof-co2-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT.parent / "mof_results" / "task_060"
EXCLUDED_METHODS = {"agent_task_060_codex_free"}
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
    ("agent_task_060_codex", "Agent (OpenAI, GPT-6.1 Sol Ultra)", "#FF6434", "-"),
)


def load_trace(path: Path) -> dict[int, float]:
    """Validate saved observations and return physical best uptake by index."""
    metadata = json.loads(path.with_name("meta.json").read_text())
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
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Figure destination (default: <results-dir>/figures)",
    )
    args = parser.parse_args()
    task_dir = args.results_dir / "co2_uptake_d188_i1"
    output_dir = args.output_dir if args.output_dir is not None else args.results_dir / "figures"

    traces = {}
    known = {method[0] for method in METHODS}
    required = {name for name in known if not name.startswith("agent_")}
    discovered = {
        method_dir.name
        for method_dir in task_dir.iterdir()
        if method_dir.is_dir()
        and method_dir.name not in EXCLUDED_METHODS
        and next(method_dir.rglob("trace.csv"), None) is not None
    }
    if not required <= discovered or not discovered <= known:
        raise ValueError(f"Unexpected or missing methods: {(required - discovered) | (discovered - known)}")
    methods = [method for method in METHODS if method[0] in discovered]
    has_agent = any(key.startswith("agent_") for key, _, _, _ in methods)
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
        is_agent = key.startswith("agent_")
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
    ax.set_ylabel("Best CO₂ uptake at LP (mol/kg)", labelpad=12)
    ax.set_xlim(0, 2000)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MultipleLocator(250))
    ax.yaxis.set_major_locator(MultipleLocator(2))
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
    comparison_note = (
        "Each curve shows the mean of the recorded best-so-far uptake at each evaluation; the Agent has one run.\n"
        if has_agent else
        "Each curve shows the mean of the recorded best-so-far uptake at each evaluation.\n"
    )
    fig.text(
        0.075, 0.935, "Best CO₂ uptake at LP vs. evaluation",
        fontsize=22, ha="left", va="center",
    )
    fig.text(
        0.075, 0.89, "Mean best-so-far uptake across seeds · higher is better",
        fontsize=13, color="#606060", ha="left", va="center",
    )
    fig.text(
        0.075, 0.065,
        comparison_note + "Curves end at their last recorded evaluation; evaluation budgets differ across methods.",
        fontsize=10.5, color="#737373", ha="left", va="center", linespacing=1.6,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        path = output_dir / f"best_co2_uptake_vs_evaluation.{extension}"
        fig.savefig(path, dpi=240)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
