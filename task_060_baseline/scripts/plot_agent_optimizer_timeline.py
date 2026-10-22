"""Plot the recorded OpenAI Agent optimizer phases without running an evaluator.

Run with .venv/bin/python scripts/plot_agent_optimizer_timeline.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import groupby
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
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT.parent / "mof_results" / "task_060"
# Baseline optimizers retain the colors from the uptake comparison. Coordinate
# search variants share green; custom probes and mutation have distinct colors.
STYLES = {
    "sobol_search": ("Sobol", "#1BBC8D", "white"),
    "lbfgsb": ("L-BFGS-B", "#FC4C56", "white"),
    "cma_es": ("CMA-ES", "#F59E0B", "#202020"),
    "sobol_explore": ("Sobol exploration", "#1BBC8D", "white"),
    "response_probe": ("Local response probes", "#64748B", "white"),
    "coordinate_poll": ("Coordinate polling", "#159623", "white"),
    "sparse_mutation": ("Sparse\nmutation", "#7052ED", "white"),
    "subspace_cma": ("Subspace\nCMA-ES", "#F59E0B", "#202020"),
    "coordinate_refine": ("Coordinate refinement", "#159623", "white"),
    "coordinate_endpoint": ("Upper-endpoint\ntrials", "#159623", "white"),
    "coordinate_midpoint": ("Midpoint\ntrials", "#159623", "white"),
    "coordinate_levels": ("Targeted\nlevels", "#159623", "white"),
}


@dataclass(frozen=True)
class Phase:
    label: str
    start: int
    end: int

    @property
    def count(self) -> int:
        return self.end - self.start + 1

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2

    @property
    def interval(self) -> str:
        return f"{self.start:,}–{self.end:,}"


def load_phases(trace: Path) -> list[Phase]:
    with trace.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    metadata = json.loads(trace.with_name("meta.json").read_text())
    if not rows or len(rows) != metadata["evaluations_used"]:
        raise ValueError("Agent trace length does not match saved metadata")
    indices = [int(row["eval_idx"]) for row in rows]
    if indices != list(range(1, len(rows) + 1)):
        raise ValueError("Agent evaluations are not consecutive and unique")
    phases = []
    for label, group in groupby(rows, key=lambda row: row["optimizer"]):
        block = list(group)
        if label not in STYLES:
            raise ValueError(f"Unknown recorded Agent optimizer: {label}")
        phases.append(Phase(label, int(block[0]["eval_idx"]), int(block[-1]["eval_idx"])))
    if sum(phase.count for phase in phases) != len(rows):
        raise ValueError("Timeline does not cover every recorded evaluation")
    return phases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--output-dir", type=Path,
        help="Figure destination (default: <results-dir>/figures)",
    )
    args = parser.parse_args()
    trace = args.results_dir / "co2_uptake_d188_i1" / "agent_task_060_codex" / "trace.csv"
    phases = load_phases(trace)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(19, 5.3), facecolor="white")
    fig.subplots_adjust(left=0.18, right=0.985, top=0.81, bottom=0.28)
    bar_bottom, bar_height = 0.0, 0.68
    ax.set_xlim(0, phases[-1].end + 0.5)
    ax.set_ylim(-0.34, 1.65)
    ax.set_yticks([bar_bottom + bar_height / 2])
    ax.set_yticklabels(["Agent (OpenAI,\nGPT-6.1 Sol Ultra)"], fontsize=13, linespacing=1.5)
    ax.tick_params(axis="y", length=0, pad=18)
    ax.tick_params(axis="x", labelsize=12, length=5, pad=9, colors="#666666")
    ax.xaxis.set_major_locator(MultipleLocator(250))
    ax.spines["bottom"].set_color("#B8B8B8")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#DDDDDD", linewidth=1)
    ax.set_xlabel("Evaluation", fontsize=14, labelpad=12)

    for phase in phases:
        label, color, text_color = STYLES[phase.label]
        # Half-integer edges put each integer evaluation inside its phase and
        # retain exact contiguous coverage, including the first and final call.
        ax.broken_barh(
            [(phase.start - 0.5, phase.count)], (bar_bottom, bar_height),
            facecolors=color, edgecolors="white", linewidth=2, zorder=3,
        )
        if phase.label in {"sobol_search", "sobol_explore"}:
            ax.annotate(
                f"{label}\n{phase.interval}",
                xy=(phase.center, bar_bottom + bar_height),
                xytext=(15, 1.27), ha="left", va="bottom", fontsize=11.5,
                linespacing=1.5, annotation_clip=False,
                arrowprops={"arrowstyle": "-", "color": "#808080", "lw": 1.1},
            )
        elif phase.label in {"lbfgsb", "response_probe"}:
            ax.annotate(
                f"{label}\n{phase.interval}",
                xy=(phase.center, bar_bottom + bar_height),
                xytext=(205, 0.83), ha="left", va="bottom", fontsize=11.5,
                linespacing=1.5, annotation_clip=False,
                arrowprops={"arrowstyle": "-", "color": "#808080", "lw": 1.1},
            )
        else:
            font_size = 12 if phase.count >= 500 else 10
            ax.text(
                phase.center, bar_bottom + bar_height / 2,
                f"{label}\n{phase.interval}", ha="center", va="center",
                color=text_color, fontsize=font_size, linespacing=1.4,
                zorder=4,
            )
        print(f"{phase.interval}: {phase.label} ({phase.count} evaluations)")

    fig.text(
        0.18, 0.935, "Which optimizer the Agent used, by evaluation",
        fontsize=21, ha="left", va="center",
    )
    fig.text(
        0.18, 0.08,
        "Segments follow the optimizer labels recorded in the Agent trace; evaluation ranges are inclusive.\n"
        "Colors match the optimizer comparison where applicable. Adjustments within a phase are not separate optimizer switches.",
        fontsize=10.5, color="#737373", ha="left", va="center", linespacing=1.6,
    )
    output_dir = args.output_dir if args.output_dir is not None else args.results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        path = output_dir / f"agent_optimizer_timeline.{extension}"
        fig.savefig(path, dpi=240)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
