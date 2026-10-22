"""Animated regret-vs-evaluation GIF replicating the exact visual style of
a reference "mean incumbent regret" chart: transparent background, bold
headline, a live "Evaluation N / Budget" counter, a dashed vertical
cursor at the current evaluation, thick step-style lines with a leading
dot, and a top-right legend.

Usage
-----
    python scripts/plot_mean_incumbent_style.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from io import BytesIO
from PIL import Image


def _grab_frame_fixed(self, **savefig_kwargs):
    """matplotlib 3.10.0's PillowWriter.grab_frame appends a transparent
    frame to a nonexistent self._frame (typo for self._frames), raising
    AttributeError and silently dropping every transparent frame. Fixed
    here so transparent-background GIFs actually work."""
    buf = BytesIO()
    self.fig.savefig(buf, **{**savefig_kwargs, "format": "rgba", "dpi": self.dpi})
    im = Image.frombuffer("RGBA", self.frame_size, buf.getbuffer(), "raw", "RGBA", 0, 1)
    if im.getextrema()[3][0] < 255:
        self._frames.append(im)
    else:
        self._frames.append(im.convert("RGB"))


PillowWriter.grab_frame = _grab_frame_fixed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from plot_dionysus_vs_agents import baseline_trajectory  # noqa: E402

INK = "#172334"       # title + agent line
GRIDLINE = "#eceef1"
CURSOR_GRAY = "#b8bfc8"
AXIS_GRAY = "#4a5160"


def load_rows(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def agent_curve(task: str, agent_dir: str, f_opt: float, budget: int):
    rows = load_rows(ROOT / "results" / f"dionysus_{task}" / agent_dir / "trace.csv")
    y = np.array([float(r["y_best_so_far"]) for r in rows]) - f_opt
    eval_idx = np.array([int(r["eval_idx"]) for r in rows])
    keep = eval_idx <= budget
    return eval_idx[keep], np.maximum(y[keep], 1e-9)


def main():
    task = "delaney"
    f_opt = -11.6
    budget = 100
    agent_dir = "claude_agent_run_hptuning_tweak"
    series = [
        ("gp", "mordred", "GP (mordred features)", "#da8332"),
        ("gp", "mfp", "GP (fingerprint features)", "#8e5eb3"),
    ]

    data_dir = ROOT / "data" / "dionysus" / task
    ax_curve, ay_curve = agent_curve(task, agent_dir, f_opt, budget)

    baseline_curves = []
    for model, feature, label, color in series:
        evals, regret = baseline_trajectory(data_dir, model, feature, f_opt, "minimize")
        keep = evals <= budget
        baseline_curves.append((evals[keep], np.maximum(regret[keep], 1e-9), label, color, 30))

    y_max = max(ay_curve.max(), max(c[1].max() for c in baseline_curves)) * 1.08

    fig = plt.figure(figsize=(12.8, 7.6), dpi=100, facecolor="white")
    ax = fig.add_axes([0.085, 0.12, 0.856, 0.74])
    ax.set_facecolor("white")

    fig.text(0.085, 0.934, "Low-solubility molecule discovery",
              fontsize=22, fontweight="bold", color=INK, ha="left", va="top")
    counter_text = fig.text(0.945, 0.934, "", fontsize=13, fontweight="bold", color=INK,
                             ha="right", va="top")

    ax.set_xlim(0, budget)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Budget", fontsize=16, color=AXIS_GRAY)
    ax.set_ylabel("Regret", fontsize=16, color=AXIS_GRAY)
    ax.tick_params(labelsize=14, colors=AXIS_GRAY)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1.0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#c7ccd3")
    ax.spines["bottom"].set_color("#c7ccd3")

    (cursor_line,) = ax.plot([], [], color=CURSOR_GRAY, linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)

    (agent_line,) = ax.plot([], [], color=INK, linewidth=3.4, zorder=5,
                             label="Agent")
    (agent_dot,) = ax.plot([], [], "o", color=INK, markersize=5.5, zorder=6)

    base_lines, base_dots = [], []
    for _, _, label, color, n in baseline_curves:
        (ln,) = ax.plot([], [], color=color, linewidth=2.2, zorder=4, label=label)
        (dot,) = ax.plot([], [], "o", color=color, markersize=4.5, zorder=4.5)
        base_lines.append(ln)
        base_dots.append(dot)

    legend = ax.legend(loc="upper right", frameon=False, fontsize=11, labelcolor=INK,
                        handlelength=1.8, bbox_to_anchor=(1.0, 1.0))

    frames = list(range(1, budget + 1))

    def update(budget_now):
        a_mask = ax_curve <= budget_now
        agent_line.set_data(ax_curve[a_mask], ay_curve[a_mask])
        if a_mask.any():
            agent_dot.set_data([ax_curve[a_mask][-1]], [ay_curve[a_mask][-1]])
        for (bx, by, *_), ln, dot in zip(baseline_curves, base_lines, base_dots):
            b_mask = bx <= budget_now
            ln.set_data(bx[b_mask], by[b_mask])
            if b_mask.any():
                dot.set_data([bx[b_mask][-1]], [by[b_mask][-1]])
        cursor_line.set_data([budget_now, budget_now], [0, y_max])
        counter_text.set_text(f"Evaluation {budget_now} / {budget}")
        return agent_line, agent_dot, cursor_line, counter_text, *base_lines, *base_dots

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    outdir = ROOT.parent / "slide_figures" / "dionysus"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{task}_mean_incumbent_regret.gif"
    anim.save(out_path, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
