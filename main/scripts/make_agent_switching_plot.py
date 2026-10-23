"""Agent method-switching figure: what each switch between optimizer families
bought the agent on rosenbrock_d10_i1 (Astra max campaign, 2026-10-22).

The agent ran four kinds of stage, grouped here by method family from the
trace's optimizer labels:

  Evolutionary   cma_es                                       evals    1-401
  Model Based    bobyqa, bobyqa_full                           evals  402-820
  Gradient Based lbfgsb, fd_newton_trust, precond_bfgs,
                 bfgs_restart, central_bfgs                    evals 821-1459, 1556-1859
  Basin Check    basin_probe (perturbations + full-box design) evals 1460-1555

Path panel: a real 2-D slice of the objective (the task's own function, not a
schematic). The plane passes through three of the agent's incumbents: its best
point after the Evolutionary stage, its best after the Model Based stage, and
its final best (the origin). Those three lie exactly on the slice; every other
evaluated point is orthogonally projected onto it, so its position is exact in
the plane but its distance from the plane is not shown. Dots are every
evaluated point, colored by stage. Lines trace the incumbent (best-so-far
point) through every stage. The Evolutionary line zigzags because each new
incumbent comes from a fresh population sample that can land far away: broad
coverage, but no steady descent. Lines carry a white outline so they stay
visible on dark regions. Background: log10 regret on the plane as filled
contour bands, white (low) to near-black (high), with band edges spread over
the 2nd-98th percentile of the slice so the valley shape is not washed out.

Progress panel: the agent's best-so-far regret vs. budget, colored by stage.

Style follows FIGURES.md via figure_style.py. Stage colors (red, yellow, blue)
pass validate_palette.py on all pairs (--mode light --surface #ffffff, --pairs
all); yellow is under 3:1 contrast, relieved by the legend labels. Basin Check
is neutral gray, not a categorical hue.

Usage (from main/):
    python scripts/make_agent_switching_plot.py
    # writes results_linux/agent_switching_rosenbrock.png and
    # results_linux/legend_rosenbrock_stages.png: the stage legend on its own,
    # one entry per row, shared by the agent and single-optimizer figures
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import numpy as np

MAIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN / "scripts"))
from src.tasks.registry import TASKS  # noqa: E402
import figure_style as fs  # noqa: E402  (applies the shared rcParams)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RUN = MAIN / "agent_runs/astra_max_2000_20261022/analysis/rosenbrock_d10_i1"
TASK_KEY = "rosenbrock10"
F_OPT_META = MAIN / "results_full/rosenbrock_d10_i1/cma_es/seed_0/meta.json"
OUT = MAIN / "results_linux/agent_switching_rosenbrock.png"
OUT_LEGEND = MAIN / "results_linux/legend_rosenbrock_stages.png"

# (legend label, color, optimizer labels in the trace)
STAGES = [
    ("Evolutionary", "#e34948", {"cma_es"}),
    ("Model Based", "#eda100", {"bobyqa", "bobyqa_full"}),
    ("Gradient Based", "#2a78d6", {"lbfgsb", "fd_newton_trust", "precond_bfgs", "bfgs_restart", "central_bfgs"}),
    ("Basin Check", "#898781", {"basin_probe"}),
]
STAGE_OF = {label: i for i, (_, _, labels) in enumerate(STAGES) for label in labels}
GRID_N = 300
BACKGROUND = LinearSegmentedColormap.from_list("gray_ramp", ["#ffffff", "#262626"])
N_BANDS = 14
HALO = [pe.Stroke(linewidth=fs.LINE_WIDTH + 2.5, foreground="#ffffff"), pe.Normal()]
REGRET_FLOOR = 1e-12
# One Progress y-range for the agent and every single-optimizer figure on this
# task (make_single_optimizer_plot.py), so their panels can be compared directly.
# It spans the agent's 0 regret (drawn at fs.FLOOR) up to the highest starting
# regret among those runs (CMA-ES seed 1, ~4e5).
PROGRESS_YLIM = (1e-13, 1e6)


def load_run():
    rows = list(csv.DictReader(open(RUN / "trace.csv")))
    X = np.array([json.loads(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    stage = np.array([STAGE_OF[r["optimizer"]] for r in rows])
    f_opt = json.loads(F_OPT_META.read_text())["task_f_opt"]
    return X, y - f_opt, stage


def plane_basis(X, regret, stage):
    """Origin = final best; axes span the best points after stages 0 and 1."""
    best = lambda mask: np.flatnonzero(mask)[regret[mask].argmin()]
    origin = X[regret.argmin()]
    a = X[best(stage == 0)] - origin
    b = X[best(np.isin(stage, (0, 1)))] - origin
    u = a / np.linalg.norm(a)
    v = b - (b @ u) * u
    return origin, u, v / np.linalg.norm(v)


def incumbent_path(regret):
    """Indices where the best-so-far value improved."""
    return np.flatnonzero(regret < np.minimum.accumulate(np.r_[np.inf, regret[:-1]]))


def slice_geometry(X, regret, stage, f_opt):
    """The agent's plane (see plane_basis) and the objective evaluated on a grid in it.

    Any run on this task can be drawn on the same slice with project_run, so
    figures for different optimizers share one landscape.
    """
    task = TASKS[TASK_KEY]()
    origin, u, v = plane_basis(X, regret, stage)
    P = (X - origin) @ np.stack([u, v]).T
    inc = incumbent_path(regret)
    lo, hi = P[inc].min(axis=0), P[inc].max(axis=0)
    pad = 0.15 * (hi - lo)
    g1, g2 = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], GRID_N),
                         np.linspace(lo[1] - pad[1], hi[1] + pad[1], GRID_N))
    pts = origin + g1[..., None] * u + g2[..., None] * v
    Z = np.array([task._f(p) for p in pts.reshape(-1, X.shape[1])]).reshape(g1.shape) - f_opt
    logZ = np.log10(np.maximum(Z, REGRET_FLOOR))
    levels = np.linspace(*np.percentile(logZ, [2, 98]), N_BANDS + 1)
    return dict(origin=origin, basis=np.stack([u, v]), g1=g1, g2=g2, logZ=logZ, levels=levels)


def project_run(geo, X, regret, stage):
    """A run's points on the slice plus what the panels need to draw it."""
    return dict(P=(X - geo["origin"]) @ geo["basis"].T, inc=incumbent_path(regret), regret=regret, stage=stage)


def draw_path(ax, geo, run, upto, stages=STAGES):
    """Background slice plus every point and incumbent among the first `upto` evaluations."""
    P, stage, g1, g2 = run["P"], run["stage"], geo["g1"], geo["g2"]
    ax.contourf(g1, g2, geo["logZ"], levels=geo["levels"], cmap=BACKGROUND, extend="both")
    ax.contour(g1, g2, geo["logZ"], levels=geo["levels"], colors="#ffffff", linewidths=0.4, alpha=0.6)
    shown = np.arange(len(P)) < upto
    in_view = (P[:, 0] >= g1.min()) & (P[:, 0] <= g1.max()) & (P[:, 1] >= g2.min()) & (P[:, 1] <= g2.max())
    for i, (_, color, _) in enumerate(stages):
        m = (stage == i) & in_view & shown
        ax.scatter(P[m, 0], P[m, 1], s=12, color=color, alpha=0.7, linewidths=0, zorder=2)
    # One polyline per run of consecutive incumbents from the same stage, so the
    # white outline is drawn once around the run rather than around every step.
    # Each run starts at the previous run's last incumbent to keep the path joined.
    inc = run["inc"][run["inc"] < upto]
    runs = np.split(np.arange(len(inc)), np.flatnonzero(np.diff(stage[inc])) + 1) if len(inc) else []
    for r in runs:
        idx = inc[np.r_[max(r[0] - 1, 0), r]]
        ax.plot(P[idx, 0], P[idx, 1], color=stages[stage[inc[r[0]]]][1], linewidth=fs.LINE_WIDTH,
                solid_capstyle="round", solid_joinstyle="round", path_effects=HALO, zorder=3)
    ax.set_xlim(g1.min(), g1.max())
    ax.set_ylim(g2.min(), g2.max())
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Path", fontsize=fs.TITLE_SIZE)


def draw_progress(ax, run, upto, stages=STAGES, ylim=PROGRESS_YLIM):
    """Best-so-far regret over the first `upto` evaluations, on fixed axes."""
    stage = run["stage"]
    best = np.maximum(np.minimum.accumulate(run["regret"]), fs.FLOOR)
    x = np.arange(1, len(best) + 1)
    starts = np.flatnonzero(np.r_[True, stage[1:] != stage[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(best)]):
        if a >= upto:
            break
        lo = max(a - 1, 0)  # join each stage to the previous one's last point
        ax.plot(x[lo:min(b, upto)], best[lo:min(b, upto)], color=stages[stage[a]][1], linewidth=fs.LINE_WIDTH)
    fs.style_axes(ax, "Progress", 2000, ylabel=True)
    ax.set_ylim(*ylim)


def build_figure(geo, run, upto, stages=STAGES, ylim=PROGRESS_YLIM, dpi=fs.DPI):
    fig, (ax_path, ax_prog) = plt.subplots(1, 2, figsize=fs.FIGSIZE_1x2, dpi=dpi,
                                           gridspec_kw={"width_ratios": [1, 1.25]})
    draw_path(ax_path, geo, run, upto, stages)
    draw_progress(ax_prog, run, upto, stages, ylim)
    handles = [Line2D([], [], color=color, linewidth=fs.LINE_WIDTH) for _, color, _ in stages]
    fig.legend(handles, [label for label, _, _ in stages], loc="lower center", ncol=len(stages),
               frameon=False, fontsize=fs.LEGEND_SIZE, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(w_pad=4, rect=(0, 0.06, 1, 1))
    return fig


def write_legend(out, stages=STAGES):
    """The stage legend as its own image: one entry per row, nothing else."""
    fig = plt.figure(figsize=(3, 2))
    handles = [Line2D([], [], color=color, linewidth=fs.LINE_WIDTH) for _, color, _ in stages]
    legend = fig.legend(handles, [label for label, _, _ in stages], loc="center", ncol=1,
                        frameon=False, fontsize=fs.LEGEND_SIZE)
    fig.savefig(out, dpi=fs.DPI, bbox_inches=legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).padded(0.05))
    plt.close(fig)
    print("wrote", out)


def main():
    write_legend(OUT_LEGEND)
    X, regret, stage = load_run()
    f_opt = json.loads(F_OPT_META.read_text())["task_f_opt"]
    geo = slice_geometry(X, regret, stage, f_opt)
    fig = build_figure(geo, project_run(geo, X, regret, stage), upto=len(regret))
    fig.savefig(OUT, dpi=fs.DPI, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
