"""Shared style for publication figures; the rules are written out in main/FIGURES.md.

Import this module before creating any figure: importing it applies the
rcParams. Import it after any other plotting module (e.g. make_comparison_plots,
which sets its own gray-text rcParams on import). A figure script then only
chooses tasks, series and titles:

    import figure_style as fs
    fig, axes = plt.subplots(1, 2, figsize=fs.FIGSIZE_1x2)
    fs.plot_series(ax, trajs, budget, fs.SERIES_COLORS[0], "Gradient Based")
    fs.style_axes(ax, "Smooth", budget, ylabel=True)
    fs.finish(fig, axes, out_path)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- color ----
# Series hues in fixed order: series i always gets SERIES_COLORS[i]. The pair
# passes the dataviz skill's validate_palette.py (--mode light --surface
# #fcfcfb) on every check. A figure with more series must extend this list and
# re-run the validator on the full ordered list first.
SERIES_COLORS = ["#2a78d6", "#e34948"]  # blue, red
BAND_ALPHA = 0.18
TEXT = "#000000"       # every piece of text
AXIS = "#52514e"       # spines and tick marks
SURFACE = "#fcfcfb"

# ---- sizes (points) ----
TITLE_SIZE = 20
LABEL_SIZE = 13
TICK_SIZE = 12
LEGEND_SIZE = 12
LINE_WIDTH = 3
SPINE_WIDTH = 1.2
FIGSIZE_1x2 = (11, 4.2)
DPI = 200

FLOOR = 1e-12  # regret 0 is drawn at this floor on the log axis

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.labelcolor": TEXT,
    "ytick.labelcolor": TEXT,
    "axes.grid": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def plot_series(ax, trajs, budget, color, label):
    """One method's regret curve on one task.

    Several seeds: mean of log10(regret) (a geometric mean) with a +/- 1
    standard-error band, both in log space so the band is positive and
    symmetric on the log axis. One seed (a deterministic method): that run.
    A run that stops itself before the budget is held flat at its best value.
    """
    stacked = np.clip(np.stack([np.pad(t[:budget], (0, budget - len(t[:budget])), mode="edge") for t in trajs]), FLOOR, None)
    x = np.arange(1, budget + 1)
    if len(trajs) > 1:
        logs = np.log10(stacked)
        mean, se = logs.mean(axis=0), logs.std(axis=0, ddof=1) / np.sqrt(len(trajs))
        ax.fill_between(x, 10 ** (mean - se), 10 ** (mean + se), color=color, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(x, 10 ** mean, color=color, linewidth=LINE_WIDTH, label=label)
    else:
        ax.plot(x, stacked[0], color=color, linewidth=LINE_WIDTH, label=label)


def style_axes(ax, title, budget, ylabel):
    """Log regret vs. budget; y label only on the leftmost panel of a row."""
    ax.set_yscale("log")
    ax.set_xlim(0, budget)
    ax.set_xlabel("Budget", fontsize=LABEL_SIZE)
    if ylabel:
        ax.set_ylabel("Regret", fontsize=LABEL_SIZE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(SPINE_WIDTH)
    ax.tick_params(which="both", color=AXIS, labelsize=TICK_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)


def finish(fig, axes, out_path):
    """One shared legend centered below the panels, then save."""
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=LEGEND_SIZE, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(w_pad=4, rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
