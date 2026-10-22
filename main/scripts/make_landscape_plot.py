"""Schematic loss landscapes for the two sides of the crossover figure
(make_crossover_plot.py): a 2-D slice of each objective, seen from above.

  Smooth          2-D Rosenbrock: one smooth, curved valley. A finite-difference
                  gradient always points somewhere useful.
  Mixed Integer   the same function with both inputs snapped to a grid, as
                  bbob_mixint_f8 does to 8 of its 10 inputs: flat tiles
                  separated by jumps. The gradient is zero on every tile.

These are illustrations of the landscape's shape, not data: the real tasks are
10-D, and the grid spacing here (0.25) is chosen to make the tiles visible.
Color is log10(1 + f) on one shared scale for both panels (light = low loss),
so the valley floor is not flattened by the walls. Style follows FIGURES.md
where it applies (black text, 20 pt titles, no gridlines); the color is a
single-hue sequential ramp from the standard's blue. No axes: the coordinates
of a schematic carry no meaning.

Usage (from main/):
    python scripts/make_landscape_plot.py    # writes results_linux/landscape_{smooth,mixed_integer,both}.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style as fs  # noqa: E402  (applies the shared rcParams)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results_linux"
GRID_STEP = 0.25
X1, X2 = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-1, 3, 400))
# Light-to-dark ramp of the standard's first series color: low loss light, high loss dark.
RAMP = LinearSegmentedColormap.from_list("blue_ramp", ["#eaf2fc", fs.SERIES_COLORS[0], "#0d2a4d"])


def rosenbrock(x1, x2):
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2


def snap(x):
    return np.round(x / GRID_STEP) * GRID_STEP


PANELS = [  # (slug, title, log loss)
    ("smooth", "Smooth", np.log10(1 + rosenbrock(X1, X2))),
    ("mixed_integer", "Mixed Integer", np.log10(1 + rosenbrock(snap(X1), snap(X2)))),
]
NORM = Normalize(min(z.min() for *_, z in PANELS), max(z.max() for *_, z in PANELS))  # shared color scale


def draw_panel(ax, title, z):
    ax.imshow(z, origin="lower", extent=(X1.min(), X1.max(), X2.min(), X2.max()),
              cmap=RAMP, norm=NORM, interpolation="nearest")
    ax.set_axis_off()  # schematic: no ticks, spines or gridlines
    ax.set_title(title, fontsize=fs.TITLE_SIZE)


def main():
    for slug, title, z in PANELS:
        fig, ax = plt.subplots(figsize=(5, 5))
        draw_panel(ax, title, z)
        fig.savefig(RESULTS / f"landscape_{slug}.png", dpi=fs.DPI, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, len(PANELS), figsize=fs.FIGSIZE_1x2)
    for ax, (_, title, z) in zip(axes, PANELS):
        draw_panel(ax, title, z)
    fig.tight_layout(w_pad=4)
    fig.savefig(RESULTS / "landscape_both.png", dpi=fs.DPI, bbox_inches="tight")
    print("wrote", RESULTS / "landscape_both.png")


if __name__ == "__main__":
    main()
