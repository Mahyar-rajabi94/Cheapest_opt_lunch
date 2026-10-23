"""Schematic: the standard optimization loop vs. the agent-extended loop.

Schematic (no data), drawn as a design-build-test-learn style cycle: a ring of
arrow segments over a tinted panel split into one sector per step, with a line
icon in each sector. Left panel "Standard Loop": acquire data -> fit optimizer ->
propose next point -> acquire data. Right panel "Agent Loop": the same cycle
with a "Design Algorithm" step inserted between acquiring data and fitting the
optimizer, driven by an agent that draws on literature and prior knowledge
and an optimizer toolkit. Each shared step keeps its color in both panels;
everything the agent adds is red.

    ../.venv/bin/python scripts/make_agent_loop_plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Wedge

import figure_style as fs

OUT = Path(__file__).resolve().parents[1] / "results_linux" / "agent_loop.png"

BLUE, RED, AQUA, YELLOW = fs.SERIES_COLORS[:4]
DATA, FIT, PROPOSE, AGENT = BLUE, AQUA, YELLOW, RED

HALF = 2.2              # panel half-width
R_MID, R_W = 1.5, 0.28  # ring radius and thickness
GAP = 4                 # degrees between ring segments
HEAD = 13               # arrowhead length, degrees
LABEL_SIZE = 12.5
LABEL_R = 0.8
VERB_SIZE = 13
NOUN_SIZE = 11
ICON_LW = 2.4


def tint(hex_color, amount):
    """Mix a color toward white."""
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(rgb + (1 - rgb) * amount)


def polar(r, deg, c=(0, 0)):
    a = np.radians(deg)
    return c[0] + r * np.cos(a), c[1] + r * np.sin(a)


# ---- icons: line drawings in a box of half-size s around (x, y) ----

def icon_data(ax, x, y, s, color):
    """Erlenmeyer flask, partly filled, with two bubbles."""
    neck, base, top, shoulder, bottom = 0.26, 0.85, 1.0, 0.3, -0.9
    outline = [(-neck, top), (-neck, shoulder), (-base, bottom), (base, bottom), (neck, shoulder), (neck, top)]
    level = -0.3  # liquid surface, as a height on the flask's sloped walls
    half = neck + (base - neck) * (shoulder - level) / (shoulder - bottom)
    liquid = [(-half, level), (-base, bottom), (base, bottom), (half, level)]
    ax.add_patch(Polygon([(x + px * s, y + py * s) for px, py in liquid], closed=True, fc=tint(color, 0.45), ec="none"))
    ax.plot([x + px * s for px, _ in outline], [y + py * s for _, py in outline], color=color, lw=ICON_LW,
            solid_joinstyle="round", solid_capstyle="round")
    ax.plot([x - (neck + 0.14) * s, x + (neck + 0.14) * s], [y + top * s] * 2, color=color, lw=ICON_LW,
            solid_capstyle="round")
    for bx, by, br in [(-0.2, -0.55, 0.1), (0.22, -0.68, 0.07)]:
        ax.add_patch(Circle((x + bx * s, y + by * s), br * s, color="white"))


def icon_fit(ax, x, y, s, color):
    ax.plot([x - s, x - s, x + s], [y + s, y - s, y - s], color=color, lw=ICON_LW, solid_capstyle="round")
    curve = lambda t: -0.55 + 1.3 * (t + 0.8) ** 2 / 2.7
    t = np.linspace(-0.8, 0.85, 50)
    ax.plot(x + t * s, y + curve(t) * s, color=color, lw=ICON_LW)
    for dx in (-0.55, -0.05, 0.45):
        ax.add_patch(Circle((x + dx * s, y + (curve(dx) + 0.18) * s), 0.11 * s, color=color))


def icon_propose(ax, x, y, s, color):
    for r in (0.85, 0.45):
        ax.add_patch(Circle((x, y), r * s, fill=False, ec=color, lw=ICON_LW))
    ax.add_patch(Circle((x, y), 0.13 * s, color=color))
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ax.plot([x + dx * 0.6 * s, x + dx * 1.1 * s], [y + dy * 0.6 * s, y + dy * 1.1 * s], color=color, lw=ICON_LW)


def icon_design(ax, x, y, s, color):
    root = (x - 0.8 * s, y)
    leaves = [(x + 0.75 * s, y + dy * s) for dy in (0.75, 0, -0.75)]
    for i, (lx, ly) in enumerate(leaves):
        ax.plot([root[0], x - 0.05 * s, lx], [root[1], ly, ly], color=color, lw=ICON_LW, solid_joinstyle="round")
        ax.add_patch(Circle((lx, ly), 0.2 * s, fc=color if i == 0 else "white", ec=color, lw=ICON_LW, zorder=5))
    ax.add_patch(Circle(root, 0.2 * s, color=color, zorder=5))


def icon_agent(ax, x, y, s, color):
    ax.add_patch(FancyBboxPatch((x - 0.8 * s, y - 0.7 * s), 1.6 * s, 1.25 * s,
                                boxstyle=f"round,pad=0,rounding_size={0.3 * s}", fill=False, ec=color, lw=ICON_LW))
    for dx in (-0.35, 0.35):
        ax.add_patch(Circle((x + dx * s, y - 0.05 * s), 0.14 * s, color=color))
    ax.plot([x - 0.3 * s, x + 0.3 * s], [y - 0.42 * s, y - 0.42 * s], color=color, lw=ICON_LW, solid_capstyle="round")
    ax.plot([x, x], [y + 0.55 * s, y + 0.85 * s], color=color, lw=ICON_LW)
    ax.add_patch(Circle((x, y + 0.95 * s), 0.12 * s, color=color))


def icon_literature(ax, x, y, s, color):
    for side in (-1, 1):
        page = [(x, y - 0.6 * s), (x + side * 0.95 * s, y - 0.4 * s), (x + side * 0.95 * s, y + 0.75 * s), (x, y + 0.55 * s)]
        ax.add_patch(Polygon(page, closed=True, fill=False, ec=color, lw=ICON_LW, joinstyle="round"))
        for k in (0.25, 0.0, -0.25):
            ax.plot([x + side * 0.2 * s, x + side * 0.75 * s], [y + (k + 0.08) * s, y + (k + 0.15) * s],
                    color=color, lw=ICON_LW * 0.6, solid_capstyle="round")


def icon_toolkit(ax, x, y, s, color):
    """Toolbox: body, lid seam, latch and handle."""
    ax.add_patch(FancyBboxPatch((x - 0.95 * s, y - 0.7 * s), 1.9 * s, 1.2 * s,
                                boxstyle=f"round,pad=0,rounding_size={0.15 * s}", fill=False, ec=color, lw=ICON_LW))
    ax.plot([x - 0.95 * s, x + 0.95 * s], [y + 0.05 * s] * 2, color=color, lw=ICON_LW)
    ax.add_patch(FancyBboxPatch((x - 0.18 * s, y - 0.12 * s), 0.36 * s, 0.32 * s,
                                boxstyle=f"round,pad=0,rounding_size={0.06 * s}", fc=color, ec="none"))
    ax.plot([x - 0.4 * s, x - 0.4 * s, x + 0.4 * s, x + 0.4 * s], [y + 0.5 * s, y + 0.85 * s, y + 0.85 * s, y + 0.5 * s],
            color=color, lw=ICON_LW, solid_joinstyle="round")


# ---- panels ----

def ring_segment(ax, c, start, end, color):
    """Clockwise arrow along the ring from angle start to end (degrees, start > end)."""
    r_in, r_out, flare = R_MID - R_W / 2, R_MID + R_W / 2, 0.12
    body_end = end + HEAD
    a = np.linspace(start, body_end, 40)
    pts = ([polar(r_out, d, c) for d in a]
           + [polar(r_out + flare, body_end, c), polar(R_MID, end, c), polar(r_in - flare, body_end, c)]
           + [polar(r_in, d, c) for d in a[::-1]])
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="none", zorder=4))


def panel(ax, cx, title, steps):
    """steps: (center angle, label, color, icon fn, icon corner angle), listed clockwise."""
    c = (cx, 0.0)
    frame = FancyBboxPatch((cx - HALF, -HALF), 2 * HALF, 2 * HALF, boxstyle="round,pad=0,rounding_size=0.3",
                           fc="none", ec="none")
    ax.add_patch(frame)
    span = 360 / len(steps)
    for ang, label, color, icon, corner in steps:
        w = Wedge(c, 4, ang - span / 2, ang + span / 2, fc=tint(color, 0.82), ec="none", zorder=1)
        ax.add_patch(w)
        w.set_clip_path(frame)
        ring_segment(ax, c, ang + span / 2 - GAP / 2, ang - span / 2 + GAP / 2, color)
        # inside the ring, each step gets its own lighter sector holding its label
        ax.add_patch(Wedge(c, R_MID - R_W / 2 - 0.02, ang - span / 2, ang + span / 2,
                           fc=tint(color, 0.9), ec="white", lw=3, zorder=2))
        lx, ly = polar(LABEL_R, ang, c)
        verb, noun = label
        ax.text(lx, ly + 0.02, verb, ha="center", va="bottom", fontsize=VERB_SIZE, fontweight="bold", zorder=5)
        ax.text(lx, ly - 0.04, noun, ha="center", va="top", linespacing=1.1, fontsize=NOUN_SIZE, zorder=5)
        ix, iy = polar(HALF * np.sqrt(2) * 0.8, corner, c)
        icon(ax, ix, iy, 0.27, color)
    ax.text(cx, HALF + 0.2, title, ha="center", va="bottom", fontsize=fs.TITLE_SIZE)


def card(ax, x, y, w, h, label, icon, color):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.2",
                                fc=tint(color, 0.82), ec="none", zorder=1))
    icon(ax, x - w / 2 + 0.42, y, 0.25, color)
    ax.text(x - w / 2 + 0.82, y, label, ha="left", va="center", fontsize=LABEL_SIZE, fontweight="bold")


def link(ax, start, end, color):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="simple,head_length=0.9,head_width=0.9,tail_width=0.35",
                                 mutation_scale=22, fc=color, ec="none", zorder=6))


def main():
    fig, ax = plt.subplots(figsize=(16.5, 6.2))

    panel(ax, 0.0, "Standard Loop", [
        (90, ("ACQUIRE", "data"), DATA, icon_data, 135),
        (-30, ("FIT", "optimizer"), FIT, icon_fit, -45),
        (210, ("PROPOSE", "next point"), PROPOSE, icon_propose, 225),
    ])

    cx = 2 * HALF + 0.7
    panel(ax, cx, "Agent Loop", [
        (135, ("ACQUIRE", "data"), DATA, icon_data, 135),
        (45, ("DESIGN", "algorithm"), AGENT, icon_design, 45),
        (-45, ("FIT", "optimizer"), FIT, icon_fit, -45),
        (-135, ("PROPOSE", "next point"), PROPOSE, icon_propose, 225),
    ])

    kx, kw, kh = cx + HALF + 0.75 + 1.25, 2.5, 0.85
    # agent level with the Design sector, fed by knowledge from above and tools from below
    lit_y, agent_y, tool_y = 2.35, 1.0, -0.35
    card(ax, kx, lit_y, kw, kh, "LITERATURE &\nKNOWLEDGE", icon_literature, AGENT)
    card(ax, kx, agent_y, kw, kh, "AGENT", icon_agent, AGENT)
    card(ax, kx, tool_y, kw, kh, "OPTIMIZER\nTOOLKIT", icon_toolkit, AGENT)
    link(ax, (kx, lit_y - kh / 2 - 0.04), (kx, agent_y + kh / 2 + 0.04), AGENT)
    link(ax, (kx, tool_y + kh / 2 + 0.04), (kx, agent_y - kh / 2 - 0.04), AGENT)
    link(ax, (kx - kw / 2 - 0.05, agent_y), (cx + HALF - 0.05, agent_y), AGENT)

    ax.set_xlim(-HALF - 0.1, kx + kw / 2 + 0.1)
    ax.set_ylim(-HALF - 0.1, lit_y + kh / 2 + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT, dpi=fs.DPI, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
