"""Same animated regret-vs-budget gif as plot_mean_incumbent_style.py, with
a ball-and-stick 3D render of the agent's current best-found molecule in
the corner (regenerated only when the incumbent changes), its compound
name, and its regret -- matching the reference
slide_figures/mof_mean_incumbent_regret_with_structures.gif's layout.

Usage
-----
    python scripts/plot_mean_incumbent_with_structure.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from plot_dionysus_vs_agents import baseline_trajectory  # noqa: E402

INK = "#172334"
GRIDLINE = "#eceef1"
CURSOR_GRAY = "#b8bfc8"
AXIS_GRAY = "#4a5160"

# Standard CPK coloring, matching a typical ball-and-stick renderer.
ELEMENT_COLOR = {
    "C": "#909090", "O": "#e0201e", "N": "#2050e0", "H": "#f5f5f5",
    "S": "#e8c93e", "Cl": "#2fb52f", "F": "#90e050", "Br": "#a13c13",
    "P": "#e89a2e", "I": "#8a2be2",
}
# Ball-and-stick radii (not van der Waals/space-filling), in model units.
ELEMENT_RADIUS = {
    "C": 0.40, "O": 0.37, "N": 0.38, "H": 0.24, "S": 0.46, "Cl": 0.46,
    "F": 0.32, "Br": 0.52, "P": 0.46, "I": 0.58,
}
BOND_RADIUS = 0.11
BOND_COLOR = "#c7c7c7"


def _sphere_mesh(center, radius, n=16):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    return x, y, z


def _cylinder_mesh(p0, p1, radius, n=10):
    p0, p1 = np.asarray(p0), np.asarray(p1)
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-6:
        axis = np.array([0.0, 0.0, 1.0])
        length = 1e-6
    axis = axis / length
    not_axis = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp1 = np.cross(axis, not_axis)
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)
    theta = np.linspace(0, 2 * np.pi, n)
    t = np.linspace(0, 1, 2)
    theta_grid, t_grid = np.meshgrid(theta, t)
    circle = (radius * np.cos(theta_grid))[..., None] * perp1 + (radius * np.sin(theta_grid))[..., None] * perp2
    centerline = p0 + t_grid[..., None] * (p1 - p0)
    pts = centerline + circle
    return pts[..., 0], pts[..., 1], pts[..., 2]


def load_rows(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def agent_curve_and_smiles(task: str, agent_dir: str, f_opt: float, budget: int):
    rows = load_rows(ROOT / "results" / f"dionysus_{task}" / agent_dir / "trace.csv")
    rows = [r for r in rows if int(r["eval_idx"]) <= budget]
    eval_idx = np.array([int(r["eval_idx"]) for r in rows])
    y = np.maximum(np.array([float(r["y_best_so_far"]) for r in rows]) - f_opt, 1e-9)
    # The incumbent SMILES at each eval: whichever candidate last set a new
    # best-so-far (not necessarily the one just evaluated this step).
    best_smiles, best_y = None, None
    incumbent_smiles = []
    for r in rows:
        y_val = float(r["y"])
        if best_y is None or y_val < best_y:
            best_y, best_smiles = y_val, r["candidate"]
        incumbent_smiles.append(best_smiles)
    return eval_idx, y, incumbent_smiles


def molecule_3d_coords(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    seed = abs(hash(smiles)) % (2**31)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    coords -= coords.mean(axis=0)
    return coords, elements, bonds


def draw_molecule_3d(ax3d, smiles: str):
    ax3d.cla()
    ax3d.set_axis_off()
    ax3d.set_facecolor("black")

    coords, elements, bonds = molecule_3d_coords(smiles)

    # Each bond as two half-cylinders, colored like the nearer atom (the
    # usual ball-and-stick convention), with real shaded cylinder geometry.
    for i, j in bonds:
        mid = (coords[i] + coords[j]) / 2
        for end, other in ((i, mid), (j, mid)):
            color = ELEMENT_COLOR.get(elements[end], BOND_COLOR)
            x, y, z = _cylinder_mesh(coords[end], other, BOND_RADIUS)
            ax3d.plot_surface(x, y, z, color=color, shade=True, antialiased=True,
                               linewidth=0, zorder=1)

    # Each atom as a shaded sphere (not a flat scatter dot).
    for pos, el in zip(coords, elements):
        radius = ELEMENT_RADIUS.get(el, 0.4)
        x, y, z = _sphere_mesh(pos, radius)
        ax3d.plot_surface(x, y, z, color=ELEMENT_COLOR.get(el, "#c2255c"),
                           shade=True, antialiased=True, linewidth=0, zorder=2)

    span = (np.abs(coords).max() + 0.6) * 1.05
    ax3d.set_xlim(-span, span)
    ax3d.set_ylim(-span, span)
    ax3d.set_zlim(-span, span)
    ax3d.view_init(elev=15, azim=35)
    ax3d.set_box_aspect((1, 1, 1))
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_facecolor("black")
        axis.pane.set_alpha(1.0)
        axis.line.set_color("black")
    ax3d.grid(False)


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
    name_rows = load_rows(data_dir / "candidates_with_names.csv")
    name_of = {r["smiles"]: r["name"] for r in name_rows}

    ax_curve, ay_curve, incumbent_smiles = agent_curve_and_smiles(task, agent_dir, f_opt, budget)

    baseline_curves = []
    for model, feature, label, color in series:
        evals, regret = baseline_trajectory(data_dir, model, feature, f_opt, "minimize")
        keep = evals <= budget
        baseline_curves.append((evals[keep], np.maximum(regret[keep], 1e-9), label, color, 30))

    y_max = max(ay_curve.max(), max(c[1].max() for c in baseline_curves)) * 1.08

    fig = plt.figure(figsize=(12.8, 7.6), dpi=100, facecolor="white")
    ax = fig.add_axes([0.07, 0.12, 0.60, 0.74])
    ax.set_facecolor("white")
    ax3d = fig.add_axes([0.70, 0.30, 0.27, 0.40], projection="3d")
    ax3d.set_facecolor("none")
    name_text = fig.text(0.835, 0.74, "", fontsize=15, fontweight="bold", color=INK, ha="center", va="top")
    regret_text = fig.text(0.835, 0.27, "", fontsize=11.5, color="#5b6472", ha="center", va="top")

    fig.text(0.07, 0.934, "Low-solubility molecule discovery",
              fontsize=22, fontweight="bold", color=INK, ha="left", va="top")
    counter_text = fig.text(0.93, 0.934, "", fontsize=13, fontweight="bold", color=INK,
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
    (agent_line,) = ax.plot([], [], color=INK, linewidth=3.4, zorder=5, label="Agent")
    (agent_dot,) = ax.plot([], [], "o", color=INK, markersize=5.5, zorder=6)

    base_lines, base_dots = [], []
    for _, _, label, color, n in baseline_curves:
        (ln,) = ax.plot([], [], color=color, linewidth=2.2, zorder=4, label=label)
        (dot,) = ax.plot([], [], "o", color=color, markersize=4.5, zorder=4.5)
        base_lines.append(ln)
        base_dots.append(dot)

    ax.legend(loc="upper right", frameon=False, fontsize=11, labelcolor=INK,
              handlelength=1.8, bbox_to_anchor=(1.0, 1.0))

    frames = list(range(1, budget + 1))

    def update(budget_now, last_smiles):
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

        idx = int(np.searchsorted(ax_curve, budget_now, side="right")) - 1
        idx = max(idx, 0)
        smiles = incumbent_smiles[idx]
        if smiles != last_smiles:
            draw_molecule_3d(ax3d, smiles)
            name_text.set_text(name_of.get(smiles, smiles))
        regret_text.set_text(f"Incumbent regret: {ay_curve[idx]:.4f}")
        return smiles

    fig.canvas.draw()
    images = []
    last_smiles = None
    for budget_now in frames:
        last_smiles = update(budget_now, last_smiles)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        rgb = buf[:, :, :3]
        near_white = (rgb >= 250).all(axis=2)
        rgb = rgb.copy()
        rgb[near_white] = 255
        images.append(Image.fromarray(rgb, "RGB"))
        print(f"frame {budget_now}/{budget}", end="\r")

    outdir = ROOT.parent / "slide_figures" / "dionysus"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{task}_mean_incumbent_regret_with_structure.gif"
    images[0].save(out_path, save_all=True, append_images=images[1:],
                    duration=100, loop=0, optimize=False)
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
