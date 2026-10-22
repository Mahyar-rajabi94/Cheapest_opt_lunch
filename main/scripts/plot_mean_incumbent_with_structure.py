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

ELEMENT_COLOR = {
    "C": "#4d4d4d", "O": "#e03131", "N": "#3b5bdb", "H": "#dee2e6",
    "S": "#f2c94c", "Cl": "#2f9e44", "F": "#74c69d", "Br": "#8a3b12",
    "P": "#e8820c", "I": "#9c36b5",
}
ELEMENT_RADIUS = {
    "C": 90, "O": 80, "N": 80, "H": 35, "S": 110, "Cl": 100,
    "F": 60, "Br": 120, "P": 110, "I": 130,
}


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
    coords, elements, bonds = molecule_3d_coords(smiles)
    for i, j in bonds:
        ax3d.plot(*zip(coords[i], coords[j]), color="#9aa2af", linewidth=1.4, zorder=1)
    colors = [ELEMENT_COLOR.get(e, "#c2255c") for e in elements]
    sizes = [ELEMENT_RADIUS.get(e, 70) for e in elements]
    ax3d.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=sizes,
                 edgecolors="white", linewidths=0.4, zorder=2, depthshade=True)
    span = np.abs(coords).max() * 1.15
    ax3d.set_xlim(-span, span)
    ax3d.set_ylim(-span, span)
    ax3d.set_zlim(-span, span)
    ax3d.view_init(elev=18, azim=35)
    ax3d.set_box_aspect((1, 1, 1))


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
