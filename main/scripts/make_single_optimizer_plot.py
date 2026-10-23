"""Single optimizers on rosenbrock_d10_i1, drawn like the agent switching
figure so they can be compared side by side (make_agent_switching_plot.py,
make_agent_switching_gif.py). --method picks the run:

  bobyqa   the best single optimizer: lowest final regret among the nine
           standalone baselines (6.3e-13, stopping itself after 1,596
           evaluations; deterministic, one seed).
  cma_es   CMA-ES alone, the agent's first stage run for the whole budget.
           It has 10 seeds; the animation needs one trajectory, so it uses
           the seed whose final regret is closest to the 10-seed median
           (seed 1: 4.13 vs. median 4.40). Seeds range 2.8-8.6.

Runs come from results_full, the baseline set the Astra max campaign is
scored against (its summary.csv). results_linux is not used: its BOBYQA run on
this task has a 100-evaluation budget.

Comparability with the agent figure:
  - same landscape slice: the agent's plane, extents and contour bands;
    the run's points are projected onto it (positions exact in the plane,
    distance from the plane not shown);
  - same Progress axes: x 0-2000, y fixed by sw.PROGRESS_YLIM for every run;
  - same color and label for the method family as the agent's stage
    (Model Based yellow, Evolutionary red);
  - the GIF steps the same number of evaluations per frame as the agent GIF,
    so both play at the same speed.

Usage (from main/):
    python scripts/make_single_optimizer_plot.py                   # bobyqa
    python scripts/make_single_optimizer_plot.py --method cma_es
    # writes results_linux/single_optimizer_rosenbrock_<method>.{png,gif}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_agent_switching_plot as sw  # noqa: E402  (also applies figure_style rcParams)
import make_agent_switching_gif as anim  # noqa: E402
import figure_style as fs  # noqa: E402

# method -> (seed, stage label, color); label and color match the agent's stage for that family
METHODS = {
    "bobyqa": (0, "Model Based", "#eda100"),
    "cma_es": (1, "Evolutionary", "#e34948"),  # median-final seed, see module docstring
}


def load_baseline(method, seed):
    run_dir = sw.MAIN / f"results_full/rosenbrock_d10_i1/{method}/seed_{seed}"
    rows = list(csv.DictReader(open(run_dir / "trace.csv")))
    meta = json.loads((run_dir / "meta.json").read_text())
    X = np.array([[float(r[f"x{i}"]) for i in range(meta["task_dim"])] for r in rows])
    regret = np.array([float(r["y"]) for r in rows]) - meta["task_f_opt"]
    return X, regret, np.zeros(len(rows), dtype=int)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=METHODS, default="bobyqa")
    method = parser.parse_args().method
    seed, label, color = METHODS[method]
    stages = [(label, color, {method})]
    out_png = sw.MAIN / f"results_linux/single_optimizer_rosenbrock_{method}.png"
    out_gif = out_png.with_suffix(".gif")
    Xa, regret_a, stage_a = sw.load_run()
    f_opt = json.loads(sw.F_OPT_META.read_text())["task_f_opt"]
    geo = sw.slice_geometry(Xa, regret_a, stage_a, f_opt)
    run = sw.project_run(geo, *load_baseline(method, seed))

    fig = sw.build_figure(geo, run, len(run["regret"]), stages)
    fig.savefig(out_png, dpi=fs.DPI, bbox_inches="tight")
    print("wrote", out_png)

    evals_per_frame = len(regret_a) / anim.N_FRAMES
    anim.write_gif(geo, run, out_gif, stages, n_frames=round(len(run["regret"]) / evals_per_frame))


if __name__ == "__main__":
    main()
