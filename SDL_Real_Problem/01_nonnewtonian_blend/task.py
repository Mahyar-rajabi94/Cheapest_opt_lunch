"""Formulate a non-Newtonian blend to hit a target flow curve (7-D).

x = [c_carbopol, c_sepineo, c_solagum, c_peg600k (wt%), w_glycerol (wt%),
     pH, t_mix (s)]; the objective is model.BlendModel.objective (E9 in
PROBLEM.md): log10-RMSE between the blend's flow curve and the target spec,
plus ingredient-cost, mixing-time and pH-spec penalties.
f_opt is the best-known value from data/reference_optimum.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "main"))
sys.path.insert(0, str(HERE))

from src.tasks.base import Task  # noqa: E402

import model  # noqa: E402


class BlendFormulationTask(Task):
    def __init__(self):
        self.model = model.BlendModel()
        self.dim = len(model.VARIABLES)
        self.name = f"nonnewtonian_blend_d{self.dim}"
        self.bounds = np.array([[v[2], v[3]] for v in model.VARIABLES])
        self.f_opt = json.loads((HERE / "data" / "reference_optimum.json").read_text())["f_opt"]

    def _f(self, x: np.ndarray) -> float:
        return self.model.objective(x)


def make_task() -> Task:
    return BlendFormulationTask()
