"""psa_v10 black box: a user-supplied, sealed objective over a discrete set of materials.

Each material is described by 8 features scaled to [0, 1]; any point in the box
evaluates to the value of the material it maps to, so the objective is
piecewise-constant. The evaluator (`psa_v10_blackbox.py`, standard library only)
lives in `external/psa_v10/` at the repository root, which git ignores.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from src.tasks.base import Task

EVALUATOR = Path(__file__).resolve().parents[3] / "external" / "psa_v10" / "psa_v10_blackbox.py"


class PSAv10Task(Task):
    """Minimize the psa_v10 black box over [0, 1]^8.

    `f_opt` = 83.13 is a reference value supplied by the user, used only for
    offline regret.
    """

    def __init__(self):
        spec = importlib.util.spec_from_file_location("psa_v10_blackbox", EVALUATOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._blackbox = module.BlackBox()
        self.dim = self._blackbox.dim
        self.name = f"psa_v10_d{self.dim}"
        self.bounds = np.asarray(self._blackbox.bounds, dtype=float)
        self.f_opt = 83.13

    def _f(self, x: np.ndarray) -> float:
        # Optimizers can overshoot the box by rounding error; the evaluator rejects that.
        return self._blackbox(np.clip(x, self.bounds[:, 0], self.bounds[:, 1]))
