"""Organic photovoltaic (OPV) device task: maximize the power conversion
efficiency (PCE) predicted by a random-forest emulator of 526 literature
devices (Seifrid et al., J. Mater. Chem. A, 2024).

Needs only numpy. It reads two files from `task_005_baseline/` at the
repository root: search_space.json (bounds, interlayer levels and linear
constraints, written by preprocess.py) and opv_emulator_rf.npz (the forest's
trees as flat arrays, written by train_emulator.py). Predictions match
scikit-learn's RandomForestRegressor.predict, which train_emulator.py checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.tasks.base import Task

DATA_ROOT = Path(__file__).resolve().parents[3] / "task_005_baseline"

# Best-known -PCE (no exact optimum is known): max over 2^20 Sobol points,
# coordinate ascent with exact line searches from the 64 best of them, and
# 20 CMA-ES runs of 1e5 evaluations, computed 2026-10-22 by
# task_005_baseline/find_f_opt.py. All 64 ascents and 5 of 20 CMA-ES runs reach
# it (Sobol best 11.717%); no constraint is active there. HTL_energy -5.2 or
# -4.7 and ETL_energy -2.18 (mostly PFN-Br) give it.
_F_OPT = -12.955165987276994

# Same tie rule as preprocess.snap: exact ties go to the more negative level.
_SNAP_TOL = 1e-9


class OPVTask(Task):
    """Maximize emulated PCE (%), minimized here as -PCE, under the six
    required linear constraints of search_space.json.

    The box is [0, 1]^11, one coordinate per input in search_space.json
    order (HOMO_D, LUMO_D, Eg_D, HOMO_A, LUMO_A, Eg_A, DA_ratio,
    additive_conc, anneal_temp, HTL_energy, ETL_energy). A coordinate u
    decodes as lower + u (upper - lower); HTL_energy and ETL_energy then
    snap to the nearest observed interlayer level. The forest is piecewise
    constant along every coordinate. The constraints (energy-level offsets
    and gaps, in eV) are linear in the decoded values and do not involve
    HTL_energy or ETL_energy; g <= 0 is feasible. `f_opt` is a best-known value.
    """

    def __init__(self):
        space = json.loads((DATA_ROOT / "search_space.json").read_text())
        names = space["inputs"]
        cont = space["continuous"]
        self._lower = np.array([cont[n]["lower"] for n in names], dtype=float)
        self._upper = np.array([cont[n]["upper"] for n in names], dtype=float)
        self._levels = {
            i: np.sort(np.asarray(cont[n]["snap_to_levels"], dtype=float))
            for i, n in enumerate(names)
            if cont[n].get("snap_to_levels") is not None
        }
        # Each required constraint as a row of A x <= b.
        required = [c for c in space["linear_constraints"] if not c["optional"]]
        sign = {"<=": 1.0, ">=": -1.0}
        self._A = np.array([[sign[c["op"]] * c["coefficients"].get(n, 0.0) for n in names] for c in required])
        self._b = np.array([sign[c["op"]] * c["rhs"] for c in required])
        self.constraint_names = [c["name"] for c in required]

        with np.load(DATA_ROOT / "opv_emulator_rf.npz") as z:
            if [str(s) for s in z["inputs"]] != names:
                raise ValueError("opv_emulator_rf.npz and search_space.json list different inputs.")
            self._left, self._right = z["left"], z["right"]
            self._feature = z["feature"].astype(np.intp)
            self._threshold, self._value, self._roots = z["threshold"], z["value"], z["roots"]

        self.input_names = names
        self.name = "opv_pce_d11"
        self.dim = len(names)
        self.bounds = np.tile([0.0, 1.0], (self.dim, 1))
        self.n_constraints = len(required)
        self.f_opt = _F_OPT

    def decode(self, U: np.ndarray) -> np.ndarray:
        """Unit-box points, shape (n, 11), to inputs in original units with HTL/ETL snapped."""
        X = self._lower + np.clip(U, 0.0, 1.0) * (self._upper - self._lower)
        for i, levels in self._levels.items():
            dist = np.abs(X[:, i, None] - levels)
            X[:, i] = levels[np.argmax(dist <= dist.min(axis=1, keepdims=True) + _SNAP_TOL, axis=1)]
        return X

    def pce(self, X: np.ndarray) -> np.ndarray:
        """Emulated PCE (%) at decoded inputs X, shape (n, 11): the mean over trees of the leaf reached."""
        # scikit-learn compares float32-cast inputs with float64 thresholds; do the same.
        X = X.astype(np.float32).astype(np.float64)
        rows = np.arange(len(X))[:, None]
        node = np.tile(self._roots, (len(X), 1))
        while True:
            inner = self._left[node] >= 0
            if not inner.any():
                return self._value[node].mean(axis=1)
            go_left = X[rows, self._feature[node]] <= self._threshold[node]
            node = np.where(inner, np.where(go_left, self._left[node], self._right[node]), node)

    def _f(self, x: np.ndarray) -> float:
        return -float(self.pce(self.decode(x[None, :]))[0])

    def _g(self, x: np.ndarray) -> np.ndarray:
        return self._A @ self.decode(x[None, :])[0] - self._b
