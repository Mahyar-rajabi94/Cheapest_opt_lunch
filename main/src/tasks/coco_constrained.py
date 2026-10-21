"""Constrained BBOB tasks from COCO's `bbob-constrained` suite, via `cocoex`.

Requires `coco-experiment==2.8.2` (import name `cocoex`). It is not in
requirements.txt (sandbox code does not need it); install it into `.venv`
by hand.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from src.tasks.base import Task


class COCOConstrainedTask(Task):
    """One `bbob-constrained` function (e.g. 20: rotated ellipsoid,
    38: different powers, 44: Rastrigin) with its 3 linear constraints.

    The box is [-5, 5] per dimension. COCO exposes no public optimum, so
    `f_opt` is the objective at the constrained optimum x* that COCO writes
    out for the instance; x* is checked to be feasible.
    """

    def __init__(self, function_id: int, dim: int = 10, instance: int = 1):
        import cocoex

        suite = cocoex.Suite("bbob-constrained", "", "")
        self._problem = suite.get_problem_by_function_dimension_instance(function_id, dim, instance)
        if self._problem is None:
            raise ValueError(f"No bbob-constrained function {function_id}, dim {dim}, instance {instance}.")

        self.name = f"bbob_constrained_f{function_id}_d{dim}_i{instance}"
        self.dim = dim
        self.bounds = np.column_stack(
            [self._problem.lower_bounds, self._problem.upper_bounds]
        ).astype(float)
        self.n_constraints = int(self._problem.number_of_constraints)

        # Private API: the only way to get x*. It writes a file into the CWD,
        # so run it in a temporary directory.
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                self._problem._best_parameter("print")
                with open("._bbob_problem_best_parameter.txt") as f:
                    x_star = np.array(f.read().split(), dtype=float)
            finally:
                os.chdir(cwd)
        if np.any(np.asarray(self._problem.constraint(x_star)) > 0.0):
            raise ValueError(f"{self.name}: COCO's reported optimum is infeasible.")
        self.f_opt = float(self._problem(x_star))

    def _f(self, x: np.ndarray) -> float:
        return float(np.asarray(self._problem(x)).reshape(-1)[0])

    def _g(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._problem.constraint(x))
