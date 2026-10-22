"""Constrained tasks G04 and G07 from the classic G-suite, via `pymoo`.

Requires `pymoo==0.6.1.6`. It is not in requirements.txt (sandbox code does
not need it); install it into `.venv` by hand.
"""

from __future__ import annotations

import numpy as np

from src.tasks.base import Task


class PymooGTask(Task):
    """One pymoo G-suite problem ("g4" or "g7"), minimized.

    The box is pymoo's own xl/xu, which is not uniform across dimensions.
    `f_opt` is the best-known optimum that pymoo reports as its Pareto front.
    """

    def __init__(self, problem_name: str):
        from pymoo.problems import get_problem

        self._problem = get_problem(problem_name)
        self.name = f"pymoo_g{int(problem_name[1:]):02d}"
        self.dim = int(self._problem.n_var)
        self.bounds = np.column_stack([self._problem.xl, self._problem.xu]).astype(float)
        self.n_constraints = int(self._problem.n_ieq_constr)
        self.f_opt = float(np.asarray(self._problem.pareto_front()).reshape(-1)[0])

    def _evaluate(self, x: np.ndarray):
        out = self._problem.evaluate(x, return_values_of=["F", "G"], return_as_dictionary=True)
        return out["F"], out["G"]

    def _f(self, x: np.ndarray) -> float:
        return np.asarray(self._evaluate(x)[0]).reshape(-1)[0]

    def _g(self, x: np.ndarray) -> np.ndarray:
        return self._evaluate(x)[1]
