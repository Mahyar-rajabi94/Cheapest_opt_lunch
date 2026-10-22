"""Mixed-integer BBOB tasks from COCO's `bbob-mixint` suite, via `cocoex`.

Requires `coco-experiment==2.8.2` (import name `cocoex`). It is not in
requirements.txt (sandbox code does not need it); install it into `.venv`
by hand.
"""

from __future__ import annotations

import numpy as np

from src.tasks.base import Task
from src.tasks.coco_constrained import coco_x_star


class COCOMixintTask(Task):
    """One `bbob-mixint` function (e.g. 8: Rosenbrock, 15: Rastrigin,
    17: Schaffer F7) with the suite's integer coordinates.

    The first `problem.number_of_integer_variables` coordinates are integer
    (at D=10: bounds [0, 1], [0, 1], [0, 3], [0, 3], [0, 7], [0, 7], [0, 15],
    [0, 15]); the rest are continuous in [-5, 5]. The box is COCO's own. COCO
    rounds integer coordinates internally, so optimizers can propose any real
    point and the objective is piecewise constant along those coordinates.
    `f_opt` is the objective at the x* COCO writes out for the instance; x* is
    checked to be integral on the integer coordinates.
    """

    def __init__(self, function_id: int, dim: int = 10, instance: int = 1):
        import cocoex

        suite = cocoex.Suite("bbob-mixint", "", "")
        self._problem = suite.get_problem_by_function_dimension_instance(function_id, dim, instance)
        if self._problem is None:
            raise ValueError(f"No bbob-mixint function {function_id}, dim {dim}, instance {instance}.")

        self.name = f"bbob_mixint_f{function_id}_d{dim}_i{instance}"
        self.dim = dim
        self.bounds = np.column_stack(
            [self._problem.lower_bounds, self._problem.upper_bounds]
        ).astype(float)

        x_star = coco_x_star(self._problem)
        n_int = int(self._problem.number_of_integer_variables)
        if not np.array_equal(x_star[:n_int], np.round(x_star[:n_int])):
            raise ValueError(f"{self.name}: COCO's reported optimum is not integral.")
        self.f_opt = float(self._problem(x_star))

    def _f(self, x: np.ndarray) -> float:
        return float(np.asarray(self._problem(x)).reshape(-1)[0])
