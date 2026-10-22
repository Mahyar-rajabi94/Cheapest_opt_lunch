"""BBOB task: any noiseless BBOB function (f1-f24), via the `ioh` package."""

from __future__ import annotations

import ioh
import numpy as np

from src.tasks.base import Task


class BBOBTask(Task):
    """One BBOB function, dimension, and instance.

    Parameters
    ----------
    name : str
        Function name used as the prefix of `self.name`
        (e.g. "rosenbrock" -> "rosenbrock_d10_i1").
    function_id : int
        BBOB function id (1-24).
    dim : int
        Number of dimensions (default 10).
    instance : int
        BBOB instance id, selecting the random shift/rotation applied to
        this copy of the function (default 1).

    BBOB's standard box is [-5, 5] per dimension. IOH exposes the exact
    optimum of each instance, so `f_opt` is known exactly.
    """

    def __init__(self, name: str, function_id: int, dim: int = 10, instance: int = 1):
        self._problem = ioh.get_problem(
            function_id,
            instance=instance,
            dimension=dim,
            problem_class=ioh.ProblemClass.BBOB,
        )

        self.name = f"{name}_d{dim}_i{instance}"
        self.dim = dim
        self.bounds = np.column_stack(
            [self._problem.bounds.lb, self._problem.bounds.ub]
        ).astype(float)
        self.f_opt = float(self._problem.optimum.y)

    def _f(self, x: np.ndarray) -> float:
        return self._problem(x)
