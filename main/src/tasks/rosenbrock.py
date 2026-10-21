"""Rosenbrock (BBOB f8 / f9) task wrapper.

Wraps the BBOB Rosenbrock functions, via the `ioh` package, behind this
project's standard Task interface: a fixed search space, a known optimum
for offline regret computation, and a single `evaluate()` method that calls
the objective and times it.

f8 = Rosenbrock, original (ill-conditioned, axis-aligned curved valley)
f9 = Rosenbrock, rotated  (same shape, but rotated so the valley is not
     axis-aligned -- harder for separable / coordinate-wise methods)

Both are unimodal. By convention, this task MINIMIZES (matching BBOB and
this project's optimizers in src/optimizers/).
"""

from __future__ import annotations

import time

import ioh
import numpy as np


class RosenbrockTask:
    """BBOB Rosenbrock task (f8 original, or f9 rotated).

    Parameters
    ----------
    dim : int
        Number of dimensions (default 10).
    instance : int
        BBOB instance id, selecting the random shift/rotation applied to
        this copy of the function (default 1). BBOB ships 15 instances per
        function by convention; use different instances as different
        random seeds when averaging results across runs.
    rotated : bool
        False (default) selects f8 (original Rosenbrock, axis-aligned);
        True selects f9 (rotated Rosenbrock).

    Attributes
    ----------
    name : str
    direction : str
        Always "min".
    dim : int
    bounds : np.ndarray of shape (dim, 2)
        [lower, upper] bound per dimension; BBOB's standard box is [-5, 5].
    f_opt : float
        The true minimum value for this specific instance. IOH exposes it
        directly, so it is known exactly here -- it is used only for
        offline regret computation, never passed to the optimizer.

    Example
    -------
        task = RosenbrockTask(dim=10, instance=1)
        y, elapsed = task.evaluate(x)   # x has length task.dim
        regret = y - task.f_opt
    """

    def __init__(self, dim: int = 10, instance: int = 1, rotated: bool = False):
        function_id = 9 if rotated else 8
        self._problem = ioh.get_problem(
            function_id,
            instance=instance,
            dimension=dim,
            problem_class=ioh.ProblemClass.BBOB,
        )

        variant = "_rotated" if rotated else ""
        self.name = f"rosenbrock{variant}_d{dim}_i{instance}"
        self.direction = "min"
        self.dim = dim
        self.bounds = np.column_stack(
            [self._problem.bounds.lb, self._problem.bounds.ub]
        ).astype(float)
        self.f_opt = float(self._problem.optimum.y)

    def evaluate(self, x):
        """Evaluate one point `x` (length == self.dim).

        Returns (y, elapsed_seconds).
        """
        x = np.asarray(x, dtype=float)
        start = time.perf_counter()
        y = float(self._problem(x))
        elapsed = time.perf_counter() - start
        return y, elapsed
