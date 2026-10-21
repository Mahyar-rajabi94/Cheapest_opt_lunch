"""Weierstrass (BBOB f16) task wrapper.

Wraps the BBOB Weierstrass function, via the `ioh` package, behind this
project's standard Task interface: a fixed search space, a known optimum
for offline regret computation, and a single `evaluate()` method that calls
the objective and times it.

f16 = Weierstrass -- highly rugged and multimodal: a sum of cosines at
geometrically increasing frequencies gives a continuous but nowhere-smooth
landscape with many local optima and no exploitable global structure at
small scales. It stresses whether a method keeps exploring rather than
committing to the first basin it finds.

By convention, this task MINIMIZES (matching BBOB and this project's
optimizers in src/optimizers/).
"""

from __future__ import annotations

import time

import ioh
import numpy as np


class WeierstrassTask:
    """BBOB Weierstrass task (f16).

    Parameters
    ----------
    dim : int
        Number of dimensions (default 10).
    instance : int
        BBOB instance id, selecting the random shift/rotation applied to
        this copy of the function (default 1). BBOB ships 15 instances per
        function by convention; use different instances as different seeds
        when averaging results across runs.

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
        task = WeierstrassTask(dim=10, instance=1)
        y, elapsed = task.evaluate(x)   # x has length task.dim
        regret = y - task.f_opt
    """

    def __init__(self, dim: int = 10, instance: int = 1):
        self._problem = ioh.get_problem(
            16,
            instance=instance,
            dimension=dim,
            problem_class=ioh.ProblemClass.BBOB,
        )

        self.name = f"weierstrass_d{dim}_i{instance}"
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
