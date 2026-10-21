"""Bent Cigar (BBOB f12) task wrapper.

Wraps the BBOB Bent Cigar function, via the `ioh` package, behind this
project's standard Task interface: a fixed search space, a known optimum
for offline regret computation, and a single `evaluate()` method that calls
the objective and times it.

f12 = Bent Cigar -- unimodal but extremely ill-conditioned (one direction is
~1e6 times steeper than the rest) and non-separable. Unlike Rosenbrock's
curved valley, the difficulty here is pure conditioning: it stresses whether
a method notices the anisotropy and adapts (e.g. CMA-ES's covariance
adaptation) rather than relying on coordinate-wise steps.

By convention, this task MINIMIZES (matching BBOB and this project's
optimizers in src/optimizers/).
"""

from __future__ import annotations

import time

import ioh
import numpy as np


class BentCigarTask:
    """BBOB Bent Cigar task (f12).

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
        task = BentCigarTask(dim=10, instance=1)
        y, elapsed = task.evaluate(x)   # x has length task.dim
        regret = y - task.f_opt
    """

    def __init__(self, dim: int = 10, instance: int = 1):
        self._problem = ioh.get_problem(
            12,
            instance=instance,
            dimension=dim,
            problem_class=ioh.ProblemClass.BBOB,
        )

        self.name = f"bent_cigar_d{dim}_i{instance}"
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
