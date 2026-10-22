"""CMA-ES (plain, no restarts) -- wraps pycma behind the standard ask/tell
interface shared by every optimizer in this package.

Standard library defaults only: population size, learning rates, and all
other internal hyperparameters are left at pycma's own values (no tuning).
This is plain CMA-ES, with no restart strategy (IPOP/BIPOP); those would be
a separate, explicitly different tool if added later.

By convention, this optimizer MINIMIZES the objective (matching
RandomSearch and SobolSearch). To maximize, negate the objective values
before calling `tell`.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

# pycma warns at import time if matplotlib isn't available; harmless (we
# never call cma.plot()), so it is silenced here rather than left to print.
warnings.filterwarnings("ignore", message="Could not import matplotlib.*")

import cma  # noqa: E402  (import after the warnings filter, deliberately)


class CMAES:
    """Covariance Matrix Adaptation Evolution Strategy, single run.

    Wraps `cma.CMAEvolutionStrategy` (the pycma reference implementation).
    CMA-ES proposes and updates on a whole generation (population) at a
    time internally, but this wrapper hides that bookkeeping so it can be
    driven exactly like RandomSearch/SobolSearch: ask for any number of
    points, including one at a time, and tell them back in the same order
    they were asked. The underlying population-level ask/tell happens
    transparently once a full generation has been told.

    The starting point and step size are derived from `bounds` using
    CMA-ES's own standard recommendation for a box-constrained problem
    (not a tuned choice): x0 = the midpoint of the box, sigma0 = one
    quarter of the (mean) box width. Box constraints are then enforced
    throughout the run via pycma's own `bounds` option. Population size
    and all other internal hyperparameters are left at pycma's defaults.

    IMPORTANT: unlike RandomSearch/SobolSearch, `tell` must be called with
    exactly the points most recently returned by `ask`, in the same order
    -- CMA-ES's update depends on matching each proposed point to its
    evaluated value within its generation.

    Parameters
    ----------
    bounds : array-like of shape (d, 2)
        Lower and upper bound for each of the d dimensions.
    seed : int, optional
        Seed for pycma's random number generator, for reproducibility.

    Example
    -------
        opt = CMAES(bounds=[(-5, 5)] * 10, seed=0)
        X = opt.ask(n=1)             # propose 1 point, shape (1, 10)
        y = [f(x) for x in X]        # evaluate with the task's objective
        opt.tell(X, y)                # record the result
        x_best, y_best = opt.best()  # best point found so far
    """

    def __init__(self, bounds, seed: Optional[int] = None):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError(
                "bounds must have shape (d, 2): one (low, high) pair per dimension"
            )
        self.dim = self.bounds.shape[0]

        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        x0 = (lower + upper) / 2
        sigma0 = float(np.mean(upper - lower)) / 4.0

        opts = {
            "bounds": [lower.tolist(), upper.tolist()],
            "verbose": -9,  # suppress pycma's default per-generation printing
        }
        if seed is not None:
            opts["seed"] = seed

        self._es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        # FIFO buffers that let the population-based `ask`/`tell` of pycma
        # be driven point-by-point from the outside.
        self._pending_population: list = []  # asked, not yet told
        self._pending_told: list = []  # told so far for the current generation

        self.X_ = np.empty((0, self.dim))
        self.y_ = np.empty((0,))

    def ask(self, n: int = 1) -> np.ndarray:
        """Propose `n` new candidate points, shape (n, dim)."""
        out = []
        for _ in range(n):
            if not self._pending_population:
                self._pending_population = list(self._es.ask())
            out.append(self._pending_population.pop(0))
        return np.array(out)

    def tell(self, X, y) -> None:
        """Record evaluated points `X` (n, dim) and their values `y` (n,).

        Must be called with exactly the points most recently returned by
        `ask`, in the same order (see class docstring).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))

        self.X_ = np.vstack([self.X_, X])
        self.y_ = np.concatenate([self.y_, y])

        for xi, yi in zip(X, y):
            self._pending_told.append((xi, float(yi)))
            if len(self._pending_told) == self._es.popsize:
                xs = [p[0] for p in self._pending_told]
                ys = [p[1] for p in self._pending_told]
                self._es.tell(xs, ys)
                self._pending_told = []

    def best(self):
        """Return (x_best, y_best) among all points told so far (minimum y)."""
        if len(self.y_) == 0:
            raise RuntimeError("No evaluations recorded yet; call tell() first.")
        i = np.argmin(self.y_)
        return self.X_[i], self.y_[i]
