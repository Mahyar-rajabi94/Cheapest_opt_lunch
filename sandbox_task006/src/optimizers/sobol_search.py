"""Sobol' quasi-random search — a minimal, low-discrepancy baseline optimizer.

Standard library defaults only; no tuning (scipy's own default, scramble=True,
is used as-is). Intended as a generic baseline that can be dropped into any
bounded continuous optimization task.

By convention, this optimizer MINIMIZES the objective (matching scipy and
most of the optimization literature). To maximize, negate the objective
values before calling `tell`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import qmc


class SobolSearch:
    """Quasi-random search using a scrambled Sobol' low-discrepancy sequence.

    Proposes points from a Sobol' sequence (`scipy.stats.qmc.Sobol`), scaled
    to the given box bounds. Unlike plain random search, the sequence covers
    the space more evenly. The sampler's internal state advances with every
    call to `ask`, so repeated calls continue the same sequence rather than
    restarting it.

    Note: Sobol' sequences have their best (balance) properties when the
    *cumulative* number of points drawn so far is a power of 2. This
    implementation accepts any batch size per call, which scipy allows, but
    for the most even coverage, prefer asking in batches whose running total
    is a power of 2 (e.g. 1, 2, 4, 8, 16, ...).

    This class does not evaluate the objective itself: the caller asks for
    candidate points, evaluates them with the task's own objective function,
    and reports the results back via `tell`.

    Parameters
    ----------
    bounds : array-like of shape (d, 2)
        Lower and upper bound for each of the d dimensions, e.g.
        [(-5, 5), (-5, 5), ...].
    seed : int, optional
        Seed for the sequence's scrambling, for reproducibility.

    Example
    -------
        opt = SobolSearch(bounds=[(-5, 5)] * 10, seed=0)
        X = opt.ask(n=16)             # propose 16 points, shape (16, 10)
        y = [f(x) for x in X]         # evaluate with the task's objective
        opt.tell(X, y)                 # record the results
        x_best, y_best = opt.best()   # best point found so far
    """

    def __init__(self, bounds, seed: Optional[int] = None):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError(
                "bounds must have shape (d, 2): one (low, high) pair per dimension"
            )
        self.dim = self.bounds.shape[0]
        self._sampler = qmc.Sobol(d=self.dim, scramble=True, seed=seed)

        self.X_ = np.empty((0, self.dim))
        self.y_ = np.empty((0,))

    def ask(self, n: int = 1) -> np.ndarray:
        """Propose `n` new candidate points, shape (n, dim)."""
        unit_cube_sample = self._sampler.random(n)
        return qmc.scale(unit_cube_sample, self.bounds[:, 0], self.bounds[:, 1])

    def tell(self, X, y) -> None:
        """Record evaluated points `X` (n, dim) and their values `y` (n,)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        self.X_ = np.vstack([self.X_, X])
        self.y_ = np.concatenate([self.y_, y])

    def best(self):
        """Return (x_best, y_best) among all points told so far (minimum y)."""
        if len(self.y_) == 0:
            raise RuntimeError("No evaluations recorded yet; call tell() first.")
        i = np.argmin(self.y_)
        return self.X_[i], self.y_[i]
