"""Uniform random search — a minimal, stateless baseline optimizer.

Standard library defaults only; no tuning. Intended as a generic baseline
that can be dropped into any bounded continuous optimization task.

By convention, this optimizer MINIMIZES the objective (matching scipy and
most of the optimization literature). To maximize, negate the objective
values before calling `tell`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class RandomSearch:
    """Uniform random search over a bounded continuous domain.

    Each proposed point is drawn independently and uniformly at random
    from the given box bounds. The optimizer holds no adaptive state beyond
    the evaluation history, so points can be asked for and told about in
    any order or batch size.

    This class does not evaluate the objective itself: the caller asks for
    candidate points, evaluates them with the task's own objective function,
    and reports the results back via `tell`.

    Parameters
    ----------
    bounds : array-like of shape (d, 2)
        Lower and upper bound for each of the d dimensions, e.g.
        [(-5, 5), (-5, 5), ...].
    seed : int, optional
        Seed for the random number generator, for reproducibility.

    Example
    -------
        opt = RandomSearch(bounds=[(-5, 5)] * 10, seed=0)
        X = opt.ask(n=20)            # propose 20 points, shape (20, 10)
        y = [f(x) for x in X]        # evaluate with the task's objective
        opt.tell(X, y)                # record the results
        x_best, y_best = opt.best()  # best point found so far
    """

    def __init__(self, bounds, seed: Optional[int] = None):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError(
                "bounds must have shape (d, 2): one (low, high) pair per dimension"
            )
        self.dim = self.bounds.shape[0]
        self.rng = np.random.default_rng(seed)

        self.X_ = np.empty((0, self.dim))
        self.y_ = np.empty((0,))

    def ask(self, n: int = 1) -> np.ndarray:
        """Propose `n` new candidate points, shape (n, dim)."""
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return self.rng.uniform(low, high, size=(n, self.dim))

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
