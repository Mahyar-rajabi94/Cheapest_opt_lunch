"""Uniform random selection from a fixed pool of candidates, without
replacement -- the pool counterpart of RandomSearch.

At every step each candidate not yet proposed is equally likely, so this is
the pure-luck reference for a pool task: the expected number of picks to
reach one of K target candidates in a pool of N is (N + 1)/(K + 1).
RandomSearch on a pool task is not this: its uniform points favour
candidates with much empty space around them.

Only for pool tasks: it needs one point per candidate, which
scripts/run_campaign.py passes from the task (`task.candidates`). It
proposes those exact points, each of which the task maps to its own
candidate.

By convention, this optimizer MINIMIZES the objective (it ignores the
values when choosing).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.optimizers import OptimizerExhausted


class PoolRandomSearch:
    """Uniform random selection from a pool, without replacement.

    Parameters
    ----------
    bounds : array-like of shape (d, 2)
        Lower and upper bound for each of the d dimensions.
    seed : int, optional
        Seed for the random number generator, for reproducibility.
    candidates : array-like of shape (n, d)
        One point per candidate in the pool.

    Example
    -------
        opt = PoolRandomSearch(bounds=task.bounds, seed=0, candidates=task.candidates)
        X = opt.ask(n=1)             # the point of one random untried candidate
        opt.tell(X, [task.evaluate(X[0]).y])
        x_best, y_best = opt.best()
    """

    pool_only = True  # run_campaign passes candidates=task.candidates; sweeps skip non-pool tasks

    def __init__(self, bounds, seed: Optional[int] = None, candidates=None):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError(
                "bounds must have shape (d, 2): one (low, high) pair per dimension"
            )
        self.dim = self.bounds.shape[0]
        if candidates is None:
            raise ValueError("PoolRandomSearch needs the pool's candidates")
        self.candidates = np.asarray(candidates, dtype=float)
        if self.candidates.ndim != 2 or self.candidates.shape[1] != self.dim:
            raise ValueError(f"candidates must have shape (n, {self.dim})")
        self.rng = np.random.default_rng(seed)
        self._order = self.rng.permutation(len(self.candidates))
        self._next = 0

        self.X_ = np.empty((0, self.dim))
        self.y_ = np.empty((0,))

    def ask(self, n: int = 1) -> np.ndarray:
        """Propose the points of `n` candidates not proposed before, shape (n, dim)."""
        if self._next + n > len(self._order):
            raise OptimizerExhausted("every candidate in the pool has been proposed")
        idx = self._order[self._next:self._next + n]
        self._next += n
        return self.candidates[idx].copy()

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
