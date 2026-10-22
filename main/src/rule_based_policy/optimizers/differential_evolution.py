"""Differential Evolution (classic DE/rand/1/bin), Storn & Price (1997).

No common Python library exposes DE through a plain-array ask/tell
interface the way pycma does for CMA-ES: scipy's `differential_evolution`
calls the objective function itself internally rather than accepting
values back from outside, and pymoo/Nevergrad wrap points in their own
object types instead of plain arrays. The algorithm itself is simple and
standard enough to implement directly, so this module does that.

Standard literature defaults only (no tuning): population size
NP = 10 * dim, mutation factor F = 0.8, crossover rate CR = 0.9, strategy
DE/rand/1/bin -- the values most commonly cited as the classic Storn &
Price recommendation.

By convention, this optimizer MINIMIZES the objective (matching
RandomSearch, SobolSearch, and CMAES).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class DifferentialEvolution:
    """Classic Differential Evolution (DE/rand/1/bin).

    On each generation, every population member ("target") is challenged
    by a trial vector: a mutant built from three other, mutually distinct
    population members (mutation: a + F * (b - c)), recombined with the
    target via binomial crossover. The trial replaces the target if it
    scores at least as well (greedy selection).

    Like CMA-ES, DE is population-based: it proposes and updates a whole
    generation at a time internally. This wrapper hides that bookkeeping
    so it can still be driven exactly like RandomSearch/SobolSearch/CMAES:
    ask for any number of points (including one at a time) and tell them
    back in the same order; the underlying generation-level update happens
    transparently once a full generation has been told.

    The first `pop_size` points asked are the random initial population
    itself (there is no separate "warm-up" argument); real mutation and
    crossover only begin once all of those have been told back. Because
    pop_size = 10 * dim here, a budget smaller than 10 * dim will never
    reach a real generation -- see the standard regime notes in
    docs/optimizer_landscape.md (DE needs large budgets, typically
    10^3-10^4 x dim, to show its strength; it is not meant for tiny
    budgets).

    IMPORTANT: as with CMAES, `tell` must be called with exactly the
    points most recently returned by `ask`, in the same order.

    Parameters
    ----------
    bounds : array-like of shape (d, 2)
        Lower and upper bound for each of the d dimensions.
    seed : int, optional
        Seed for the random number generator, for reproducibility.

    Example
    -------
        opt = DifferentialEvolution(bounds=[(-5, 5)] * 10, seed=0)
        X = opt.ask(n=1)
        y = [f(x) for x in X]
        opt.tell(X, y)
        x_best, y_best = opt.best()
    """

    def __init__(self, bounds, seed: Optional[int] = None, init_population=None):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError(
                "bounds must have shape (d, 2): one (low, high) pair per dimension"
            )
        self.dim = self.bounds.shape[0]
        self.rng = np.random.default_rng(seed)

        # Classic Storn & Price (1997) defaults.
        self.pop_size = 10 * self.dim
        self.F = 0.8
        self.CR = 0.9

        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        self._population = self.rng.uniform(
            lower, upper, size=(self.pop_size, self.dim)
        )
        # Warm start: the agent can seed the initial population with good
        # archive points (agent/warmstart.py). Rows beyond pop_size are
        # ignored; missing rows stay uniform random. These seeded points are
        # re-evaluated in the first generation (DE needs their fitness in its
        # own bookkeeping), which costs len(init_population) evaluations.
        if init_population is not None:
            P = np.clip(np.atleast_2d(np.asarray(init_population, float)), lower, upper)
            k = min(len(P), self.pop_size)
            self._population[:k] = P[:k]
        self._pop_fitness = np.full(self.pop_size, np.nan)

        self._phase = "init"  # "init" (first generation) or "evolve"
        self._to_ask: list = []  # [(idx, vector), ...] generated, not yet asked
        self._awaiting_tell: list = []  # [(idx, vector), ...] asked, not yet told

        self.X_ = np.empty((0, self.dim))
        self.y_ = np.empty((0,))

    def ask(self, n: int = 1) -> np.ndarray:
        """Propose `n` new candidate points, shape (n, dim)."""
        out = []
        for _ in range(n):
            if not self._to_ask:
                self._generate_next_batch()
            idx, vec = self._to_ask.pop(0)
            self._awaiting_tell.append((idx, vec))
            out.append(vec)
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

        for yi in y:
            idx, vec = self._awaiting_tell.pop(0)
            if self._phase == "init":
                self._pop_fitness[idx] = yi
                if not np.isnan(self._pop_fitness).any():
                    self._phase = "evolve"
            else:
                if yi <= self._pop_fitness[idx]:
                    self._population[idx] = vec
                    self._pop_fitness[idx] = yi

    def _generate_next_batch(self) -> None:
        if self._phase == "init":
            self._to_ask = [
                (i, self._population[i].copy()) for i in range(self.pop_size)
            ]
            return

        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        batch = []
        for i in range(self.pop_size):
            other_idxs = np.array([j for j in range(self.pop_size) if j != i])
            r1, r2, r3 = self._population[
                self.rng.choice(other_idxs, size=3, replace=False)
            ]
            mutant = np.clip(r1 + self.F * (r2 - r3), lower, upper)

            trial = self._population[i].copy()
            cross_points = self.rng.random(self.dim) < self.CR
            if not cross_points.any():
                cross_points[self.rng.integers(self.dim)] = True
            trial[cross_points] = mutant[cross_points]

            batch.append((i, trial))
        self._to_ask = batch

    def best(self):
        """Return (x_best, y_best) among all points told so far (minimum y)."""
        if len(self.y_) == 0:
            raise RuntimeError("No evaluations recorded yet; call tell() first.")
        i = np.argmin(self.y_)
        return self.X_[i], self.y_[i]

    def state(self) -> dict:
        """Internal state for the agent's StateReport."""
        fit = self._pop_fitness[~np.isnan(self._pop_fitness)]
        spread = float(np.mean(np.std(self._population, axis=0)))
        return {
            "n_told": int(len(self.y_)),
            "phase": self._phase,
            "pop_size": int(self.pop_size),
            "pop_fitness_spread": float(fit.max() - fit.min()) if len(fit) else None,
            "pop_x_spread": spread,
            "exhausted": False,
        }

    def close(self) -> None:
        pass
