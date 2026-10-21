"""Task interface shared by every benchmark task.

A task is a fixed search space, a known optimum for offline regret
computation, and a single `evaluate()` method that calls the objective and
times it. Subclasses set the attributes and implement `_f`; scripts only use
the interface below. A constrained task also sets `n_constraints` and
implements `_g`.

By convention, every task MINIMIZES (matching this project's optimizers in
src/optimizers/). A maximization problem negates its value inside `_f`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# Same feasibility tolerance as the optbench reference (normalize_constraints).
FEASIBILITY_TOL = 1e-10


@dataclass(frozen=True)
class Evaluation:
    """Result of one `Task.evaluate`: objective, constraint values, timing."""

    y: float
    g: tuple[float, ...]  # inequality constraints, feasible iff all <= 0; empty if unconstrained
    elapsed: float

    @property
    def violation(self) -> float:
        return sum(max(v, 0.0) for v in self.g)

    @property
    def feasible(self) -> bool:
        return self.violation <= FEASIBILITY_TOL


class Task(ABC):
    """Base class for a benchmark task.

    Attributes
    ----------
    name : str
        Results folder name; encodes function, dim, and instance
        (e.g. "rosenbrock_d10_i1").
    direction : str
        Always "min".
    dim : int
    bounds : np.ndarray of shape (dim, 2)
        [lower, upper] bound per dimension.
    f_opt : float
        The true (or best-known) minimum. Used only for offline regret
        computation, never passed to the optimizer. For a constrained task
        it is the constrained optimum.
    n_constraints : int
        Number of inequality constraints (default 0, unconstrained).

    Example
    -------
        task = TASKS["rosenbrock10"]()
        ev = task.evaluate(x)           # x has length task.dim
        regret = ev.y - task.f_opt      # ev.g, ev.feasible for constrained tasks
    """

    name: str
    direction = "min"
    dim: int
    bounds: np.ndarray
    f_opt: float
    n_constraints = 0

    @abstractmethod
    def _f(self, x: np.ndarray) -> float:
        """Objective value at one point `x` (length == self.dim)."""

    def _g(self, x: np.ndarray) -> np.ndarray:
        """Inequality constraints at `x`, feasible iff all <= 0. Constrained
        tasks override this."""
        return np.empty(0)

    def evaluate(self, x) -> Evaluation:
        """Evaluate one point `x` (length == self.dim); the objective and the
        constraints are timed together."""
        x = np.asarray(x, dtype=float)
        start = time.perf_counter()
        y = float(self._f(x))
        g = tuple(float(v) for v in np.asarray(self._g(x)).reshape(-1))
        elapsed = time.perf_counter() - start
        return Evaluation(y, g, elapsed)
