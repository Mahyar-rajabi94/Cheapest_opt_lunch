"""Task interface shared by every benchmark task.

A task is a fixed search space, a known optimum for offline regret
computation, and a single `evaluate()` method that calls the objective and
times it. Subclasses set the attributes and implement `_f`; scripts only use
the interface below.

By convention, every task MINIMIZES (matching this project's optimizers in
src/optimizers/). A maximization problem negates its value inside `_f`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np


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
        computation, never passed to the optimizer.

    Example
    -------
        task = TASKS["rosenbrock10"]()
        y, elapsed = task.evaluate(x)   # x has length task.dim
        regret = y - task.f_opt
    """

    name: str
    direction = "min"
    dim: int
    bounds: np.ndarray
    f_opt: float

    @abstractmethod
    def _f(self, x: np.ndarray) -> float:
        """Objective value at one point `x` (length == self.dim)."""

    def evaluate(self, x):
        """Evaluate one point `x` (length == self.dim).

        Returns (y, elapsed_seconds).
        """
        x = np.asarray(x, dtype=float)
        start = time.perf_counter()
        y = float(self._f(x))
        elapsed = time.perf_counter() - start
        return y, elapsed
