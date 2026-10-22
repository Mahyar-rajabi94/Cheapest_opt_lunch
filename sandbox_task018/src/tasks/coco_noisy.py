"""Noisy BBOB tasks from COCO's `bbob-noisy` suite, via `cocoex`.

Requires `coco-experiment==2.8.2` (import name `cocoex`); not in
requirements.txt, install it into `.venv` by hand.

COCO's noise RNG is global in its C library, cannot be seeded, and resets
whenever a problem is constructed. Every campaign therefore sees the same
noise stream by call index, whatever its seed; `reseed` does nothing.
"""

from __future__ import annotations

import numpy as np

from src.tasks.base import Task
from src.tasks.coco_constrained import coco_x_star


class COCONoisyTask(Task):
    """One noisy Rosenbrock from `bbob-noisy` (110: Gaussian, 111: uniform,
    112: Cauchy noise), with COCO's noise as-is.

    The box is [-5, 5] per dimension. f110-f112 of one instance share x* and
    the noise-free Rosenbrock; `f_opt` is f110 at x*, where its noise
    vanishes. `y_true` reimplements COCO's noise-free Rosenbrock.
    """

    noisy = True

    def __init__(self, function_id: int, dim: int = 5, instance: int = 1):
        import cocoex

        suite = cocoex.Suite("bbob-noisy", "", "")
        reference = suite.get_problem_by_function_dimension_instance(110, dim, instance)
        self._x_star = coco_x_star(reference)
        self.f_opt = float(reference(self._x_star))
        # Constructed last: construction resets COCO's noise stream.
        self._problem = suite.get_problem_by_function_dimension_instance(function_id, dim, instance)
        if self._problem is None or function_id not in (110, 111, 112):
            raise ValueError(f"No noisy Rosenbrock f{function_id}, dim {dim}, instance {instance}.")
        if not np.array_equal(coco_x_star(self._problem), self._x_star):
            raise ValueError(f"f{function_id}: x* differs from f110's.")

        self.name = f"bbob_noisy_f{function_id}_d{dim}_i{instance}"
        self.dim = dim
        self.bounds = np.column_stack(
            [self._problem.lower_bounds, self._problem.upper_bounds]
        ).astype(float)
        self._scale = max(1.0, np.sqrt(dim) / 8.0)

    def _f(self, x: np.ndarray) -> float:
        return float(np.asarray(self._problem(x)).reshape(-1)[0])

    def _f_true(self, x: np.ndarray) -> float:
        z = self._scale * (x - self._x_star) + 1.0
        return float(np.sum(100.0 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1.0) ** 2) + self.f_opt)
