"""Noisy tasks from nevergrad's `ArtificialFunction`, with our own noise.

Requires `nevergrad==1.0.12`; not in requirements.txt, install it into
`.venv` by hand.

Nevergrad draws its noise from numpy's global RNG at call time, tangling it
with optimizer randomness. So the instance (the shift) is fixed, built once
from global seed 0, and the noise is regenerated here with nevergrad's exact
formula from a task-owned RNG that `reseed` sets per campaign.
"""

from __future__ import annotations

import numpy as np

from src.tasks.base import Task

NOISE_LEVEL = 10.0
# Unrotated nevergrad functions have their optimum at shift + this offset.
OPTIMUM_OFFSET = {"cigar": 0.0, "rosenbrock": 1.0}


class NevergradNoisyTask(Task):
    """One shifted, unrotated nevergrad function with noise
    NOISE_LEVEL * N(0, 1) * (f(x + N(0, I)) - f(x)).

    The box is [-5, 5] per dimension. The noise-free minimum is 0, so
    `f_opt` = 0; x* is checked to lie inside the box.
    """

    noisy = True

    def __init__(self, function_name: str, dim: int = 20):
        from nevergrad.functions import ArtificialFunction

        np.random.seed(0)
        func = ArtificialFunction(
            name=function_name, block_dimension=dim, noise_level=NOISE_LEVEL, rotation=False
        )
        func.transform_var._initialize()
        self._transform = func._transform
        self._func = func.function_from_transform

        self.name = f"nevergrad_noisy_{function_name}_d{dim}"
        self.dim = dim
        self.bounds = np.tile([-5.0, 5.0], (dim, 1))
        self.f_opt = 0.0
        x_star = func.transform_var._transforms[0].translation + OPTIMUM_OFFSET[function_name]
        if np.any(np.abs(x_star) > 5.0):
            raise ValueError(f"{self.name}: x* lies outside the box.")
        self.reseed(0)

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def _f(self, x: np.ndarray) -> float:
        # Same draw order as nevergrad's _noisy_call: side point, then scale.
        fx = self._func(self._transform(x))
        side = self._func(self._transform(x + self._rng.normal(0, 1, size=len(x))))
        return float(fx + NOISE_LEVEL * self._rng.normal(0, 1) * (side - fx))

    def _f_true(self, x: np.ndarray) -> float:
        return float(self._func(self._transform(x)))
