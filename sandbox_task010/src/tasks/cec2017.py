"""CEC-2017 tasks F5, F6, F8 (shifted/rotated), via direct equations.

Requires `opfunu==1.0.1`, used only as the source of the official shift
vectors and rotation matrices. Its evaluators are not called. It is not in
requirements.txt (sandbox code does not need it) and imports `pkg_resources`,
so it needs `setuptools<81`.
"""

from __future__ import annotations

import numpy as np

from src.tasks.base import Task

_NAMES = {5: "f5", 6: "f6", 8: "f8"}


class CEC2017Task(Task):
    """One CEC-2017 function (5: Rastrigin, 6: Expanded Schaffer F6,
    8: Non-Continuous Rastrigin) at dimension `dim`.

    The named equation takes priority over a conflicting CEC-2017 ID, so the
    values can differ from opfunu's own evaluators. The box is
    [-half_width, half_width] per dimension (CEC's own is 100); the optimum is
    the shift vector scaled by half_width / 100, with value 100 * function_id.
    A smaller box shrinks the search space relative to the function's
    features: on F6 at the CEC box, every method stays at random-search level
    within a few thousand evaluations.
    """

    def __init__(self, function_id: int, dim: int, half_width: float = 100.0):
        if function_id not in _NAMES:
            raise ValueError(f"Only CEC-2017 F5, F6, F8 are implemented, got F{function_id}.")
        from opfunu.cec_based import cec2017

        data = getattr(cec2017, f"F{function_id}2017")(ndim=dim)
        self.function_id = function_id
        self.shift = np.asarray(data.f_shift, dtype=float).reshape(-1)[:dim] * half_width / 100.0
        self.rotation = np.asarray(data.f_matrix, dtype=float)[:dim, :dim]

        self.name = f"cec2017_{_NAMES[function_id]}_d{dim}" + ("" if half_width == 100.0 else f"_b{half_width:g}")
        self.dim = dim
        self.bounds = np.tile([-half_width, half_width], (dim, 1))
        self.f_opt = 100.0 * function_id

    def _f(self, x: np.ndarray) -> float:
        delta = x - self.shift
        if self.function_id == 8:
            delta = np.where(
                np.abs(delta) > 0.5, np.floor(2.0 * delta + 0.5) / 2.0, delta
            )
        scale = 0.0512 if self.function_id in (5, 8) else 1.0
        z = self.rotation @ (scale * delta)
        if self.function_id == 6:
            z_next = np.roll(z, -1)
            r2 = z * z + z_next * z_next
            value = np.sum(0.5 + (np.sin(np.sqrt(r2)) ** 2 - 0.5) / (1.0 + 0.001 * r2) ** 2)
        else:
            value = np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0)
        return value + self.f_opt
