"""Black-box evaluation harness for Agent Task 006.

Do not inspect this file or try to work around the budget enforcement --
see agent_briefs/task_006.md for the rules of this exercise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.agent_harness.harness import BlackBoxHarness, BudgetExhausted

__all__ = ["AgentTaskHarness", "BudgetExhausted"]

SANDBOX_ASSETS = ("assets/task_006.npz",)
_TIE_TOL = 1e-9


class AgentTaskHarness(BlackBoxHarness):
    def __init__(self, budget: int, outdir: str = "results/agent_task_006"):
        with np.load(Path(__file__).resolve().parent / "assets" / "task_006.npz") as z:
            points, values = z["points"], z["values"]
        dim = points.shape[1]
        used = np.zeros(len(points), dtype=bool)

        def objective(x):
            if x.shape != (dim,):
                raise ValueError(f"x must have shape ({dim},), got {x.shape}")
            if not np.isfinite(x).all():
                raise ValueError("x must contain only finite values")
            if np.any(x < 0.0) or np.any(x > 1.0):
                raise ValueError("x must lie within the task bounds")
            if used.all():
                raise RuntimeError("no further values are available")
            dist = np.sqrt(((points - x) ** 2).sum(axis=1))
            dist[used] = np.inf
            i = int(np.flatnonzero(dist <= dist.min() + _TIE_TOL)[0])
            used[i] = True
            return float(values[i])

        bounds = np.tile([0.0, 1.0], (dim, 1))
        super().__init__(objective, bounds, budget, outdir, task_id="task_006")
