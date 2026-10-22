"""Black-box evaluation harness for Agent Task 060.

Do not inspect this file or try to work around the budget enforcement --
see agent_briefs/task_060.md for the rules of this exercise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.agent_harness.harness import BlackBoxHarness, BudgetExhausted

__all__ = ["AgentTaskHarness", "BudgetExhausted"]

SANDBOX_ASSETS = ("assets/task_060.model", "assets/task_060_bounds.json")


class AgentTaskHarness(BlackBoxHarness):
    def __init__(self, budget: int, outdir: str = "results/agent_task_060"):
        assets = Path(__file__).resolve().parent / "assets"
        bounds = np.asarray(
            json.loads((assets / "task_060_bounds.json").read_text()), dtype=float
        )
        if (
            bounds.ndim != 2
            or bounds.shape[1] != 2
            or not len(bounds)
            or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] > bounds[:, 1])
        ):
            raise ValueError("Invalid task bounds")

        problem = xgb.Booster(params={"nthread": 1})
        problem.load_model(str(assets / "task_060.model"))
        dim = int(problem.num_features())
        if bounds.shape != (dim, 2):
            raise ValueError("Task bounds do not match the input dimension")

        def objective(x):
            if x.shape != (dim,):
                raise ValueError(f"x must have shape ({dim},), got {x.shape}")
            if not np.isfinite(x).all():
                raise ValueError("x must contain only finite values")
            if np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1]):
                raise ValueError("x must lie within the task bounds")
            return -float(problem.inplace_predict(x.reshape(1, -1))[0])

        super().__init__(objective, bounds, budget, outdir, task_id="task_060")
