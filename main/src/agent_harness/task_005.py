"""Black-box evaluation harness for Agent Task 005.

Do not inspect this file or try to work around the budget enforcement --
see agent_briefs/task_005.md for the rules of this exercise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.agent_harness.harness import BlackBoxHarness, BudgetExhausted

__all__ = ["AgentTaskHarness", "BudgetExhausted"]

SANDBOX_ASSETS = ("assets/task_005.npz",)
_SNAP_TOL = 1e-9


class AgentTaskHarness(BlackBoxHarness):
    def __init__(self, budget: int, outdir: str = "results/agent_task_005"):
        with np.load(Path(__file__).resolve().parent / "assets" / "task_005.npz") as z:
            lower, upper = z["lower"], z["upper"]
            levels = {int(i): z[f"levels_{i}"] for i in z["snap_dims"]}
            left, right, feature = z["left"], z["right"], z["feature"].astype(np.intp)
            threshold, value, roots = z["threshold"], z["value"], z["roots"]
        dim = len(lower)

        def objective(x):
            if x.shape != (dim,):
                raise ValueError(f"x must have shape ({dim},), got {x.shape}")
            if not np.isfinite(x).all():
                raise ValueError("x must contain only finite values")
            if np.any(x < 0.0) or np.any(x > 1.0):
                raise ValueError("x must lie within the task bounds")
            X = lower + x[None, :] * (upper - lower)
            for i, lv in levels.items():
                dist = np.abs(X[:, i, None] - lv)
                X[:, i] = lv[np.argmax(dist <= dist.min(axis=1, keepdims=True) + _SNAP_TOL, axis=1)]
            X = X.astype(np.float32).astype(np.float64)
            node = roots[None, :].copy()
            while True:
                inner = left[node] >= 0
                if not inner.any():
                    return -float(value[node].mean(axis=1)[0])
                go_left = X[0, feature[node]] <= threshold[node]
                node = np.where(inner, np.where(go_left, left[node], right[node]), node)

        bounds = np.tile([0.0, 1.0], (dim, 1))
        super().__init__(objective, bounds, budget, outdir, task_id="task_005")
