"""Budget-limited evaluation harness, generic over any objective.py.

Reads DIM, BOUNDS, DIRECTION, NOISY and fn(x) from objective.py (at the
sandbox root) and wraps them with budget enforcement and automatic trace
logging. See Agent.md for the protocol this is used under.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import objective as _objective


class BudgetExhausted(Exception):
    """Raised by evaluate() once the task's evaluation budget is used up."""


class AgentTaskHarness:
    """Budget-limited objective with automatic trace logging.

    Parameters
    ----------
    budget : int
        Maximum number of evaluate() calls allowed. Enforced: a call past
        this limit raises BudgetExhausted rather than returning a value.
    outdir : str
        Where trace.csv and meta.json are written (created automatically).

    Usage
    -----
        harness = AgentTaskHarness(budget=2000)
        harness.set_active_optimizer("cma_es")
        y = harness.evaluate(x)          # x: length harness.dim
        harness.set_active_optimizer("lbfgsb")   # call again whenever you switch
        ...
        harness.save()                    # or let it auto-save at budget exhaustion
    """

    def __init__(self, budget: int, outdir: str = "results/agent_run"):
        if _objective.DIM is None or _objective.BOUNDS is None:
            raise RuntimeError(
                "objective.py is not configured: set DIM, BOUNDS, DIRECTION "
                "and implement fn(x) before creating the harness."
            )

        self.dim = int(_objective.DIM)
        self.bounds = [[float(lb), float(ub)] for lb, ub in _objective.BOUNDS]
        self.direction = _objective.DIRECTION
        self.noisy = bool(getattr(_objective, "NOISY", False))
        self._fn = _objective.fn
        self.budget = int(budget)

        self._n_used = 0
        self._active_optimizer = "unspecified"
        self._rows: list = []
        self._best_so_far = None
        self._cumulative_seconds = 0.0
        self._saved = False

        self._outdir = Path(outdir)
        self._outdir.mkdir(parents=True, exist_ok=True)
        self._start_time_utc = datetime.now(timezone.utc).isoformat()
        self._start_wall = time.perf_counter()

    def set_active_optimizer(self, name: str) -> None:
        """Tag every subsequent evaluate() call with this label, until
        you call this again with a new one. Call it before the first
        evaluate() too, and every time you switch methods."""
        self._active_optimizer = str(name)

    def evaluate(self, x) -> float:
        """Evaluate one point `x` (length == self.dim). Returns y.

        Raises BudgetExhausted if the budget has already been used up.
        """
        if self._n_used >= self.budget:
            raise BudgetExhausted(
                f"Budget of {self.budget} evaluations has been used up; "
                "no further evaluate() calls are allowed."
            )

        x = np.asarray(x, dtype=float)
        start = time.perf_counter()
        y = float(self._fn(x))
        elapsed = time.perf_counter() - start

        self._n_used += 1
        self._cumulative_seconds += elapsed
        is_better = (
            self._best_so_far is None
            or (self.direction == "min" and y < self._best_so_far)
            or (self.direction == "max" and y > self._best_so_far)
        )
        if is_better:
            self._best_so_far = y

        row = {
            "eval_idx": self._n_used,
            "batch_idx": self._n_used - 1,
            "optimizer": self._active_optimizer,
        }
        for d in range(self.dim):
            row[f"x{d}"] = float(x[d])
        row["y"] = y
        row["y_best_so_far"] = self._best_so_far
        row["eval_seconds"] = elapsed
        row["cumulative_seconds"] = self._cumulative_seconds
        self._rows.append(row)

        if self._n_used >= self.budget:
            self.save()

        return y

    @property
    def evaluations_used(self) -> int:
        return self._n_used

    @property
    def evaluations_remaining(self) -> int:
        return self.budget - self._n_used

    def save(self) -> None:
        """Write trace.csv and meta.json for whatever has been completed
        so far. Called automatically once the budget is exhausted; call
        it yourself if you stop before that point. Safe to call more than
        once (later calls overwrite with the latest state)."""
        end_time_utc = datetime.now(timezone.utc).isoformat()
        total_wall_seconds = time.perf_counter() - self._start_wall

        trace_path = self._outdir / "trace.csv"
        if self._rows:
            with open(trace_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
                writer.writeheader()
                writer.writerows(self._rows)
        else:
            trace_path.write_text("")

        meta = {
            "task_dim": self.dim,
            "task_direction": self.direction,
            "noisy": self.noisy,
            "requested_budget": self.budget,
            "evaluations_used": self._n_used,
            "stopped_early": self._n_used < self.budget,
            "start_time_utc": self._start_time_utc,
            "end_time_utc": end_time_utc,
            "total_wall_seconds": total_wall_seconds,
            "numpy_version": np.__version__,
        }
        with open(self._outdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._saved = True
