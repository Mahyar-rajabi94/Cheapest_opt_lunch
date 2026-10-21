"""Budget-enforced evaluation with trace logging. Every evaluation in a run
goes through Evaluator.evaluate(), which also writes it to the archive.
The trace format matches the existing campaign/harness trace.csv."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from agent.archive import Archive


class BudgetExhausted(Exception):
    pass


class Evaluator:
    def __init__(self, objective, bounds, budget: int, archive: Archive):
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=float)
        self.budget = int(budget)
        self.archive = archive
        self.rows: list[dict] = []
        self._cum = 0.0
        self._t0 = time.perf_counter()
        self.start_time_utc = datetime.now(timezone.utc).isoformat()

    @property
    def used(self) -> int:
        return len(self.rows)

    @property
    def left(self) -> int:
        return self.budget - self.used

    def evaluate(self, x, tag: str) -> float:
        if self.left <= 0:
            raise BudgetExhausted(f"budget of {self.budget} evaluations used")
        x = np.clip(np.asarray(x, dtype=float), self.bounds[:, 0], self.bounds[:, 1])
        t = time.perf_counter()
        y = float(self.objective(x))
        dt = time.perf_counter() - t
        self._cum += dt
        self.archive.add(x, y, tag)
        best = float(self.archive.best_curve()[-1])
        row = {"eval_idx": self.used + 1, "batch_idx": self.used, "optimizer": tag}
        row.update({f"x{i}": float(v) for i, v in enumerate(x)})
        row.update({"y": y, "y_best_so_far": best, "eval_seconds": dt,
                    "cumulative_seconds": self._cum})
        self.rows.append(row)
        return y

    def save(self, outdir, meta_extra: dict | None = None) -> None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        if self.rows:
            with open(outdir / "trace.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
                w.writeheader()
                w.writerows(self.rows)
        meta = {
            "task_dim": int(self.bounds.shape[0]),
            "task_direction": "min",
            "requested_budget": self.budget,
            "evaluations_used": self.used,
            "stopped_early": self.used < self.budget,
            "start_time_utc": self.start_time_utc,
            "end_time_utc": datetime.now(timezone.utc).isoformat(),
            "total_wall_seconds": time.perf_counter() - self._t0,
            "numpy_version": np.__version__,
        }
        meta.update(meta_extra or {})
        with open(outdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
