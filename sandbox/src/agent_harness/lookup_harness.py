"""Budget-limited evaluation harness for lookup-table (fixed finite candidate
pool) tasks, generic over any lookup_objective.py.

Reads CANDIDATES (list of opaque candidate labels, e.g. SMILES strings),
DIRECTION, NOISY and fn(idx) from lookup_objective.py (at the sandbox root)
and wraps them with budget enforcement, no-repeat-selection enforcement, and
automatic trace logging. Mirrors AgentTaskHarness's protocol (see Agent.md)
but the action is "pick one candidate index from the pool" instead of
"propose one continuous point": there is no gradient/interpolation structure
between candidates, so this is a pure discrete selection-without-replacement
problem over a finite, fully known set of options.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import lookup_objective as _objective


class BudgetExhausted(Exception):
    """Raised by evaluate() once the task's evaluation budget is used up."""


class CandidateAlreadyEvaluated(Exception):
    """Raised by evaluate() if the given candidate index was already queried.
    Each candidate may only be evaluated once, matching this benchmark's own
    baselines (no re-measurement of an already-sampled molecule). Does not
    consume budget."""


class LookupTaskHarness:
    """Budget-limited lookup-table objective with automatic trace logging.

    Parameters
    ----------
    budget : int
        Maximum number of evaluate() calls allowed. Enforced: a call past
        this limit raises BudgetExhausted rather than returning a value.
    outdir : str
        Where trace.csv and meta.json are written (created automatically).

    Usage
    -----
        harness = LookupTaskHarness(budget=275)
        print(harness.num_candidates, harness.direction)
        print(harness.candidates[0])      # e.g. a SMILES string
        harness.set_active_optimizer("greedy_similarity")
        y = harness.evaluate(17)          # 17: an integer index into harness.candidates
        harness.set_active_optimizer("random_explore")
        ...
        harness.save()                    # or let it auto-save at budget exhaustion
    """

    def __init__(self, budget: int, outdir: str = "results/agent_run"):
        if _objective.CANDIDATES is None:
            raise RuntimeError(
                "lookup_objective.py is not configured: set CANDIDATES, "
                "DIRECTION and implement fn(idx) before creating the harness."
            )

        self.candidates = list(_objective.CANDIDATES)
        self.num_candidates = len(self.candidates)
        self.direction = _objective.DIRECTION
        self.noisy = bool(getattr(_objective, "NOISY", False))
        self._fn = _objective.fn
        self.budget = int(budget)

        self._n_used = 0
        self._active_optimizer = "unspecified"
        self._rows: list = []
        self._best_so_far = None
        self._cumulative_seconds = 0.0
        self._used_idxs: set[int] = set()
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

    def evaluate(self, idx: int) -> float:
        """Evaluate candidate `idx` (0 <= idx < self.num_candidates). Returns y.

        Raises BudgetExhausted if the budget has already been used up, or
        CandidateAlreadyEvaluated if this index was already queried (neither
        consumes budget in the latter case).
        """
        idx = int(idx)
        if not (0 <= idx < self.num_candidates):
            raise IndexError(f"idx {idx} out of range [0, {self.num_candidates})")
        if idx in self._used_idxs:
            raise CandidateAlreadyEvaluated(
                f"Candidate {idx} was already evaluated; pick an unused index."
            )
        if self._n_used >= self.budget:
            raise BudgetExhausted(
                f"Budget of {self.budget} evaluations has been used up; "
                "no further evaluate() calls are allowed."
            )

        start = time.perf_counter()
        y = float(self._fn(idx))
        elapsed = time.perf_counter() - start

        self._used_idxs.add(idx)
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
            "candidate_idx": idx,
            "candidate": self.candidates[idx],
            "y": y,
            "y_best_so_far": self._best_so_far,
            "eval_seconds": elapsed,
            "cumulative_seconds": self._cumulative_seconds,
        }
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
            "task_num_candidates": self.num_candidates,
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
