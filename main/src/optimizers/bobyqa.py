"""BOBYQA (Powell, 2009) -- a model-based trust-region local solver.

BOBYQA is closed-loop: like most classical derivative-free solvers, the
underlying library calls the objective function itself rather than
accepting values back from outside. This project's own survey of the
gradient/DFO landscape (docs/.../literature/A2_gradient_dfo_exact.md,
section 1.1) already identifies the standard fix: run the solver in a
background thread, and have its objective-function callback post each
proposed point to a queue and block until the main thread supplies the
value. That is what this module does, giving BOBYQA the same ask/tell
interface as every other optimizer in this package.

NEWUOA is BOBYQA's unconstrained counterpart (same underlying algorithm,
no box constraints); it is not used here since bounds are always
available from the task.

Implementation: Py-BOBYQA (pip package `Py-BOBYQA`, import name
`pybobyqa`; Cartis, Fiala, Marteau & Roberts, ACM TOMS 2019), a pure-Python
implementation. Two other options were tried first and do not work in
this project's venv: the PyPI package literally named `prima` is an
unrelated, broken placeholder (version "0.0", no importable module), and
`pdfo` (its real, legitimate dependency, also cited in this project's own
optimizer survey) ships compiled extensions built against the numpy 1.x
ABI, which crash under this venv's numpy 2.x (pulled in by `ioh`). Being
pure Python, Py-BOBYQA has no such binary-compatibility issue.

Standard, no-tuning defaults:
- Starting point: the midpoint of the box (same convention as CMAES).
- `scaling_within_bounds=True`, Py-BOBYQA's own built-in convenience for
  box-constrained problems: it internally rescales the search to roughly
  the unit cube using the given bounds, which lets the initial
  trust-region radius (`rhobeg`) default to the library's own standard
  value for that case (0.1) instead of a hand-picked one.
- Final trust-region radius (`rhoend`) is left at the library's own
  default (1e-8). The maximum number of function evaluations (`maxfun`)
  defaults to 20,000 (overridable), since the library's own default
  (min(100*(dim+1), 1000)) would cut the search off early -- this
  project's external budget (how many times `ask`/`tell` are called) is
  what controls how long the search runs, exactly as for every other
  optimizer here.

RESOLVED (2026-10-21): this wrapper used to pass maxfun=10**9, and Py-BOBYQA
allocates Python lists of length maxfun (e.g. `[False]*maxfun` for its
restart bookkeeping) -- about 8 GB before the first evaluation. That caused
the previously documented "impractically slow past ~500 evaluations" (high
*system* CPU time = memory paging), and an out-of-memory kill on smaller
machines. With maxfun=20,000 the wrapper runs 1,500+ evaluations on 10-D
Rosenbrock in ~10 s using ~130 MB, so the old 500-evaluation ceiling no
longer applies.

By convention, this optimizer MINIMIZES the objective (matching the rest
of this package).
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
import pybobyqa

from src.optimizers import OptimizerExhausted


class _StopSolver(Exception):
    """Raised inside the solver thread to make it return early (close())."""


_STOP = object()


class BOBYQA:
    """BOBYQA, driven one point at a time via a background solver thread.

    This optimizer only supports asking for one point at a time (n=1): it
    is a strictly sequential local solver, with no batch/population
    concept at all.

    Parameters
    ----------
    bounds : array-like of shape (d, 2)
        Lower and upper bound for each of the d dimensions.
    seed : int, optional
        Unused -- BOBYQA is a deterministic algorithm given a starting
        point and bounds. Accepted for interface consistency with the
        other optimizers in this package.

    Example
    -------
        opt = BOBYQA(bounds=[(-5, 5)] * 10)
        X = opt.ask(n=1)
        y = [f(x) for x in X]
        opt.tell(X, y)
        x_best, y_best = opt.best()
    """

    def __init__(self, bounds, seed: Optional[int] = None, x0=None, maxfun: int = 20_000, **solver_kwargs):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError(
                "bounds must have shape (d, 2): one (low, high) pair per dimension"
            )
        self.dim = self.bounds.shape[0]

        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        # Default start: box midpoint (original behavior). The agent passes
        # x0 = archive incumbent to warm-start instead (agent/warmstart.py).
        if x0 is None:
            self.x0 = (lower + upper) / 2
        else:
            self.x0 = np.clip(np.asarray(x0, dtype=float), lower, upper)
        self._exhausted = False

        self._ask_q: "queue.Queue" = queue.Queue()
        self._tell_q: "queue.Queue" = queue.Queue()
        self.result = None

        def objective(x):
            self._ask_q.put(np.array(x, dtype=float))
            v = self._tell_q.get()
            if v is _STOP:
                raise _StopSolver()
            return v

        def run_solver():
            try:
                self.result = pybobyqa.solve(
                    objective,
                    self.x0,
                    bounds=(lower, upper),
                    scaling_within_bounds=True,
                    # NOT 10**9: Py-BOBYQA allocates per-maxfun Python lists
                    # (e.g. restarts' [False]*maxfun), so a huge maxfun costs
                    # ~8 GB of RAM before the first evaluation. The caller
                    # (campaign runner / agent controller) enforces the real
                    # budget anyway; this is only an upper bound.
                    maxfun=maxfun,
                    do_logging=False,
                    **solver_kwargs,
                )
            except _StopSolver:
                pass  # close() was called: the agent abandoned this solver
            finally:
                self._exhausted = True
                self._ask_q.put(None)  # sentinel: the solver thread has returned

        self._thread = threading.Thread(target=run_solver, daemon=True)
        self._thread.start()

        self.X_ = np.empty((0, self.dim))
        self.y_ = np.empty((0,))

    def ask(self, n: int = 1) -> np.ndarray:
        """Propose 1 new candidate point, shape (1, dim)."""
        if n != 1:
            raise NotImplementedError(
                "BOBYQA only supports asking for one point at a time (n=1); "
                "it is a strictly sequential local solver."
            )
        x = self._ask_q.get()
        if x is None:
            self._exhausted = True
            raise OptimizerExhausted(
                "BOBYQA's solver thread has already returned (converged, or "
                "hit a numerical stopping condition) and will not propose "
                f"further points. Final result: {self.result}"
            )
        return x.reshape(1, -1)

    def tell(self, X, y) -> None:
        """Record the evaluated point `X` (1, dim) and its value `y` (1,)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        self.X_ = np.vstack([self.X_, X])
        self.y_ = np.concatenate([self.y_, y])
        self._tell_q.put(float(y[0]))

    def best(self):
        """Return (x_best, y_best) among all points told so far (minimum y)."""
        if len(self.y_) == 0:
            raise RuntimeError("No evaluations recorded yet; call tell() first.")
        i = np.argmin(self.y_)
        return self.X_[i], self.y_[i]

    def state(self) -> dict:
        """Internal state for the agent's StateReport."""
        return {
            "n_told": int(len(self.y_)),
            "exhausted": self._exhausted,
        }

    def close(self) -> None:
        """Stop the background solver thread (safe to call more than once).
        Used when the agent abandons this optimizer mid-run, so idle solver
        threads don't pile up."""
        if not self._exhausted and self._thread.is_alive():
            self._tell_q.put(_STOP)
        self._exhausted = True
