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
  is set to an effectively unreachable value (10**9), since its own
  default (min(100*(dim+1), 1000)) would otherwise cut the search off
  early -- this project's external budget (how many times `ask`/`tell`
  are called) is what controls how long the search runs, exactly as for
  every other optimizer here.

KNOWN LIMITATION (investigated, not fixed): driven through this
thread+queue wrapper, BOBYQA reliably works up to a few hundred
evaluations (tested cleanly to budget=500) but becomes impractically slow
somewhere in the next few hundred (observed stalling between roughly 400
and 2000, depending on exact settings). This was tracked down to a
thread-scheduling interaction specific to this library, not the objective
function or any of rhoend/scaling_within_bounds/BLAS-thread settings:
driven synchronously (no thread), the same solver with the same settings
runs to completion in seconds; the same code through this wrapper shows
high *system* (not *user*) CPU time and visibly slows down a few dozen
evaluations in, and this reproduces even with a trivial pure-numpy
objective with no other part of this project involved. A real fix would
likely mean running the solver in a separate process instead of a thread
(no shared GIL/scheduler), which is a larger change than this module
currently makes. Until then, treat budget=500 as this optimizer's tested
practical ceiling here, not an arbitrary choice.

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

    deterministic = True  # seed is unused; one seed is enough in sweeps

    def __init__(self, bounds, seed: Optional[int] = None):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError(
                "bounds must have shape (d, 2): one (low, high) pair per dimension"
            )
        self.dim = self.bounds.shape[0]

        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        self.x0 = (lower + upper) / 2

        self._ask_q: "queue.Queue" = queue.Queue()
        self._tell_q: "queue.Queue" = queue.Queue()
        self.result = None

        def objective(x):
            self._ask_q.put(np.array(x, dtype=float))
            return self._tell_q.get()

        def run_solver():
            res = pybobyqa.solve(
                objective,
                self.x0,
                bounds=(lower, upper),
                scaling_within_bounds=True,
                maxfun=10**9,
                do_logging=False,
            )
            self.result = res
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
