"""L-BFGS-B with finite-difference gradients (Byrd, Lu, Nocedal & Zhu, 1995).

L-BFGS-B is closed-loop, like BOBYQA and NelderMead: scipy's
implementation calls the objective function itself rather than accepting
values back from outside. This module reuses the exact same thread+queue
pattern as bobyqa.py and nelder_mead.py (see those modules' docstrings,
and this project's own survey, docs/.../literature/A2_gradient_dfo_exact.md
section 1.1, for why) to give it the same ask/tell interface as every
other optimizer in this package.

No analytic gradient is supplied, deliberately: this is meant to be a
fair black-box baseline, not one that secretly sees the objective's
closed form. Leaving scipy's `jac` argument unset makes L-BFGS-B estimate
the gradient itself via forward finite differences (its own built-in
behaviour, not an extra flag) -- which means each internal "step" costs
roughly dim + 1 objective evaluations (one for the current point, one per
coordinate to estimate the gradient), not one. That cost is the whole
point of comparing this baseline against ones with cheaper or no gradient
estimation.

Standard, no-tuning defaults:
- Starting point: the midpoint of the box (same convention as the other
  local solvers in this package).
- `ftol`, `gtol`, and the finite-difference step `eps` are left at
  scipy's own defaults.
- `maxfun`/`maxiter` are set to effectively unreachable values (10**9) so
  the solver's own default caps (15000 each) never cut the search off
  early -- this project's external budget controls how long the search
  runs, exactly as for every other optimizer here.

Like Nelder-Mead, this is a local method: it is expected to converge (and
stop proposing points) well before a large budget is used. If `ask` is
called after it has converged, `OptimizerExhausted` is raised, which
scripts/run_campaign.py already handles by stopping the campaign cleanly.

By convention, this optimizer MINIMIZES the objective (matching the rest
of this package).
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from src.optimizers import OptimizerExhausted


class _StopSolver(Exception):
    """Raised inside the solver thread to make it return early (close())."""


_STOP = object()


class LBFGSB:
    """L-BFGS-B, driven one point at a time via a background solver thread.

    This optimizer only supports asking for one point at a time (n=1): it
    is a strictly sequential local solver, with no batch/population
    concept at all. Each internal gradient estimate costs roughly dim + 1
    of those individual asks, not one.

    Parameters
    ----------
    bounds : array-like of shape (d, 2)
        Lower and upper bound for each of the d dimensions.
    seed : int, optional
        Unused -- L-BFGS-B is a deterministic algorithm given a starting
        point and bounds. Accepted for interface consistency with the
        other optimizers in this package.

    Example
    -------
        opt = LBFGSB(bounds=[(-5, 5)] * 10)
        X = opt.ask(n=1)
        y = [f(x) for x in X]
        opt.tell(X, y)
        x_best, y_best = opt.best()
    """

    def __init__(self, bounds, seed: Optional[int] = None, x0=None, **solver_kwargs):
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
                self.result = minimize(
                    objective,
                    self.x0,
                    method="L-BFGS-B",
                    bounds=[tuple(b) for b in self.bounds],
                    options={"maxfun": 10**9, "maxiter": 10**9, **solver_kwargs},
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
                "LBFGSB only supports asking for one point at a time (n=1); "
                "it is a strictly sequential local solver."
            )
        x = self._ask_q.get()
        if x is None:
            self._exhausted = True
            raise OptimizerExhausted(
                "L-BFGS-B has already converged (or stopped) and will not "
                f"propose further points. Final result: {self.result}"
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
