"""Nelder-Mead simplex method (Nelder & Mead, 1965; adaptive variant:
Gao & Han, Comput. Optim. Appl. 2012).

Nelder-Mead is closed-loop, like BOBYQA: scipy's implementation calls the
objective function itself rather than accepting values back from outside.
This module uses the same thread+queue pattern as bobyqa.py (see that
module's docstring, and this project's own survey,
docs/.../literature/A2_gradient_dfo_exact.md section 1.1, for why) to give
it the same ask/tell interface as every other optimizer in this package.

Standard, no-tuning defaults:
- Starting point: the midpoint of the box (same convention as CMAES and
  BOBYQA).
- `adaptive=True`: scipy's built-in Gao & Han (2012) parameter scaling,
  the standard recommendation for anything beyond very low dimensions
  (this project's own optimizer survey explicitly notes this setting).
- `maxfev` is set to an effectively unreachable value (10**9) so the
  solver's own default cap (200 * dim) never cuts the search off early --
  this project's external budget controls how long the search runs,
  exactly as for every other optimizer here.
- `xatol`/`fatol` (the convergence tolerances) are left at scipy's own
  defaults (1e-4 each). Unlike BOBYQA's `rhoend`, this was checked
  empirically (see the project history) and found not to cause the same
  kind of pathological slow-down if tightened, but Nelder-Mead is
  documented in this project (docs/claude_deep_serach/optimizer_landscape.md,
  section 2.4) as a local refiner, not a global search method: it is
  expected to converge and stop proposing points well before a large
  budget is used, and that is treated here as normal behaviour, not an
  error to be engineered around with restarts.

By convention, this optimizer MINIMIZES the objective (matching the rest
of this package).
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from src.rule_based_policy.optimizers import OptimizerExhausted


class _StopSolver(Exception):
    """Raised inside the solver thread to make it return early (close())."""


_STOP = object()


class NelderMead:
    """Nelder-Mead, driven one point at a time via a background solver thread.

    This optimizer only supports asking for one point at a time (n=1): it
    is a strictly sequential local solver, with no batch/population
    concept at all.

    As a local method, it is expected to converge (and stop proposing new
    points) well before a large budget is exhausted. If `ask` is called
    after it has converged, a clear RuntimeError is raised rather than
    silently returning something meaningless.

    Parameters
    ----------
    bounds : array-like of shape (d, 2)
        Lower and upper bound for each of the d dimensions.
    seed : int, optional
        Unused -- Nelder-Mead is a deterministic algorithm given a
        starting point and bounds. Accepted for interface consistency
        with the other optimizers in this package.

    Example
    -------
        opt = NelderMead(bounds=[(-5, 5)] * 10)
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
                    method="Nelder-Mead",
                    bounds=[tuple(b) for b in self.bounds],
                    options={"maxfev": 10**9, "adaptive": True, **solver_kwargs},
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
                "NelderMead only supports asking for one point at a time "
                "(n=1); it is a strictly sequential local solver."
            )
        x = self._ask_q.get()
        if x is None:
            self._exhausted = True
            raise OptimizerExhausted(
                "Nelder-Mead has already converged (or stopped) and will "
                f"not propose further points. Final result: {self.result}"
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
