"""Rule-based adaptive router: probe the landscape, then switch between this
project's own optimizers by fixed heuristics (no LLM).

This is a port of the "scripted" policy developed on the `agent-v2` branch
(see main_agent_v2/design/agent_design.md and CLAUDE.md): a round-based
controller runs landscape probes (space-filling scan, fitness-distance
correlation, a quadratic fit, multi-start local runs, gradient-consistency
checks -- src/rule_based_policy/probes.py), classifies the landscape as
roughly smooth-unimodal / multi-basin / rugged, and routes to CMA-ES,
L-BFGS-B, BOBYQA, etc. accordingly, carrying every evaluation forward in a
shared archive so a method switch never restarts from the box midpoint
(src/rule_based_policy/warmstart.py). No learning or external API calls are
involved -- it is a fixed set of rules over measured features, used as the
baseline that an LLM-driven strategist is compared against.

Like BOBYQA (src/optimizers/bobyqa.py), the underlying controller is
closed-loop (it calls the objective itself rather than accepting values
back from outside), so it is driven the same way: a background thread runs
it, and its objective callback posts each proposed point to a queue and
blocks for the value supplied back through tell().

IMPORTANT: unlike this package's other optimizers, the controller plans
probe/round budgets from a *known total budget*, decided once at
construction (`budget=` kwarg, default 2000 -- this project's standard).
If actually driven for a different number of evaluations than that, its
internal planning (round size, probe budget) will not match, though it
remains correct (it still enforces its own budget internally and simply
stops proposing points once it does, same as any other closed-loop
solver here).

By convention, this optimizer MINIMIZES the objective (matching the rest
of this package).
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np

from src.optimizers import OptimizerExhausted
from src.rule_based_policy.controller import Controller
from src.rule_based_policy.intake import Intake
from src.rule_based_policy.policies.scripted import ScriptedPolicy

DEFAULT_BUDGET = 2000


class RuleBased:
    """Rule-based (scripted) adaptive router, driven one point at a time.

    This optimizer only supports asking for one point at a time (n=1): the
    underlying controller is strictly sequential.

    Parameters
    ----------
    bounds : array-like of shape (d, 2)
        Lower and upper bound for each of the d dimensions.
    seed : int, optional
        Seed for every stochastic step inside the controller (probes,
        CMA-ES, etc.).
    budget : int, optional
        Total evaluations this run is planned for (default 2000, this
        project's standard budget). See module docstring.

    Example
    -------
        opt = RuleBased(bounds=[(-5, 5)] * 10, seed=0)
        X = opt.ask(n=1)
        y = [f(x) for x in X]
        opt.tell(X, y)
        x_best, y_best = opt.best()
    """

    def __init__(self, bounds, seed: Optional[int] = None, budget: int = DEFAULT_BUDGET):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError(
                "bounds must have shape (d, 2): one (low, high) pair per dimension"
            )
        self.dim = self.bounds.shape[0]

        self._ask_q: "queue.Queue" = queue.Queue()
        self._tell_q: "queue.Queue" = queue.Queue()
        self._tag_q: "queue.Queue" = queue.Queue()
        self.result = None

        def objective(x):
            self._ask_q.put(np.array(x, dtype=float))
            return self._tell_q.get()

        intake = Intake(dim=self.dim, bounds=self.bounds.tolist(), budget_evals=int(budget))
        policy = ScriptedPolicy()
        self._controller = Controller(objective, intake, policy, seed=int(seed or 0))

        # The controller's internal round/probe/method tag (e.g. "probe:P1",
        # "cma_es", "race:turbo") is known to its own Evaluator.evaluate(x,
        # tag) at call time, but never reaches the plain `objective(x)`
        # callback above. Wrap it so each tag is queued right before the
        # matching point -- consumed in ask(), in lock-step with X/y, so
        # run_campaign.py's trace can tag each row with the real sub-method
        # instead of a single flat "rule_based" label.
        _orig_evaluate = self._controller.ev.evaluate

        def _evaluate_with_tag(x, tag):
            self._tag_q.put(tag)
            return _orig_evaluate(x, tag)

        self._controller.ev.evaluate = _evaluate_with_tag

        def run_controller():
            res = self._controller.run()
            self.result = res
            self._ask_q.put(None)  # sentinel: the controller has returned

        self._thread = threading.Thread(target=run_controller, daemon=True)
        self._thread.start()

        self.X_ = np.empty((0, self.dim))
        self.y_ = np.empty((0,))
        self.tags_: list[str] = []

    def ask(self, n: int = 1) -> np.ndarray:
        """Propose 1 new candidate point, shape (1, dim)."""
        if n != 1:
            raise NotImplementedError(
                "RuleBased only supports asking for one point at a time (n=1); "
                "the underlying controller is strictly sequential."
            )
        x = self._ask_q.get()
        if x is None:
            raise OptimizerExhausted(
                "RuleBased's controller thread has already returned (budget "
                "exhausted, or it chose to stop early) and will not propose "
                f"further points. Final result: {self.result}"
            )
        self.tags_.append(self._tag_q.get())
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
