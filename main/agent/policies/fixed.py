"""Fixed single-method baseline through the same controller: start
`method` (cold, from the box center) and keep running it. With
restarts=True, an exhausted/converged method is restarted (IPOP-style for
CMA-ES) instead of stopping. Used for comparisons and ablations."""

from __future__ import annotations

from agent.schema import Action


class FixedPolicy:
    def __init__(self, method: str, restarts: bool = False):
        self.method = method
        self.restarts = restarts
        self.name = f"fixed_{method}" + ("_restarts" if restarts else "")
        self._n_restart = 0

    def meta(self) -> dict:
        return {"fixed_method": self.method, "restarts": self.restarts}

    def decide(self, r: dict) -> Action:
        if r["evals_used"] == 0:
            return Action(kind="switch", method=self.method, start="center")
        if r.get("last_event") in ("exhausted", "converged"):
            if not self.restarts:
                return Action(kind="stop")
            self._n_restart += 1
            return Action(kind="restart", method=self.method, start="random",
                          popsize_factor=min(2.0 ** self._n_restart, 8.0)
                          if self.method == "cma_es" else None)
        return Action(kind="continue")
