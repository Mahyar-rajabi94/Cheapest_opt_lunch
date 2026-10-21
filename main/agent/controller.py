"""Round-based controller (design §3).

The controller owns all numerics: it runs optimizers, probes and races,
enforces the budget, and builds the StateReport. The policy (scripted rules
or the LLM strategist) only sees StateReports and returns Actions.

    ctl = Controller(objective, intake, policy, seed=0)
    result = ctl.run()
    ctl.save(outdir)
"""

from __future__ import annotations

import traceback
from typing import Optional

import numpy as np
from pydantic import ValidationError

from agent.archive import Archive
from agent.decision_log import DecisionLog
from agent.evaluator import BudgetExhausted, Evaluator
from agent.intake import Intake
from agent.schema import Action
from agent.warmstart import LOCAL_METHODS, METHODS, make_optimizer
from src.optimizers import OptimizerExhausted


def _sig(v, n=6):
    """Round to n significant digits for compact reports."""
    if v is None or not np.isfinite(v):
        return v
    return float(f"{v:.{n}g}")


class Controller:
    def __init__(self, objective, intake: Intake, policy, seed: int = 0,
                 max_rounds: int = 500):
        self.intake = intake
        self.policy = policy
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.bounds = np.asarray(intake.bounds, dtype=float)
        self.dim = intake.dim
        self.archive = Archive(self.bounds)
        self.ev = Evaluator(objective, self.bounds, intake.budget_evals, self.archive)
        self.log = DecisionLog()
        self.k = intake.round_size()
        self.allowed = intake.allowed_methods()
        self.max_rounds = max_rounds

        self.round = 0
        self.active: Optional[dict] = None     # {method, opt, start_eval, start_best}
        self.method_stats: dict[str, dict] = {}
        self.probe_features: Optional[dict] = None
        self.last_race: Optional[dict] = None
        self.last_event: Optional[str] = None
        self.last_action: Optional[Action] = None
        self._round_start_best: Optional[float] = None
        self._n_seed = 0

    # ================================================================ running
    def _eval(self, x, tag):
        return self.ev.evaluate(x, tag)

    def run_optimizer(self, opt, n: int, tag: str, check_converged: bool = True):
        """Ask/tell `opt` for up to n evaluations. Returns (n_done, event),
        event in {None, 'exhausted', 'converged', 'budget'}."""
        done = 0
        n = min(n, self.ev.left)
        while done < n:
            try:
                X = opt.ask(1)
            except OptimizerExhausted:
                return done, "exhausted"
            try:
                y = self._eval(X[0], tag)
            except BudgetExhausted:
                return done, "budget"
            opt.tell(X, [y])
            done += 1
            st = self._stats(tag)
            st["evals"] += 1
            st["best"] = min(st["best"], y)
            if check_converged and done % max(1, self.dim) == 0:
                s = opt.state()
                if s.get("converged"):
                    return done, "converged"
        return done, ("budget" if self.ev.left <= 0 else None)

    def _stats(self, tag):
        return self.method_stats.setdefault(tag, {"evals": 0, "best": np.inf, "rounds": 0})

    def _new_opt(self, method, start="incumbent", **kw):
        self._n_seed += 1
        return make_optimizer(method, self.bounds, self.archive, start=start,
                              seed=self.seed * 1000 + self._n_seed, **kw)

    def _activate(self, method, opt):
        if self.active is not None:
            self.active["opt"].close()
        best = self.archive.incumbent()[1] if len(self.archive) else np.inf
        self.active = {"method": method, "opt": opt, "start_eval": self.ev.used,
                       "start_best": best}

    # ============================================================== execution
    def execute(self, a: Action) -> str:
        """Run one action. Returns a note for the decision log."""
        self.last_event = None
        k = min(self.k, self.ev.left)

        if a.kind == "stop":
            return "stop"

        if a.kind == "continue":
            if self.active is None:
                raise ValueError("'continue' with no active method")
            m = self.active["method"]
            _, ev = self.run_optimizer(self.active["opt"], k, m)
            self.last_event = ev
            return f"continued {m}"

        if a.kind in ("switch", "restart"):
            m = a.method or (self.active["method"] if self.active else None)
            if m is None:
                raise ValueError("'restart' with no active method and no `method`")
            self._check_method(m)
            start = a.start or ("incumbent" if a.kind == "switch" else "diverse")
            if start == "active_best":
                start = "incumbent"  # switch/restart: treat as incumbent
            kw = {}
            if a.popsize_factor and m == "cma_es":
                kw["popsize"] = int(round((4 + 3 * np.log(self.dim)) * a.popsize_factor))
            if start != "incumbent" and m in ("cma_es",):
                kw["sigma0"] = 0.2 * float(np.mean(self.archive.width))
            opt = self._new_opt(m, start=start, **kw) if len(self.archive) else \
                self._new_opt(m, start="center", **kw)
            self._activate(m, opt)
            _, ev = self.run_optimizer(opt, a.budget or k, m)
            self.last_event = ev
            return f"{a.kind} -> {m} (start={start}{', ' + str(kw) if kw else ''})"

        if a.kind == "polish":
            m = a.method
            self._check_method(m)
            if m not in LOCAL_METHODS:
                raise ValueError(f"polish needs a local method, got {m}")
            if not len(self.archive):
                opt = self._new_opt(m, start="center")
            elif a.start == "active_best" and self.active and len(self.active["opt"].y_):
                opt = self._new_opt(m, x0=self.active["opt"].best()[0])
            else:
                opt = self._new_opt(m, start="incumbent")
            self._activate(m, opt)
            budget = a.budget or self.ev.left
            _, ev = self.run_optimizer(opt, budget, m)
            self.last_event = ev
            return f"polish {m} ({ev or 'budget reached'})"

        if a.kind == "probe":
            from agent.probes import run_probes
            budget = a.budget or self.intake.probe_budget()
            feats = run_probes(self, a.probe_ids, budget)
            self.probe_features = {**(self.probe_features or {}), **feats}
            return f"probes {sorted(feats)}"

        if a.kind == "race":
            from agent.racing import successive_halving
            for c in a.candidates:
                self._check_method(c)
            budget = a.budget or int(0.2 * self.intake.budget_evals)
            budget = min(budget, int(0.3 * self.intake.budget_evals), self.ev.left)
            res = successive_halving(self, a.candidates, budget)
            self.last_race = res
            self._activate(res["winner"], res["winner_opt"])
            return f"race winner {res['winner']}"

        raise ValueError(f"unknown action {a.kind}")

    def _check_method(self, m):
        if m not in METHODS:
            raise ValueError(f"unknown method {m!r}")
        if m not in self.allowed:
            raise ValueError(f"method {m!r} not allowed for this problem ({self.allowed})")

    # ================================================================ report
    def report(self) -> dict:
        n = self.ev.used
        rep: dict = {
            "round": self.round,
            "evals_used": n,
            "evals_left": self.ev.left,
            "round_size": self.k,
            "problem": {
                "dim": self.dim,
                "budget": self.intake.budget_evals,
                "bounds_width": _sig(float(np.mean(self.archive.width))),
                "eval_cost": self.intake.eval_cost,
                "noise": self.intake.noise,
                "allowed_methods": self.allowed,
            },
        }
        if n == 0:
            rep.update({"incumbent": None, "progress": None, "active": None,
                        "methods_tried": {}, "probes": None, "race": None,
                        "last_event": None, "last_action_check": None,
                        "history": []})
            return rep

        x_inc, y_inc = self.archive.incumbent()
        i_inc = self.archive.incumbent_index()
        bc = self.archive.best_curve()
        cps = sorted(set(np.linspace(0, n - 1, min(n, 12)).astype(int)))
        prog = {
            "best_curve": [[int(i + 1), _sig(float(bc[i]))] for i in cps],
            "stall_evals": int(self.archive.evals_since_improvement()),
            "best_at_round_start": _sig(self._round_start_best),
        }
        y = self.archive.y
        prog["y_quantiles_all"] = {q: _sig(float(np.quantile(y, q / 100))) for q in (0, 10, 50, 90)}
        dist_to_bounds = np.minimum(x_inc - self.archive.lower, self.archive.upper - x_inc)
        rep["incumbent"] = {
            "y": _sig(y_inc, 10),
            "found_at_eval": i_inc + 1,
            "by": self.archive.tags[i_inc],
            "n_coords_near_bound": int(np.sum(dist_to_bounds < 0.01 * self.archive.width)),
        }
        rep["progress"] = prog

        if self.active:
            m = self.active["method"]
            used = n - self.active["start_eval"]
            gain = self.active["start_best"] - y_inc if np.isfinite(self.active["start_best"]) else None
            st = {k: (_sig(v) if isinstance(v, float) else v)
                  for k, v in self.active["opt"].state().items()}
            rep["active"] = {"method": m, "evals_in_tenure": used,
                             "improvement_in_tenure": _sig(gain), "state": st}
        else:
            rep["active"] = None

        rep["methods_tried"] = {
            t: {"evals": s["evals"], "best": _sig(s["best"])}
            for t, s in self.method_stats.items()
        }
        rep["probes"] = self.probe_features
        rep["race"] = ({k: v for k, v in self.last_race.items() if k != "winner_opt"}
                       if self.last_race else None)
        rep["last_event"] = self.last_event
        la = self.last_action
        if la is not None and la.expect_best_at_most is not None:
            rep["last_action_check"] = {
                "expected_best_at_most": la.expect_best_at_most,
                "actual_best": _sig(y_inc, 10),
                "held": bool(y_inc <= la.expect_best_at_most),
            }
        else:
            rep["last_action_check"] = None
        rep["history"] = [
            {"round": e["round"], "action": {k: v for k, v in e["action"].items()
                                             if v is not None and k != "rationale"},
             "note": e["note"]}
            for e in self.log.entries[-6:]
        ]
        return rep

    # ================================================================== loop
    def _fallback_action(self) -> Action:
        if self.active is None:
            return Action(kind="switch", method="cma_es",
                          rationale="fallback: no active method")
        if self.last_event in ("exhausted", "converged"):
            return Action(kind="restart", method="cma_es", start="diverse",
                          popsize_factor=2.0, rationale="fallback: active method finished")
        return Action(kind="continue", rationale="fallback")

    def run(self) -> dict:
        while self.ev.left > 0 and self.round < self.max_rounds:
            rep = self.report()
            self._round_start_best = rep["incumbent"]["y"] if rep["incumbent"] else None
            source = getattr(self.policy, "name", "policy")
            try:
                action = self.policy.decide(rep)
                if isinstance(action, dict):
                    action = Action(**action)
            except Exception as e:  # invalid output, API error, ... -> never crash a run
                action, source = self._fallback_action(), f"fallback ({type(e).__name__}: {str(e)[:200]})"

            try:
                note = self.execute(action)
            except (ValueError, NotImplementedError) as e:
                fb = self._fallback_action()
                note = f"invalid action ({e}); fallback: " + self.execute(fb)
                action, source = fb, f"fallback (invalid: {e})"
            self.log.record(rep, action.model_dump(), source, note)
            self.last_action = action
            if self.active:
                self._stats(self.active["method"])["rounds"] += 1
            self.round += 1
            if action.kind == "stop":
                break
        if self.active:
            self.active["opt"].close()
        x, y = self.archive.incumbent()
        return {"best_y": y, "best_x": x.tolist(), "evals_used": self.ev.used,
                "rounds": self.round}

    def save(self, outdir, meta_extra: dict | None = None) -> None:
        from pathlib import Path
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        self.log.save(outdir / "decisions.jsonl")
        if hasattr(self.policy, "save_transcript"):
            self.policy.save_transcript(outdir / "llm_transcript.jsonl")
        self.intake.save(outdir / "intake.yaml")
        meta = {"policy": getattr(self.policy, "name", "?"), "seed": self.seed,
                "rounds": self.round, "round_size": self.k}
        meta.update(getattr(self.policy, "meta", lambda: {})())
        meta.update(meta_extra or {})
        self.ev.save(outdir, meta)
