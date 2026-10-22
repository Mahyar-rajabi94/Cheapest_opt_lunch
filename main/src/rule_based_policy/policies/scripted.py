"""Scripted (rule-based) policy: the no-LLM baseline.

It uses the same StateReport and Action menu as the LLM strategist, with
fixed rules on the probe features. Comparing the LLM against this policy
measures what the LLM's reasoning adds beyond hand-written rules.
"""

from __future__ import annotations

from src.rule_based_policy.schema import Action

LOCAL = ("lbfgsb", "bobyqa", "nelder_mead")


def classify(probes: dict | None) -> str:
    """'smooth_single_basin' | 'multi_basin' | 'rugged' | 'unknown'."""
    if not probes:
        return "unknown"
    p4, p5, p2 = probes.get("P4") or {}, probes.get("P5") or {}, probes.get("P2") or {}
    cos = p5.get("cosine_min")
    smooth = cos is not None and cos > 0.9
    contraction = p4.get("endpoint_contraction")
    y_spread = p4.get("endpoint_y_spread")
    gain = p4.get("rel_gain_median")
    fdc = p2.get("fdc_spearman")
    if smooth and gain is not None and gain > 0.5 and (y_spread is None or y_spread < 0.5):
        return "smooth_single_basin"
    if smooth or (fdc is not None and fdc > 0.3):
        return "multi_basin"
    return "rugged"


class ScriptedPolicy:
    name = "scripted"

    def __init__(self, use_probes: bool = True):
        self.use_probes = use_probes
        self.landscape = "unknown"

    def meta(self) -> dict:
        return {"policy_use_probes": self.use_probes, "landscape_guess": self.landscape}

    def _local(self, allowed):
        for m in LOCAL:
            if m in allowed:
                return m
        return "cma_es"

    def decide(self, r: dict) -> Action:
        allowed = r["problem"]["allowed_methods"]
        k = r["round_size"]
        if r["evals_used"] == 0:
            if self.use_probes:
                return Action(kind="probe", rationale="no data yet: characterize the landscape first")
            return Action(kind="switch", method="cma_es", start="center",
                          rationale="no probes: CMA-ES is the robust default")

        if r["round"] == 1 and r.get("probes"):
            self.landscape = classify(r["probes"])
            if self.landscape == "smooth_single_basin":
                return Action(kind="polish", method=self._local(allowed),
                              rationale="probes: smooth single basin -> local solver from incumbent")
            if self.landscape == "multi_basin":
                return Action(kind="switch", method="cma_es",
                              rationale="probes: several basins -> CMA-ES from incumbent, restarts later")
            return Action(kind="switch", method="cma_es", start="center", popsize_factor=2.0,
                          rationale="probes: rugged -> CMA-ES with larger population")

        active = (r.get("active") or {}).get("method")
        event = r.get("last_event")
        stall = (r.get("progress") or {}).get("stall_evals", 0)

        # a local solver finished: look for another basin, then polish again
        if active in LOCAL and event == "exhausted":
            if r["evals_left"] < 3 * k:
                return Action(kind="stop", rationale="local solver converged, little budget left")
            return Action(kind="restart", method="cma_es", start="diverse", popsize_factor=2.0,
                          rationale="local solver converged: search another basin")
        # a global method converged or stalled: polish its best point
        if active not in LOCAL and (event == "converged" or stall >= 2 * k):
            return Action(kind="polish", method=self._local(allowed), start="active_best",
                          rationale=f"{active} converged/stalled ({stall} evals) -> polish its best point")
        if active in LOCAL and stall >= 2 * k:
            return Action(kind="restart", method="cma_es", start="diverse", popsize_factor=2.0,
                          rationale="local solver stalled")
        return Action(kind="continue", rationale="improving")
