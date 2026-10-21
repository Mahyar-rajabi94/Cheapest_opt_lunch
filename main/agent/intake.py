"""Problem intake (design §7): what is known about the problem before any
evaluation is spent, and the defaults that follow from it."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

import yaml

EvalCost = Literal["cheap", "moderate", "expensive", "unknown"]
Noise = Literal["deterministic", "noisy", "unknown"]


@dataclass
class Intake:
    dim: int
    bounds: list
    budget_evals: int
    variable_types: list = field(default_factory=list)  # default: all continuous
    eval_cost: EvalCost = "unknown"
    noise: Noise = "unknown"
    gradients_available: bool = False
    batch_size: int = 1
    constraints: str = "none"
    prior_knowledge: str = ""
    warm_start_points: list = field(default_factory=list)

    def __post_init__(self):
        if not self.variable_types:
            self.variable_types = ["continuous"] * self.dim
        if any(t != "continuous" for t in self.variable_types):
            raise NotImplementedError("v2 supports continuous variables only.")

    # ------------------------------------------------------------- defaults
    def round_size(self) -> int:
        """Evaluations per decision round (design §3.1)."""
        k = max(5 * self.dim, round(0.05 * self.budget_evals))
        return int(min(k, self.budget_evals))

    def probe_budget(self) -> int:
        """Default total budget for landscape probes (design §4.1)."""
        B, d = self.budget_evals, self.dim
        if self.eval_cost == "expensive":
            return int(max(2 * d + 2, round(0.05 * B)))
        # ~12% of the budget, but at least enough for all probes (~25*d)
        # as long as that stays under 30% of the budget.
        return int(max(2 * d + 2, round(0.12 * B), min(25 * d, round(0.3 * B))))

    def allowed_methods(self) -> list[str]:
        """Methods the strategist may choose, given what is known."""
        m = ["cma_es", "de", "gp_bo", "turbo", "bobyqa", "nelder_mead",
             "lbfgsb", "sobol", "random_search"]
        if self.noise == "noisy":
            m.remove("lbfgsb")          # finite differences break under noise
        if self.budget_evals < 20 * self.dim:
            m.remove("de")              # 10*dim population: too few generations
        if self.eval_cost in ("cheap", "unknown") and self.budget_evals > 400:
            m.remove("gp_bo")           # exact-GP cost per step; use TuRBO
        return m

    # -------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load(cls, path) -> "Intake":
        with open(path) as f:
            return cls(**yaml.safe_load(f))
