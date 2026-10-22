"""StateReport (what the strategist sees) and Action (what it may do).
Design §3.2-3.3. The Action JSON schema exported here is what the LLM is
forced to produce, and what ROLE.md's action-menu section is generated from.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

ActionKind = Literal["continue", "switch", "restart", "probe", "race", "polish", "stop"]
StartKind = Literal["incumbent", "diverse", "center", "random", "active_best"]

ACTION_HELP = {
    "continue": "Run the active method for another round.",
    "switch": "Stop the active method; start `method` warm-started from the archive (start defaults to incumbent).",
    "restart": "Fresh instance of `method` (default: active method) from `start` (default: diverse); `popsize_factor` > 1 grows the CMA-ES population (IPOP-style).",
    "probe": "Run landscape probes `probe_ids` (default: all) within `budget` evals. Probe evaluations go into the archive.",
    "race": "Successive-halving race between 2-4 `candidates`, all warm-started, within `budget` evals; the winner becomes the active method.",
    "polish": "Local refinement with `method` (bobyqa | lbfgsb | nelder_mead) from `start` = incumbent (default) or active_best (best point of the currently active method, e.g. after a restart in a new region), until it converges or `budget` (default: all remaining) is used.",
    "stop": "Save and finish early.",
}


class Action(BaseModel):
    kind: ActionKind
    method: Optional[str] = None
    start: Optional[StartKind] = None
    popsize_factor: Optional[float] = Field(None, ge=1.0, le=8.0)
    probe_ids: Optional[list[str]] = None
    candidates: Optional[list[str]] = None
    budget: Optional[int] = Field(None, ge=1)
    rationale: str = Field("", description="1-3 sentences: why, citing evidence from the report.")
    expect_best_at_most: Optional[float] = Field(
        None,
        description="Falsifiable expectation: the best value you expect to have reached by the "
                    "end of this action's round. Checked and reported back next round.",
    )

    @model_validator(mode="after")
    def _check(self):
        if self.kind in ("switch", "polish") and not self.method:
            raise ValueError(f"'{self.kind}' requires `method`")
        if self.kind == "race" and (not self.candidates or not 2 <= len(self.candidates) <= 4):
            raise ValueError("'race' requires 2-4 `candidates`")
        return self


def action_json_schema() -> dict:
    return Action.model_json_schema()


def action_menu_markdown(allowed_methods: list[str] | None = None) -> str:
    lines = ["| kind | effect |", "|---|---|"]
    for k, v in ACTION_HELP.items():
        lines.append(f"| `{k}` | {v} |")
    if allowed_methods:
        lines.append("")
        lines.append("Methods: " + ", ".join(f"`{m}`" for m in allowed_methods))
    return "\n".join(lines)
