"""Black-box evaluation harness for Agent Task 003.

Do not inspect this file or try to work around the budget enforcement --
see agent_briefs/task_003.md for the rules of this exercise.
"""

from __future__ import annotations

import ioh
import numpy as np

from src.agent_harness.harness import BlackBoxHarness, BudgetExhausted

__all__ = ["AgentTaskHarness", "BudgetExhausted"]


class AgentTaskHarness(BlackBoxHarness):
    def __init__(self, budget: int, outdir: str = "results/agent_task_003"):
        problem = ioh.get_problem(
            12, instance=1, dimension=10, problem_class=ioh.ProblemClass.BBOB
        )
        bounds = np.column_stack([problem.bounds.lb, problem.bounds.ub])
        super().__init__(problem, bounds, budget, outdir, task_id="task_003")
