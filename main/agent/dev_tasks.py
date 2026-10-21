"""Development smoke tasks (BBOB via ioh). FOR DEVELOPMENT ONLY: these are
not the team's official benchmark suite, and results on them are not
reported as results. The task name is never shown to the strategist."""

from __future__ import annotations

import ioh
import numpy as np

# id -> (BBOB function id, short landscape type, used in dev reports only)
DEV_FUNCTIONS = {
    1: "separable smooth unimodal",
    8: "smooth unimodal, curved valley",
    10: "ill-conditioned unimodal",
    15: "multimodal, global structure",
    21: "multimodal, weak structure",
}


class DevTask:
    def __init__(self, fid: int, dim: int = 10, instance: int = 1):
        self._p = ioh.get_problem(fid, instance=instance, dimension=dim,
                                  problem_class=ioh.ProblemClass.BBOB)
        self.name = f"dev_f{fid}_d{dim}_i{instance}"
        self.dim = dim
        self.bounds = np.column_stack([self._p.bounds.lb, self._p.bounds.ub]).astype(float)
        self.f_opt = float(self._p.optimum.y)  # scoring only, never shown to the agent

    def __call__(self, x) -> float:
        return float(self._p(np.asarray(x, dtype=float)))


def get_task(name: str) -> DevTask:
    """'task001' (= the Task 001 problem) or 'dev_f<id>_d<dim>[_i<inst>]'."""
    if name == "task001":
        return DevTask(8, 10, 1)
    parts = name.split("_")
    fid = int(parts[1][1:]); dim = int(parts[2][1:])
    inst = int(parts[3][1:]) if len(parts) > 3 else 1
    return DevTask(fid, dim, inst)
