# Task 001 — example user prompt

This is an optional saved example. New tasks start with a terminal conversation;
the agent writes their task definitions during planning. This file is not
required and should only be used when the user explicitly references it.

Help me define and then run the following black-box optimization task.

## 1. Problem statement

- Minimize a single scalar objective; lower returned values are better.
- The input `x` is a **10-dimensional continuous**, real-valued vector.
- Every coordinate must lie in **[-5, 5]**. There are no additional constraints.
- You may generate any new point inside that box. There is **no predefined
  candidate pool** or tabular dataset.
- The objective is **deterministic**: the same `x` returns the same value.
- The evaluation budget is **2000 calls in total** for one new campaign.
  Every successful `evaluate(x)` call costs one evaluation, including repeated
  points. There are no initial observations or previous budget usage to reuse.
- Evaluate sequentially through one harness instance. There is no separate
  wall-time limit. Stop at budget exhaustion, or save early if the chosen
  strategy terminates. Report unexpected evaluation errors instead of retrying
  automatically; do not reset the budget.
- The function's identity and implementation are sealed; only its public
  interface below is available to the optimization agent.
- Use `results/agent_task_001/` in a fresh workspace for the run outputs. If it
  already contains a campaign, resolve a fresh destination with me first.

## 2. Python evaluator and public calling contract

The supplied Python module is **`src/agent_harness/task_001.py`**. This is the
actual evaluator; the following is its public usage example, not permission to
inspect its implementation or start evaluating before task confirmation.
Run Python from this workspace's root so `src` is importable.

```python
from src.agent_harness.task_001 import AgentTaskHarness, BudgetExhausted

# Initialize once, after the problem definition has been confirmed.
harness = AgentTaskHarness(budget=2000, outdir="results/agent_task_001")
print(harness.dim, harness.bounds, harness.direction)
# 10, [[-5.0, 5.0], ...], "min"

# After writing the initial roadmap, for each feasible length-10 list/array x:
harness.set_active_optimizer("cma_es")
# y = harness.evaluate(x)  # returns one float; consumes one evaluation

# When switching methods:
# harness.set_active_optimizer("lbfgsb")

# On early stopping (automatic saving also occurs at budget exhaustion):
# harness.save()
```

- `harness.evaluate(x)` accepts a length-10 numerical list or array and returns
  one scalar `float`. All submitted points must respect the stated bounds.
- Calls after the budget is used raise `BudgetExhausted`.
- `harness.evaluations_used` and `harness.evaluations_remaining` expose the
  counters without evaluating the objective.
- `harness.set_active_optimizer(name)` labels subsequent evaluations with the
  supplied method name.
- `harness.save()` writes the recorded `trace.csv` and `meta.json` to `outdir`;
  this also happens automatically on the last budgeted evaluation. `outdir`
  can be changed at initialization for a fresh run without changing the task.
