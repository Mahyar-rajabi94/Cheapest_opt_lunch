# Public evaluation interface

Minimize a noisy scalar black-box objective over 5 real-valued
coordinates with bounds [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]. Each evaluation returns a noisy observation of the objective; the campaign is scored by the noise-free value at the point with the lowest observed y. There are no additional constraints.
There are no prior observations. The hard budget is 2000 objective evaluations,
including repeated points, initialization, line searches and finite differences.
The objective identity, implementation and optimum are private.

Use `.venv/bin/python` from the workspace root. Put scripts in `work/`.

```python
from src.agent_harness.harness import AgentTaskHarness, BudgetExhausted
h = AgentTaskHarness(budget=2000, outdir="results/agent_run")
h.set_active_optimizer("your_method")
# y = h.evaluate(x)
h.save()
```

Public metadata: dim, bounds, direction (min), noisy (True),
n_constraints (0), budget.
Live counters: evaluations_used and evaluations_remaining.
The client is inspectable transport code; evaluation runs in a separate process.
Reconnecting or creating another client cannot reset the campaign budget.

Follow AGENTS.md. Record only the first block in results/agent_run/roadmap_initial.md
before any evaluation. Keep it unchanged; append each subsequent decision to
changelog.md before its block. Choose the method, relevant hyperparameters and
block length yourself. Return control to the LLM at every checkpoint.
The evaluator snapshots these documents and rejects retrospective edits.
It writes trace.csv and meta.json; call save() at checkpoints to refresh them.

Installed packages include numpy, scipy, cma, torch, botorch, gpytorch, pybobyqa,
pandas and matplotlib. Wrappers are in src/optimizers. You may use their libraries
or write adapters in work/. Network access and dependency installation are disabled.
On an evaluator error, stop and report; do not retry an unknown evaluation.
