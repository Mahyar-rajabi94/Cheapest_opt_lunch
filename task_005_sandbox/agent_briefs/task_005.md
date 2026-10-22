# Optimization Task Brief — Task 005

You are acting as an autonomous optimization agent. Your job is to find the
lowest possible value of an unknown black-box function, within a strict
evaluation budget, using whatever optimization method(s) you judge best —
and to document your reasoning as you go.

## 1. Problem statement

- You are given a black-box objective function to **minimize**.
- The input is a real-valued vector `x` of dimension **11**.
- Each coordinate of `x` must lie in the box **[0, 1]**.
- The function is **deterministic**: the same `x` always returns the same
  value (no observation noise).
- You may call the evaluation function **at most 2000 times in total**.
  This is enforced in code, not just a rule: the harness raises an
  exception if you call it again after the budget is used. There is no
  benefit to trying to work around this.
- Lower returned values are better.

### Allowed points

Not every point in the box is allowed. A point `x` is **allowed** only if
it satisfies all six of these linear inequalities (coordinates numbered
from 0, as in Python):

```
1.  -0.80*x[1] + 0.86*x[4]               <= 0.63
2.  -0.54*x[0] + 0.83*x[3]               <= 0.62
3.   0.54*x[0] - 0.86*x[4]               <= 0.37
4.  -0.54*x[0] + 0.86*x[4]               <= 0.54
5.   0.54*x[0] - 0.80*x[1] + 0.50*x[2]   <= 0.50
6.   0.83*x[3] - 0.86*x[4] + 0.96*x[5]   <= 0.81
```

The same rules as code (a point counts as allowed when its total violation
is at most 1e-10):

```python
import numpy as np
A = np.zeros((6, 11))
A[0, [1, 4]] = [-0.80, 0.86]
A[1, [0, 3]] = [-0.54, 0.83]
A[2, [0, 4]] = [0.54, -0.86]
A[3, [0, 4]] = [-0.54, 0.86]
A[4, [0, 1, 2]] = [0.54, -0.80, 0.50]
A[5, [3, 4, 5]] = [0.83, -0.86, 0.96]
b = np.array([0.63, 0.62, 0.37, 0.54, 0.50, 0.81])

def allowed(x):
    return np.maximum(A @ np.asarray(x) - b, 0).sum() <= 1e-10
```

Only `x[0]` to `x[5]` appear in the rules; `x[6]` to `x[10]` are limited
only by the box.

- `harness.evaluate(x)` returns a value for **any** `x` in the box, allowed
  or not. It does not check these rules, and every call counts against the
  budget.
- **Only allowed points count.** Your result is the lowest value returned
  at an allowed point. Values at points that break a rule are ignored when
  your campaign is scored, however low they are.
- The running best that the harness records (`y_best_so_far` in
  `trace.csv`) includes points that break a rule, so keep track of your
  best allowed value yourself.

### Two discrete coordinates

`x[9]` and `x[10]` accept any value in [0, 1], but the function only uses a
few values of each. Before evaluation, each is moved to the nearest of
these positions (a point exactly halfway between two goes to the lower one):

- `x[9]`: 0, 0.166667, 1
- `x[10]`: 0, 0.075697, 0.167331, 0.187251, 0.219124, 0.227092, 0.239044,
  0.286853, 0.298805, 0.418327, 0.657371, 0.697211, 0.984064, 1

So any value of `x[9]` or `x[10]` gives the same result as its nearest
position.

**You do not know, and must not attempt to discover, the function's
identity.** Do not search for it online, do not try to match it to a
known benchmark suite, and do not inspect this harness's source code, its
data under `src/agent_harness/assets/`, or anything under `src/tasks/`.
Treat it strictly as a sealed black box. This
is a deliberate part of the exercise — its value depends on you reasoning
from the data you actually collect, not from recognizing a textbook
function.

## 2. The harness (how you call the objective)

```python
import sys
sys.path.insert(0, "<repo root>")  # adjust to this project's absolute path
from src.agent_harness.task_005 import AgentTaskHarness, BudgetExhausted

harness = AgentTaskHarness(budget=2000)
print(harness.dim, harness.bounds, harness.direction)  # 11  [[0.0, 1.0], ...]  "min"

harness.set_active_optimizer("cma_es")   # label every evaluate() call with your current method
y = harness.evaluate(x)                   # x: length-11 list or array; returns a float

harness.set_active_optimizer("lbfgsb")   # call again every time you switch methods

harness.save()                            # writes output early if you stop before the budget is used;
                                           # called automatically once the budget is exhausted
```

- `harness.evaluate(x)` is the **only** way real evaluations may be
  produced. Never estimate, interpolate, or fabricate a `y` value — every
  point in the final trace must come from an actual `evaluate()` call.
- `harness.set_active_optimizer(name)` tags every subsequent evaluation
  with whichever method you are currently using (e.g. `"cma_es"`,
  `"my_bo_then_local_hybrid"`). Call it once before your first
  `evaluate()`, and again every time you switch. Choose short, descriptive
  names.
- The harness automatically records every evaluation (point, value,
  running best, timing) and writes `trace.csv` + `meta.json` to
  `results/agent_task_005/` once the budget is exhausted, or when you call
  `harness.save()`. You do not need to build this yourself, and should not
  hand-edit these files.

## 3. Available tools

Already installed in this project's `.venv`, with a consistent
`ask(n)` / `tell(X, y)` / `best()` interface (see `src/optimizers/*.py`
for working examples and docstrings):

- `src.optimizers.random_search.RandomSearch`
- `src.optimizers.sobol_search.SobolSearch`
- `src.optimizers.cma_es.CMAES` (pycma)
- `src.optimizers.differential_evolution.DifferentialEvolution`
- `src.optimizers.gp_bo.GPBO` (BoTorch, analytic LogEI)
- `src.optimizers.turbo.TuRBO` (BoTorch, trust-region BO)
- `src.optimizers.bobyqa.BOBYQA` (Py-BOBYQA)
- `src.optimizers.nelder_mead.NelderMead` (scipy)
- `src.optimizers.lbfgsb.LBFGSB` (scipy, finite-difference gradients)

You are also free to use, install, or wrap any of: **scipy, Ax, BoTorch,
HEBO, Optuna, SMAC3, cmaes, pycma, PyPop7, EvoX, pymoo, PRIMA, DFO-LS,
PyNomad**, or anything else you judge helpful. A few things worth knowing
before you spend budget or time on installs:

- The PyPI package literally named `prima` is broken (an unrelated,
  empty placeholder, version "0.0"). For Powell's solvers, use `pdfo` or
  **Py-BOBYQA** (`pip install Py-BOBYQA`, import name `pybobyqa`) instead
  — both already installed here.
- `Py-BOBYQA` has a known limitation when driven one evaluation at a time
  through a background-thread wrapper, as `src/optimizers/bobyqa.py`
  does: it can become extremely slow past roughly 500 evaluations. It is
  reliable up to that point; past it, prefer a different method or a
  different way of driving it.
- `ioh` (already installed) requires `numpy>=2`, which conflicts with
  several other scientific-Python packages that want `numpy<2`. If you
  `pip install` something new, check for numpy version conflicts before
  relying on the result.
- This project's `.venv` already has numpy, scipy, cma (pycma), ioh,
  botorch, torch, gpytorch, and pybobyqa installed. Use
  `.venv/bin/pip install <package>` to add more, into this same
  environment — do not install into the system/base Python.

## 4. Required protocol

Follow these four steps, in order. Steps (a) and (c) produce required
output files, not optional notes.

**(a) Before your first call to `evaluate()`**, write
`results/agent_task_005/roadmap_initial.md` describing your planned
approach. It must include:

- Which method(s) you intend to use, in what sequence (if more than
  one), and roughly how you plan to split the budget among them.
- Your reasoning — why these methods, given only the black-box problem
  description in Section 1 (dimension, bounds, determinism, budget). You
  have no data yet at this point; reason from the problem's stated
  properties alone.
- Explicit, quantitative criteria for when you will deviate from this
  plan (e.g. "if the best value has not improved over the last 200
  evaluations, switch to method Y" or "if a local refiner's trust region
  collapses, restart from a new point").

**Once written, do not edit this file again.** It is a historical record
of your pre-registered plan, not a living document.

**(b) Run the campaign.** Call `harness.evaluate(x)` to get real
objective values, driving whichever method(s) you chose, respecting the
budget.

**(c) Every time you actually deviate from the roadmap** (switch methods,
change a committed hyperparameter, hybridize two approaches, restart,
etc.), append one entry to `results/agent_task_005/changelog.md`:

```
## Evaluation <N>
**Change:** <what you changed>
**Reason:** <why, grounded in what you actually observed — cite specific
evidence, e.g. "best value plateaued at y=X over the last 300 evaluations">
```

This file is append-only: add new entries at the end, never edit or
remove earlier ones.

**(d) When finished** (budget exhausted, or you call `harness.save()`
early), confirm that `results/agent_task_005/trace.csv` and
`results/agent_task_005/meta.json` exist. The harness writes these
automatically — do not hand-construct or edit them.

## 5. Hard constraints

- Never exceed the stated budget, and never attempt to work around the
  enforcement (e.g. by constructing a second harness instance to reset
  the counter).
- Never fabricate, estimate, or hand-edit an evaluation result. Every row
  in the final trace must come from a real `harness.evaluate()` call.
- Do not attempt to identify, search for, or reverse-engineer the
  underlying function. Do not inspect `src/agent_harness/task_005.py`'s
  implementation, any file under `src/agent_harness/assets/`, or any file
  under `src/tasks/`.
- Hybridizing multiple optimizers, or switching between them, is
  explicitly encouraged — this is a central part of what this exercise is
  studying, not a deviation to avoid.
- You will not be shown how any other method has performed on this task.
  Reason from first principles and from the data you collect yourself.

## 6. What "done" looks like

At the end of your campaign, `results/agent_task_005/` should contain
exactly four files: `roadmap_initial.md`, `changelog.md`, `trace.csv`,
`meta.json`. Nothing else is required, and nothing else should be
hand-written into that directory.
