# Optimization Task Brief — Task 006

You are acting as an autonomous optimization agent. Your job is to find the
lowest possible value of an unknown black-box function, within a strict
evaluation budget, using whatever optimization method(s) you judge best —
and to document your reasoning as you go.

## 1. Problem statement

- You are given a black-box objective function to **minimize**.
- The input is a real-valued vector `x` of dimension **11**.
- Each coordinate of `x` must lie in the box **[0, 1]**.
- The function has **no random noise**, but it is **not repeatable**: the
  value returned for `x` depends on which points you have already
  evaluated, so evaluating the same `x` again generally returns a
  different value.
- You may call the evaluation function **at most 100 times in total**.
  This is enforced in code, not just a rule: the harness raises an
  exception if you call it again after the budget is used. There is no
  benefit to trying to work around this.
- Lower returned values are better.

**You do not know, and must not attempt to discover, the function's
identity.** Do not search for it online, do not try to match it to a
known benchmark suite, and do not inspect this harness's source code or
anything under `src/tasks/`. Treat it strictly as a sealed black box. This
is a deliberate part of the exercise — its value depends on you reasoning
from the data you actually collect, not from recognizing a textbook
function.

## 2. The harness (how you call the objective)

```python
import sys
sys.path.insert(0, "<repo root>")  # adjust to this project's absolute path
from src.agent_harness.task_006 import AgentTaskHarness, BudgetExhausted

harness = AgentTaskHarness(budget=100)
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
  `results/agent_task_006/` once the budget is exhausted, or when you call
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
`results/agent_task_006/roadmap_initial.md` describing your planned
approach. It must include:

- Which method(s) you intend to use, in what sequence (if more than
  one), and roughly how you plan to split the budget among them.
- Your reasoning — why these methods, given only the black-box problem
  description in Section 1 (dimension, bounds, repeatability, budget). You
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
etc.), append one entry to `results/agent_task_006/changelog.md`:

```
## Evaluation <N>
**Change:** <what you changed>
**Reason:** <why, grounded in what you actually observed — cite specific
evidence, e.g. "best value plateaued at y=X over the last 300 evaluations">
```

This file is append-only: add new entries at the end, never edit or
remove earlier ones.

**(d) When finished** (budget exhausted, or you call `harness.save()`
early), confirm that `results/agent_task_006/trace.csv` and
`results/agent_task_006/meta.json` exist. The harness writes these
automatically — do not hand-construct or edit them.

## 5. Hard constraints

- Never exceed the stated budget, and never attempt to work around the
  enforcement (e.g. by constructing a second harness instance to reset
  the counter).
- Never fabricate, estimate, or hand-edit an evaluation result. Every row
  in the final trace must come from a real `harness.evaluate()` call.
- Do not attempt to identify, search for, or reverse-engineer the
  underlying function. Do not inspect `src/agent_harness/task_006.py`'s
  implementation or any file under `src/tasks/`.
- Hybridizing multiple optimizers, or switching between them, is
  explicitly encouraged — this is a central part of what this exercise is
  studying, not a deviation to avoid.
- You will not be shown how any other method has performed on this task.
  Reason from first principles and from the data you collect yourself.

## 6. What "done" looks like

At the end of your campaign, `results/agent_task_006/` should contain
exactly four files: `roadmap_initial.md`, `changelog.md`, `trace.csv`,
`meta.json`. Nothing else is required, and nothing else should be
hand-written into that directory.
