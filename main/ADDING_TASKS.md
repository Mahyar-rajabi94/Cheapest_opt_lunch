# Adding a New Task

Checklist for adding a benchmark task: baseline campaigns first, then (optionally) a blind agent run.

This file names task identities. Keep it in `main/`; never copy it into a sandbox.

## Overview

A task exists in two forms:

| | Standard task | Blind harness |
| --- | --- | --- |
| File | `src/tasks/<name>.py` | `src/agent_harness/task_<NNN>.py` |
| Used by | `scripts/run_campaign.py` (fixed-optimizer baselines) | LLM agent in a sandbox |
| Knows `f_opt` | Yes (offline regret only) | No |
| Budget | Capped by the campaign loop | Enforced in code (`BudgetExhausted`) |
| `evaluate(x)` returns | `(y, elapsed_seconds)` | `y` |

The blind harness is self-contained on purpose: it must not import from `src/tasks/`, so a sandbox can ship without that folder.

## 1. Task Wrapper

Create `src/tasks/<name>.py`, using `src/tasks/rosenbrock.py` as the template. Required interface:

| Attribute / method | Meaning |
| --- | --- |
| `name` | Results folder name; encode function, dim, instance (e.g. `rosenbrock_d10_i1`) |
| `direction` | Always `"min"` (maximization: negate inside `evaluate`) |
| `dim` | Number of variables |
| `bounds` | `np.ndarray` of shape `(dim, 2)`, `[lower, upper]` per variable |
| `f_opt` | Known optimum, for offline regret only; never passed to optimizers |
| `evaluate(x)` | Returns `(y, elapsed_seconds)` |

IOH/BBOB functions take a few lines: `ioh.get_problem(fid, instance=..., dimension=..., problem_class=ioh.ProblemClass.BBOB)`, and `problem.optimum.y` gives `f_opt`. A custom function needs its own known optimum (or a best-known reference value, documented as such).

## 2. Register

Add one entry to `TASKS` in `scripts/run_campaign.py`:

```python
TASKS = {
    "rosenbrock10": lambda: RosenbrockTask(dim=10, instance=1, rotated=False),
    "<key>": lambda: MyTask(dim=10, instance=1),
}
```

Nothing else in that script changes.

## 3. Baselines

From `main/`, smoke-test into a temporary directory first:

```bash
../.venv/bin/python scripts/run_campaign.py --task <key> --optimizer random_search --seed 0 --budget 20 --outdir /tmp/smoke
```

Then run every optimizer over seeds 0–4. Task 001 used these budgets (lower for slow or limited optimizers):

| Optimizer | Budget |
| --- | --- |
| `random_search`, `sobol`, `cma_es`, `de`, `turbo`, `nelder_mead`, `lbfgsb` | 2000 |
| `bobyqa` | 500 (Py-BOBYQA slows sharply past ~500 one-at-a-time evals) |
| `gp_bo` | 100 |

Outputs land in `results/<task.name>/<optimizer>/seed_<n>/` (`trace.csv`, `meta.json`). Check `eval_seconds` in the smoke run before choosing budgets for an expensive task.

## 4. Blind Agent Task (Optional)

1. **Harness:** copy `src/agent_harness/task_001.py` to `task_<NNN>.py`. Change:
   - the objective construction in `__init__` (inline it; do not import from `src/tasks/`)
   - the default `outdir` (`results/agent_task_<NNN>`)
   - `task_id` in `save()`
   - the docstring reference to the brief

   Keep `dim`, `bounds`, and `direction` public; never expose `f_opt` or the function name.
2. **Brief:** copy `agent_briefs/task_001.md` to `task_<NNN>.md`. Update:
   - Section 1 (dimension, bounds, noise, budget)
   - Section 2 (harness import and budget)
   - every `agent_task_001` / `task_001` path

   Describe only properties the agent may know; nothing that identifies the function.
3. **Sandbox:** create `task_<NNN>_sandbox_<agent>/` at the repo root containing only:

   ```
   agent_briefs/task_<NNN>.md
   requirements.txt          # copy from an existing sandbox
   src/__init__.py
   src/optimizers/           # copy of main/src/optimizers/
   src/agent_harness/__init__.py
   src/agent_harness/task_<NNN>.py
   results/                  # empty
   ```

   No `src/tasks/`, no `main/results/`, no other harnesses, nothing that names the function.
4. **Leak check:** search the sandbox for the function name, BBOB function id, and `f_opt`. Only the harness's objective construction may identify it.

   ```bash
   grep -rin "<function name>" task_<NNN>_sandbox_<agent>/
   ```

5. **Run:** start a brand-new agent session with the sandbox as its working directory (never one that has seen `main/`). First message only:

   > Read `agent_briefs/task_<NNN>.md` and carry out everything it describes.

6. **Score:** from `main/`, compare `results/agent_task_<NNN>/trace.csv` against `f_opt` and the step 3 baselines.

## Checklist

- [ ] `src/tasks/<name>.py` with `name`, `direction`, `dim`, `bounds`, `f_opt`, `evaluate`
- [ ] `TASKS` entry in `scripts/run_campaign.py`
- [ ] Smoke run passes
- [ ] Baselines: all optimizers × seeds 0–4
- [ ] (Blind) harness `task_<NNN>.py`, self-contained, no `f_opt`
- [ ] (Blind) brief `task_<NNN>.md`, no identifying details
- [ ] (Blind) sandbox built and leak-checked
- [ ] (Blind) agent run in a fresh session; scored from `main/`
