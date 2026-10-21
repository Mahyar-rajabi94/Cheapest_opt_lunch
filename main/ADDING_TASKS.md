# Adding a New Task

Checklist for adding a benchmark task. Every task needs both parts: fixed-optimizer baselines and a blind agent run.

This file names task identities. Keep it in `main/`; never copy it into a sandbox.

## Overview

A task exists in two forms:

| | Standard task | Blind harness |
| --- | --- | --- |
| File | entry in `src/tasks/registry.py` | `src/agent_harness/task_<NNN>.py` |
| Used by | `scripts/run_campaign.py` (fixed-optimizer baselines) | LLM agent in a sandbox |
| Knows `f_opt` | Yes (offline regret only) | No |
| Budget | Capped by the campaign loop | Enforced in code (`BudgetExhausted`) |
| `evaluate(x)` returns | `Evaluation(y, g, elapsed)` | `y` |

The blind harness must not import from `src/tasks/`, so a sandbox can ship without that folder. Shared code lives in `src/tasks/base.py` (`Task`) and `src/agent_harness/harness.py` (`BlackBoxHarness`); a new task adds only its objective.

## 1. Register the Task

Add one entry to `TASKS` in `src/tasks/registry.py`. Any noiseless BBOB function is one line:

```python
"<key>": lambda: BBOBTask("<name>", <function_id>, dim=10),
```

`BBOBTask` names its results folder `<name>_d<dim>_i<instance>` and takes `f_opt` from IOH. Nothing else changes; `run_campaign.py`, `run_sweep.py`, and `slurm/sweep_all.sh` read the registry.

Non-BBOB families exist: `CEC2017Task` (`src/tasks/cec2017.py`, F5/F6/F8) is `CEC2017Task(<function_id>, dim=<n>)`. It needs `opfunu==1.0.1` (official shifts/rotations only) and `setuptools<81` (opfunu imports `pkg_resources`); install both into `.venv` by hand, they are not in `requirements.txt` because sandboxes do not need them.

Constrained families: `PymooGTask("g4" | "g7")` (`src/tasks/pymoo_g.py`, needs `pymoo==0.6.1.6`) and `COCOConstrainedTask(<function_id>, dim=<n>)` (`src/tasks/coco_constrained.py`, COCO `bbob-constrained`, needs `coco-experiment==2.8.2`, import name `cocoex`). Install both into `.venv` by hand; they are not in `requirements.txt`. A constrained task sets `n_constraints` and implements `_g(x)`, the inequality vector, feasible iff all `g <= 0`. `Evaluation.violation` is `sum(max(g, 0))` and `feasible` is `violation <= 1e-10`. `run_campaign.py` gives optimizers the merit `y` if feasible, else `1e12 + 1e6 * violation + tanh(y)`, and adds `merit`, `violation`, `feasible` columns to `trace.csv`; `y_best_so_far` is the best feasible `y` (blank until the first feasible point) and `meta.json` gains `task_n_constraints`. `f_opt` is the constrained optimum. Plots skip seeds with no feasible point yet and label how many found one.

Noisy families: `COCONoisyTask(<110|111|112>, dim=5)` (`src/tasks/coco_noisy.py`, COCO `bbob-noisy` Rosenbrock with Gaussian/uniform/Cauchy noise, needs `coco-experiment==2.8.2`) and `NevergradNoisyTask("rosenbrock" | "cigar", dim=20)` (`src/tasks/nevergrad_noisy.py`, needs `nevergrad==1.0.12`, installed into `.venv` by hand, not in `requirements.txt`). A noisy task sets `noisy = True`, returns the observation from `_f` and the noise-free value from `_f_true`, and may override `reseed(seed)`, which `run_campaign.py` calls once per campaign. Optimizers see only the observed `y`; `Evaluation.y_true` and a `y_true` column in `trace.csv` record the noise-free value, and `y_best_so_far` is the `y_true` of the point with the lowest observed `y` so far (what returning the best observed point delivers), so plots and regret work unchanged; `meta.json` gains `task_noisy`. COCO's noise RNG is global in its C library, cannot be seeded, and resets on problem construction, so every campaign sees the same noise stream by call index whatever its seed (a deterministic optimizer gives identical traces across seeds). Nevergrad's own noise draws from numpy's global RNG, so `NevergradNoisyTask` fixes the instance (built from global seed 0) and regenerates nevergrad's noise formula from a task-owned RNG seeded by `reseed`.

A new non-BBOB function subclasses `Task` (`src/tasks/base.py`) in a new `src/tasks/<family>.py`: set `name`, `dim`, `bounds` (`np.ndarray`, shape `(dim, 2)`), `f_opt`, and implement `_f(x)`. `Task.evaluate(x)` adds timing and returns an `Evaluation` with `y`, `g` (constraint values, empty if unconstrained) and `elapsed`. Minimize only (negate a maximization inside `_f`). `f_opt` is for offline regret only; if no exact optimum is known, use a best-known reference value and document it as such.

## 2. Smoke Test

From `main/`, smoke-test into a temporary directory first:

```bash
../.venv/bin/python scripts/run_campaign.py --task <key> --optimizer random_search --seed 0 --budget 20 --outdir /tmp/smoke
```

## 3. Baselines

Run the sweep (every optimizer, resumable):

```bash
../.venv/bin/python scripts/run_sweep.py --task <key> --workers <n>
```

Budget 2000; `gp_bo`, `turbo`, `bobyqa` capped at 100 via `BUDGET_CAPS`. Seeds 0–9. Deterministic optimizers (`bobyqa`, `nelder_mead`, `lbfgsb`) run seed 0 only. Reruns skip completed campaigns; a budget mismatch with existing results errors (use a new `--outdir`).

Outputs land in `results_linux/<task.name>/<optimizer>/seed_<n>/` (`trace.csv`, `meta.json`). Run sweeps on Linux (Trillium: `sbatch -p debug scripts/slurm/sweep_all.sh <key>`): results differ bit-for-bit across platforms, so `results_linux/` holds Linux runs only. `results/` holds upstream's original (macOS) baselines. Check `eval_seconds` in the smoke run before sweeping an expensive task.

Figures (needs `matplotlib`): `../.venv/bin/python scripts/make_comparison_plots.py [--task <task.name>]` writes `plot1_counterfactual_bar.png`, `plot2_regret_vs_eval.png` and, once an agent run exists, `plot3_agent_method_timeline.png` into `results_linux/<task.name>/`. Add the task's sandbox results folder to `AGENT_RUNS` in that script. Commit results and figures.

## 4. Blind Agent Task

1. **Harness:** copy `src/agent_harness/task_001.py` to `task_<NNN>.py`. Change only the objective construction (inline it; do not import from `src/tasks/`), the default `outdir` (`results/agent_task_<NNN>`), `task_id`, and the docstring's brief reference. Budget enforcement and logging come from `BlackBoxHarness`. Never expose `f_opt` or the function name.
2. **Sandbox:** from `main/`, run

   ```bash
   ../.venv/bin/python scripts/make_sandbox.py --task-id <NNN> --budget 2000 --outdir ../task_<NNN>_sandbox_<agent>
   ```

   It renders the brief from `agent_briefs/blind_brief_template.md` (dim and bounds from the harness; uniform box, deterministic), copies `requirements.txt`, `src/optimizers/`, and the harness, and creates an empty `results/`. No `src/tasks/`, no other harnesses.
3. **Leak check:** `make_sandbox.py` searches every sandbox file for registered task names and `f_opt`, and deletes the sandbox on a hit. Also check by hand for anything the template could not know about (e.g. a new function's name not yet in the registry).
4. **Brief changes:** if the task needs a property the template does not state (noise, non-uniform bounds), edit the rendered brief in the sandbox; describe only properties the agent may know.
5. **Run:** start a brand-new agent session with the sandbox as its working directory (never one that has seen `main/`). First message only:

   > Read `agent_briefs/task_<NNN>.md` and carry out everything it describes.

6. **Score:** from `main/`, rerun `scripts/make_comparison_plots.py --task <task.name>` to compare the agent trace against `f_opt` and the step 3 baselines.

## Checklist

- [ ] `TASKS` entry in `src/tasks/registry.py` (new `Task` subclass only for a non-BBOB family)
- [ ] Smoke run passes
- [ ] Baselines: `run_sweep.py` (budget 2000, expensive methods 100, seeds 0–9, on Linux into `results_linux/`)
- [ ] Harness `task_<NNN>.py` (objective only), no `f_opt`
- [ ] Sandbox built with `make_sandbox.py`; leak check passed
- [ ] Agent run in a fresh session; scored from `main/`
