# Adding a New Task

Checklist for adding a benchmark task. Every task needs both parts: fixed-optimizer baselines and a blind agent run.

This file names task identities. Keep it in `main/`; never copy it into a sandbox.

## Overview

A task is implemented once in `src/tasks/` and registered in `src/tasks/registry.py`.
The baseline runner calls it directly. The isolated agent runner keeps it private
and exposes `src/agent_harness/harness.py`, a generic transport client. There are
no numbered agent harnesses or per-task copies of the objective.

The native runner accepts every registered task. It exposes only the public
contract below; a new task needs nothing runner-specific.

## 1. Register the Task

Add one entry to `TASKS` in `src/tasks/registry.py`. Any noiseless BBOB function is one line:

```python
"<key>": lambda: BBOBTask("<name>", <function_id>, dim=10),
```

`BBOBTask` names its results folder `<name>_d<dim>_i<instance>` and takes `f_opt` from IOH. Nothing else changes; `run_campaign.py`, `run_sweep.py`, and `slurm/sweep_all.sh` read the registry.

Non-BBOB families exist: `CEC2017Task` (`src/tasks/cec2017.py`, F5/F6/F8) is `CEC2017Task(<function_id>, dim=<n>[, half_width=<h>])`; `half_width` (default 100, CEC's box) shrinks the box to [-h, h] and scales the shift with it. It needs `opfunu==1.0.1` (official shifts/rotations only) and `setuptools<81` (opfunu imports `pkg_resources`); install both into `.venv` by hand, they are not in `requirements.txt` because sandboxes do not need them.

Constrained families: `PymooGTask("g4" | "g7")` (`src/tasks/pymoo_g.py`, needs `pymoo==0.6.1.6`) and `COCOConstrainedTask(<function_id>, dim=<n>)` (`src/tasks/coco_constrained.py`, COCO `bbob-constrained`, needs `coco-experiment==2.8.2`, import name `cocoex`). Install both into `.venv` by hand; they are not in `requirements.txt`. A constrained task sets `n_constraints` and implements `_g(x)`, the inequality vector, feasible iff all `g <= 0`. `Evaluation.violation` is `sum(max(g, 0))` and `feasible` is `violation <= 1e-10`. `run_campaign.py` gives optimizers the merit `y` if feasible, else `1e12 + 1e6 * violation + tanh(y)`, and adds `merit`, `violation`, `feasible` columns to `trace.csv`; `y_best_so_far` is the best feasible `y` (blank until the first feasible point) and `meta.json` gains `task_n_constraints`. `f_opt` is the constrained optimum. Plots skip seeds with no feasible point yet and label how many found one.

Noisy families: `COCONoisyTask(<110|111|112>, dim=5)` (`src/tasks/coco_noisy.py`, COCO `bbob-noisy` Rosenbrock with Gaussian/uniform/Cauchy noise, needs `coco-experiment==2.8.2`) and `NevergradNoisyTask("rosenbrock" | "cigar", dim=20)` (`src/tasks/nevergrad_noisy.py`, noise level 0.1, x* redrawn uniformly in [-4, 4]; needs `nevergrad==1.0.12`, installed into `.venv` by hand, not in `requirements.txt`). A noisy task sets `noisy = True`, returns the observation from `_f` and the noise-free value from `_f_true`, and may override `reseed(seed)`, which `run_campaign.py` calls once per campaign. Optimizers see only the observed `y`; `Evaluation.y_true` and a `y_true` column in `trace.csv` record the noise-free value, and `y_best_so_far` is the `y_true` of the point with the lowest observed `y` so far (what returning the best observed point delivers), so plots and regret work unchanged; `meta.json` gains `task_noisy`. COCO's noise RNG is global in its C library, cannot be seeded, and resets on problem construction, so every campaign sees the same noise stream by call index whatever its seed (a deterministic optimizer gives identical traces across seeds). Nevergrad's own noise draws from numpy's global RNG, so `NevergradNoisyTask` fixes the instance (built from global seed 0) and regenerates nevergrad's noise formula from a task-owned RNG seeded by `reseed`.

Mixed-integer family: `COCOMixintTask(<function_id>, dim=10)` (`src/tasks/coco_mixint.py`, COCO `bbob-mixint`, needs `coco-experiment==2.8.2`). At D=10 the first 8 coordinates are integer (bounds [0, 1], [0, 1], [0, 3], [0, 3], [0, 7], [0, 7], [0, 15], [0, 15]), the last 2 continuous in [-5, 5]. COCO rounds integer coordinates internally, so every optimizer runs unchanged and sees a piecewise-constant objective along them. `f_opt` is the objective at COCO's x* (checked integral).

HPO family: `YAHPOTask("rbv2_svm" | "rbv2_xgboost", instance="40981")` (`src/tasks/yahpo.py`), YAHPO Gym surrogate logloss on an OpenML instance at full fidelity (`trainsize=1.0`, `repl=1`). `yahpo-gym==1.0.2` pins numpy<2, which conflicts with `ioh` (and every result here used numpy 2), so the task evaluates YAHPO's ONNX surrogates directly with `onnxruntime` (install `onnxruntime==1.31.0` into `.venv` by hand, not in `requirements.txt`), using yahpo_gym's input encoding; single-point outputs match yahpo_gym exactly (batched ONNX calls differ by ~1e-6). Model files are git-ignored; download them into `external/yahpo_data/` at the repository root:

```bash
for s in rbv2_svm rbv2_xgboost; do for f in config_space.json encoding.json model.onnx; do
  curl -sfL --create-dirs -o external/yahpo_data/$s/$f https://raw.githubusercontent.com/slds-lmu/yahpo_data/v1.0.2/$s/$f
done; done
```

The box is [0, 1]^D (D = 6 / 14), one coordinate per searched hyperparameter in `config_space.json` order, decoded inside `_f`: floats interpolate (in log space if log-scaled), ints are the same rounded, a categorical with k choices takes choice floor(u k), and a hyperparameter whose condition fails (e.g. `degree` unless `kernel = polynomial`) is passed as missing, so its coordinate has no effect. No exact optimum is known; `f_opt` is a best-known value (min over 2^20 Sobol points and 20 CMA-ES runs of 1e5 evaluations), so regret can be slightly negative in principle.

Pool family: `OPVPoolTask()` (`src/tasks/opv_pool.py`), the 526 OPV devices behind `opv_pce` as a fixed pool, read from `task_006_baseline/` (built by `preprocess.py` there). Each evaluation maps x in [0, 1]^11 (min–max scaled device features) to the nearest untested device (ties to the lowest candidate id) and returns its measured −PCE, so the same x evaluated again returns a different device, and a campaign allows at most 526 evaluations. A pool task sets `pool = True`, `n_candidates`, `candidates` and `last_candidate`; `run_campaign.py` adds a `candidate_id` column to `trace.csv` and `meta.json` gains `task_n_candidates`. The pool-only baseline `pool_random` (uniform picks without replacement from `candidates`) runs on pool tasks only; the agent runner tells the agent the objective is not repeatable and never copies `pool_random.py` into its workspace. `f_opt` is the best measured device (exact). Commands to run it: `task_006_baseline/README.md`.

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

Budget 2000 for every optimizer (`BUDGET_CAPS` is empty; `results_linux/` predates this and has `gp_bo`, `turbo`, `bobyqa` capped at 100). Seeds 0–9. Deterministic optimizers (`bobyqa`, `nelder_mead`, `lbfgsb`) run seed 0 only. Reruns skip completed campaigns; a budget mismatch with existing results errors (use a new `--outdir`).

Outputs land in `results_linux/<task.name>/<optimizer>/seed_<n>/` (`trace.csv`, `meta.json`). Run sweeps on Linux (Trillium: `sbatch -p debug scripts/slurm/sweep_all.sh <key>`): results differ bit-for-bit across platforms, so `results_linux/` holds Linux runs only. `results/` holds upstream's original (macOS) baselines. Check `eval_seconds` in the smoke run before sweeping an expensive task.

Figures (needs `matplotlib`): `../.venv/bin/python scripts/make_comparison_plots.py [--task <task.name>]` writes `plot1_counterfactual_bar.png`, `plot2_regret_vs_eval.png` and, once an agent run exists, `plot3_agent_method_timeline.png` into `results_linux/<task.name>/`. Add the task's sandbox results folder to `AGENT_RUNS` in that script. Commit results and figures.

## 4. Isolated Agent Run

Run from the repository root:

```bash
.venv/bin/python main/scripts/run_agent.py --task <key> --budget 100 --check-only
.venv/bin/python main/scripts/run_agent.py --task <key> --budget 100
```

Follow the root [README](../README.md) for the separate optimizer environment,
isolation boundary, outputs and plotting command. The runner copies the canonical
root AGENTS.md and generic client into a fresh workspace. It gives the agent only
dimension, bounds, direction, noise status, number of constraints and budget.
Objective construction and scoring remain outside that workspace. Do not add a
`task_<NNN>.py` or a new prompt.

Discrete, categorical and conditional coordinates are decoded inside `_f`, so the
agent sees the same box as the baselines. Constrained tasks return `(y, g)` from
`evaluate`. Noisy tasks are reseeded with `--seed`; the private journal keeps
`y_true`, and `private/scored_trace.csv` gives `y_best_so_far` in the baseline
convention (blank before the first feasible point).

## Checklist

- [ ] Register the task and record its domain and constraints accurately.
- [ ] Smoke-test baseline evaluation and update TASKS.md.
- [ ] Run matched-budget baselines into a fresh output directory.
- [ ] Pass the native runner's zero-evaluation isolation check.
- [ ] Run a fresh isolated agent campaign with the canonical prompt.
- [ ] Score and plot outside the agent workspace; preserve source and runtime provenance.
