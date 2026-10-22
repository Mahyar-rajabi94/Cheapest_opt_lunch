# CO₂ uptake at low pressure baselines

This package reproduces the fixed optimizer baselines for Task 060. The
objective minimizes **negative predicted CO₂ uptake at LP** from the supplied
XGBoost surrogate. The input has 188 descriptors, including 28 fixed
coordinates; the optimizers search the remaining 160 coordinates in their
original units. Descriptor combinations within the box need not represent a
real MOF.

The canonical saved traces, Agent records, and figures are in
[`../mof_results/task_060`](../mof_results/task_060/). Local duplicate results
have been removed.

## Setup

Run these commands from this directory using Python 3.12:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Saved baseline metadata records NumPy 2.4.6, SciPy 1.17.1, and XGBoost 2.1.4.
Use those versions when comparing numerical trajectories; timing varies by
machine. Requirements include the backends for all nine optimizers and
Matplotlib for figure reproduction.

## Reproduce baseline campaigns

```bash
.venv/bin/python scripts/run_sweep.py --task co2_uptake --workers 4
```

The default sweep uses a budget of 2,000 evaluations and seeds 0–9 for random
search, Sobol, CMA-ES, differential evolution, GP-BO, and TuRBO. GP-BO and
TuRBO are capped at 100 evaluations. BOBYQA, Nelder-Mead, and L-BFGS-B use
seed 0 only; BOBYQA is also capped at 100 evaluations. Local solvers may stop
before their budget when they converge. These limits match the saved baseline
procedure.

Fresh runs write `results_linux/co2_uptake_d188_i1/` on Linux and
`results/co2_uptake_d188_i1/` elsewhere. Set `--outdir` to choose another
destination. Matching completed campaigns are skipped; conflicting or partial
campaigns require a fresh destination and are preserved. For a small check:

```bash
.venv/bin/python scripts/run_sweep.py --task co2_uptake \
  --optimizers random_search sobol --budget 20 --seeds 2 \
  --workers 2 --outdir /tmp/co2-baseline-smoke
.venv/bin/python -m unittest discover -s tests -v
```

The sweep reproduces fixed optimizer campaigns. The Agent's recorded
Task 060 trajectory is saved evidence from a separate adaptive campaign;
rerunning the baseline sweep does not generate that trajectory.

## Reproduce the figures

Both scripts read the canonical Task 060 results by default and save PNG,
PDF, and SVG files to its `figures/` directory:

```bash
.venv/bin/python scripts/plot_co2_uptake_results_linux.py
.venv/bin/python scripts/plot_agent_optimizer_timeline.py
```

Use `--results-dir PATH` to select a complete results collection and
`--output-dir /tmp/co2-figures` to write figures elsewhere. The comparison
also accepts a fresh full baseline sweep without an Agent trace, for example
`--results-dir results_linux`. It requires all nine baseline methods and
includes `agent_task_060_codex` when present. The comparison
plots the mean of `-y_best_so_far` across each method's seeds, only over
evaluation indices present in every seed. The timeline uses the saved
`agent_task_060_codex` trace. The separate `agent_task_060_codex_free` run is
excluded from the comparison.

## Required model and reference inputs

`src/agent_harness/models/co2_uptake_1.model` is the fixed, trained surrogate;
campaigns load it without retraining. The only reference data needed by the
evaluator are:

- `data/chemunity/descriptors/ChemUnity_Full_geo.csv`
- `data/chemunity/descriptors/core_desc_df_full.csv`
- `data/chemunity/csv/ChemUnity_all_descriptors.csv`

These paths are relative to `src/agent_harness/`. The two descriptor headers
define feature order, and complete finite rows of the combined descriptor
table define the original per-feature minimum and maximum bounds. The
original files and model are retained unchanged.
