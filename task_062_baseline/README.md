# CH4 uptake at high pressure

This workspace reproduces the fixed-optimizer baselines for predicted methane
uptake at **298 K and 6.5 MPa (65 bar)**, target
`pure_uptake_methane_298.00_6500000` in mol/kg. `CH4UptakeTask` minimizes
negative predicted uptake; physical uptake is `-y`.

Run these commands from this directory. Set up the environment with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Saved baseline metadata records NumPy 2.4.6, SciPy 1.17.1, and XGBoost 2.1.4.
Use those versions when matching the original numerical environment; timing
depends on the machine and optimizer-library versions also affect fresh runs.

Reproduce the baseline sweep:

```bash
.venv/bin/python scripts/run_sweep.py --task ch4_uptake --workers 4
```

The sweep uses 2,000 evaluations and seeds 0–9. GP-BO, TuRBO, and BOBYQA are
capped at 100 evaluations; BOBYQA, Nelder-Mead, and L-BFGS-B use seed 0 only.
Converged optimizers may finish early. New results default to
`results_linux/ch4_uptake_d188_i1/<optimizer>/seed_<seed>/{trace.csv,meta.json}`
on Linux (`results/` elsewhere). Use `--outdir` for another destination.
Existing matching completed campaigns are skipped; conflicting or incomplete
campaigns require a fresh output directory. For one method, use
`scripts/run_campaign.py --task ch4_uptake --optimizer random_search --seed 0 --budget 2000`.

The canonical saved results, including `agent_task_062`, are in
[`../mof_results/task_062`](../mof_results/task_062/). Regenerate its figures
from recorded observations with:

```bash
.venv/bin/python scripts/plot_ch4_uptake_results_linux.py --results-dir ../mof_results/task_062
.venv/bin/python scripts/plot_agent_optimizer_timeline.py --results-dir ../mof_results/task_062
```

Both plotting scripts default to this canonical directory and save PNG, PDF,
and SVG files under its `figures/` folder. Use `--output-dir /tmp/ch4_figures`
to write elsewhere. The comparison averages `-y_best_so_far` across saved
seeds at shared evaluation indices and ends each curve at its last recorded
evaluation. The timeline replays the Agent's recorded optimizer labels.
Its label is `Agent (OpenAI, GPT-6.1 Sol Ultra)`. Replotting the saved Agent
trace does not run a fresh adaptive Agent optimization; that campaign's
historical files remain in the canonical results directory.

The scientific inputs retained under `src/agent_harness/` are:

- `xgb/ch4_uptake_1.model`: the supplied, already trained XGBoost surrogate.
- `data/chemunity/descriptors/ChemUnity_Full_geo.csv` and
  `data/chemunity/descriptors/core_desc_df_full.csv`: their column headers
  define the original ordered 188 descriptors.
- `data/chemunity/csv/ChemUnity_all_descriptors.csv`: complete finite rows
  define each descriptor's lower and upper bounds.

Optimizers receive the 160 nonconstant coordinates; traces preserve all 188.
Arbitrary combinations in this descriptor box may not represent realizable
MOFs. The workspace uses the supplied model without retraining it and claims
no reference optimum or regret.

Run the evaluator and runner checks with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

For a small sweep, add `--optimizers random_search sobol --budget 20 --seeds 2
--workers 2 --outdir /tmp/ch4_sweep_smoke` to the sweep command. Tests and
smoke sweeps use temporary outputs; the full baseline sweep is separate.
