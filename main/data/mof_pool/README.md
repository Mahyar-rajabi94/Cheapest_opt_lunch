# MOF exact candidate pool

`data.csv` is an unchanged copy of `src/agent_harness/data_.csv` from the
user-supplied `task_061_sandbox_openai.zip`. Its SHA-256 and archive provenance
are recorded in `provenance.json`. It contains **1,000 rows and 193 features**,
including 28 constant features. The archive README describes 5,882 rows; that
larger table was not supplied. No rows were removed, reordered, or regenerated.

The registered task is `mof_pool`, implemented in `main/src/tasks/mof_pool.py`.
Every purchase selects an exact, previously untested row. Duplicate feature
vectors, if present, retain separate row identities. Features are public;
material names and the complete `y` column stay private. The isolated workspace
gets only an anonymous feature matrix and the generic evaluator client.

The original objective maximizes raw `y`; the repository minimizes `-y`.
Thus regret is `max(raw_y) - best_observed_raw_y`. No physical unit or meaning
for `y` was included in the supplied data contract. Global ranks and material
IDs may be added to trusted post-run reports, never to agent feedback.

## Agent campaigns

Run from the repository root after the environment setup in the root README:

```bash
.venv/bin/python main/scripts/run_agent.py --task mof_pool --budget 100 --check-only
.venv/bin/python main/scripts/run_agent.py --task mof_pool --budget 100 --seed 0 \
  --model gpt-6-astra --effort max
```

Use seeds 0, 1, and 2 for the requested three independent campaigns. The current
root `AGENTS.md` governs each campaign: the agent selects its own initialization,
method, hyperparameters, and evaluation block length. The archive's old prompt,
fixed ten-point initialization, GPBO/TuRBO restriction, and 110-call brief are
not used for these agent runs. The confirmed total budget is 100, including all
initialization. Model generation itself is not made deterministic by this seed.

## Baseline compatibility

The colleague's feature-only pool GPBO and TuRBO implementations are retained as
`src/optimizers/pool_gp_bo.py` and `pool_turbo.py`, with configurable `n_init`
(default 10). They fit only purchased observations and select from unused rows.
Existing continuous optimizer wrappers are unchanged. Exact-pool baseline
dispatch supports `gp_bo`, `turbo`, and `pool_random`; continuous box optimizers
are not silently projected onto rows. GP baselines retain ten shared seeded
uniform initial rows, counted within their total budget. Rank is never a model
training target.

```bash
.venv/bin/python main/scripts/run_campaign.py --task mof_pool --optimizer gp_bo \
  --seed 0 --budget 100 --outdir runs/mof-baselines-100
.venv/bin/python main/scripts/run_sweep.py --task mof_pool --budget 100 \
  --seeds 10 --workers 4 --outdir runs/mof-baselines-100
```

These are reproduction commands; the published comparison uses the supplied
`results_linux.zip` GP-BO and TuRBO traces (10 seeds each), preserved in the
[materials result package](../../agent_runs/materials_astra_max_100_3seeds_20261022/README.md).
The supplied traces contain 110 evaluations. The comparison uses calls 1–100,
including their ten random initialization calls, and discards calls 101–110.
Every recorded candidate ID, feature vector, label, rank and incumbent was
checked against this registered pool. Raw supplied traces maximize positive y;
the report converts them to regret using the same pool maximum as the agent.
Archive provenance, verification code and sample standard deviations are in
the result package. No new baseline sweep was run here.

Before comparing any future rerun, verify the dataset hash, row order, sign
convention, seed, and total evaluation budget.
The previous sandbox exposed the entire label CSV to the agent process and
relied on instructions; new runs use the repository's native filesystem
isolation and separate durable evaluator instead.
