# OPV device pool task (`opv_pool`)

The 526 OPV devices of Seifrid et al. (2024) as a fixed pool. The search box is
[0, 1]^11 (min–max scaled device features). Each evaluation maps `x` to the
nearest device not yet tested in this campaign (Euclidean distance, ties to the
lowest `candidate_id`) and returns that device's measured −PCE (%), so the same
`x` evaluated twice returns two different devices and a campaign allows at most
526 evaluations. `f_opt` is the best measured device (exact). The task is
`main/src/tasks/opv_pool.py`, registered as `opv_pool`; its results folder name
is `opv_pool_pce_d11`.

This folder names the task's identity. Never copy it into an agent workspace
(the runner never does).

## Files

| File | Role |
| --- | --- |
| `opv_preprocessing_prompt_pool.md` | Specification of the pool |
| `pce_cleaned_dataset.csv` | Input: cleaned device dataset |
| `preprocess.py` | Builds the pool from the dataset and checks it |
| `pool_features.csv`, `pool_labels.csv`, `pool_spec.json` | The pool, read by the task |
| `initial_sets.json`, `preprocessing_report.md` | Other outputs of `preprocess.py`: initial designs (not used by the task) and the checks with every deviation from the specification |

## Run it

Set up both environments and the Codex CLI once, as in the root `README.md`
("Install once"). Run every command from the repository root. `runs/` is
git-ignored, so results stay local unless you commit them.

1. Optional: rebuild the pool. The committed files are already its output.

   ```bash
   .agent-venv/bin/python task_006_baseline/preprocess.py
   ```

2. Smoke test:

   ```bash
   .venv/bin/python main/scripts/run_campaign.py --task opv_pool \
     --optimizer random_search --seed 0 --budget 20 --outdir runs/smoke
   ```

3. Baselines at budget 100: every optimizer, including the pool-only
   `pool_random` (uniform picks without replacement). Ten seeds, one for the
   deterministic BOBYQA, Nelder–Mead and L-BFGS-B, so 73 campaigns. They are
   committed, with their plots, in `main/results_linux/opv_pool_pce_d11/`. To
   rerun them (TuRBO traces vary slightly from run to run, on every task):

   ```bash
   .venv/bin/python main/scripts/run_sweep.py --task opv_pool \
     --budget 100 --seeds 10 --workers 4 --outdir runs/baselines-100
   ```

4. Agent run. Check isolation first (zero evaluations, no model call), then
   run. Codex is the default agent; add `--agent claude` for Claude Code.

   ```bash
   .venv/bin/python main/scripts/run_agent.py --task opv_pool --budget 100 --check-only
   .venv/bin/python main/scripts/run_agent.py --task opv_pool --budget 100
   ```

   The agent is told only the dimension, the [0, 1] box, the direction, the
   budget and that the objective is not repeatable. It never sees the pool,
   the devices or `pool_random.py`.

5. Plots, with the run ID the runner printed (against the committed
   baselines; use `--results runs/baselines-100` for your own rerun):

   ```bash
   .agent-venv/bin/python main/scripts/make_comparison_plots.py \
     --results main/results_linux --task opv_pool_pce_d11 \
     --agent-run runs/run-<id>/workspace/results/agent_run \
     --outdir runs/run-<id>/figures
   ```

   Regret is the best device's PCE minus the best PCE found. It is exactly 0
   once the best device is found, drawn at the bottom of the log axis (1e-12).

Budgets above 526 are refused by `run_campaign.py`, `run_sweep.py` and
`run_agent.py`.
