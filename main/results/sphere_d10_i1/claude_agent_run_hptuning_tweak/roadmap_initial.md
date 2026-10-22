# Roadmap — Initial Step

## Understood problem

- Sealed black-box objective, treated as unknown structure (no attempt to
  identify it).
- Dimension: 10. Bounds: [-5, 5] per coordinate (box-constrained).
- Direction: minimize.
- Noise: deterministic — no observation noise, so repeated evaluation of
  the same `x` would return the same `y`; no replication is needed to
  manage measurement noise.
- Total evaluation budget: 100 calls to `harness.evaluate(x)`, hard-enforced
  by the harness (`BudgetExhausted` once used up).
- Harness: `AgentTaskHarness(budget=100)` from
  `src/agent_harness/harness.py`, with `set_active_optimizer(name)` tagging
  every evaluation and automatic trace/meta logging.

## Key assumption driving the first step

With only 100 evaluations in 10 dimensions, the budget is very small
relative to the dimensionality (10x the dimension, not the 100x-1000x+
typical for population-based evolutionary methods to converge well). This
favors starting with a compact, space-covering design rather than committing
early to a single local solver or a large population size, so the very
first decision is simply to gather an unbiased, well-spread sample of the
landscape and use it as evidence for the next method choice (e.g. whether
the function looks smooth/unimodal enough for a local trust-region method,
or multimodal/noisy enough to warrant a global/model-based approach such as
TuRBO or CMA-ES).

## Step 1: method, hyperparameters, initialization

- **Method:** Sobol' quasi-random sampling (`src/optimizers/sobol_search.py`,
  `SobolSearch`), using scipy's `qmc.Sobol` with `scramble=True` (the
  wrapper's own documented default — no tuning).
- **Hyperparameters:**
  - `bounds = [(-5, 5)] * 10` (as given).
  - `seed = 0` (for reproducibility; no other run-time meaning since the
    objective is deterministic).
  - No other hyperparameters are exposed by this wrapper.
- **Initialization:** fresh `SobolSearch` instance, first call to `ask`.
- This step produces the campaign's only "pure exploration" design; all
  later steps will be chosen using its results plus whatever happens after.

## Step length

- **Evaluations this step:** 20 (a conventional 2×dim low-discrepancy
  initial design size, matching the convention used by this project's own
  TuRBO/GP-BO wrappers for initial designs — not a special tuning choice).
  This uses 20% of the total budget.
- **Cumulative count at pause:** 20.
- **Why this length:** 20 points in 10 dimensions is enough to get a rough
  sense of the objective's scale, range, and whether nearby Sobol points
  produce similar or wildly different values (a cheap proxy for
  smoothness/ruggedness), without committing a large fraction of the very
  tight 100-call budget to pure exploration before any adaptive method has
  had a chance to run.

## What I want to learn / evidence to inspect at the checkpoint

- The observed range and distribution of `y` across the 20 points (min,
  max, spread) — to gauge scale and whether there's an obvious promising
  region.
- Whether the best few points cluster in a region of the box or are
  scattered — a weak signal for uni-modality vs. multi-modality.
- Rough local variation (e.g. are nearby-ish Sobol points' y-values
  similar or very different) — informs whether a smooth local method
  (BOBYQA, CMA-ES) is likely to work well starting from the current best,
  versus needing a broader/model-based search (TuRBO, GP-BO) first.

## Early-pause conditions

- None expected to trigger within 20 evaluations other than reaching the
  step limit; this is a fixed-size initial design, not an adaptive loop,
  so there is no earlier stopping signal to monitor mid-step.
