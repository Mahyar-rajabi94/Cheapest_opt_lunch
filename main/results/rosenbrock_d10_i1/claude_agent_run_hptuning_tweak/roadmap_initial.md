# Roadmap — initial step only

## Problem understanding
- Objective: sealed black-box function `fn(x)`, `x` a real vector of dimension
  10, each coordinate bounded in `[-5, 5]`.
- Direction: minimize.
- Noise: deterministic (no observation noise) — repeated evaluation of the
  same `x` is expected to return the same `y`, so no replication is needed to
  manage noise.
- Budget: 100 `evaluate()` calls total, enforced by `AgentTaskHarness`.
- The function's identity is unknown and must stay unknown to this campaign;
  all decisions below are made only from observed `(x, y)` pairs collected
  through the harness, not from recognizing a textbook function.

## Assumptions
- No assumption is made yet about convexity, multimodality, separability, or
  conditioning of the objective. Precisely resolving that uncertainty is the
  purpose of the first step.
- Budget is small relative to dimension (100 evals for d=10), so the overall
  campaign will need to spend evaluations efficiently: a short, unbiased,
  low-discrepancy look at the space before committing real budget to any
  structural assumption (e.g. local trust-region vs. global/evolutionary vs.
  surrogate-based search).

## Step 1 — Sobol' space-filling design
- **Method:** `SobolSearch` (this sandbox's `src/optimizers/sobol_search.py`,
  wrapping `scipy.stats.qmc.Sobol`).
- **Hyperparameters / initialization:** `bounds = [[-5, 5]] * 10`, `seed = 0`
  (for reproducibility only — no tuning effect). All other behavior is
  scipy's own `qmc.Sobol` default (`scramble=True`), inherited as-is per the
  wrapper's own documented convention of "no tuning" for this method.
- **Why this method for this step:** Sobol' gives low-discrepancy coverage of
  the box with no prior assumption about the objective's shape, and is cheap.
  It is the right tool for the *first* step specifically because nothing is
  yet known about the landscape — a space-filling design is the generic,
  assumption-free way to get that first read, and the resulting points/values
  are reusable observations (nothing is wasted) regardless of what method is
  chosen next.
- **Step length:** 16 evaluations. Cumulative count at which to pause: 16.
  Chosen as a power of 2 (Sobol' sequences have their best balance
  properties when the cumulative count drawn is a power of 2 — see the
  wrapper's docstring) that is close to, but a bit below, the "2×dim" initial
  design size this project's own TuRBO wrapper uses by convention for a
  dimension-10 problem. 16/100 = 16% of the total budget — enough points to
  characterize the box's range and rough structure in 10 dimensions, while
  leaving the large majority of the budget (84 evaluations) for an adaptive
  method once there is real evidence to choose and configure it.
- **What I want to learn / evidence to inspect at the checkpoint:**
  - The range and distribution of observed `y` values (best, worst, spread)
    — to gauge the objective's scale and whether it varies smoothly or
    erratically across the box.
  - Whether the best point(s) cluster in one region of the box or are
    scattered — a first signal on uni- vs multi-modality / rough vs smooth
    structure, informing whether to start a local refinement (e.g. BOBYQA,
    CMA-ES from the best point) or keep exploring more globally first
    (e.g. more Sobol', or a surrogate/evolutionary method with broader
    scope).
  - Evaluation timing (`eval_seconds`), to confirm the objective is cheap
    enough that later methods with more per-iteration overhead (e.g.
    GP-based) remain practical within the remaining wall-clock.
- **Early-pause conditions:** none — this batch consists of 16 independent,
  non-adaptive Sobol' draws (each point does not depend on the objective
  value of previous points within this step), so there is no evidence
  available mid-batch that could usefully change the remaining draws of
  this same step. The checkpoint after all 16 is the first point at which
  new evidence exists.
