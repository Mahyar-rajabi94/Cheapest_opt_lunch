# Roadmap — initial step only

## Understood problem

- Sealed black-box merit function `merit(x) = y(x) + PENALTY_LAMBDA * total_constraint_violation(x)`,
  to be **minimized**. The minimum of the merit coincides with the true constrained optimum, so the
  harness's scalar return can be treated as a single ordinary (if possibly sharply penalized)
  objective.
- Dimension: 10. Bounds (read from `harness.bounds` after construction): `[-10, 10]` per coordinate
  (box, all ten dimensions identical).
- Direction: minimize.
- Noise: deterministic (`harness.noisy == False`), confirmed from the harness instance.
- Total evaluation budget: 100 `evaluate()` calls, hard-enforced by the harness.
- No information about the function's identity, shape, or the constraints' nature (number, type,
  linear/nonlinear) is available or will be sought. I only know that infeasible regions are
  penalized additively with a (likely large) `PENALTY_LAMBDA`, so merit values far above the bulk
  of observed values most likely indicate infeasibility rather than a genuinely large `y`.

## Key assumption

With only 100 evaluations in a 10-dimensional box, the budget is tight relative to the
dimensionality (10x the dimension). I am assuming the merit surface may have substantial flat or
heavily-penalized (infeasible) regions, so the very first priority is a cheap, space-filling look
at the box to learn (a) the rough scale/range of merit values, (b) whether most of the box is
feasible or only a small sub-region is, and (c) whether there is any visible structure to exploit.
This favors spending a modest slice of the budget on quasi-random global coverage before
committing the bulk of the budget to a local/adaptive refinement method, rather than immediately
starting a single local optimizer from the box center (which could land in an infeasible or
unrepresentative region).

## Step 1: method and hyperparameters

- **Method:** Sobol' quasi-random search (`src/optimizers/sobol_search.py`, wraps
  `scipy.stats.qmc.Sobol`, scrambled, scipy's own defaults — no tuning).
- **Hyperparameters:**
  - `bounds`: `[[-10, 10]] * 10` (from `harness.bounds`).
  - `seed`: 0, for reproducibility. This is the only parameter the wrapper exposes beyond bounds
    (inherited default: `scramble=True`, scipy's own default, not changed).
  - Batch size for `ask`: 16 points in one call. 16 is a power of two, which gives the Sobol
    sequence its best (balanced/low-discrepancy) coverage property for the cumulative count drawn,
    per the wrapper's own docstring guidance.
- **Initialization:** fresh `SobolSearch` instance; no prior observations exist yet.

## Step 1 length

- Maximum additional evaluations this step: **16** (one Sobol batch).
- Pause at cumulative evaluation count: **16**.
- Why this length: 16 points (~1.6x the dimension) is enough to get a first read on the merit
  range and roughly how much of the box looks feasible vs. heavily penalized, while still leaving
  84 of the 100-call budget (the large majority) for a more sample-efficient adaptive method once
  that picture is available. A single Sobol batch (rather than committing to a larger
  space-filling design) keeps this decision revisable: if the 16 points already show a clear,
  learnable structure, I can move to local refinement; if they look uninformative or highly
  degenerate (e.g. almost everything infeasible), I can decide whether a second exploratory batch
  is warranted before switching methods.

## What I want to learn / checkpoint evidence

At the checkpoint after these 16 evaluations I will inspect:
- The range (min/max/median) of observed merit values, to gauge scale and spot a likely penalty
  regime (e.g. a cluster of very large outlier values vs. a tighter band of plausibly-feasible
  values).
- The location(s) of the best point(s) found, to decide a sensible initialization/region for the
  next method (e.g. restart a local optimizer near the best point rather than the box center).
- Any visible per-coordinate pattern (e.g. consistent sign or magnitude on the best points) that
  might inform bounds/step-size choices for the next step.

Conditions for pausing sooner than 16: none — a single small batch of 16 deterministic,
independent evaluations carries no risk of runaway cost or divergence, so there is no reason to
interrupt it before completion.
