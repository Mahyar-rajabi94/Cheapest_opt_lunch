# Roadmap — Initial Step

## Understood problem

- Sealed black-box merit function `merit(x) = y(x) + PENALTY_LAMBDA * total_constraint_violation(x)`,
  to be **minimized** over `x in R^10`.
- Dimension: 10. Bounds, as reported by `harness.bounds` after construction:
  `[-5, 5]` per coordinate (all 10 dimensions identical).
- Deterministic objective (harness reports `noisy == False`): no need for
  replication or noise-robust acquisition; a single evaluation per point is
  trustworthy.
- Hard evaluation budget: 100 `harness.evaluate()` calls total, enforced by
  the harness (raises `BudgetExhausted` past the limit; auto-saves
  `trace.csv`/`meta.json` at exhaustion).
- The merit value bakes constraints in as a penalty: a value far outside the
  typical range likely reflects heavy infeasibility rather than an extreme
  true objective, so I should expect some evaluations to look like outliers
  without concluding the underlying objective itself is pathological there.
- I do not know and will not attempt to infer the function's identity; all
  decisions are based only on observed (x, merit) pairs collected from
  `harness.evaluate()`.

## Key constraint shaping this campaign: budget = 100, dim = 10

A budget of 100 in 10 dimensions is very tight relative to typical
population-based global optimizers (e.g. CMA-ES's default population size
for dim=10 is on the order of 10, giving only ~10 generations total) and
relative to typical space-filling design sizes (10x the dimension, i.e. 100
points, would already be the entire budget). This means the campaign must
spend evaluations deliberately: a modest initial exploration to learn the
coarse landscape (including where the merit penalty is small/zero, i.e.
where the point looks feasible), followed by committing most of the budget
to refining around the most promising region(s) found. I will decide the
balance step by step rather than fixing it now.

## First step: Sobol space-filling design

- **Method:** `SobolSearch` (src/optimizers/sobol_search.py), a scrambled
  Sobol' low-discrepancy sequence via `scipy.stats.qmc.Sobol`, scaled to
  `harness.bounds`. This is a pure exploration step with no adaptation, used
  to get an unbiased, well-spread first look at the merit landscape
  (feasible vs. heavily-penalized regions, rough scale of `y`) before
  committing to any local or model-based method.
- **Hyperparameters:** `seed=0` for reproducibility (otherwise the
  wrapper's only parameter). `scramble=True` is the wrapper's hard-coded,
  library-default choice (scipy's own default), not changed here. No other
  tuning is applicable to this method.
- **Initialization:** fresh `SobolSearch(bounds=harness.bounds, seed=0)`.
  I will draw points in one batch of `n=16` (a power of two, which is the
  size at which a Sobol' sequence's low-discrepancy balance property
  actually holds, per the wrapper's own docstring recommendation).
- **Step length:** 16 objective evaluations. Cumulative evaluation count at
  which I will pause: **16** (out of 100; 84 remaining after this step).
  Rationale for this length: 16 points (1.6x the dimension) is enough to see
  a spread of merit values across the box, including some near-feasible and
  some strongly infeasible points, without spending a large fraction of the
  very tight 100-evaluation budget on pure exploration before any
  refinement begins. Using a power-of-two batch also preserves the Sobol'
  sequence's intended coverage quality for this first draw.

## What I want to learn / checkpoint evidence

- The range and distribution of observed merit values (min, max, median),
  to gauge how much of the box is near-feasible vs. heavily penalized.
- The location(s) of the best point(s) found, to decide whether a local
  refinement method (e.g. BOBYQA, CMA-ES) should be initialized there next,
  or whether more global exploration is warranted first.
- Whether the merit landscape looks smooth/unimodal near the best point
  (checked informally from the spread of values) or highly rugged, which
  would influence the next method choice.

No early-pause condition is set for this step: 16 cheap, independent Sobol'
evaluations carry no risk of wasting the budget chasing a bad trajectory
(unlike an adaptive method), so the step runs to its full length before
inspection.
