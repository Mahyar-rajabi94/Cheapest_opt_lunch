# Roadmap — sandbox_task004

## Problem as stated
- dim = 10, bounds = [-5, 5]^10, direction = minimize
- deterministic (no observation noise)
- budget = 100 `evaluate()` calls total, hard-enforced by the harness
- objective: sealed black box, identity unknown and not to be investigated

## Planned method sequence

1. **TuRBO-1** (trust-region Bayesian optimization), ~80 evaluations
   (roughly evaluations 1–80, in its own batches of 4).
2. **BOBYQA** local polish from the best point TuRBO found, ~20 evaluations
   (roughly evaluations 81–100).

### Why this split

- With only 100 evaluations in 10 dimensions, sample efficiency matters far
  more than asymptotic convergence: 100 points is barely 10 per dimension,
  nowhere near enough for population-based global methods (CMA-ES,
  differential evolution) to do more than a handful of generations, and
  nowhere near enough for plain quasi-random/grid coverage of a 10-D box.
- TuRBO is specifically designed for this regime: it builds a local GP
  surrogate inside a shrinking/expanding trust region and uses Thompson
  sampling to pick informative points, which historically needs far fewer
  evaluations than CMA-ES or DE to make progress in <=20D problems. Its own
  initial design is 2*dim = 20 points (quasi-random, Sobol-based),
  consuming a reasonable fraction of the budget but leaving ~60 evaluations
  for trust-region-guided search afterward.
- The function is deterministic, so there is no need to spend budget on
  replication/noise-robust acquisition; every evaluation can go straight
  toward exploration/exploitation of new points.
- Reserving the final ~20 evaluations for BOBYQA (a model-based
  trust-region local solver) is meant to squeeze out a final local
  refinement near TuRBO's best point once its global trust-region search
  has used most of the budget: BOBYQA is a strong closing move when a
  promising basin has already been located, since it builds a local
  quadratic model and converges fast locally, something TuRBO (restarted
  from scratch whenever its own trust region collapses) is not optimized
  for in the last few evaluations of a small budget.
- Plain local solvers (L-BFGS-B, Nelder-Mead) are not used first because
  gradient/finite-difference-based descent from a single arbitrary start
  risks getting stuck in a local basin with no exploration budget left to
  recover, given we know nothing about the function's shape a priori.

## Explicit deviation criteria

- If TuRBO's trust region collapses and restarts (length shrinks below its
  own `length_min`) **before evaluation 50**, cut the TuRBO phase short at
  the next restart point and switch to BOBYQA polishing early — a collapse
  that early suggests TuRBO has already converged on what it can find
  locally and further evaluations are better spent on precise local
  refinement.
- If the running best value has not improved at all over the most recent
  30 evaluations while still in the TuRBO phase, switch to BOBYQA early
  rather than waiting out the full 80-evaluation allocation.
- If BOBYQA's trust region collapses (hits its own `rhoend`) with budget
  still remaining, restart it from a new point: the current best if not
  yet tried as a BOBYQA start, otherwise a random point from the
  best-TuRBO basin (small random perturbation of the best point found so
  far), to use any remaining evaluations productively instead of idling.
- Any such change will be logged in `changelog.md` with the evaluation
  index and the evidence that triggered it.
