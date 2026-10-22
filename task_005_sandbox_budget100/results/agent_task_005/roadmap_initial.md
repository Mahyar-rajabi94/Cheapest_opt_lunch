# Roadmap (pre-registered, written before any evaluation) — Task 005

## Problem as stated

- Minimize a deterministic black box over x in [0,1]^11, **100 evaluations total**.
- Six known linear inequality constraints on x[0..5]. Only allowed points count.
  A Monte Carlo check (no evaluations used) shows ~54% of the box is allowed.
  Constraint 6 is the most restrictive (cuts ~23% of the box), then 5 (~12%) and 4 (~11%).
- x[9] (3 levels) and x[10] (14 levels) are effectively discrete: 42 combinations.
- x[6..8] are box-only continuous coordinates.

## Method choice and reasoning

With 100 evaluations in 11 dimensions (~9 evaluations per dimension), there is no
room for population methods (CMA-ES, DE) to converge: one CMA-ES generation in 11-D
is ~10 points, so 100 evaluations is ~10 generations, and DE would be worse. Local
DFO (BOBYQA/Nelder–Mead) needs 2d+1 = 23 points just to build its first model, and
it only finds a local optimum from a single start. The function is deterministic, so
there's no noise to average out. That makes every evaluation informative and
favours a **surrogate model (Gaussian process) Bayesian optimization** approach.

The constraints are **known and linear**, so I won't learn them. I'll enforce
them exactly when choosing points: every point I evaluate will be allowed, with a
small safety margin of 1e-6 on each right-hand side so it passes the 1e-10 check.
No budget goes on disallowed points.

The two discrete coordinates are always sent at their exact level values (snapped
before evaluation), so the data the GP sees matches what the function actually used.
Inside the GP they are treated as ordinal numeric inputs at their level positions
(one-hot would add 17 dimensions, which 100 points cannot support).

## Planned sequence and budget split

1. **Evals 1–20: space-filling initial design ("init_sobol_feasible").**
   Scrambled Sobol points in the 9 continuous coordinates, keeping only allowed
   points, with greedy maximin selection from a large allowed pool. x[9] and x[10]
   levels are assigned stratified, so all 3 x[9] levels and as many x[10] levels as
   possible appear.

2. **Evals 21–85: global constrained GP-BO ("gpbo_logei_lincon").**
   SingleTaskGP (BoTorch defaults: dimension-scaled lengthscale prior, ARD),
   standardized outputs, and analytic LogEI with best_f = best allowed value.
   I'll optimize the acquisition over the allowed polytope with BoTorch's
   `inequality_constraints`. Discrete coordinates are handled by scoring a large
   allowed random and local-perturbation candidate pool across the (x9, x10) combos.
   Then I run gradient-based refinement of the continuous coordinates for the top
   starts, with the discrete combo fixed, followed by re-enumeration of all 42 combos
   at the refined continuous point. One point per iteration (fully sequential).

3. **Evals 86–100: local trust-region exploitation ("trbo_local").**
   A box trust region around the best allowed point, intersected with the
   polytope. Initial half-width is 0.15 per coordinate, scaled by the GP's
   relative lengthscales. I'll maximize LogEI inside it using the GP fit on all
   data. After 3 consecutive non-improving evaluations the region halves; on
   improvement it expands ×1.5, capped at 0.3. If it shrinks below a half-width of
   0.01, it restarts around the second-best distinct basin.

## Pre-registered criteria for deviating from this plan

- **Heavy-tailed outputs.** If, after the initial design, max(y) − median(y) >
  10 × (median(y) − min(y)), or the sample skewness of y exceeds 2, I'll switch the
  GP target from standardized y to a monotone transform: log(y − min(y) + δ), with δ
  = 0.1 × IQR(y) or a rank-Gaussian transform. I'll keep that until the criterion no
  longer holds.
- **Poor surrogate.** Every 10 BO iterations, I'll compute the leave-one-out
  Spearman rank correlation between GP predictions and observations. If it is below
  0.3 at two consecutive checks, I'll stop global BO and move to the local
  trust-region phase early, anchored at the best point.
- **Stagnation.** If the best allowed value hasn't improved by more than 1e-3 ×
  (y range) over **15 consecutive** BO evaluations, I'll move to the local phase
  early. Any unspent global budget goes to local refinement plus at most one
  restart in the next-best basin.
- **Acquisition collapse.** If the proposed point is within normalized
  Euclidean distance 1e-3 of an existing evaluation, I'll replace it with the best
  candidate at least 0.05 away. If LogEI is uninformative everywhere (max EI below
  1e-8 × y-std), I'll use one pure-exploration point: the maximum posterior std
  over the allowed region.
- **Late improvement in the global phase.** If the global BO is still improving
  (an improvement within its last 5 evaluations) at evaluation 85, I'll extend it
  by up to 5 evaluations before switching to local.
- **Constraint safety.** If any proposal fails the allowed check after optimization,
  I'll project it onto the polytope (minimum-distance QP, 1e-6 margin) before
  evaluating. If a disallowed point is ever evaluated, it will be logged as a
  deviation.

All deviations will be logged in `changelog.md` with the evaluation number and
the observed evidence.
