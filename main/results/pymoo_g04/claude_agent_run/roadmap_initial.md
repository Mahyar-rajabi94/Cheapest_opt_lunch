# Roadmap — initial plan

## Problem as stated
- dim = 5, bounds (per coordinate, asymmetric box):
  x0 in [78, 102], x1 in [33, 45], x2 in [27, 45], x3 in [27, 45], x4 in [27, 45]
- direction: minimize
- noise: deterministic (no observation noise)
- budget: 100 evaluate() calls total
- Objective returns a single penalized merit value: true_objective + lambda *
  violation. Feasible region's optimum coincides with the merit optimum
  (violation=0 there). I cannot separate true objective from penalty from a
  single scalar, so per the task instructions I will treat merit as the one
  scalar to minimize directly, and treat any sudden large upward jump in
  observed merit (relative to nearby points) as a signal of heavy
  infeasibility in that region to move away from, not as real objective
  curvature to chase or model literally.

## Budget is tight relative to dimension
100 evals / 5 dims = 20 evals-per-dim. This is a small budget, so the plan
favors a short, information-dense global scan followed by aggressive
exploitation, rather than a large model-based BO campaign (surrogate fitting
needs more points per dimension than we can afford for a reliable multi-stage
GP pipeline here, and with hidden constraints a GP over raw merit would be
badly distorted by penalty cliffs).

## Planned method sequence and budget split
1. **Sobol / LHS space-filling design — 20 evals** (`"sobol_init"`).
   Purpose: get a coarse, unbiased read of the merit landscape over the full
   box, including the chance of landing in deeply infeasible (high-penalty)
   regions, so later local search is seeded away from those and starts near
   the best feasible-looking basin found.
2. **CMA-ES from the best point(s) found — ~55 evals** (`"cma_es"`).
   Reasoning: CMA-ES is robust to the non-smooth, cliff-like structure a
   penalized merit surface can have (it doesn't need gradients and adapts
   its covariance/step size to local curvature it actually observes,
   including shrinking away from directions where it keeps hitting penalty
   cliffs). Initialize mean at the best Sobol point, initial sigma ~15% of
   each coordinate's range, with bounds passed in so CMA-ES's own
   box-constraint handling keeps samples inside the stated domain.
3. **BOBYQA local polish — ~25 evals** (`"bobyqa_polish"`).
   Reasoning: once CMA-ES has located a promising basin, BOBYQA's local
   quadratic-model trust-region approach is efficient at squeezing out the
   last bit of precision in a low-eval-count regime, provided we're already
   in a locally smooth (feasible, non-penalized) region.

Total: 20 + 55 + 25 = 100.

## Deviation criteria (quantitative, pre-registered)
I will deviate from the above and log it in changelog.md if any of these
trigger:
- **Sobol stage finds no finite-looking merit** (e.g. all 20 points show
  merit values clustered within a narrow band with no variation, suggesting
  the box is almost entirely one infeasible plateau): extend the init stage
  by up to 10 more Sobol points before moving to CMA-ES, borrowed from the
  CMA-ES budget.
- **CMA-ES stalls**: if the best-so-far merit has not improved for 15
  consecutive CMA-ES evaluations (roughly 2 generations at popsize 8), stop
  CMA-ES early and move the remaining budget to BOBYQA polish from the
  current best point.
- **CMA-ES repeatedly lands on penalty cliffs** (e.g. >50% of a generation's
  samples show merit at least 3x higher than the running best, a sign the
  step size pushed the population into infeasible territory): shrink sigma
  by 2x and re-center at the current best point rather than letting the
  adaptive step size keep exploring that direction.
- **BOBYQA's trust region collapses** (reported trust-region radius below
  ~1e-3 of the coordinate range before the polish budget is used up): restart
  BOBYQA from the second-best distinct basin found so far with the remaining
  budget, instead of continuing to spend evals on a converged point.
- **Any stage's budget allocation is overrun or underrun** because of early
  stopping above: unused evals roll forward to the next stage; if the final
  stage finishes early, apply remaining budget as an additional short
  BOBYQA/CMA-ES polish from the overall best point.

All deviations will be logged to changelog.md with the evaluation index at
which the change was made and the evidence that triggered it.
