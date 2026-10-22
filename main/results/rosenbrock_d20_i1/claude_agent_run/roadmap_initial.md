# Roadmap — Task 005 (dim=20, bounds=[-5,5]^20, minimize, deterministic, budget=100)

## Problem read

- Dimension 20, budget 100 — only 5x dim. This is a *very* small budget
  relative to dimension: far too small for population-based global methods
  (CMA-ES's default population here is ~4+3*ln(20)≈13, giving only ~7-8
  generations; DE's default pop_size=10*dim=200 would never even complete
  one generation) or for local derivative-based/model-based solvers that
  need many points just to build their first local model (BOBYQA's default
  initial interpolation set alone would consume a large fraction of the
  budget; L-BFGS-B's finite-difference gradient costs dim+1≈21 evals per
  step, i.e. ~5 steps total for the whole budget).
- The function is deterministic (no noise), so there is no need to spend
  budget on replication, and any method that assumes/benefits from
  noise-robustness (e.g. noisy-EI variants) is unnecessary overhead.
- This regime (continuous, box-constrained, deterministic, budget on the
  order of 5-10x dim) is exactly the documented sweet spot for sample-
  efficient sequential Bayesian optimization in this project's own
  optimizer survey (gp_bo.py docstring: "only practical at small budgets
  ... budgets of order 10 x dim, not 1000+ x dim"), which is the closest
  match among the available optimizers to this task's actual budget.

## Planned method sequence

1. **GP-BO (LogEI)** — primary method, intended to run for as much of the
   100-evaluation budget as it continues to be useful.
   - Its own internal design: 2*(dim+1) = 42 Sobol initial points, then
     up to 58 sequential LogEI-acquisition points (ask n=1 at a time).
   - Rationale: GP surrogates make maximal use of every expensive
     evaluation, which matters most when the budget is this tight. This
     is the project's own documented reasoning for when to prefer GP-BO
     over population-based or purely local methods.
2. **Local refiner (BOBYQA)**, held in reserve for the tail of the budget
   — only triggered if the deviation criteria below fire. BOBYQA is a
   trust-region model-based local solver that needs comparatively few
   evaluations to polish around an already-good point, which is a better
   use of a handful of remaining evaluations than continuing to fit a GP
   whose acquisition has stopped finding improvements.
3. **CMA-ES**, held in reserve only as a fallback if GP-BO's GP fitting or
   acquisition optimization fails outright (numerical error), since it is
   a robust, dependency-light alternative that needs no gradient or
   second-order model.

No blind global exploration phase (plain Sobol/random) is planned beyond
GP-BO's own built-in 42-point Sobol design — with only 100 evaluations
total for 20 dimensions, spending further budget on undirected search
before modeling would leave too little for the acquisition-driven phase
that is expected to deliver most of the improvement.

## Explicit deviation criteria

- **Plateau -> switch to BOBYQA local refinement:** once GP-BO has used at
  least 70 of the 100 evaluations (i.e. is in its acquisition-driven phase)
  AND `y_best_so_far` has not improved over the most recent 15 consecutive
  evaluations, stop GP-BO and spend the remaining budget running BOBYQA
  seeded at the current best point (with a tight initial trust region
  around it, respecting the global bounds), since this indicates GP-BO's
  acquisition has stopped finding useful structure and a cheap local
  polish is a better use of the remaining calls than more GP-driven
  global search.
- **Numerical failure -> fall back to CMA-ES:** if GP-BO's `ask()` raises
  an exception (e.g. GP fitting fails to converge, or `optimize_acqf`
  errors) at any point, switch immediately to CMA-ES for the rest of the
  budget, seeded with all points evaluated so far added to its population
  history where possible, otherwise started fresh from the box midpoint.
- **Non-improving/duplicate proposals:** if GP-BO proposes a point
  essentially identical (L2 distance < 1e-6 in normalized space) to one
  already evaluated, treat this the same as a plateau signal counting
  toward the 15-evaluation no-improvement window above.
- Any such switch will be logged in `changelog.md` with the evaluation
  index and the observed evidence (e.g. the actual no-improvement streak
  length or the exception raised).

## Budget allocation summary (planned, subject to the above triggers)

- ~42 evaluations: GP-BO's own Sobol initial design.
- ~58 evaluations: GP-BO LogEI acquisition loop (default plan), unless the
  plateau criterion above redirects the tail of this phase to BOBYQA.
