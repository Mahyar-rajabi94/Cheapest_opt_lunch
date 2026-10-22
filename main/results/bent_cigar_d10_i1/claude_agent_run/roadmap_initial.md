# Roadmap — Task 003 (dim=10, bounds=[-5,5]^10, minimize, deterministic, budget=100)

## Plan

1. **Sobol quasi-random initial design — 20 evaluations (evals 1-20).**
   Use `sobol_search.SobolSearch` to draw 20 low-discrepancy points across
   the full box. With only 100 evals total for a 10-dimensional space, a
   small but well-spread design is needed to locate a promising basin
   before committing the bulk of the budget to local refinement. 20 points
   (2x the dimension) is enough to get a rough read on the landscape
   (unimodal vs. multimodal-looking, scale of variation) without spending
   so much of the tiny budget on pure exploration that nothing is left to
   exploit it.

2. **BOBYQA local refinement from the best Sobol point — remaining ~80
   evaluations (evals 21-100).**
   BOBYQA is a model-based trust-region local solver that is highly
   sample-efficient on smooth deterministic objectives — exactly the
   setting here (noise=deterministic). Rather than using this project's
   BOBYQA wrapper's own fixed box-midpoint start, I will seed it by
   constructing it normally (it starts from the box midpoint internally)
   fed through this same harness; see deviation note below if this turns
   out to be a meaningful waste of budget relative to restarting a fresh
   BOBYQA run from the Sobol-phase best point. Given the dimension (10)
   and budget (100), a single local solver run initialized near the best
   found region should be able to make real progress in ~80 evaluations,
   far more sample-efficient per-eval than a population method (CMA-ES
   needs a handful of generations just to get going) or a full GP-BO
   (expensive per-iteration GP fits are not worth it for an 80-eval
   budget on top of only 20 prior points).

## Why these methods (reasoning from problem properties alone)

- **Deterministic, no noise** → no need for replication or noise-robust
  acquisition; every evaluation's y is trustworthy as-is, so a
  trust-region local method that trusts its model fits is appropriate.
- **Budget is very small (100) relative to dimension (10)** → favors
  methods with low per-evaluation overhead and fast convergence once in a
  good basin (BOBYQA), over methods that need many evaluations just to
  calibrate (full Bayesian optimization with hyperparameter fitting,
  CMA-ES population dynamics) or that never converge locally (pure
  random/Sobol search).
- **Unknown landscape (unimodal vs. multimodal)** → a short, cheap
  exploration phase (Sobol) before committing to a local method hedges
  against committing immediately to one local basin that might be poor;
  if the Sobol phase reveals large variation across the box, this is
  evidence worth reacting to (see deviation criteria).

## Explicit deviation criteria

I will deviate from the plan above, and log it in `changelog.md`, if any
of the following triggers:

- **Trigger 1 — Sobol phase looks flat/unhelpful:** if, after the 20 Sobol
  evaluations, the best and worst observed y values differ by less than
  1% of their mean magnitude (i.e., the landscape looks nearly flat over
  the box at this resolution), I will spend only 10 more evaluations on
  a second exploration batch (CMA-ES early generations) before switching
  to local refinement, rather than immediately trusting the single best
  Sobol point as a good starting basin.
- **Trigger 2 — BOBYQA converges/exhausts early:** BOBYQA's solver thread
  may terminate (converge to a local optimum, or hit an internal
  stopping condition) before the ~80 allotted evaluations are used. If
  `OptimizerExhausted` is raised with budget remaining, I will restart a
  fresh BOBYQA instance from a different starting point (e.g. the next
  best Sobol point, or a small random perturbation of the current best)
  to use the remaining budget, rather than leaving evaluations unused.
- **Trigger 3 — BOBYQA plateaus:** if 25 consecutive BOBYQA evaluations
  produce no improvement to the running best y, I will treat the current
  trust region as likely stuck in a local optimum and restart BOBYQA from
  a new point (best alternate Sobol point not yet locally refined, or a
  random perturbation around the current global best) for the remaining
  budget.
- **Trigger 4 — budget exhaustion:** regardless of the above, the harness
  hard-stops at 100 evaluations; no method switch will be attempted past
  that point.
