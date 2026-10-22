# Roadmap — task011 (dim=30, bounds=[-5,5]^30, minimize, deterministic, budget=100)

## Context / key constraint

Budget (100) is only ~3.3x the dimension (30). This is far below the regime
any of the available optimizers are designed/tuned for by default:

- `GPBO`'s own docstring says it targets budgets "of order 10 x dim" and its
  default initial design alone is `2*(dim+1) = 62` points (62% of our total
  budget) before a single acquisition-driven point is even proposed.
- `TuRBO`'s default initial design is `2*dim = 60` points (60% of budget),
  also leaving little room for the trust-region loop to adapt.
- `BOBYQA` is a strictly local, sequential trust-region solver seeded at the
  box midpoint; it typically wants on the order of `2*dim+1 = 61` points just
  to build its first fully-determined quadratic model, and its wrapper does
  not accept a custom starting point.
- `CMAES`'s default population size for dim=30 is `4 + floor(3*ln(30))  ≈ 14`,
  so the full 100-eval budget is only ~7 generations — few, but CMA-ES still
  makes monotonic (per-generation) progress even in a handful of generations,
  and it isn't gated behind a large fixed up-front design the way GP-BO/TuRBO
  are.

Given this, spending the majority of the budget on an initial design sized
for a ~10x-dim regime (GP-BO, TuRBO) is not a good use of 100 evaluations,
and a from-scratch BOBYQA run from the midpoint may not even finish building
its first local model. Plain CMA-ES, with its built-in standard
box-based initialization (x0 = box midpoint, sigma0 = quarter mean box
width), gets some exploration and some adaptation within the same budget,
which fits the regime better than the other three.

## Planned method sequence

1. **Sobol' global scan — 20 evaluations** (`sobol_search`, seed=0).
   A small, cheap, low-discrepancy scan of the full box to get a rough sense
   of where the landscape's better region(s) are, without spending the
   majority of the budget the way GP-BO/TuRBO's own default init would.
   20 points in 30 dimensions is sparse, but it is purely for picking a
   sensible region to seed the next phase, not for building a surrogate.

2. **CMA-ES local refinement — remaining 80 evaluations** (`cma_es`, seed=0),
   restricted to a box of half-width 2.0 per coordinate (clipped to the
   original [-5, 5] bounds) centered on the best point found in phase 1.
   Shrinking the box both lowers CMA-ES's own default sigma0 (quarter of the
   box's mean width) and re-centers its default start point (box midpoint)
   on the most promising region found so far — using the library's own
   standard box-based defaults, just on a smaller, better-centered box,
   rather than hand-picking x0/sigma0 directly. With popsize ≈ 14 for a
   width-4 box in 30D, this gives on the order of 5-6 generations of local
   adaptation.

This is a deliberate two-phase hybrid (global scan → local refine), chosen
specifically because of the budget/dimension mismatch, not a default
off-the-shelf config for either sub-method.

## Reasoning recap

- No data yet exists, so this plan is based purely on the stated problem
  shape (dim=30, tight box, deterministic, minimize, budget=100) and each
  optimizer's own documented behavior/defaults (read from in-repo docstrings
  only — the objective itself has not been inspected or evaluated).
- Deterministic/noiseless objective means no replication budget is needed,
  and single-point acquisition/refinement is safe (no risk of chasing noise).
- A pure global method (Sobol alone) for all 100 evaluations would not
  exploit any structure found; a pure local method from the box midpoint
  (BOBYQA/CMA-ES over the full box) risks wasting evaluations if the
  midpoint region is poor. A short scan-then-refine hybrid hedges between
  the two given the small budget.

## Pre-registered deviation criteria

I will deviate from the plan above, and log it in `changelog.md`, if any of
the following are observed:

- **CMA-ES stalls:** if, during phase 2, the best-so-far value does not
  improve over any stretch of 20 consecutive evaluations (i.e. roughly
  1.5 generations with the above popsize), I will restart CMA-ES with a new,
  re-centered box around the current global best point (a "soft restart"),
  rather than continuing to spend budget on a stalled search.
- **Sobol phase finds a very flat/uninformative landscape** (all 20 values
  within a tiny relative range, e.g. < 1e-6 relative spread): in that case
  I will keep the box used for phase 2 centered on the box midpoint instead
  of the best Sobol point, since "best of 20 near-identical values" would be
  noise-level, not signal.
- **Running low on budget mid-generation:** if fewer evaluations remain than
  CMA-ES's current population size, I will fall back to finishing with the
  single best point found so far rather than submitting a partial, unscored
  generation (CMA-ES's `tell` only updates once a full generation is told,
  so a partial last generation would otherwise never be consumed).
- If any optimizer call raises an unexpected error (e.g. a numerical
  failure), I will fall back to plain random sampling within bounds for any
  remaining budget rather than stopping early, so the budget is not wasted.
