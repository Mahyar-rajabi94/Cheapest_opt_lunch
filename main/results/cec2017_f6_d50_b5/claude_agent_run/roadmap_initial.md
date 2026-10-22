# Roadmap — task012 (dim=50, bounds=[-5,5]^50, minimize, deterministic, budget=100)

## Problem framing

Budget (100) is only ~2x the dimension (50). This is an extremely tight
ratio relative to the conventions this project's optimizers were built
around, so the first job was to work out which of the available tools can
even produce a real optimization step inside this budget, before picking
a sequence.

Checked (by instantiating each optimizer class locally, bounds only, no
objective calls — this touches no black-box evaluations and consumes no
budget):

- **DifferentialEvolution**: pop_size = 10*dim = 500. The first generation
  alone exceeds the whole budget by 5x; it would never complete even one
  real generation. Ruled out.
- **GP-BO**: initial Sobol design = 2*(dim+1) = 102 points, already over
  budget before the first acquisition-driven point. Ruled out.
- **TuRBO**: initial design = 2*dim = 100 points — exactly the entire
  budget, leaving zero evaluations for any actual trust-region/acquisition
  step. Ruled out.
- **BOBYQA**: default interpolation set size npt = 2*dim+1 = 101 > budget.
  It would spend the whole budget building its initial model and likely
  never take a real trust-region step. Ruled out as a primary/standalone
  method.
- **Nelder-Mead**: needs dim+1 = 51 points just to build the initial
  simplex, leaving only ~49 evals for actual simplex operations in 50
  dimensions. Weak, but not immediately disqualified; not chosen as
  primary because CMA-ES below dominates it at this budget.
- **L-BFGS-B**: each finite-difference gradient estimate costs dim+1 = 51
  evals, so the budget allows only ~2 gradient steps total from a single
  starting point (box midpoint). Too little information to be a serious
  primary method, but cheap enough to keep in mind as a last-resort local
  polish if nothing else fits.
- **CMA-ES**: instantiated locally to check its actual population size for
  this dim/bounds: `popsize = 15`, `sigma0 = 2.5` (pycma's own standard
  box-constrained defaults: x0 = box midpoint, sigma0 = 1/4 mean box
  width). 100/15 = 6 full generations (90 evals) plus a 10-eval remainder.
  Unlike the methods above, CMA-ES's "setup" (its first generation) is
  itself a real, informative optimization step — nothing is wasted purely
  on model-building before the method starts adapting.

CMA-ES is the only candidate whose up-front cost doesn't consume the
entire (or nearly the entire) budget before producing useful adaptation,
so it is the primary method.

## Planned method sequence

**Phase 1 — CMA-ES (seed=0), full box bounds=[-5,5]^50.**
Drive it generation-by-generation: `ask(15)` / evaluate all 15 / `tell`.
Track the running best after each generation.

**Phase 2 (conditional) — CMA-ES restart on a shrunk box ("zoom"),
seed=1.** If triggered (see deviation criteria below), construct a fresh
`CMAES` instance with bounds = `[clip(x_best_i - 1.5, -5, 5), clip(x_best_i
+ 1.5, -5, 5)]` per coordinate `i`, where `x_best` is the incumbent best
point from Phase 1. This exploits the wrapper's own convention (x0 = box
midpoint, sigma0 = 1/4 box width) to automatically re-center and shrink
the step size around the current best region for local exploitation,
using whatever budget remains.

**End-of-budget handling.** If at any point fewer than 15 evaluations
remain (not enough for a full generation), stop asking full generations
and instead call `ask(n=1)` one point at a time on the currently active
CMA-ES instance for the remaining evals, so no budget is wasted on an
uncompletable generation. (Per the optimizer's docstring, the wrapper
buffers partial generations internally — these are still real, useful
evaluations even if they don't trigger a full population-level CMA
update.)

Budget split (expected, not guaranteed — actual split depends on when/if
the deviation trigger fires): roughly 45–90 evals to Phase 1, remainder
(10–55 evals) to Phase 2 if triggered, else all 100 stay in Phase 1.

## Reasoning

- 50 dimensions with 100 evals is a regime where almost every method in
  this toolbox (built with budgets of 10x–10,000x dim in mind) cannot
  even complete its documented initialization. CMA-ES is the exception
  because its initialization (generation 1) already doubles as a real,
  adaptive optimization step.
- A single CMA-ES run of ~6 generations is short — plain CMA-ES normally
  needs tens of generations to meaningfully adapt its covariance in 50D —
  so there is a real risk it stalls (covariance barely adapts, mean
  barely moves) well before the budget is used. The shrink-and-restart
  (Phase 2) is a cheap, principled hedge against that: if global-scale
  CMA-ES stops improving, re-center a smaller-step-size search on the
  best region found so far rather than continuing to spend evals on a
  stalled global covariance estimate.
- No initial Sobol/random scan is used before Phase 1: CMA-ES's own first
  generation (sigma0=2.5 around the box midpoint) already covers a
  comparable amount of the space, and a separate scan would not be able
  to warm-start the CMAES object (its public interface takes no x0/mean
  argument), so it would only cost evaluations without feeding back into
  the optimizer.

## Explicit deviation criteria

1. **Stagnation → Phase 2 trigger.** After completing generation g ≥ 3
   (i.e., ≥ 45 evals used in Phase 1), if the running best value has not
   strictly improved over the most recent 2 full generations (30 evals),
   stop Phase 1 and start Phase 2 (shrunk-box CMA-ES restart) with
   whatever budget remains.
2. **Partial-generation handling.** Whenever the evaluations remaining
   drop below 15 (one full generation) under whichever phase is active,
   switch to asking/telling one point at a time (`n=1`) instead of a full
   batch, so the exact remaining budget is used without overshooting.
3. **No second restart.** Phase 2, once triggered, runs to budget
   exhaustion; it will not itself trigger a further shrink-restart (not
   enough budget would remain for that to be informative at this scale).
   If this assumption looks wrong once Phase 2 is underway, it will be
   logged as a further deviation in changelog.md rather than silently
   acted on.
4. If an optimizer raises `OptimizerExhausted` or any unexpected error
   before budget exhaustion, fall back to spending remaining budget on
   `RandomSearch` around the incumbent best (full box) rather than
   leaving evaluations unused, and log this in changelog.md.
