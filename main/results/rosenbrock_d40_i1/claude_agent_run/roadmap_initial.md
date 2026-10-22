# Roadmap — Task 006 (dim=40, budget=100, minimize, deterministic)

## Problem read

- dim = 40, bounds = [-5, 5]^40, direction = minimize, noise = none (deterministic).
- Budget = 100 evaluations total — only 2.5x dim. This is far below the
  regime any of the provided optimizers document as their comfortable
  operating range (CMA-ES/DE typically want O(10^2)-O(10^3) x dim; the
  project's own GP-BO/TuRBO wrappers are documented for "budgets of order
  10 x dim"; here we have 2.5 x dim). No optimizer here will reach real
  convergence. The goal is to make the best use of very few samples, not
  to fully solve the problem.
- Nothing is known about modality, separability, or conditioning beyond
  "sealed black box, deterministic, box-constrained, 40-d real". I will
  not assume unimodality or separability going in.

## Method sequence and budget split

Three phases, 100 evals total, each phase handing its best-known region to
the next (since none of the provided ask/tell wrappers expose a custom
starting point — CMA-ES/BOBYQA/Nelder-Mead/L-BFGS-B all start at the
midpoint of whatever `bounds` they're constructed with) — handoff between
phases is therefore done by constructing the next phase's optimizer over a
**shrunk sub-box re-centered on the current best point**, clipped to stay
inside the original [-5,5]^40 box. This is a standard trust-region "zoom"
trick, not a change to any optimizer's internal hyperparameters.

1. **Sobol' (25 evals)** — quasi-random space-filling design over the full
   box. With only 100 evals in 40 dims we cannot afford a large initial
   design, but some unbiased global read is needed before committing to
   any local region, since nothing is known about where the optimum lies.
   25 points is a compromise: enough to get a low-discrepancy sample of
   the space without spending the majority of the budget on pure
   exploration.

2. **CMA-ES (55 evals)**, constructed over a sub-box of half-width 40% of
   the full range (i.e. +-2.0 per coordinate, clipped to [-5,5]) centered
   on the best point found in phase 1. CMA-ES's default population size
   for dim=40 is popsize = 4 + floor(3*ln(40)) = 15, so 55 evals give
   ~3-4 generations — not enough for full covariance adaptation, but CMA-ES
   is the best-suited provided method for exploiting a promising region
   under a tight, still population-sized budget (more sample-efficient
   per-generation than DE, which needs pop_size=10*dim=400 and would never
   complete one generation here).

3. **BOBYQA (20 evals)**, constructed over a further-shrunk sub-box of
   half-width 10% of the full range (+-0.5, clipped) centered on the best
   point found in phase 2. A model-based trust-region local solver is the
   right tool to squeeze final precision out of the last ~20 evals once a
   promising basin has been identified; BOBYQA's default interpolation set
   size (2*dim+1=81) will not be fully built in only 20 evals, but the
   trust-region steps it takes while building that set still move
   monotonically in an improving direction, so partial use is still
   useful, not wasted.

Total: 25 + 55 + 20 = 100.

## Reasoning summary

- No noise -> no need to replicate points or use noise-robust acquisition.
- Budget is too small for a GP surrogate to pay for itself: the project's
  own GP-BO wrapper hardcodes a 2*(dim+1)=82-point Sobol initial design
  before it ever proposes an acquisition-driven point, which would leave
  only 18 evals of actual Bayesian optimization out of our 100 — worse
  than spending those 82 points more cheaply (Sobol here) and reserving
  more of the budget for an adaptive/local method afterward. Same
  argument rules out TuRBO (2*dim=80-point init design).
- DE is explicitly documented (in this project's own optimizer notes) as
  needing 10^3-10^4 x dim evals; at 100 evals total it would not complete
  a single generation (pop_size=10*dim=400). Ruled out.
- L-BFGS-B's finite-difference gradient costs ~dim+1=41 evals per internal
  step; at most 2 steps fit in 100 evals. Too coarse to be useful here;
  ruled out in favor of BOBYQA, which makes direct use of the same budget
  via trust-region model updates instead of one-shot gradient steps.
- Nelder-Mead's initial simplex alone needs dim+1=41 points in 40-d,
  leaving very little for the method's own iteration; BOBYQA's partial
  interpolation set is expected to make comparably better use of a small
  budget as a local refiner, so it is preferred.
- Explore-then-exploit (coarse global -> regional -> local) is a standard,
  defensible strategy when the budget is too small for any single method
  to run in its own comfortable regime.

## Explicit deviation criteria

I will deviate from the plan above, and log it in changelog.md, if any of
the following are observed:

1. **Early CMA-ES stagnation**: if CMA-ES's running best (`y_best_so_far`)
   has not improved at all over 2 full consecutive generations (30 evals)
   within phase 2, I will stop phase 2 early and move the remaining
   phase-2 budget into phase 3 (more local-polish evals) instead of
   letting CMA-ES keep sampling a region it has stopped improving in.
2. **Optimizer self-reports exhaustion** (`OptimizerExhausted`, e.g. if
   BOBYQA's internal solver converges/stops before its allotted 20 evals
   are used): any unused evals from that phase will be redirected to a
   fresh CMA-ES restart over a sub-box re-centered on the then-current
   global best (shrunk box, half-width 20% of full range), rather than
   wasting the remaining budget.
3. **Degenerate shrunk box**: if re-centering a sub-box on the current
   best point and clipping to [-5,5]^40 would leave a half-width below
   0.05 in any coordinate (best point essentially on the global boundary),
   I will floor that coordinate's half-width at 0.05 so the next phase's
   optimizer still has a usable (non-degenerate) box.
4. **Numerical failure**: if any `evaluate()` call returns NaN/Inf, or an
   optimizer's internal fit fails (e.g. GP fitting divergence — not
   expected here since no GP phase is planned, but noting for
   completeness), I will fall back to Sobol sampling over the current
   phase's box for the remainder of that phase's budget rather than
   trusting a corrupted model.

No other changes to method choice, budget split, or hyperparameters are
planned. Any such change made anyway will be logged in changelog.md with
the concrete evidence that motivated it.
