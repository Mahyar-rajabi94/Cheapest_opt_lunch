# Roadmap — task013 (dim=30, bounds=[-100,100]^30, minimize, deterministic, budget=100)

## Problem framing

Budget (100) is only ~3.3x the dimension (30). This is far below the
"budget ~ 10x dim" regime that most black-box optimizers assume, and much
smaller than typical convention elsewhere in this project (2000-eval
runs). This single fact dominates the method choice:

- Population-based global methods (CMA-ES, classic DE) need at least one
  population's worth of evaluations per generation (CMA-ES default
  popsize for dim=30 is ~4+3*ln(30) ≈ 14; DE's NP=10*dim=300 alone
  exceeds the entire budget). With budget=100 these methods would get at
  most a handful of generations — not enough for covariance adaptation or
  differential mutation to meaningfully outperform quasi-random sampling.
- TuRBO's standard initial design alone is 2*dim = 60 points, plus it
  needs enough data per trust region to fit a useful local GP; with
  batch size 4 that leaves only ~10 batches, little room for the
  trust-region adaptation logic to do anything beyond its first region.
- GP-BO (SingleTaskGP + analytic LogEI) is explicitly documented in this
  project as suited to budgets "of order 10 x dim" (still larger than
  100 here, but closer than any alternative) and is the most
  sample-efficient method available for a small, deterministic,
  continuous, box-constrained problem: it extracts information from
  *every* observation via a single surrogate fit refit over the full
  history, rather than relying on population diversity or trust-region
  heuristics that need many evaluations to pay off.
- Local closed-loop refiners (BOBYQA, L-BFGS-B, Nelder-Mead) are strong
  once near a good basin, but blind in 30-D from a single starting point
  (e.g. the box midpoint) with no global search first; used alone they
  risk converging to a poor local optimum with no way to know from one
  run whether a better basin exists elsewhere.

## Planned method sequence

Single method for the full budget: **GP-BO (LogEI)**, using its own
standard, untuned configuration (`src/optimizers/gp_bo.py`):

1. Initial design: the optimizer's own built-in Sobol' design,
   n_init = 2*(dim+1) = 62 points (evals 1-62). Tagged
   `set_active_optimizer("gp_bo_sobol_init")`.
2. Acquisition-driven phase: for the remaining 38 evaluations (63-100),
   each `ask()` fits a fresh SingleTaskGP on all data collected so far
   and proposes the point maximizing analytic LogEI, found via
   multi-start L-BFGS (`optimize_acqf`, num_restarts=10,
   raw_samples=512 — BoTorch tutorial defaults). Tagged
   `set_active_optimizer("gp_bo_acquisition")`.

No hand-tuning of GP-BO's internals (kernel, priors, init size,
acquisition-optimizer restarts) — all left at the library defaults
documented in the module's own docstring.

This is deliberately a single, non-hybrid plan: the budget is too small
to split further (e.g. reserving evaluations for a second method) without
undermining both the initial design's coverage and the number of
GP-informed exploitation steps. If execution reveals a reason to
hybridize (see deviation criteria below), the changelog will record it.

## Deviation criteria (quantitative, pre-registered)

I will deviate from the plan above, and log it in `changelog.md`, if and
only if one of the following is observed:

1. **GP-BO internals fail.** If `ask()` or `tell()` on the `GPBO` object
   raises an exception (e.g. numerical failure fitting `SingleTaskGP`, or
   `optimize_acqf` failing to return a candidate) at any point after the
   Sobol init phase, I will fall back to Sobol' quasi-random sampling
   (`SobolSearch`) for all remaining evaluations rather than retry
   indefinitely or hand-fix GP internals.
2. **No improvement during the BO phase.** If, after at least 15
   acquisition-driven evaluations (i.e. by eval 77), `y_best_so_far` has
   not improved at all relative to its value at the end of the Sobol
   init phase (eval 62), I will treat this as evidence the GP surrogate
   is not finding productive search directions in 30-D with this little
   data, and will switch the remaining budget to a local refinement
   method (BOBYQA, started from the current best point found so far)
   to at least exploit the best basin found, rather than continue an
   acquisition loop that is not working.
3. **Budget bookkeeping.** The budget of 100 is a hard ceiling; I will
   never call `evaluate()` beyond it and will never construct a second
   harness instance to bypass the counter.

Absent any of the above, the plan is to run GP-BO's init + acquisition
phases end-to-end for the full 100-evaluation budget as described.
