# Roadmap — Task 001 (pre-registered, written before any `evaluate()` call)

## What I know going in

Only the stated problem properties: dimension 10, box bounds [-5, 5] per
coordinate, deterministic (noise-free), budget of 2000 evaluations, single
scalar objective to minimize. No data yet, no assumptions about modality,
smoothness, or separability beyond what these properties imply.

A budget of 2000 for dim=10 is ~200x dimension — large enough for a
population-based global method to do real work, and large enough that an
exact-GP method (cubic cost in number of points) is not viable across the
whole budget, but still small enough that pure random/grid coverage of a
10-D box would be hopeless. That shapes the plan below: spend a modest
slice on unbiased global coverage, spend the bulk on a global optimizer
that scales to hundreds of evaluations, and reserve a meaningful tail for
local polishing once a promising basin is found.

## Planned sequence and budget split (2000 total)

1. **Sobol quasi-random scan — 150 evaluations** (`sobol`).
   Purpose: unbiased, low-discrepancy coverage of the 10-D box before
   committing to any adaptive method, and a cheap way to read off basic
   landscape character (is the top decile of points clustered in one
   region, or scattered — i.e., how multimodal/rugged does this look)
   before spending the bulk of the budget. 150 = 15x dim, in the same
   order as a standard DOE-style initial design, rounded to keep later
   numbers tidy; not meant to find the optimum itself.

2. **CMA-ES, global phase — up to 1100 evaluations** (`cma_es`, possibly
   `cma_es_restart_2`, `cma_es_restart_3` if restarts happen — see
   deviation criteria below).
   Purpose: CMA-ES is the standard default for continuous, derivative-free,
   possibly non-separable, possibly moderately multimodal optimization in
   this budget regime (hundreds of evaluations, dim ~10); it adapts step
   size and a full covariance matrix, so it doesn't assume separability or
   a particular axis-aligned scale, which nothing in the problem statement
   rules out. It's used here as the main workhorse, not DE, because DE's
   classic recommended population (10*dim = 100) makes each generation
   expensive relative to this budget, while CMA-ES's adaptation is more
   sample-efficient per generation at this dimensionality.

3. **Local refinement cascade — remaining ~750 evaluations**
   (`bobyqa_local`, then `lbfgsb_local` and/or `nelder_mead_local` as
   needed).
   Purpose: once the global phase has located a promising basin, a
   model-based local solver converges to its optimum far more
   evaluation-efficiently than continuing a global/population method.
   BOBYQA goes first (trust-region model-based, good constrained local
   convergence, no gradient needed). Per the brief's own warning, BOBYQA
   driven one evaluation at a time through the thread wrapper gets very
   slow past ~500 evaluations, so it is capped at 300 evaluations here;
   if it hasn't converged or stalled by then, finish with L-BFGS-B
   (finite-difference gradient) or Nelder-Mead instead. All local solvers
   are warm-started at the current incumbent best by re-centering their
   bounds argument on it (narrow box clipped to [-5, 5]), since the
   wrappers always start at the midpoint of whatever bounds they're given.

150 + 1100 + 750 = 2000.

## Why not other tools, up front

- **GP-BO / TuRBO**: GP-BO's own docstring states it's intended for
  budgets of order 10x dim, not 1000+; exact GP fitting cost grows
  cubically with data, so running it across most of a 2000-evaluation
  budget is not a good use of compute or evaluations relative to CMA-ES.
  TuRBO fits its GP only on the current trust region's local data (not
  full history), so it doesn't have the same scaling problem — it's kept
  in reserve as a candidate replacement for the local-refinement cascade
  if BOBYQA/L-BFGS-B/Nelder-Mead all plateau without a satisfying
  convergence (see deviation criteria).
- **Differential Evolution**: kept in reserve as an alternate global
  method if CMA-ES's covariance adaptation gets stuck in a way that looks
  like a bad local trap (see below) — DE's mutation/crossover dynamics
  explore differently and may escape a basin CMA-ES has committed to.
- **Random search**: no adaptive benefit over Sobol for the initial scan;
  not planned for direct use.

## Explicit, quantitative deviation criteria

I will deviate from the above and log it in `changelog.md` if any of the
following trigger:

1. **Sobol phase shape check (after the 150 Sobol evals, before starting
   CMA-ES):** if the 15 best Sobol points (top 10%) are spread so that
   their pairwise distances span more than half the box diagonal (i.e.,
   they are not clustered in one region), treat the landscape as
   apparently multimodal and plan for at least one CMA-ES restart from a
   distinct region rather than trusting a single run; if instead they
   cluster tightly, a single CMA-ES run is expected to suffice and the
   restart budget may instead be redirected to local refinement.
2. **CMA-ES stagnation:** if the running best has not improved by more
   than 1e-6 over the last 200 evaluations within a CMA-ES run, stop that
   run and either (a) restart a fresh CMA-ES instance with a new random
   seed and bounds re-centered away from the exhausted region (if the
   global-phase budget remains and criterion 1 suggested multimodality),
   or (b) move directly into the local-refinement cascade early (if the
   remaining global-phase budget is small or criterion 1 suggested
   unimodality).
3. **CMA-ES early convergence:** if CMA-ES's internal step size collapses
   (population effectively stops moving) before the 1100-evaluation
   global-phase budget is used, stop early and carry the unused
   evaluations into the local-refinement phase rather than continuing to
   ask a converged CMA-ES for points.
4. **BOBYQA cap/stall:** stop BOBYQA at 300 evaluations regardless of
   convergence status (per the brief's documented slow-down past ~500),
   or earlier if it raises `OptimizerExhausted`, or earlier still if its
   best value hasn't improved by more than 1e-8 over its last 100
   evaluations; in any of these cases, move to L-BFGS-B next.
5. **L-BFGS-B / Nelder-Mead stall or exhaustion:** if the active local
   solver raises `OptimizerExhausted` or stalls (best value improvement
   under 1e-8 over its last 100 evaluations) with more than 100
   evaluations still remaining in the local-refinement budget, switch to
   the next local method in the cascade (L-BFGS-B -> Nelder-Mead, or
   Nelder-Mead -> TuRBO as a sample-efficient trust-region alternative if
   both classical local solvers have stalled).
6. **Global regression check:** if any local solver's own converged value
   is worse than the best value already found during the Sobol/CMA-ES
   global phase, restart that local solver warm-started at the
   global-phase incumbent instead of its own converged point, rather than
   accepting the worse local result.
7. **DE fallback:** if, after one CMA-ES restart (criterion 2a), the
   second run also stagnates per criterion 2 without improving on the
   first run's best by more than 1% (relative), switch the remaining
   global-phase budget to Differential Evolution instead of a third
   CMA-ES restart, on the reasoning that CMA-ES's covariance adaptation
   may be poorly matched to this landscape's structure.

Every one of these, if triggered, gets one dated entry in
`changelog.md` citing the specific numbers observed (evaluation count,
best value, stall window) before the change is made.
