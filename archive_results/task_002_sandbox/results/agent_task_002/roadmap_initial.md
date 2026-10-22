# Roadmap — Task 002 (pre-registered, before any real evaluation)

## Problem as given

- Minimize a deterministic, black-box `f: [-5, 5]^10 -> R`.
- Budget: 2000 evaluations, no observation noise, no gradient access.
- Nothing is known about multimodality, separability, conditioning, or
  smoothness, so the plan below is chosen for robustness to any of those,
  not tuned to a specific guess about the function's identity.

## Plan and budget split

Three phases, run in sequence, total budget 2000:

1. **Sobol' quasi-random design — 100 evaluations (`sobol_init`).**
   A low-discrepancy space-filling design, not plain random search,
   because in 10 dimensions a uniform random sample has much higher
   gap/clump variance than a Sobol' sequence for the same sample size.
   Purpose: get a first, unbiased read on the landscape (value range,
   crude roughness) and bank 100 real, well-spread (x, y) pairs. 100 =
   10 * dim, a standard rule-of-thumb initial-design size that is big
   enough to be informative but small (5% of budget) relative to the
   2000-evaluation budget.

2. **CMA-ES as the primary global search — target ~1400 evaluations
   (`cma_es`, possibly `cma_es_restart_k` if restarted).**
   CMA-ES is chosen as the main workhorse because, for a continuous,
   unconstrained-in-structure, 10-dimensional problem with a budget of
   this size (hundreds to low thousands of evaluations), it is the
   default strong choice in the black-box optimization literature: it
   adapts a full covariance matrix (so it is not hurt by an unknown
   non-separable or ill-conditioned landscape, unlike coordinate-wise
   methods), it does not require gradients, and it tends to outperform
   Bayesian optimization once the budget is large enough that GP
   surrogate-fitting cost/scaling stops paying for itself (BO's main
   advantage is sample efficiency when evaluations are very expensive
   and budgets are in the tens-to-low-hundreds; here we have 2000).
   Each CMA-ES run starts from the box midpoint with sigma0 = a quarter
   of the box width (the wrapper's standard, untuned convention) — a
   reasonable default when nothing is known about where the optimum
   lies.

3. **Local refinement/polish — remaining evaluations (~500), method
   `local_polish` (BOBYQA first, L-BFGS-B as fallback/second pass).**
   Because the function is deterministic (no noise), a derivative-free
   local solver can converge to near-exact local-optimum precision from
   CMA-ES's final best point, squeezing out accuracy that a
   population-based method's stochastic sampling leaves on the table.
   BOBYQA is tried first (model-based trust region, typically sample-
   efficient for local polishing); L-BFGS-B (finite-difference
   gradient) is used as a fallback/cross-check, or if BOBYQA's known
   performance issue past ~500 evaluations (see brief, Section 3)
   becomes a concern. Starting point: CMA-ES's best `x` found so far,
   with a tight local box clipped to the global bounds (so the local
   solver's own "midpoint" convention lands exactly on that point).

## Why this order (Sobol -> CMA-ES -> local polish), not something else

- Starting with a cheap space-filling design before committing to any
  directed method is a standard, low-risk way to avoid anchoring a
  global optimizer on one arbitrary starting assumption about the
  landscape, for only 5% of the budget.
- CMA-ES before a local solver, not the reverse: a purely local method
  started from the box midpoint has no mechanism to escape a bad local
  basin if the landscape is multimodal, which is unknown here. CMA-ES's
  population/covariance adaptation gives it some global reach while
  still converging to a precise optimum given enough evaluations; it is
  spent before the final local-polish phase, when returns on broad
  search are expected to be lowest.
- Local polish last, not skipped: with 2000 evaluations and a noise-free
  function, the marginal evaluations are cheap enough to afford a
  dedicated exact-convergence phase after the global search has found a
  good basin, and population methods like CMA-ES are not designed to
  drive the last few digits of precision as efficiently as a local
  trust-region/quasi-Newton method can.
- TuRBO/GP-BO (Bayesian optimization) are available but not chosen as
  the primary method: their main strength is sample efficiency at very
  small budgets (tens to a couple hundred evaluations) where every
  evaluation is precious; with 2000 evaluations in only 10 dimensions,
  CMA-ES is expected to make better use of the larger budget and scales
  better per-evaluation (no GP refit cost growing with history size).
  TuRBO is kept as the designated fallback (see deviation criteria)
  specifically because it combines a trust-region idea with a GP
  surrogate — useful if CMA-ES's evidence suggests a narrow, smooth
  basin where a surrogate model would help more than further undirected
  population search.

## Explicit, quantitative deviation criteria

These are checked while running; any time one fires, the change and the
observed evidence that triggered it are logged in `changelog.md`.

1. **CMA-ES stagnation -> restart or switch.** If the running best value
   has not improved by more than `1e-6 * |best|` (or `1e-6` in absolute
   terms if `best` is near 0) over any window of 200 consecutive
   `cma_es` evaluations, AND pycma's own internal state has not already
   signalled convergence (tiny sigma / flat fitness), restart a new
   CMA-ES instance with a different seed, consuming at most 2 restarts
   total before abandoning CMA-ES in favor of TuRBO for the rest of the
   global-search budget.
2. **CMA-ES early convergence -> reallocate to local polish early.** If
   pycma's internal stopping condition triggers (e.g. `tolx`/`tolfun`
   style convergence — sigma collapses well below 1e-6 relative to the
   initial box width) before the ~1400-evaluation budget for this phase
   is used, stop CMA-ES immediately and move the unused evaluations into
   the local-polish phase rather than restarting pointlessly.
3. **BOBYQA reliability ceiling.** Per the brief's documented limitation,
   BOBYQA driven through the thread+queue wrapper becomes impractically
   slow past roughly 500 evaluations. Local polish therefore caps any
   single BOBYQA run at 400 evaluations as a hard, pre-committed limit
   (not something decided ad hoc while it's running); if BOBYQA instead
   raises `OptimizerExhausted` (converged) before that cap, switch
   immediately to L-BFGS-B from the same best point, using the leftover
   local-polish budget.
4. **Local polish plateau -> stop early and bank budget.** If local
   polish improves the incoming best value by less than `1e-8` in
   absolute terms, consider it converged and stop drawing further
   evaluations from that phase, even if evaluations remain in its
   sub-budget; unused evaluations are *not* reallocated elsewhere, this
   simply ends the campaign early via `harness.save()`. (There is no
   benefit to spending evaluations once a deterministic, noise-free
   local solver has converged to a given point's local optimum.)
5. **Hard global ceiling.** Never exceed 2000 total evaluations across
   all phases, under any of the above; the plan reserves the final
   phase's budget deliberately smaller than what's left after phases 1-2
   finish on schedule, to absorb any overrun from restarts under
   criterion 1.
