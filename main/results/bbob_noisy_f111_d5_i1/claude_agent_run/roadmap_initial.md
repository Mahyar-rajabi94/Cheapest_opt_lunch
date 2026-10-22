# Roadmap — sandbox_task015

## Problem summary
- Dimension: 5, bounds [-5, 5]^5, direction: minimize.
- Noisy: evaluate(x) called twice at the same x can return different values
  (real observation noise).
- Budget: 100 evaluate() calls (small — roughly 20x dim, not the 2000-eval
  convention used elsewhere in this project).

## Planned method sequence

**Phase 1 — GP-BO (project's `src/optimizers/gp_bo.py`), ~80 evaluations.**
- Rationale: GP-BO's own docstring states it targets exactly this regime —
  "expensive, continuous problems, budgets of order 10 x dim, not 1000+ x
  dim" — which matches 100 evals at dim=5 (20x dim) far better than
  population-based methods (CMA-ES, differential evolution) that typically
  need hundreds-to-thousands of evaluations to converge, or finite-difference
  local methods (L-BFGS-B) that are fragile under observation noise because
  a finite-difference gradient estimate is itself a noisy quantity built
  from differences of noisy values — small denominators amplify noise into
  large gradient errors.
- GP-BO's SingleTaskGP fits an inferred (homoskedastic) observation-noise
  term rather than interpolating exactly through every point, so it treats
  each observed y as a noisy estimate of the true surface rather than
  ground truth — a meaningfully better fit to this task's noise model than
  trust-region/pattern-search methods (BOBYQA, Nelder-Mead) that are built
  around the assumption of (near-)exact function values and can be misled
  or destabilized by a single noisy sample.
- Uses its built-in initial design: 2*(dim+1) = 12 Sobol points, followed by
  ~68 single-point LogEI-acquisition steps (68 = 80 - 12).

**Phase 2 — Noise-averaging exploitation, ~20 evaluations.**
- Rationale: because the objective is noisy, the single best observed y in
  Phase 1's trace could be a lucky noisy draw rather than the location of
  the true best mean value. Before finishing, re-sample the most promising
  candidates to average out noise and get a more trustworthy estimate of
  which point is genuinely best.
- Plan: take the top 4 distinct-x points by observed y from Phase 1's full
  history (GP-BO's internal `X_`/`y_`), and evaluate each one 5 more times
  (4 x 5 = 20 evaluations), under optimizer tag `noise_averaging`. Compute
  the mean y for each of the 4 candidates (combining the new repeats with
  any original single observation) and report/finish on whichever has the
  lowest mean. This uses only real evaluate() calls; no values are
  estimated or fabricated.

Total planned: 12 (Sobol init) + 68 (GP-BO acquisition) + 20 (noise
averaging) = 100 evaluations, matching the budget exactly.

## Explicit deviation criteria (pre-registered)

1. **GP fitting instability:** if GP-BO's `ask()` raises an exception (e.g.
   numerical failure in `fit_gpytorch_mll`) on two consecutive calls, stop
   using GP-BO for the remainder of Phase 1 and switch to TuRBO
   (`src/optimizers/turbo.py`, also BoTorch/GP-based and noise-tolerant)
   for the rest of the Phase-1 budget.
2. **Stagnation:** if `y_best_so_far` has not improved at all over the most
   recent 30 consecutive evaluations within Phase 1, end Phase 1 early
   (even if its 80-evaluation allowance is not exhausted) and move directly
   to Phase 2 with whatever budget remains, since further acquisition
   steps are unlikely to be worth their cost relative to spending that
   budget confirming the true best point under noise.
3. **Noise magnitude larger than expected:** if, during Phase 2, the
   standard deviation of the 5 repeats for the current-best candidate is
   larger than the gap between the top two candidates' Phase-1 single
   observations, increase replication depth for the remaining candidates
   (i.e. spend more of the remaining Phase-2 budget on fewer candidates,
   evaluating them more times each) rather than spreading the same budget
   thinly across 4 candidates.
4. Any such change will be logged in `changelog.md` with the evaluation
   index at which it occurred and the evidence that triggered it.

## What will NOT change
- The overall two-phase structure (explore-and-model, then
  confirm-under-noise) is fixed in advance.
- No evaluate() result will ever be fabricated, estimated, or hand-edited;
  every reported value comes from a real harness.evaluate() call.
