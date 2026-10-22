# Roadmap — task023 (dim=10, bounds=[-5,5]^10, minimize, deterministic,
budget=100, hidden-constraint penalized merit)

## Problem read

- 10-dimensional, box-bounded, deterministic (no observation noise),
  minimize. Budget of 100 `evaluate()` calls total.
- The scalar returned by `evaluate(x)` is `true_objective + penalty_lambda *
  violation`, with violation = 0 at feasible points (including the true
  optimum). I have no separate feasibility signal, no constraint gradients,
  and am told not to try to reverse-engineer the penalty structure. I will
  therefore treat the returned value as a single, possibly non-smooth /
  occasionally spiky scalar merit surface and optimize it directly, the same
  way I would optimize any other scalar black box — large upward jumps are
  just data (a bad region), not something to special-case with custom logic.
- Budget (100) is almost exactly 10x the dimension (10), which several of
  the available optimizers' own docstrings (gp_bo.py, turbo.py) flag as
  precisely the regime GP-based Bayesian optimization is designed for, and
  where population-based methods (CMA-ES, DE) with their usual
  recommended population sizes (10*dim = 100 for DE alone) would not even
  complete one generation.

## Planned method sequence

**Phase 1 — GP-BO with LogEI (`gp_bo`), up to ~85 evaluations.**
- Uses the project's `GPBO` class: an internal 2*(dim+1) = 22-point Sobol
  initial design (for GP data, not a separate phase I drive myself), then
  sequential LogEI-acquisition points, one at a time, refit each step.
- Rationale: dim=10 with budget=100 is squarely BoTorch's own documented
  "small budget, expensive/continuous" regime for this acquisition
  function. A surrogate model that pools information across all points is
  far more sample-efficient here than a population-based evolutionary
  method, which would barely get through a handful of generations at this
  budget. Standard GP output standardization also gives it some natural
  robustness to the occasional large penalized value without any
  hand-tuning on my part.
- No hyperparameter tuning: library defaults throughout (per the module's
  own docstring), consistent with the generic black-box framing of this
  task.

**Phase 2 — Local refinement (`bobyqa_local`), remaining budget (~15 evals,
or whatever is left).**
- BOBYQA always starts from the midpoint of whatever bounds it is given.
  To use it as a *local* refiner around the Phase-1 incumbent (rather than
  re-running a global search from the box center), I will construct it
  with bounds tightened to a box of half-width ~10% of the full range
  (i.e. 1.0 per coordinate, clipped to the original [-5,5]) centered on
  the best point found in Phase 1. This keeps it a legitimate trust-region
  polish step, not a parameter change to the optimizer's own internals.
- Rationale: BOBYQA is a model-based trust-region method well suited to
  squeezing out the last bit of local precision once global search has
  located the right basin, and it is cheap (one evaluation per step, no
  finite-difference gradient overhead like L-BFGS-B).

## Budget split (nominal, see deviation criteria for exact trigger)

- Phase 1 (GP-BO): ~85 evaluations.
- Phase 2 (BOBYQA local refine): ~15 evaluations (whatever remains).

## Explicit deviation criteria

1. **Early handoff to Phase 2:** if the running best merit has not
   improved over the most recent 20 consecutive Phase-1 evaluations, stop
   Phase 1 early and start Phase 2 immediately with whatever budget
   remains, even if fewer than 85 evaluations have been used.
2. **BOBYQA exhaustion:** if BOBYQA's internal solver converges
   (`OptimizerExhausted`) before the remaining budget is used, and the
   refined point improved on the Phase-1 incumbent, restart BOBYQA with a
   fresh, re-centered trust region (same half-width rule) around the new
   best point, for the remaining budget. If it did not improve at all, or
   if a restart also exhausts immediately, switch the remaining budget to
   Nelder-Mead (`nelder_mead`) local refinement from the same tightened
   region (implemented the same way, via shrunk bounds), since it is a
   different local-search geometry that is less sensitive to an
   unhelpful local quadratic model.
3. **Any leftover evaluations** after both local refiners are exhausted
   (converged) will be spent on small random perturbations around the
   final incumbent (tagged `local_perturbation`), evaluated directly via
   `harness.evaluate`, rather than left unused.
4. **Hidden-constraint handling:** no explicit constraint-aware machinery
   is used — per the task framing, the merit is optimized directly. If an
   evaluation returns a merit far outside the range seen so far from
   nearby points (a likely heavily-penalized point), I treat it as
   ordinary high-merit data and let the optimizer's own selection pressure
   (GP posterior / BOBYQA's local model) steer away from it; I will not
   hand-filter, clip, or exclude such points from the optimizers' `tell`
   calls, since doing so would amount to inventing feasibility information
   I was not given.

Any time one of these triggers actually fires, or any other change is made
on the fly, an entry will be appended to `changelog.md`.
