# Roadmap — Task 007 (dim=10, bounds=[-5,5]^10, minimize, deterministic, budget=100)

## Problem reading

- Budget is only 10x dimension (100 evals, dim 10). This is a small-budget,
  "expensive black-box" regime, not the 2000-eval convention used elsewhere
  in this project. Methods that need hundreds/thousands of evaluations to
  pay off (e.g. CMA-ES, pure random search, large-population DE) are a poor
  fit here: they would still be in an exploratory phase when the budget ran
  out.
- Function is deterministic (no observation noise), so there is no need to
  replicate points or use noise-robust acquisition — every evaluation is
  fully informative and worth modeling exactly.
- This project's own `gp_bo.py` docstring explicitly documents GP-based BO
  with the analytic LogEI acquisition as designed for exactly this regime
  ("expensive, continuous problems, budgets of order 10 x dim"), using a
  Sobol initial design of `2*(dim+1) = 22` points followed by sequential
  single-point acquisition-driven proposals. That matches this task almost
  exactly (budget 100 ≈ 10x dim=10), so it is the natural primary method.
- A pure global BO run over the whole budget risks spending its last
  evaluations still exploring moderately-uncertain regions rather than
  tightly polishing the best basin found. Since the function is
  deterministic, a cheap, derivative-free local refinement
  (BOBYQA, trust-region model-based) around the best point GP-BO finds can
  extract extra precision from the last fraction of the budget at very low
  marginal cost per evaluation and with a complementary search behavior
  (exploitative local polish vs. GP-BO's global modeling).

## Planned method sequence

1. **Phase 1 — GP-BO** (primary, global phase): `GPBO(bounds=harness.bounds, seed=0)`
   from `src/optimizers/gp_bo.py`. Budget allocation: up to 80 evaluations
   total (22 Sobol initial-design evaluations + up to 58 sequential
   LogEI-guided evaluations), ask/tell one point at a time.
2. **Phase 2 — BOBYQA local refinement** (exploitation phase): `BOBYQA` from
   `src/optimizers/bobyqa.py`, instantiated with a shrunk box of half-width
   1.0 per coordinate (clipped to the original [-5,5]^10 bounds) centered on
   the best point found in Phase 1. Because `BOBYQA`'s wrapper always starts
   from the midpoint of whatever `bounds` it is given, centering a tighter
   box on the Phase-1 incumbent effectively "warm-starts" BOBYQA's
   trust-region search from that incumbent. Remaining budget (nominally 20
   evaluations, more if Phase 1 ends early — see deviation criteria) is
   spent here.

Both phases log under `set_active_optimizer("gp_bo")` and
`set_active_optimizer("bobyqa_refine")` respectively.

## Explicit deviation criteria

- **Early switch to Phase 2:** if, after the 22-point initial design, the
  running best value during Phase 1 has not improved for 15 consecutive
  GP-BO-proposed evaluations, stop Phase 1 early and move into Phase 2 with
  whatever budget remains (giving the local refiner more evaluations rather
  than continuing an apparently-stalled global search).
- **GP-BO failure fallback:** if `GPBO.ask`/`tell` raises an exception
  (e.g. a GP-fitting numerical failure) before Phase 1's budget is used,
  stop Phase 1 immediately at that point and move to Phase 2 with the
  remaining budget, using whatever incumbent has been found so far.
- **BOBYQA exhaustion fallback:** if `BOBYQA.ask` raises
  `OptimizerExhausted` (its internal solver thread has converged) before
  Phase 2's allotted evaluations are used, spend any remaining budget on
  small random perturbations (uniform samples inside the same shrunk local
  box used for Phase 2) around the current global incumbent, logged under
  `set_active_optimizer("local_random_polish")`, rather than leaving budget
  unused.
- **Hard stop:** regardless of the above, never exceed the 100-evaluation
  budget; the harness enforces this directly.

## Budget summary (nominal, subject to the deviation rules above)

- Phase 1 (GP-BO): ~80 evaluations (22 init + up to 58 adaptive).
- Phase 2 (BOBYQA local refinement): ~20 evaluations.
