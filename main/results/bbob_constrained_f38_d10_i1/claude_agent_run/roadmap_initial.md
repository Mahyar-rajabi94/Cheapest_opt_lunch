# Roadmap (pre-registered, written before any evaluate() call)

## Problem as given

- dim = 10, bounds = [-5, 5]^10, direction = minimize, deterministic (no
  observation noise), budget = 100 evaluate() calls total.
- The objective is a sealed black box with hidden inequality constraints.
  `evaluate(x)` returns a single penalized merit value
  `true_objective + penalty_lambda * violation`, violation = 0 at the
  (feasible) optimum. I am told to optimize this merit value directly as
  my scalar objective, not to try to separate objective from penalty, and
  to treat sudden large jumps in merit as a signal of heavy infeasibility
  (move away), not as real objective structure to chase.
- Budget (100) is only ~10x dim, which is a small/expensive-style budget
  for a 10-D problem.

## Method sequence and budget split

1. **Sobol quasi-random search** (`SobolSearch`), 20 evaluations.
   Purpose: cheap, space-filling initial exploration of the whole box.
   With hidden constraints and penalty jumps, I have no prior idea where
   the feasible region sits, so a low-discrepancy global scan is safer
   than starting a local solver blind at the box midpoint — it gives a
   rough picture of where merit is low/moderate vs. where it spikes
   (likely-infeasible) before committing to a local/model-based method.

2. **TuRBO-1** (`TuRBO`, BoTorch trust-region BO), up to 60 evaluations
   (batch size 4, ~15 batches, with its own 2*dim=20-point init design
   counted inside this budget). Purpose: a budget-appropriate
   (BoTorch's own documented regime is ~10x dim) Bayesian-optimization
   method whose trust region naturally contracts away from regions that
   produce large penalized merit values, which is exactly the behavior
   wanted here — it should implicitly learn to avoid the infeasible side
   of the hidden constraints without me having to model the penalty
   explicitly. This is the main budget phase.

3. **BOBYQA** local polish (`BOBYQA`), remaining evaluations (~20),
   restarted from the best point found by phases 1-2 rather than the box
   midpoint (its default). Purpose: a trust-region model-based local
   solver to squeeze out final precision near the best basin found,
   since by this point the search should already be in/near the feasible
   optimum's neighborhood and a classical local method converges fast
   there. Py-BOBYQA's closed-loop solve starts from a fixed x0 argument
   at construction time, so I will not literally reuse the packaged
   `BOBYQA` wrapper's default (box midpoint) — see deviation note below
   if this requires a small code-level adjustment.

Total: 20 + 60 + 20 = 100, matching the budget exactly. All phase
boundaries are soft (see deviation criteria) — evaluations can shift
between phases 2 and 3 depending on observed behavior.

## Reasoning (from problem properties alone, no data yet)

- Dim=10 with budget=100 (~10x dim) rules out population-based global
  methods that expect hundreds-to-thousands of evaluations (plain CMA-ES,
  DE) as the primary driver — they would barely complete a few
  generations. TuRBO and GP-BO are explicitly documented (in this
  project's own optimizer docstrings) as suited to budgets of this
  order; TuRBO is chosen over plain GP-BO because its trust-region
  mechanism should handle the deterministic-but-spiky penalized-merit
  surface more gracefully (it shrinks away from bad regions rather than
  letting a single global GP get distorted by one very large penalty
  outlier).
- A short Sobol warm-up is cheap insurance against the hidden feasible
  region being a small subset of the box; it costs only 20 evaluations
  but gives both TuRBO-independent context and (together with TuRBO's
  own ask/tell interface) a sense of the merit landscape's range before
  committing to model-based search.
- A short closed-loop local solver (BOBYQA) at the end is standard
  practice to convert a good-but-imprecise global optimum into a precise
  one, since local trust-region solvers converge much faster per-eval
  than BO once near the optimum's basin.
- Gradient-based L-BFGS-B is not used as a primary method: with 10
  dimensions it costs ~11 evaluations per finite-difference gradient
  step, which is too expensive relative to a 100-evaluation budget, and
  finite-difference gradients are fragile right at a constraint boundary
  where the merit can jump sharply.

## Deviation criteria (explicit, quantitative)

- If, during Sobol exploration, more than 50% of the 20 points show
  wildly divergent merit values (e.g. the top few points are >10x the
  interquartile range above the median), treat this as evidence of a
  small feasible region and bias TuRBO's initial design toward the
  best Sobol points rather than a fresh random design, if the wrapper
  allows seeding via its own init — otherwise note this and proceed, since
  the TuRBO docstring doesn't expose starting from specific points
  without a code edit, so this may remain a note not an action.
- If TuRBO's trust region restarts (collapses below length_min) more
  than twice within its allotted budget without the global best
  improving by more than ~1% in between restarts, cut phase 2 short and
  move the remaining budget to phase 3 (BOBYQA polish) early.
- If BOBYQA (phase 3) raises `OptimizerExhausted` (converges) before
  using its full allotted budget, stop the campaign at that point and
  call `harness.save()` rather than forcing further evaluations — this
  is normal behavior for a converged local solver, not an error to
  route around with restarts, as the project's own optimizer docstrings
  state.
- If BOBYQA's closed-loop API does not actually expose a way to set
  a non-default starting point x0 without editing `bobyqa.py` (which I
  will not do, to avoid hand-modifying shared project code for a one-off
  run), fall back to running a small local refiner I drive myself
  (e.g. a tiny manual trust-region/pattern search around the best point,
  or `NelderMead` initialized implicitly at its own default) — any such
  substitution will be logged in changelog.md with the reasoning.
- Any merit value that is more than ~50x the typical (non-spike) range
  seen so far will be treated purely as a "move away from here" signal,
  not investigated further or treated as meaningful objective structure.

## Output

`harness.set_active_optimizer(...)` will be called before each phase (and
again on any mid-phase deviation) so `trace.csv` correctly attributes
every evaluation to the method that produced it.
