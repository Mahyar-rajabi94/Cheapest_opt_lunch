# Roadmap — agent_run (dim=10, bounds=[-5,5]^10, minimize, deterministic, budget=100)

## Problem read

- 10-dimensional, box-bounded, deterministic (noise-free), minimize, budget
  of only 100 `evaluate()` calls total — i.e. 10x the dimension. This is a
  tight budget: not enough for a population method (CMA-ES) to run many
  generations (default popsize ~10-11 in 10D gives only ~9 generations),
  and not enough for a pure random/Sobol sweep to characterize the
  landscape on its own. This regime (roughly 5-20x dim, noise-free,
  moderate dimension) is exactly where trust-region Bayesian optimization
  (TuRBO) and model-based derivative-free trust-region solvers (BOBYQA)
  are reported to be most sample-efficient, so the plan centers on those
  two rather than population-based global search or a finite-difference
  gradient method (L-BFGS-B would burn ~11 evals per gradient estimate,
  i.e. under 10 effective steps for the whole budget — too expensive here).
- No assumption is made about modality/smoothness beyond what's given; the
  plan is designed to adapt based on observed progress rather than guess.

## Planned method sequence

1. **TuRBO-1** (`src/optimizers/turbo.py`), up to 80 evaluations.
   - TuRBO already begins with its own internal 2*dim = 20-point scrambled
     Sobol space-filling design before fitting any GP / trust region, so
     this phase doubles as both global space-filling and local refinement
     — no separate standalone Sobol phase is needed first.
   - After its init phase, TuRBO fits a local GP and does trust-region
     Thompson sampling, shrinking/growing/restarting the region based on
     observed progress — a reasonable default global+local strategy for a
     black box of unknown modality at this budget.
2. **BOBYQA** (`src/optimizers/bobyqa.py`) local polish, remaining budget
   (minimum 15, up to ~20 evaluations), warm-started conceptually at the
   best point TuRBO found (practically: since this wrapper's BOBYQA always
   starts from the box midpoint internally, the polish value-add is
   judged by whether it beats TuRBO's best; if BOBYQA's own midpoint-start
   trajectory does not reach TuRBO's best within its allotted evals, TuRBO's
   point remains the reported best — see evaluate-budget note below).
   Rationale: BOBYQA's trust-region quadratic model is well suited to
   squeezing out final local precision that a Thompson-sampling batch
   selector (TuRBO) may leave on the table, cheaply (no GP refit cost).

Target split: ~80 / ~20, adaptive per the criteria below.

## Explicit deviation criteria (quantitative)

- **Early handoff from TuRBO to BOBYQA:** if the global best-so-far value
  has not improved at all over the most recent 30 evaluations while in
  the TuRBO phase (checked after each batch of 4), stop TuRBO early and
  move to the BOBYQA polish phase with whatever budget remains, down to a
  floor of at least 15 evaluations reserved for polish.
- **Floor on polish budget:** regardless of how TuRBO is doing, at least
  15 evaluations are always reserved for the BOBYQA phase (i.e. TuRBO is
  capped at 85 evaluations even if still improving), so the run never
  ends without a local-refinement pass.
- **BOBYQA exhausts early:** if BOBYQA's internal solver thread converges
  and raises `OptimizerExhausted` before its allotted evaluations are
  used, spend the remaining budget resuming TuRBO (a fresh trust region,
  i.e. effectively a restart) rather than leaving budget unused.
- **TuRBO repeated restarts without progress:** if TuRBO's trust region
  collapses and restarts (internal mechanism, triggered when its length
  drops below the minimum) more than twice with no improvement in the
  global best between restarts, treat that as evidence the GP-based local
  search isn't adding value here and move to the BOBYQA polish phase
  immediately with all remaining budget, instead of waiting for the
  30-evaluation stall rule above.
- Every time any of the above actually fires, an entry will be appended
  to `changelog.md` citing the specific observed numbers (eval index,
  best-so-far value, stall length, etc.) that triggered it.

## Labels

- `"turbo"` for all TuRBO evaluations.
- `"bobyqa"` for all BOBYQA evaluations.
