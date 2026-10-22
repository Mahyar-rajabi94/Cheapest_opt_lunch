# Roadmap — initial step only

## Understood problem

- Dimension: 10. Bounds (0-indexed): x0 in [0,1], x1 in [0,1], x2 in [0,3],
  x3 in [0,3], x4 in [0,7], x5 in [0,7], x6 in [0,15], x7 in [0,15],
  x8 in [-5,5], x9 in [-5,5]. Confirmed via `harness.bounds` (matches the
  task statement exactly).
- Mixed-integer: x0..x7 (8 of 10 coordinates) are integer-valued; the
  evaluator rounds non-integer values passed there internally, so the
  objective is piecewise-constant along those 8 axes. x8, x9 are
  continuous. Gradient-based / smooth local methods are not meaningful on
  x0..x7 in isolation; any continuous optimizer driven over the whole
  vector (e.g. Sobol, CMA-ES) still produces valid, evaluable points here
  because the evaluator's internal rounding handles the discretization —
  it is just not informative to assume smoothness in those 8 coordinates.
- Direction: minimize. Noise: deterministic (confirmed `harness.noisy ==
  False`), so no replication is needed and every observed y is exact.
- Budget: 100 `evaluate()` calls total, hard-enforced by the harness.
- Objective: sealed black box. No attempt will be made to identify it;
  decisions are based only on observed (x, y) pairs from this run.

## Assumption

The combinatorial space over x0..x7 is large (2 x 2 x 4 x 4 x 8 x 8 x 16 x
16 = 524,288 integer combinations) relative to the 100-evaluation budget,
and interacts with a 2-D continuous subspace. With so few evaluations
available, no method can do more than get a coarse read on structure
before most of the budget is committed to exploitation. The plan is
therefore to spend an initial, deliberately small step on a
space-filling design across the *full* mixed box (continuous Sobol
samples, relying on the evaluator's internal rounding for x0..x7), purely
to see the scale/shape of y and whether any coordinate shows an obvious
trend, before committing to a local-search strategy for the integer block.

## Step 1: Sobol space-filling design

- Method: `SobolSearch` from `src/optimizers/sobol_search.py`, wrapping
  `scipy.stats.qmc.Sobol` with `scramble=True` (scipy's own default,
  unmodified) over the full 10-d box exactly as returned by
  `harness.bounds`.
- Hyperparameters: `seed=0` for reproducibility (no other tuning knobs
  exposed by this wrapper). Batch size for this step: 16, drawn in one
  `ask(16)` call — 16 is a power of 2, which is the condition under which
  Sobol sequences have their best (lowest-discrepancy) balance property,
  per the wrapper's own docstring.
- Initialization: fresh `SobolSearch(bounds=harness.bounds, seed=0)`; no
  prior observations exist.
- `harness.set_active_optimizer("sobol_init")` before these calls.

## Step length and pause point

- Additional evaluations this step: 16.
- Will pause at cumulative evaluation count 16 (16 used / 100 total, 84
  remaining).
- Rationale for length: 16 is enough points to get a first read on the
  objective's scale and on whether specific integer coordinates or
  regions of the continuous pair look favorable, while keeping the
  commitment small (16% of budget) since this step is exploratory, not
  exploitative — the real work (whatever method is chosen next) needs the
  bulk of the budget.

## What I want to learn / checkpoint evidence

- The observed range and distribution of y across these 16 points, and
  the current best (x, y).
- Any visible association between specific integer-coordinate values
  (x0..x7) and lower y, suggestive of which discrete settings deserve
  focused search.
- Any visible trend in the continuous pair (x8, x9) suggestive of a
  promising region for local continuous refinement.
- No condition to pause earlier than 16 evaluations: this step is a
  single fixed-size batch (Sobol' balance degrades if interrupted
  mid-sequence at a non-power-of-2 count) and 16 evaluations is already a
  small fraction of the budget, so there is little benefit to stopping
  sooner.
