# Roadmap — initial step

## Understood problem

- Dimension 10, minimize a deterministic (noise-free) black-box `fn(x)`.
- Per-coordinate bounds (confirmed via `harness.bounds`):
  - x0 in [0,1], x1 in [0,1] (integer)
  - x2 in [0,3], x3 in [0,3] (integer)
  - x4 in [0,7], x5 in [0,7] (integer)
  - x6 in [0,15], x7 in [0,15] (integer)
  - x8 in [-5,5], x9 in [-5,5] (continuous)
- x0..x7 are constrained to integer values; the evaluator rounds non-integer
  inputs on those coordinates internally, so the objective is piecewise
  constant along those 8 axes. Gradient/covariance-based reasoning is
  meaningless there; only combinatorial/local-search reasoning applies.
  x8, x9 are genuinely continuous.
- Total evaluation budget: 100 calls to `harness.evaluate()`, hard-enforced
  by the harness. No replication is needed since the function is
  deterministic (noisy=False confirmed via harness.noisy).
- The integer sub-space alone has 2*2*4*4*8*8*16*16 = 2,097,152 combinations,
  far more than the budget allows to enumerate; the campaign must rely on
  cheap structure-finding (space-filling sample) followed by targeted local
  search rather than any exhaustive or dense-grid approach.

## Step 1 plan

- **Method:** Sobol' low-discrepancy quasi-random search
  (`src/optimizers/sobol_search.py`, wraps `scipy.stats.qmc.Sobol`,
  `scramble=True`), sampling jointly over the full 10-dimensional box
  (all original per-coordinate bounds, continuous for x8/x9). Sampled
  points on x0..x7 are rounded to the nearest integer within their bounds
  before calling `evaluate()` (the harness/evaluator would round internally
  anyway; rounding explicitly before the call keeps the recorded `x` values
  in the trace consistent with what was actually evaluated and avoids two
  distinct Sobol draws silently landing on the same integer cell without my
  noticing).
- **Why Sobol for this step:** with no observations yet, the priority is to
  get even, unbiased coverage of the whole mixed space (joint over all 10
  dims, not axis-by-axis) with no risk of early bias toward one region. This
  is a standard choice for an uninformed first look at a black box, and
  costs nothing beyond the evaluation itself.
- **Hyperparameters:** `seed=0` for reproducibility (otherwise library
  default `scramble=True`, scipy's own default — no additional tuning).
  Batch requested in one `ask(n=16)` call, since Sobol sequences have the
  best balance properties when the cumulative count is a power of 2.
- **Step length:** 16 evaluations (cumulative count after this step: 16).
  Chosen as a power of 2 (matches Sobol's balance property) and as a modest
  fraction (16%) of the 100-call budget — enough points to see rough
  structure (which integer cells and which sign/magnitude of x8,x9 look
  promising) while preserving the large majority of the budget (84 calls)
  for whatever targeted method the first checkpoint's evidence favors
  (e.g. integer coordinate descent anchored at the best point found here,
  combined with continuous refinement on x8,x9).
- **What I want to learn / evidence to inspect at the checkpoint:**
  - The range and spread of observed y values (is the landscape flat,
    smoothly varying, or highly discontinuous across the sampled points?).
  - Which integer settings (x0..x7) and which region of (x8,x9) are
    associated with the best observed y so far, as an anchor for local
    search.
  - Whether 16 points already reveal a clearly dominant region (in which
    case local refinement from there is well justified) or the values are
    too noisy/varied to generalize (in which case a slightly larger
    space-filling sample before committing to local search may be
    warranted).
  - No early-pause condition within this step: 16 evaluations is already
    the full step, and all 16 are planned as a single batch.
