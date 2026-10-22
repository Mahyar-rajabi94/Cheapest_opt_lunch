# Roadmap — initial step only

## Understood problem

- Dimension 10. Minimize a deterministic (noiseless) sealed black-box
  objective `f(x)`.
- Coordinates x0..x7 are constrained to **integer** values on these bounds:
  x0 in [0,1], x1 in [0,1], x2 in [0,3], x3 in [0,3], x4 in [0,7], x5 in
  [0,7], x6 in [0,15], x7 in [0,15]. x8, x9 are continuous in [-5,5].
- The evaluator rounds any non-integer value passed for x0..x7 internally,
  so the objective is piecewise-constant along those 8 axes: gradient
  information there is meaningless, and the effective search space on
  those axes is a finite (if large: ~1.05e6 combinations) integer lattice.
  x8, x9 are genuinely continuous.
- Total evaluation budget: 100 `harness.evaluate()` calls, hard-enforced
  by the harness (a call past budget raises `BudgetExhausted`). No
  replication is useful since the function is deterministic.
- This budget (100 calls for a 10-d mixed problem, with ~1e6 integer
  combinations alone) is far too small for any exhaustive or even
  moderately-dense coverage. The practical approach has to be: (1) spend a
  modest slice of budget on a space-filling initial design to learn rough
  structure (which integer settings tend to help, roughly where x8/x9
  should sit), then (2) spend the bulk of the budget on local,
  evidence-driven refinement (integer coordinate descent / discrete local
  search on x0..x7, paired with continuous local refinement on x8,x9),
  because a population-based global method (CMA-ES, DE, etc.) would spend
  most of a 100-evaluation budget just forming its first generation in 10
  dimensions and would not have enough generations left to adapt usefully,
  especially given the piecewise-constant axes it isn't designed for.
- No assumption is made yet about separability, unimodality, or relative
  importance of the integer vs. continuous block; the first step is
  designed to gather evidence on exactly that, not to assume it.

## Step 1 — space-filling initial design

**Method:** Scrambled Sobol' low-discrepancy sequence
(`src/optimizers/sobol_search.py`, wrapping `scipy.stats.qmc.Sobol`),
sampled over the full 10-d box `harness.bounds`. Sobol is chosen over
uniform random search for its lower-discrepancy (more even) coverage of
the joint 10-d space per point spent, which matters given how few points
we can afford. Candidates for x0..x7 are rounded to the nearest integer
(and re-clipped to bounds) only when constructing the vector passed to
`harness.evaluate`; the Sobol sampler itself continues to operate on
continuous proposals internally (its own bookkeeping is on the
unrounded reals), since rounding its state would corrupt the
low-discrepancy property of the sequence for future draws.

**Hyperparameters / initialization:**
- `SobolSearch(bounds=harness.bounds, seed=0)` — inherited defaults from
  the wrapper: scipy's own `qmc.Sobol` with `scramble=True` (scipy
  default), no tuning. `seed=0` chosen only for reproducibility of this
  run, not a tuned value.
- Batch size per `ask()` call: 16, drawn twice to reach 32... (see step
  length below) — Sobol sequences have their best balance properties when
  the *cumulative* count is a power of 2, so draws are taken in
  power-of-two-aligned chunks.

**Step length:** Draw and evaluate **20** points (two Sobol batches: 16,
then 4, keeping the running cumulative total moving toward the next power
of two without overspending budget on pure exploration). Pause at
cumulative evaluation **20** (of 100 total; 80 remaining after this step).
20 was chosen as roughly a fifth of the total budget: enough points to
see whether any integer sub-blocks or x8/x9 regions show a consistent
trend across a space-filling sample, while leaving the large majority of
the budget (80 evaluations) for the local refinement phase that a 100-eval
mixed-integer budget realistically depends on for its final quality.

**What I want to learn / evidence to inspect at the checkpoint:**
- The best `y` found and the point that achieves it, as a starting
  incumbent for local refinement.
- The spread of `y` values across the 20 points (range, rough variance) —
  whether the landscape looks highly sensitive to the integer coordinates,
  to x8/x9, or both, which will shape how the refinement budget is split
  between discrete coordinate descent and continuous local search.
- Any visually obvious pattern (e.g. a coordinate whose observed values
  correlate with better y) that would justify weighting local search
  toward particular coordinates first.

**Conditions for pausing sooner:** none expected — 20 fixed evaluations
is already a small, deliberately bounded block; there is no mid-block
signal (e.g. budget exhaustion risk) that would warrant stopping earlier
than 20 within this first step.
