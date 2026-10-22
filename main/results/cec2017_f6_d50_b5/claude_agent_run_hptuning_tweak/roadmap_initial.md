# Roadmap — Initial Step

## Understood problem

- Sealed black-box objective `fn(x)`, accessed only via `AgentTaskHarness.evaluate(x)`.
- Dimension: 50. Bounds: `[-5, 5]` per coordinate (a 50-dimensional box).
- Direction: minimize.
- Noise: deterministic (`NOISY=False` expected and will be confirmed from
  `harness.noisy` at startup) — no need for replication or noise-robust
  acquisition.
- Total evaluation budget: 100 `evaluate()` calls, hard-enforced by the
  harness (`BudgetExhausted` once exceeded).
- Critical framing assumption: 100 evaluations for a 50-dimensional problem
  is an extremely tight budget-to-dimension ratio (2x), far below what most
  of the available optimizers assume by default. For reference, this
  project's own wrappers default to: BOBYQA's underlying library wants
  `npt = 2*dim+1 = 101` points just to build its first full quadratic model
  (more than the entire budget); TuRBO's documented default initial design
  is `2*dim = 100` points (the entire budget) before its first trust-region
  update; CMA-ES's default population size at dim=50 is
  `4 + floor(3*ln(50)) ≈ 15`, which is survivable but only allows ~6-7
  generations if used for the whole budget with no initial scan. None of
  these defaults can be used as-is; whichever method is chosen will need
  deliberately scaled-down hyperparameters (smaller init design, smaller
  population, or a reduced model size) to leave any evaluations for actual
  directed search. This also means a quadratic-model-fitting local solver
  (BOBYQA with full npt) is not a good first move at this ratio — it would
  spend the whole first step only building a local model with no walk
  toward a optimum yet.
- No information about the function's identity, separability, smoothness,
  or multimodality is available yet, and none will be sought beyond the
  stated interface (sealed black box; no inspection of `objective.py`
  beyond the documented `DIM`/`BOUNDS`/`DIRECTION`/`NOISY`/`fn` contract
  already read via the harness).

## Step 1 plan

**Method:** Scrambled Sobol' low-discrepancy quasi-random scan
(`src/optimizers/sobol_search.SobolSearch`), standard library defaults
(scipy `qmc.Sobol`, `scramble=True`), seeded with `seed=0` for
reproducibility.

**Why this method for this step:** With zero observations so far, the most
valuable use of a small slice of budget is cheap, even coverage of the
50-dimensional box to get a first read on the objective's scale, spread,
and any detectable large-scale trend, before committing the bulk of the
budget to a directed (local or population-based) method. A low-discrepancy
scan is preferred over plain uniform random because it spreads points more
evenly per unit of budget spent, which matters when the sample size (a
small fraction of what a 50-D space would ideally want) is itself tiny.

**Hyperparameters / initialization for this step:**
- `SobolSearch(bounds=harness.bounds, seed=0)` — all defaults inherited
  from `scipy.stats.qmc.Sobol` (`scramble=True`); no tuning applicable to
  a pure quasi-random sampler beyond its seed.
- Batch size for this step: ask for all 16 points in a single `ask(n=16)`
  call. Sobol sequences have their best balance properties when the
  cumulative count is a power of 2, so 16 (rather than an arbitrary round
  number like 20) is chosen to respect that property exactly while still
  being a small, affordable fraction of the budget.

**Step length:** 16 objective evaluations (this is the entire step — one
batch, asked and told in full, then a checkpoint). Cumulative evaluation
count at pause: 16 / 100. Remaining after this step: 84.

**Why this length:** 16 is large enough to span a meaningful, well-balanced
sample of a 50-D box and to give a first (coarse) read on the range and
spread of objective values, but small enough to preserve the large majority
of the budget (84 of 100 evaluations) for whatever directed method the
first checkpoint's evidence favors. Given the extreme scarcity of budget
relative to dimension, no more than this is spent on pure exploration
before reassessing.

**What I want to learn / evidence to inspect at the checkpoint:**
- The range (min/max) and spread (e.g. std) of the 16 observed `y` values,
  to gauge the objective's scale and whether there is substantial variation
  across the box (vs. a nearly flat landscape, which would suggest a very
  different strategy).
- The best point found so far (`x_best`, `y_best`) as a candidate seed
  location for a subsequent local/population method.
- Whether a simple linear fit of `y` against the 16 `x` points shows a
  strong, consistent direction of improvement (suggesting large-scale
  structure a directed method can exploit) or looks closer to noise-free
  but unstructured variation (suggesting a more global/population method
  is safer than committing immediately to a local descent from one point).
- No condition for pausing earlier than 16: this step is a single Sobol
  batch kept intentionally short already, and Sobol's sequential ask/tell
  interface does not provide intermediate feedback that would justify
  stopping mid-batch.
