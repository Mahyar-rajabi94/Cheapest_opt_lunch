# Roadmap — initial step only

## Understood problem

- Black-box objective, sealed identity (not to be reverse-engineered or
  inspected beyond the documented harness/optimizer interfaces).
- Dimension: 40. Bounds: [-5, 5] per coordinate (40 identical box
  constraints, confirmed via `harness.dim` / `harness.bounds` before any
  evaluation). Direction: minimize. Noise: deterministic (`harness.noisy
  == False`), confirmed from the harness.
- Total evaluation budget: 100 calls to `harness.evaluate()`, hard-enforced
  by the harness (raises `BudgetExhausted` past the limit). This is an
  unusually small budget relative to dimension (2.5x dim, versus the more
  common ~10x-dim or larger conventions), so every evaluation is precious
  and no method that relies on large initial designs or many generations
  can be used at its "textbook" settings.

## Assumption surfaced

No replication is planned anywhere in this campaign: the objective is
stated deterministic, so repeated evaluations at the same point would
waste budget for zero information. All methods below are used in their
noiseless form.

## Step 1 — method, hyperparameters, and why

**Method:** Scrambled Sobol' low-discrepancy space-filling design
(`src/optimizers/sobol_search.py`, wraps `scipy.stats.qmc.Sobol`,
`scramble=True`), evaluated one point at a time via `harness.evaluate`.

**Hyperparameters / initialization:**
- `bounds`: the 40 `[-5, 5]` box constraints reported by the harness.
- `seed`: 0, for reproducibility (scipy `qmc.Sobol` default scrambling
  otherwise; no other tuning — this optimizer has no other exposed
  hyperparameters per its own docstring).
- Batch size for this step: **n = 16**. Sobol sequences have their best
  (lowest-discrepancy) coverage when the *cumulative* count is a power of
  two (per the module's own docstring), so 16 is chosen over an arbitrary
  number like 15 or 20. 16 points is roughly 0.4x the dimension — far
  short of any rule-of-thumb "enough points to fit a 40-D GP"
  (commonly 2*(dim+1) = 82, which would consume 82% of the entire budget
  on initialization alone and is not viable at this budget) — but enough
  to get a first, evenly-spread read on the objective's scale and rough
  behavior before committing more budget to any model-based method.

**Why this method for this step:** With no observations yet, nothing is
known about the function's smoothness, scale, separability, or whether it
has a single dominant basin. A space-filling design is the standard,
assumption-free way to gather that first evidence cheaply and without bias
toward any particular optimizer's internal model. Given the extremely
tight budget (2.5x dim), committing a large fraction of it to a single
untested model-based method (e.g. the packaged GP-BO's default 82-point
init) before knowing anything about the landscape would be premature;
16 points is a deliberately small opening bet.

## Step length and pause point

- Maximum additional evaluations this step: **16**.
- Cumulative evaluation count at which I will pause: **16 / 100**.
- Budget remaining after this step: 84.

## What I want to learn, and what I will inspect at the checkpoint

- The range and distribution of observed `y` values (scale of the
  objective, whether it spans orders of magnitude suggesting a
  log-transform or standardization matters for any surrogate).
- Whether the best-so-far value drops sharply from just a few points
  (suggesting a strong, perhaps near-convex, global structure reachable by
  a local refiner) or looks flat/erratic across the sampled points
  (suggesting more global/multimodal structure, favoring continued
  global search or a trust-region method).
- Whether 16 points already suffice to make the choice between (a)
  switching to a GP-based method (GP-BO or a trust-region BO such as
  TuRBO) with a reduced, campaign-tuned initial design, (b) seeding a
  CMA-ES run at the best point found with a tuned step size, or (c)
  drawing more Sobol points first because the landscape looks too
  irregular to trust a surrogate yet.

No early-pause condition applies within this first 16-point block: Sobol
points are independent of each other, so there is no useful intermediate
checkpoint before all 16 are drawn.
