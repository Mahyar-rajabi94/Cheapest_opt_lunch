# Roadmap — Initial Step

## Understood problem

- Black-box objective `fn(x)`, sealed (identity unknown, not to be inspected
  or reverse-engineered beyond the harness's documented public interface).
- Dimension: 20 (confirmed via `harness.dim`).
- Bounds: `[-5, 5]` per coordinate, all 20 dims (confirmed via
  `harness.bounds`).
- Direction: minimize (confirmed via `harness.direction`).
- Noise: `harness.noisy == True`. Stated to be relatively mild,
  roughly multiplicative-scale, not heavy-tailed. Repeated `evaluate()`
  calls at the same `x` can differ, so single-point comparisons (and
  finite-difference gradients in particular) are unreliable without
  replication or averaging.
- Budget: 100 `evaluate()` calls total, hard-enforced by the harness
  (`BudgetExhausted`). This is a very small budget relative to dimension
  (5x dim, versus the ~10x-dim regime some of the available surrogate
  methods, e.g. GP-BO, TuRBO, are documented for with their default
  "standard" initial-design sizes of 2*(dim+1)=42 and 2*dim=40
  respectively). Spending 40%+ of the whole budget purely on a surrogate's
  fixed initial design is a real risk at this budget and will factor into
  later method choice.

## Assumption

- "Mild, roughly multiplicative-scale noise" is read as: not so large that
  a single evaluation is uninformative, but large enough that a handful of
  nearby/repeated evaluations can disagree in rank. No assumption is made
  yet about the exact noise magnitude — this first step is partly designed
  to produce direct evidence about it.

## Step 1 plan

**Method:** Sobol' low-discrepancy quasi-random search
(`src/optimizers/sobol_search.py`, wraps `scipy.stats.qmc.Sobol`,
`scramble=True`). No tuning: this method has no relevant hyperparameters
beyond the bounds and an RNG seed (used only for reproducibility, not a
quality lever). Seed fixed to `0`.

**Why this method/step:** With only 100 total evaluations for a 20-D
space, there is no budget to waste on an uninformative or risky first
move. A space-filling design is the standard, low-risk way to get a first
read on the landscape (scale of `y`, rough location of promising regions)
before committing most of the budget to a model-based or population-based
method whose own hyperparameters (population size, initial-design size,
trust-region shape) should be chosen with actual evidence in hand rather
than guessed blind. Sobol is preferred over plain random search for its
better coverage per point, which matters when every point is precious.

**Step composition (18 evaluate() calls total):**
- 16 fresh Sobol points (a power of 2, as recommended for Sobol's balance
  properties), covering the box broadly. This also gives a first read on
  `y`'s scale and whether the landscape is is smooth/unimodal-looking or
  rugged, across these 16 widely separated samples.
- 3 repeated evaluations of one of those 16 points (chosen as the first
  Sobol point drawn), to get a direct, dedicated estimate of the noise's
  magnitude at a fixed x (sample std of 4 draws at the same point) before
  any budget is committed to a method whose hyperparameters (e.g.
  population size, whether to average repeats) would benefit from knowing
  that magnitude.

Total for this step: 16 + 3 = 19 evaluate() calls (cumulative count after
this step: 19 of 100). I will pause at cumulative evaluation 19, or
earlier only if `BudgetExhausted` is somehow raised prematurely (should
not happen since 19 < 100).

**What I want to learn / evidence to inspect at the checkpoint:**
- The range and distribution of the 16 Sobol `y` values: scale, whether one
  region stands out as much better, roughness (do nearby-ish points in the
  16 give wildly different `y`, suggesting high effective dimensionality
  or rugged landscape).
- Direct noise estimate: std / spread of the 4 values (1 original + 3
  repeats) at the fixed repeated point, and how that compares to the
  overall spread across the 16 distinct points (i.e., is the noise small
  relative to the signal we'd be optimizing against, or large enough to
  require averaging/robust handling in the next method).
- Based on this, choose the next method (e.g., CMA-ES, TuRBO, GP-BO, or
  continued Sobol) and its hyperparameters with actual evidence, per the
  protocol — not decided in advance here.

No early-pause condition is set for this step: 19 calls is already a small,
bounded block and each point is cheap to interpret only once all are in
hand (the repeats in particular need the other 3 draws before they're
informative).
