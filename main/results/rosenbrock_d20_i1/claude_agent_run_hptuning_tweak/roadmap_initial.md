# Roadmap — Initial Step

## Understood problem

- Sealed black-box objective, treated as unknown apart from the stated
  problem-specific parameters. No attempt will be made to identify it.
- Dimension: 20. Bounds: [-5, 5] per coordinate (box-constrained).
- Direction: minimize.
- Noise: deterministic (no observation noise) — stated explicitly, so no
  replication strategy is needed; a single evaluation per point is
  trustworthy evidence for comparisons.
- Total evaluation budget: 100 `evaluate()` calls, hard-enforced by the
  harness (`BudgetExhausted`). This is only 5x the dimension — a very tight
  budget relative to the generic 2000-eval convention used elsewhere in
  this project. Every evaluation is relatively precious; initial-design
  size and local-refinement length both need to be chosen with this in
  mind rather than reusing defaults sized for a 100x-dim budget.
- Assumption: no further structure (e.g. separability, smoothness class) is
  known a priori. This will be inferred only from observed data.

## Method for this step: Sobol' quasi-random space-filling search

Rationale: with no observations yet, the only thing worth doing is a cheap,
unbiased space-filling read of the 20-dimensional box. A small Sobol' design
(`src/optimizers/sobol_search.py`, wrapping `scipy.stats.qmc.Sobol`,
`scramble=True`) gives more even coverage than plain random search for the
same number of points, which matters when so few points can be afforded.
This also directly informs the next decision: whether the landscape looks
amenable to a single local refinement from the best point found (e.g.
BOBYQA, CMA-ES) or whether a model-based approach that reuses this exact
batch as its own initial design (GP-BO, which conventionally starts from
2*(dim+1) = 42 Sobol points before switching to acquisition-driven points)
is a better fit for the remaining budget.

No hyperparameters are being tuned for this step; this is a deliberately
generic, assumption-free opening move, not an optimizer being configured.

**Hyperparameters / initialization:**
- `SobolSearch(bounds=[[-5,5]]*20, seed=0)` — scipy's own defaults
  (`scramble=True`) are used, no tuning. Seed fixed only for
  reproducibility of this record, not a tuned choice.
- No prior points to seed the sequence with; cumulative count will be 16
  after this step, chosen to be a power of 2 (Sobol's own documented
  preference for discrepancy/balance properties) while remaining a small
  fraction (16%) of the total 100-evaluation budget.

## Step length

- **Step length:** 16 evaluations.
- **Cumulative evaluation count at pause:** 16 (of 100 total; 84 remaining).
- **Why this length:** 16 is the smallest power-of-2 batch that is still
  large enough, relative to dim=20, to give a non-degenerate first read of
  the function's range and rough shape (e.g. whether values cluster near
  the box center vs. vary strongly with distance from it, whether the
  range of observed y spans orders of magnitude suggesting a steep/uneven
  landscape). Going larger here (e.g. straight to 42, GP-BO's own
  recommended init size) would commit over 40% of the entire budget before
  any decision point, which conflicts with deciding one step at a time;
  going smaller than a power of 2 would waste some of Sobol's balance
  properties for no benefit.

## What I want to learn / evidence to inspect at the checkpoint

- The spread (min, max, std) of the 16 observed y values, and how the best
  point found compares to a few others — a rough signal of landscape
  ruggedness and whether there's a clear promising region or the response
  looks closer to uniform/flat within this coarse sample.
- Whether any evaluation failed or returned a degenerate value (e.g. NaN,
  identical value at all points), which would indicate a problem with the
  wrapper/bounds rather than the objective itself.
- Based on this, decide at the next checkpoint between: (a) continuing
  Sobol to build out a larger space-filling set (e.g. toward 42) feeding
  into GP-BO's acquisition phase, (b) switching straight to a local
  solver (BOBYQA or CMA-ES) seeded near the best Sobol point if the
  landscape already looks like it rewards local refinement, or (c) some
  other adjustment justified by the data. No method beyond this first
  Sobol batch is preselected.

## Conditions for pausing sooner

None expected — 16 cheap, independent evaluations of a declared
deterministic objective carry negligible risk of needing an early pause
(no iterative solver state to go unstable, no long-running single call
expected). If any single `evaluate()` call raises an unexpected exception
(distinct from `BudgetExhausted`), that would be inspected immediately
before continuing.
