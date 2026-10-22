# Roadmap — Initial Step

## Understood problem

- Sealed black-box merit function `merit(x) = y(x) + PENALTY_LAMBDA *
  total_constraint_violation(x)` over `x in R^10`, to be **minimized**.
  The minimum of `merit` coincides with the true constrained optimum. A
  merit value far outside the typical range likely reflects heavy
  infeasibility at that point, not an extreme underlying objective —
  this will be used to interpret outliers, not to try to separate `y`
  from the penalty term (the two are not separable from the sealed
  interface).
- Dimension: 10. Bounds, read directly from the constructed harness:
  `[-5, 5]` per coordinate (confirmed via `harness.bounds`, not assumed).
- Direction: minimize (confirmed via `harness.direction == "min"`).
- Noise: deterministic (confirmed via `harness.noisy == False`). No
  replication is needed to average out observation noise; repeated
  evaluation of the same point would only waste budget.
- Total evaluation budget: 100 `evaluate()` calls, hard-enforced by the
  harness (`BudgetExhausted` once exceeded). This is a very tight budget
  relative to the dimension (10x the dimension), which rules out
  population-based global methods that need large populations/generations
  to make progress (e.g. CMA-ES default population ~4+3*ln(10)≈11,
  needing many generations; full DE with NP=10*dim=100 would consume the
  *entire* budget on one generation) and favors sample-efficient
  surrogate-based methods.
- Assumption: because of the unknown penalty structure, the merit
  landscape may have sharp cliffs/plateaus near constraint boundaries in
  addition to smooth variation from the underlying objective. This is a
  genuine uncertainty that first-step evidence should speak to, not an
  assumption baked into the plan beyond this step.
- I will not attempt to identify, search for, or reverse-engineer the
  underlying function or inspect `objective.py`/harness internals beyond
  the public `AgentTaskHarness` interface.

## Method and hyperparameters for this (first) step

**Method:** Scrambled Sobol' low-discrepancy space-filling design
(`src/optimizers/sobol_search.py`, wraps `scipy.stats.qmc.Sobol`,
`scramble=True`, scipy's own default — no tuning).

**Why this method for step 1:** With no observations yet, the priority is
to get an unbiased, well-spread picture of the merit landscape over the
whole box — including how often random points are feasible/near-feasible
vs. heavily penalized — before committing to any surrogate model or local
solver. A Sobol design is the lowest-bias way to do this with a small
number of points (better coverage per point than uniform random), and its
size (2*(dim+1) = 22) is exactly the standard initial-design rule this
project's own GP-BO wrapper (`src/optimizers/gp_bo.py`) uses before
fitting its first GP — so this step doubles as that GP's initial design
if GP-BO is chosen next, with no wasted evaluations either way.

**Hyperparameters:**
- `bounds`: `harness.bounds` (`[-5, 5]` x 10), read from the harness, not
  hand-entered.
- `seed=0`: fixed for reproducibility of this run; no other run will be
  compared against it, so the specific value is not consequential.
- `scramble=True`: scipy `qmc.Sobol`'s own default (inherited, not
  overridden).
- Batch size for this step: 22 points, asked and told in one `ask(n=22)`
  call (Sobol sequences are most evenly distributed when the cumulative
  count is a power of 2 — 22 is not a power of 2, but the only relevant
  power-of-2 thresholds near the desired initial-design size are 16 (a
  bit small for 10-D) and 32 (uses nearly a third of the entire 100-call
  budget on pure exploration before any adaptive method runs). 22 matches
  the standard initial-design heuristic exactly and is a reasonable
  compromise; evenness only matters up to the fraction of the budget
  spent here, and 22/100 is small enough that this is a minor
  consideration, not worth sacrificing the standard heuristic for.

## Step length

- Maximum additional evaluations this step: **22**.
- Cumulative evaluation count at which I will pause: **22** (out of 100).
- No earlier pausing condition: a space-filling design has no intermediate
  decision point — all 22 points are decided before any evaluation, and
  there is nothing to learn until the batch is complete that would change
  how the remaining points in this same batch should be chosen.

## What I want to learn / evidence to inspect at the checkpoint

- The range and distribution of observed merit values: are most points
  heavily penalized (suggesting a small feasible region relative to the
  box) or are many points in a "normal" range (suggesting a large/loose
  feasible region)?
- Whether there is any visible structure (e.g. merit correlating with
  distance from the box center or from specific coordinates) worth
  exploiting, or whether the landscape looks unstructured at this
  resolution.
- The best point found, to serve as a reference for judging whether a
  subsequent local or surrogate-based method is making real progress.
- Based on this evidence, whether to proceed with GP-BO (surrogate model,
  well suited to this budget/dimension ratio), a local derivative-free
  solver seeded at the best point found (if the landscape looks smooth
  near the best point), or another approach — this choice is deferred to
  the first changelog decision, after seeing the data.
