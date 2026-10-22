# Roadmap — Initial Step

## Understood problem

- Dimension: 10, continuous, box bounds `[-5, 5]` per coordinate (confirmed
  from `harness.bounds` after construction).
- Direction: minimize.
- Noise: deterministic (no observation noise), stated by the user.
- Objective: a sealed black-box "merit" value — the true objective plus a
  penalty term for constraint violation (`merit = y + PENALTY_LAMBDA *
  violation`). Lower is better, and the merit minimum coincides with the
  true constrained optimum. I will not inspect `objective.py` or try to
  infer the function's identity; I will reason only from observed merit
  values. A key implication: a very large merit value likely reflects a
  heavily infeasible point rather than an extreme underlying objective, so
  early high-merit outliers from a space-filling design should not be
  over-weighted when judging the landscape — but they do still mark region
  to avoid.
- Total evaluation budget: 100 `harness.evaluate()` calls, hard limit
  enforced by the harness.

## Assumptions

- No further structure (e.g. separability, smoothness) is known in
  advance; with only 100 evaluations for a 10-D problem, the budget is
  tight relative to dimensionality (10 pts/dimension at most), so the
  campaign should spend only a modest fraction of budget on pure
  exploration before shifting toward exploitation.
- Because the merit already folds in constraint penalties into a single
  scalar, any ask/tell optimizer that treats this as an ordinary
  unconstrained (box-bounded) minimization of the merit value is valid —
  no separate constraint-handling logic is required on my side.

## Step 1 plan

- **Method:** Sobol' quasi-random sequence (`src/optimizers/sobol_search.py`,
  wrapping `scipy.stats.qmc.Sobol`, scramble=True), tagged as
  `"sobol_init"` in the harness.
- **Hyperparameters:** `seed=0` for reproducibility. No other tunable
  hyperparameters are exposed by this wrapper (scipy's own
  `scramble=True` default is used, per the wrapper's docstring/implementation,
  i.e. scipy `qmc.Sobol` defaults as of the installed scipy version).
- **Initialization:** box bounds taken directly from `harness.bounds`
  (`[-5, 5]` per coordinate, dim 10). No prior evaluations exist yet.
- **Why this method/step:** With zero observations and a budget too small
  to rely on high-dimensional gradient or surrogate structure immediately,
  a low-discrepancy space-filling design gives even coverage of the
  10-D box and a first read on where low-merit (feasible, good) regions
  might lie, at a fixed, modest cost before switching to a more targeted
  local/global method.
- **Step length:** 16 evaluations (a power of 2, matching Sobol's note
  that cumulative draws at a power of 2 give the best uniformity).
  Cumulative budget used after this step: 16 / 100. This leaves 84 calls
  for subsequent, better-informed steps — appropriate given that the goal
  of this step is only a coarse read of the landscape, not convergence.
- **What I want to learn / evidence at checkpoint:** the spread and range
  of observed merit values (to judge how much of the box is grossly
  infeasible vs. plausible), the best point and value found, and whether
  the values suggest a single promising basin or several, which will
  inform the next method choice (e.g. CMA-ES / TuRBO around the best
  point(s), vs. more exploration if the landscape looks highly
  multimodal or mostly infeasible).
- **Early-pause conditions:** none planned — 16 evaluations is already a
  small, fixed block; I will not interrupt it mid-way since Sobol draws
  are typically consumed in full batches for good coverage properties.
