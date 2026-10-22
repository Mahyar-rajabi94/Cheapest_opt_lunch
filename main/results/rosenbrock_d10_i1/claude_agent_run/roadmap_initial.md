# Roadmap — Initial Step

## Understood problem

- Objective: sealed black-box function `fn(x)`, to be **minimized**.
- Input: real vector `x` of dimension **10**, each coordinate bounded in
  **[-5, 5]**.
- Noise: **deterministic** (no observation noise stated or implied); no
  replication is needed to separate signal from noise.
- Budget: **100** `evaluate()` calls total, hard-enforced by
  `AgentTaskHarness`. No further information about the function's
  identity, shape, or difficulty is available or sought; all decisions
  will be based only on observed `(x, y)` pairs collected through the
  harness.
- Assumption: since nothing is stated about separability, multimodality,
  or ruggedness, no such structure is assumed a priori. The first step is
  designed to gather evidence about this cheaply rather than to assume it.

## Method for this step: GP-BO (LogEI), initial Sobol design phase

This project's `GPBO` wrapper (`src/optimizers/gp_bo.py`, BoTorch
`SingleTaskGP` + analytic `LogExpectedImprovement`) is explicitly
documented as suited to budgets "of order 10 x dim, not 1000+ x dim" —
our budget of 100 is almost exactly 10x the dimension (10), which matches
this optimizer's intended regime closely. For the very first step, before
any acquisition-driven proposal can be made, `GPBO.ask()` always returns
points from its own built-in Sobol' initial design
(`n_init = 2 * (dim + 1) = 22` points, scrambled, seeded) — a GP fit is
not meaningful with fewer points than this. This is the natural
self-contained first step: it does not depend on any decision about
acquisition-function behavior, since no acquisition proposals occur until
after it completes.

Hyperparameters / inherited defaults for this step (all from
`src/optimizers/gp_bo.py`, unmodified — no tuning has been applied):
- `seed=0` (for reproducibility of the Sobol sequence and later
  acquisition-optimizer restarts).
- `n_init = 2 * (dim + 1) = 22` (BoTorch's own standard rule of thumb for
  a GP's minimum pre-acquisition design size; not specific to this
  problem).
- Sobol' sampling is scrambled, via this project's own `SobolSearch`
  wrapper around `scipy.stats.qmc.Sobol`.
- GP/acquisition hyperparameters (`SingleTaskGP` default kernel/priors,
  `num_restarts=10`, `raw_samples=512` for `optimize_acqf`) are not
  exercised in this step, since no acquisition-driven point is proposed
  until after the 22nd evaluation; they will be reconsidered (or left
  unchanged, with that fact recorded) at the next checkpoint.

## Step length

**22 additional evaluations** (cumulative count at pause: **22 / 100**,
78 remaining).

Rationale: this is exactly `GPBO`'s own `n_init`, i.e. the complete Sobol
design phase and nothing more — the smallest block that is both
self-contained (no acquisition decision is made or needed partway
through it) and sufficient for the next GP fit to be meaningful. Stopping
here, rather than continuing into acquisition-driven points without
review, lets the first real decision (whether GP-BO's acquisition
proposals are worth the remaining ~78 evaluations, or whether the initial
design already suggests a different approach is more promising) be made
from actual data rather than assumed in advance.

## What I want to learn / checkpoint evidence

At the checkpoint after these 22 evaluations I will inspect:
- The best `y` found so far, and how it compares to the spread (min, max,
  median, interquartile range) of all 22 sampled `y` values — a rough,
  budget-cheap read on how much structure/gradient the space-filling
  design has revealed.
- Whether the 22 points show any sign that the problem is highly rugged
  or deceptive (e.g. best point sits at odds with any visible trend) or
  looks comparatively smooth — this informs whether to proceed with
  GP-BO's acquisition phase, switch to a trust-region/local method, or
  hybridize.
- Timing: whether the sequential GP fit + `optimize_acqf` calls needed for
  the next phase are fast enough, given 78 remaining evaluations, to be
  worth their per-point cost relative to simpler alternatives (e.g.
  CMA-ES, TuRBO) at this budget.

No early-pause condition applies to this step: all 22 points come from a
fixed, non-adaptive Sobol design, so there is no useful intermediate
checkpoint until it is complete.
