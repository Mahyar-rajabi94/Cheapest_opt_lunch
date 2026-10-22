# Roadmap — Initial Step

## Problem understood
- Black-box objective `fn(x)`, `x` a length-10 real vector, each coordinate
  bounded in `[-5, 5]`.
- Direction: **minimize**.
- Noise: **deterministic** (no observation noise) — stated explicitly, so no
  replication of points is needed to average out noise.
- Budget: **100** `evaluate()` calls total, hard-enforced by
  `AgentTaskHarness`. This is a small budget (10x dimension), not the
  2000-call convention used elsewhere in this project, so evaluations must
  be used efficiently from the very first call — there is no room for a
  long, cheap exploratory phase followed by a separate refinement phase the
  way a 2000-budget run could afford.
- Objective identity is sealed: I will not inspect `objective.py` or any
  harness/optimizer source beyond the public `ask`/`tell`/`best` interface
  and docstrings, and will not try to recognize or search for the function.
  All decisions are based only on observed `(x, y)` pairs I collect myself.

## Available tooling (from docstrings/public interface only)
This sandbox ships several optimizers behind a common `ask(n)` / `tell(X, y)`
/ `best()` interface: `RandomSearch`, `SobolSearch`, `CMAES` (pycma, no
restarts), `BOBYQA` (Py-BOBYQA, closed-loop local solver), `NelderMead`,
`DifferentialEvolution`, `GPBO` (BoTorch SingleTaskGP + analytic LogEI), and
`TuRBO` (BoTorch-based trust-region BO, Eriksson et al. 2019).

## Method chosen for this first step: GP-BO (`src/optimizers/gp_bo.py`)
Rationale:
- The GP-BO module's own docstring states its documented regime is
  "expensive, continuous problems, budgets of order 10 x dim, not 1000+ x
  dim" — this task (budget 100, dim 10) is exactly that regime, and exact-GP
  cubic cost only becomes a problem at much larger budgets than 100.
- The problem is deterministic and continuous (10 real-valued coordinates in
  a box), which matches GP-BO's noiseless analytic LogEI assumption (no need
  for a noise-robust acquisition function).
- With such a tight budget, a method that models the whole surface after
  every evaluation and chooses the single most information-dense next point
  (expected improvement) should make better use of each of the 100 calls
  than a population-based evolutionary method (CMA-ES, DE) that spends many
  evaluations per generation with weaker per-point exploitation early on, or
  a closed-loop local solver (BOBYQA) that could converge to a local optimum
  without any global exploration if the surface is multimodal (currently
  unknown).

## Hyperparameters / initialization for this step
All inherited, unmodified, from `src/optimizers/gp_bo.py`'s own documented
defaults (`GPBO`, this project's BoTorch wrapper):
- `n_init = 2 * (dim + 1) = 22` initial Sobol' points (via this project's
  `SobolSearch`), the module's own "standard rule of thumb for how many
  points are needed before a GP fit is meaningful."
- GP model: BoTorch `SingleTaskGP`, default kernel/priors (dimension-scaled
  lengthscale prior, Hvarfner et al. ICML 2024), `Normalize` input transform
  to the task bounds, `Standardize` output transform.
- Acquisition: analytic `LogExpectedImprovement` (noiseless, single-
  candidate), matching the deterministic objective.
- Acquisition optimization: multi-start L-BFGS via `optimize_acqf`,
  `num_restarts=10`, `raw_samples=512` (BoTorch's own tutorial defaults).
- `seed=0` for the Sobol' initial design and acquisition-optimizer restarts,
  for reproducibility of this run.
- I am not changing any of these for this first step: there is no
  observational evidence yet to justify deviating from the documented
  defaults, and the task explicitly says not to invent tuning without
  evidence.

## Step length for this first step
This step covers the **entire 22-point Sobol' initial design plus a few
GP-driven points**, specifically: **run GP-BO's `ask`/`tell` loop for 30
evaluations** (cumulative evaluation count 30 of 100 at the pause point).

Why 30:
- The first 22 calls are a fixed, non-adaptive Sobol' design (GPBO's `ask`
  returns Sobol' points for free until `n_init` is reached) — there is
  nothing to learn from inspecting the data mid-design, so there is no value
  in pausing before all 22 are done.
- Adding 8 more (GP-driven, acquisition-selected) calls after the initial
  design lets me see the GP actually functioning as a model (fit on >= 22
  points) and producing a short sequence of LogEI-selected points, giving
  real evidence (improvement rate, whether the acquisition is concentrating
  or still spreading out, value range/spread) about whether GP-BO is working
  well on this surface before committing a large fraction of the remaining
  70-call budget to it.
- 30 out of 100 still leaves 70 evaluations (70%) for whatever is decided
  next, so this is a genuine checkpoint rather than a budget-exhausting
  default.

## What I want to learn at the checkpoint
- The spread and minimum of the 22 Sobol' initialization values: does the
  function look smooth/unimodal, highly multimodal, or have a very different
  scale across the box? (Informs whether GP-BO's smoothness assumption is
  reasonable, or whether a more global/explorative method, or BOBYQA from
  the Sobol' incumbent, might do better.)
- Whether the 8 GP-driven points improve on the Sobol' best, and by how
  much, versus their cost (GP fit + acquisition optimization time, though
  wall-clock is not budget-limited here).
- Whether LogEI candidates are clustering (exploiting a promising basin) or
  still scattered (indicating more global uncertainty remains).

## Conditions for pausing sooner
None for the fixed 22-point Sobol' phase (nothing to decide mid-design).
During the subsequent GP-driven points (points 23-30), I will not pause
early purely for a single point's value, since deterministic single-point
behavior is expected to be noisy in trend terms over fewer than ~5 points;
I will review as planned at 30.
