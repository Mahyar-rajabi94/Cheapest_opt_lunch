# Roadmap — initial plan

## Problem

- dim = 10, bounds = [-5, 5]^10, direction = minimize, deterministic
  (no observation noise), budget = 100 `evaluate()` calls total.
- Budget is only 10x dimension — far too small for any population-based
  or many-restart method to do meaningful search. This is explicitly the
  regime this project's `gp_bo.py` documents itself as suited for
  ("expensive, continuous problems, budgets of order 10 x dim, not
  1000+ x dim").

## Method plan

**Primary (and expected sole) method: GP-BO (`src/optimizers/gp_bo.py`),
for the full 100-evaluation budget.**

- `GPBO(bounds=[(-5,5)]*10, seed=0)`.
- Initial design: library default `n_init = 2*(dim+1) = 22` Sobol' points
  (quasi-random, space-filling, no information used yet).
- Remaining `100 - 22 = 78` evaluations: sequential GP + analytic LogEI
  acquisition, one point at a time (fit GP on all data so far, optimize
  LogEI, evaluate, tell, repeat).
- No batching, no hyperparameter tuning beyond the library's own
  documented defaults (BoTorch SingleTaskGP default kernel/priors,
  `num_restarts=10`, `raw_samples=512` for the inner acquisition
  optimization) — this keeps the run a clean, reproducible application of
  a standard, well-regarded small-budget BO recipe rather than an
  ad-hoc tune.

## Why this choice, from the stated problem properties alone

- Dimension (10) and bounds are moderate and fully boxed — fine for a GP
  surrogate without special treatment.
- Deterministic, noise-free objective — no need for a noise-robust
  acquisition function (e.g. noisy EI) or replication; plain LogEI on
  exact observations is appropriate.
- Budget (100 = 10x dim) is the dominant constraint. At this scale:
  - Pure random/Sobol search has no time to exploit structure.
  - CMA-ES / DE need populations of order 4*dim–10*dim just to complete
    one or two generations; 100 evaluations would buy at most ~1-2
    generations, not enough for the covariance/population adaptation
    these methods rely on.
  - BOBYQA (local trust-region) converges well for *unimodal, smooth*
    landscapes but commits to one local basin from a single starting
    point (box midpoint); with no prior information about the
    landscape's shape, risking the entire budget on one local descent is
    risky if the function is multimodal.
  - GP-BO uses every single evaluation to update a global surrogate and
    directs search via expected improvement, which is the standard
    answer in the literature for exactly this "small/expensive budget,
    continuous, boxed, deterministic" regime, and is the one optimizer
    in this toolbox explicitly documented as designed for budgets of
    this order.
- A sequential, single-candidate acquisition loop (rather than a batch
  acquisition) is appropriate since there is no parallel evaluation
  constraint here — fully sequential GP-BO extracts the most information
  per evaluation.

## Budget split

- 22 evaluations: Sobol initial design (`optimizer` tag: `gp_bo_sobol_init`).
- 78 evaluations: GP-BO LogEI acquisition loop (`optimizer` tag: `gp_bo`).
- No other method is scheduled up front. See deviation criteria below for
  when a hybrid/fallback would be triggered.

## Deviation criteria (quantitative, pre-registered)

1. **GP fit/acquisition failure**: if `GPBO.ask()`/`tell()` raises an
   exception (e.g. numerical failure in GP fitting) on more than 1
   consecutive attempt, fall back to the remaining budget as a
   `SobolSearch` sweep (`optimizer` tag: `sobol_fallback`), logged in
   changelog.md with the exact error.
2. **Stagnation**: if `y_best_so_far` has not improved for 40 consecutive
   evaluations *after* the initial 22-point Sobol design has completed
   (i.e. by evaluation ~62 or later with no improvement since ~22), and
   at least 15 evaluations of budget remain, switch the remaining budget
   to a local polish starting from the current best point: a
   coordinate-wise pattern search / small-step local perturbation around
   the incumbent (implemented inline, since BOBYQA's wrapper fixes its
   start point to the box midpoint and cannot be warm-started from an
   arbitrary incumbent). This would be logged as a deviation with the
   observed stagnation evidence.
3. **Runtime blow-up**: if GP fitting time grows so large that completing
   the remaining acquisition steps within a reasonable wall-clock budget
   (a few minutes per step) becomes impractical, switch remaining budget
   to `sobol_fallback` and log the observed per-step timing.

Absent any of the above triggers, the plan is to run GP-BO (Sobol init +
LogEI loop) for the entire 100-evaluation budget without switching
methods.
