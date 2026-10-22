# Roadmap — Initial Step

## Understood problem

- Sealed black-box objective, treated as unknown and not to be identified.
- Dimension: 6. Bounds: [0, 1] per coordinate (all six dimensions equal box).
- Direction: minimize.
- Noise: deterministic (no observation noise) — repeated evaluation of the
  same point will return the same value, so there is no need for
  replication or noise-robust acquisition.
- Total evaluation budget: 100 `evaluate()` calls, hard-enforced by
  `AgentTaskHarness`.
- No other structural information (e.g. separability, smoothness,
  multimodality) is assumed yet; this will be read from observed data as
  the campaign proceeds.

## Method chosen for this step: GP-BO (this project's `GPBO` wrapper,
`src/optimizers/gp_bo.py`, BoTorch SingleTaskGP + analytic LogEI)

Rationale: budget (100) relative to dimension (6) is ~16.7x dim, matching
exactly the regime this wrapper's own docstring identifies as GP-BO's
practical sweet spot ("expensive, continuous problems, budgets of order
10 x dim, not 1000+ x dim"). With only 100 total calls, a sample-efficient
surrogate-based method that models the whole function and chooses each
next point to directly target expected improvement is a much better fit
than a population-based or purely local method, which would either waste
evaluations on blind coverage or risk never finding the basin of the
global optimum in so few calls. The function is deterministic, so an
exact (noiseless) GP fit via LogEI is appropriate as implemented (no
observation-noise term needed).

This first step only covers GP-BO's **initial design phase**, not yet any
acquisition-driven (GP-fit) evaluations.

### Hyperparameters / initialization for this step

- `GPBO(bounds=[[0,1]]*6, seed=0)` — seed fixed for reproducibility of the
  initial Sobol sequence and of the acquisition optimizer's later random
  restarts.
- Initial design size: `n_init = 2 * (dim + 1) = 14`, GP-BO's own inherited
  default (a standard rule of thumb for how many points are needed before
  a GP fit is meaningful — not a free hyperparameter I am tuning here,
  since it is intrinsic to this wrapper's startup logic, drawn from its
  own internal `SobolSearch(bounds=bounds, seed=seed)`).
- No GP is fit and no acquisition function is optimized during this phase
  — `GPBO.ask()` returns raw Sobol points for as long as
  `len(y_) < n_init`.
- Other GP-BO hyperparameters (`num_restarts=10`, `raw_samples=512` for
  the later acquisition-optimization phase) are inherited library/tutorial
  defaults (BoTorch's own official tutorials) and are not exercised in
  this step since no acquisition function is used yet; they will be
  reconsidered once that phase begins.

### Step length

- 14 objective evaluations (the full initial Sobol design).
- Pause at cumulative evaluation count = 14.
- Why this length: 14 is exactly GP-BO's own required initial-design size
  for `dim=6` — fitting a GP meaningfully before this many points would be
  premature (too little data to estimate lengthscales/noise), and drawing
  more would eat into the ~86 remaining evaluations that are better spent
  letting the acquisition function choose targeted points once a usable
  surrogate exists. This is a natural, method-defined boundary for a
  single step, not an arbitrary batch size.

### What I want to learn at the checkpoint

- The observed range and spread of `y` across the 14 Sobol points (scale
  of the objective, rough indication of smoothness/ruggedness from how
  much consecutive nearby Sobol points' values vary).
- The best value found during this design phase, to track progress
  against.
- Whether all 14 evaluations completed without error (confirms the
  harness/optimizer integration is working correctly before committing
  more of the budget to it).

### Conditions for pausing sooner

- None expected during this fixed initial-design phase other than an
  execution error; Sobol design points are fixed in advance by the method
  and are not adaptively reconsidered mid-phase.
