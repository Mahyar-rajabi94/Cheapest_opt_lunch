# Roadmap — initial step

## Understood problem

- Black-box objective `fn(x)`, sealed; identity not to be investigated.
- Dimension: 5. Bounds: `[-5, 5]` per coordinate (a 5-dimensional box).
- Direction: **minimize**.
- Noise: the objective is **noisy** — repeated `evaluate()` calls at the same
  `x` can return different `y`. This must be treated as real observation
  noise, not a deterministic function value; single-point readings (and
  especially finite-difference gradient estimates) cannot be trusted at face
  value.
- Total evaluation budget: **100** calls to `harness.evaluate()`, hard-enforced
  by the harness (raises `BudgetExhausted` past the limit). This is a small
  budget (~20x dimension), so the plan favors methods that are
  sample-efficient and that model (rather than ignore) noise, over
  population-based methods that typically need hundreds–thousands of evals
  per dimension to show their advantage.
- Available tooling: this project's own `ask`/`tell`/`best` optimizer
  wrappers (`src/optimizers/*`), including `SobolSearch` (scrambled Sobol'
  low-discrepancy sampler) and `GPBO` (BoTorch `SingleTaskGP` +
  `LogExpectedImprovement`, analytic, single-candidate). `GPBO`'s own
  docstring states it is intended for budgets "of order 10 x dim, not
  1000+ x dim" — a good match for this task's budget of 100 at dim 5
  (20x dim). It also documents an initial design of `2 * (dim + 1)` Sobol
  points before the first acquisition-driven point, matching the standard
  rule of thumb for how much data a GP needs before fitting is meaningful.

## Key assumption

GP-BO's `SingleTaskGP` fits a (homoskedastic) observation-noise term as part
of its likelihood, so it does not simply interpolate a single noisy
observation as if it were exact — this is the main reason GP-BO is preferred
here over, e.g., plain CMA-ES or Nelder-Mead, which are more easily misled by
a single lucky/unlucky noisy draw at small budgets. This is a property of
`SingleTaskGP`'s default likelihood, not a claim I will reverify at every
step; I will still watch for best-so-far jumps driven by an apparently noisy
single observation.

## Step 1 plan

**Method:** Sobol' low-discrepancy quasi-random sampling, via this project's
`SobolSearch` wrapper (`src/optimizers/sobol_search.py`), used directly
(not yet through `GPBO`) so that this step's purpose — pure space-filling
exploration, no model fit yet — is explicit and auditable on its own.

**Hyperparameters for this step:**
- `n = 12` points in one batch = `2 * (dim + 1)` = `2 * (5 + 1)`, which is
  the initial-design size `GPBO` itself uses internally (an inherited
  default from `src/optimizers/gp_bo.py`, chosen there as "a standard rule
  of thumb for how many points are needed before a GP fit is meaningful").
  12 is a power of 2 minus 4; Sobol' balance properties are best at
  cumulative counts that are powers of 2, so after this batch I may draw a
  few more Sobol points later to round the cumulative total up to 16 if
  that still looks useful — but that decision is deferred, not scheduled
  now.
- `seed = 0` for reproducibility of the scrambled sequence (`SobolSearch`'s
  own default behavior is `scramble=True`; a seed is supplied only so the
  run is reproducible, not to bias the sampler).
- `bounds = [[-5, 5]] * 5` (task-given).

This step has no finite-difference probes, no repeated points, and no
optimizer internal state beyond the Sobol generator's own sequence position
(which I will carry forward if more Sobol points are drawn later).

**Step length:** 12 evaluations. Cumulative count at pause: **12 / 100**
(88 remaining).

**Why this length:** This is the minimum size this project's own GP-BO
wrapper considers sufficient before attempting its first model fit, and
Sobol points here are "free" information (every point here contributes to
whatever method follows — including a future GP fit) rather than overhead
specific to one candidate method. Going further before looking at the data
would commit budget without any evidence to justify the choice of batch
size.

## What I want to learn at the checkpoint

- The rough scale/range of `y` values returned, and whether they vary
  smoothly across the sampled points or look dominated by noise (e.g. by
  comparing the best few observed values against the overall spread).
- Whether the optimization landscape shows any visible structure (e.g. one
  region of the box markedly better than others) worth exploiting next, or
  whether it looks closer to flat/noise-dominated at this scale.
- Based on this, whether to proceed directly to GP-BO's acquisition-driven
  phase (feeding these 12 points into a fresh `GPBO` instance via `tell`),
  draw a few more Sobol points first (e.g. to reach 16, a power of 2), or
  take a cheap noise-estimation sub-step (one or two repeated evaluations at
  the same point) before trusting any acquisition function's exploitation
  decisions.

No condition for pausing this step early: 12 is already the minimum useful
batch size, and all 12 points are predetermined by the Sobol sequence before
evaluation begins, so there is nothing to learn mid-batch that would justify
stopping at e.g. 6.
