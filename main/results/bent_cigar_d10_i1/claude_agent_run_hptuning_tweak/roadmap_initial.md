# Roadmap — initial step only

## Understood problem

- Sealed black-box objective `fn(x)`, `x` a length-10 real vector, each
  coordinate bounded to `[-5, 5]`.
- Direction: **minimize**.
- Deterministic (noise-free): repeated calls at the same `x` are assumed to
  return the same `y`, so there is no need for replication or noise-robust
  acquisition.
- Hard budget: **100** total `evaluate()` calls, enforced by
  `AgentTaskHarness`. The step limit below counts every objective call
  (initialization points included), not optimizer iterations.
- No information about the function's identity, shape, or separability is
  available or will be sought; all decisions below come only from the box
  bounds/dimension and whatever the first step's observations show.

## Assumption worth surfacing

- I'm assuming "deterministic" as stated means exactly repeatable floating
  point output for the same `x` (not just "low noise"). This matters only if
  I later consider re-evaluating a point to check stability — I do not plan
  to spend budget on that, since the task states it plainly.

## Step 1 — method and setup

**Method:** `SobolSearch` (`src/optimizers/sobol_search.py`), a scrambled
Sobol' low-discrepancy quasi-random sequence over the box bounds, via scipy's
`qmc.Sobol` (scipy's own defaults: `scramble=True`). This is a pure
space-filling design, chosen for the very first step because, at budget=100
in 10 dimensions (10x the dimension), no evidence yet exists to justify any
model-based or local method (GP-BO, CMA-ES, BOBYQA, TuRBO all need either an
initial design or a committed basin to work from). A low-discrepancy design
covers the 10-D box far more evenly per-sample than uniform random search,
which matters when only a handful of points can be afforded before the first
real decision.

**Hyperparameters:**
- `bounds`: the task's own `[[-5, 5]] * 10` (from `harness.bounds`).
- `seed=0`: fixed for reproducibility; no other tuning choice is exposed by
  this wrapper (scipy's Sobol defaults are used as-is, per the wrapper's own
  docstring).
- Batch size for this step: **16** points in one `ask(16)` / `tell` call.
  16 is a power of 2, which the wrapper's docstring notes gives Sobol' its
  intended even-coverage guarantee for the *cumulative* count drawn so far
  (here, cumulative = count drawn, since this is the first draw).

No other hyperparameters apply to this method.

## Step length and pause point

- Maximum additional evaluations this step: **16**.
- Pause at cumulative evaluation count: **16** (84 remaining after).
- Rationale for length: 16/100 = 16% of the total budget is a modest
  down payment for a first look — enough points in 10-D to see the rough
  scale and spread of `y` and to seed a subsequent model-based method
  (e.g. GP-BO's own recommended initial design is `2*(dim+1) = 22`, so 16
  is a deliberately smaller, cheaper first increment, not a full design)
  without committing a large fraction of the 100-call budget before any
  real decision is made.

## What I want to learn / evidence to inspect at the checkpoint

- The observed range, mean, and spread of `y` across the 16 Sobol points —
  a first, crude read on scale and whether the landscape is flat, has a
  clear gradient toward one region of the box, or is highly variable
  (suggestive of multimodality or a narrow basin).
- Whether any points are much better than the rest (a strong localized
  signal, favoring a local/model-based method next) versus a fairly uniform
  spread (favoring either a larger global design or an exploration-heavy
  method next).
- No early-pause condition applies within this first 16-point batch — it is
  already the minimum sensible look, and `ask`/`tell` here only returns
  control after the full batch (no intermediate stopping point within a
  single Sobol batch call).
