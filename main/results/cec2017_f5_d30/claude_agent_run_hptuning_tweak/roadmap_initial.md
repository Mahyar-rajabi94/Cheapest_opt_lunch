# Roadmap — Initial Step

## Understood problem
- Black-box objective, sealed: identity unknown and not to be investigated.
- Dimension: 30. Bounds: [-100, 100] per coordinate (box-constrained).
- Direction: minimize.
- Noise: deterministic (`NOISY=False`); repeated calls at the same `x` are
  assumed to return identical `y`, so no replication is needed for
  denoising.
- Total evaluation budget: 100 `evaluate()` calls, hard-enforced by
  `AgentTaskHarness`. This is a very small budget relative to dimension
  (100 evaluations for 30 dimensions — roughly 3.3 evaluations per
  dimension). Classical rules of thumb for even a single local
  model-based step (e.g. BOBYQA needs ~`dim+2` = 32 points just to build
  its first quadratic/linear model; a full factorial or dense grid is
  utterly out of reach) make clear that most of the budget must go toward
  a method that extracts information efficiently per evaluation, not
  toward broad exploration.
- Assumption: since nothing is stated about the function's smoothness,
  multimodality, or separability, I cannot assume unimodality or convexity.
  A short, cheap initial scan is useful to get a rough sense of the
  landscape (scale/range of y, whether there's a usable gradient-like
  trend) before committing most of the budget to a specific local/global
  method. I will not use more than a small fraction of the budget for this
  scan, since with dim=30 even a "small" space-filling design cannot
  meaningfully cover the space — its purpose is calibration, not coverage.

## Step 1 — method and hyperparameters
**Method:** Sobol' low-discrepancy quasi-random sampling
(`src/optimizers/sobol_search.py`, wraps `scipy.stats.qmc.Sobol`,
`scramble=True`).

**Why this method for this step:** Sobol gives more even coverage of the
30-D box than uniform random search for the same number of points, which
is exactly what a short calibration scan needs: a representative first
look at the range and spread of y-values across the domain, without
concentrating points in any one region. It needs no tuning and makes no
assumption about smoothness or unimodality, which suits the total absence
of prior evidence at this point.

**Hyperparameters:**
- `bounds`: `harness.bounds` (the task's own [-100, 100]^30 box), used
  unmodified.
- `seed=0`: arbitrary but fixed, for reproducibility of this run.
- `scramble=True`: the module's own stated default (not tuned).
- Batch size requested from `ask`: 16. The optimizer's docstring notes
  Sobol sequences have their best balance properties when the *cumulative*
  count drawn is a power of 2, so 16 (2^4) is chosen over a non-power
  count. 16 is also small enough (16% of the total budget) to leave the
  large majority of the budget for a more targeted method once this scan
  gives some evidence, while being large enough, relative to dim=30, to
  see a nontrivial spread of y across differently-located points (more
  than the bare minimum needed to say anything about scale/spread).

No other hyperparameters are exposed by this optimizer (see its
docstring: "Standard library defaults only; no tuning").

## Step length and checkpoint
- This step issues exactly 16 `evaluate()` calls (one Sobol batch of
  n=16), bringing the cumulative evaluation count to 16 out of 100 (16
  used, 84 remaining).
- I will pause earlier only if something anomalous happens, e.g. the
  objective raises an error, returns non-finite values, or the harness
  reports the budget already exhausted — none of which is expected.

## What I want to learn at the checkpoint
- The range and spread of observed y-values (min, max, median, std) —
  i.e. the rough scale of the objective and how much it varies across the
  box, which bears on step-size choices for a subsequent local method.
- The best point found so far and whether it clusters near the box
  interior or an edge (informs where to center a subsequent local search
  if a single promising region already stands out).
- Whether y-values look smooth/structured in any way I can detect from so
  few points (e.g. consistent ordering relative to distance from center)
  versus effectively unstructured given the fact that 16 points in 30-D
  can't resolve fine structure — this will temper how much weight I put
  on any apparent pattern.
- This evidence will inform the next decision (continue Sobol further,
  switch to a local solver such as CMA-ES or BOBYQA seeded at/near the
  best point found, or something else) — no method beyond this step is
  preselected.
