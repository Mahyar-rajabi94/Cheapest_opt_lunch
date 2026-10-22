# Roadmap — Initial Step

## Understood problem

- Black-box objective `fn(x)`, sealed — identity unknown and not to be
  investigated beyond the documented harness interface.
- Dimension: 30. Bounds: `[-100, 100]` per coordinate (range 200 per axis,
  so the search volume is effectively unconstrained-feeling but finite).
- Direction: minimize.
- Noise: deterministic (`NOISY=False` reported by harness / stated by user)
  — no need for replication or noise-robust acquisition; every evaluation is
  exact and reusable as-is.
- Budget: **100** `evaluate()` calls total, hard-enforced by the harness.
  This is a very small budget relative to dimension (100 evals for a 30-D
  problem is roughly 3.3x dim — far below what most derivative-free methods
  want, e.g. a default CMA-ES population alone would be ~14 points/generation,
  giving only ~7 generations across the whole budget). This scarcity is the
  dominant design constraint for the whole campaign: every phase must be
  deliberately small, and no method should be given more evaluations than it
  can plausibly turn into information or improvement.
- Interface: `harness.evaluate(x)` is the only source of real `y` values;
  `harness.set_active_optimizer(name)` tags method per call; `harness.save()`
  writes `trace.csv`/`meta.json` (also auto-called at budget exhaustion).
  There is no `ask/tell` layer provided for this task — I drive `evaluate()`
  directly from whatever optimizer logic I choose.

## Assumptions

- No prior information about the function's shape (unimodal vs multimodal,
  separable vs coupled, smooth vs rugged) is available or sought. The first
  step is explicitly designed to gather this information cheaply before
  committing to a refinement method.
- Treating the box uniformly (no coordinate is assumed more important than
  another) is the only defensible prior given a sealed black box.
- Given the extreme scarcity of budget relative to dimension, a short,
  information-dense first step (a space-filling design) is more valuable
  than jumping straight into a local optimizer from one arbitrary start
  point, since a single blind start risks wasting a large fraction of the
  total budget in a poor basin.

## Step 1 — method, hyperparameters, initialization

**Method:** Scrambled Sobol low-discrepancy sequence (via
`scipy.stats.qmc.Sobol`, scipy 1.17.1) over the full box
`[-100, 100]^30`, scaled with `qmc.scale`.

**Hyperparameters / initialization:**
- `qmc.Sobol(d=30, scramble=True, seed=0)` — scrambling avoids the
  known structure/alignment artifacts of a raw Sobol net; seed fixed for
  reproducibility. These are scipy's library defaults for the constructor
  aside from `d`, `scramble`, `seed`, which are the only arguments relevant
  here (no further tuning parameters are exposed that matter at this scale).
- Sample size `n = 16` points, generated via `Sobol.random(16)` (16 is a
  power of 2, which is what gives Sobol sequences their best uniformity
  guarantees for a fixed prefix length).
- Points are scaled from the Sobol unit cube to `[-100, 100]^30` with
  `qmc.scale`, then evaluated in the generated order.
- `harness.set_active_optimizer("sobol_init")` before the first call.

**Why this suits the problem:** A space-filling design spends the very
first, most information-poor evaluations (no data yet to exploit) on
broad coverage rather than a single guess, which is the standard
justification for quasi-random initialization before any model-based or
evolutionary refinement. 16 points is deliberately small — about half of
one default CMA-ES generation at this dimension — so that the bulk of the
100-evaluation budget remains for whatever refinement method the Sobol
results justify; this step is reconnaissance, not the main campaign.

## Step length and pause point

- Maximum additional evaluations this step: **16**.
- Cumulative evaluation count at which I will pause: **16** (out of 100).
- No early-stop condition is defined for this step: all 16 points are
  cheap, pre-committed, and individually uninformative about whether to
  stop early (there's no partial-batch signal worth checking before the
  design is complete), so the step runs to its full planned length unless
  an unexpected error occurs.

## What I want to learn / evidence to inspect at the checkpoint

- The range and spread of observed `y` values across the 16 points
  (crude sense of scale/curvature of the landscape and whether it varies
  sharply over the box).
- The best point found and its margin over the rest (is there one clear
  promising region, or are values fairly flat/uninformative at this
  resolution?).
- Whether evaluations are fast/cheap (informs whether a larger subsequent
  local-search step is affordable) — recorded via `eval_seconds` in the
  trace.
- This evidence will directly inform the choice of Step 2's method
  (e.g., seeding a local/evolutionary optimizer at the best Sobol point,
  vs. expanding the space-filling design further if the landscape looks
  highly rugged/uninformative at this density) and its hyperparameters
  (e.g., initial step size proportional to observed variation).
