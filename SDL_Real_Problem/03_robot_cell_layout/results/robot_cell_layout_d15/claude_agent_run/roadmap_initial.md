# Roadmap — task031 (15D robot cell layout, minimize, deterministic, budget=100)

## Problem summary (from stated parameters only)
- 15-D, box-bounded (bounds read from `harness.bounds`), minimize, deterministic
  (no replication needed), budget = 100 `evaluate()` calls total.
- Coordinates 0-8: continuous geometric/kinematic parameters (varied box widths,
  several on the order of radians, a couple of very narrow intervals e.g.
  dims 0,3,4 spanning <0.5 units — these likely need finer step sizes than a
  generic CMA-ES default would pick without box-aware scaling).
- Coordinates 9-14: each in [0,1] and read by the model only through a
  `>= 0.5` threshold, i.e. 6 effectively-binary IK-branch flags -> 2^6 = 64
  discrete combinations, each inducing a different (possibly discontinuous)
  continuous sub-landscape on coordinates 0-8.
- Landscape is explicitly stated to have real discontinuities at branch
  switches plus heavily penalized infeasible/collision regions. This rules
  out assuming smoothness globally; a single local optimizer started from one
  point risks being trapped in a bad discrete branch or a penalty plateau.

## Planned method sequence and budget split (100 evals total)

1. **Phase 1 — mixed space-filling scan, ~28 evals** (`sobol_scan`):
   Use `SobolSearch` over the 9 continuous dims (dims 0-8) for coverage, paired
   with a systematic/random sweep over the 6 binary flags so that most of the
   64 flag combinations get at least one sample (28 draws covers a reasonable
   fraction of the 64 combos while also varying the continuous coordinates).
   Purpose: get a first honest read of (a) which flag combinations are even
   feasible/competitive, since discontinuities mean a smooth optimizer could
   easily miss a good branch, and (b) a rough location for the continuous
   optimum within the best branch(es), without committing to one region too
   early.

2. **Phase 2 — CMA-ES refinement, ~50 evals** (`cma_es_full15d`):
   Warm-start is done implicitly by biasing Phase 1's best point as a strong
   prior: since this project's `CMAES` wrapper always starts at the box
   midpoint, instead run CMA-ES on the **full 15-D box** (letting the flag
   dims float continuously — CMA-ES can push them toward 0 or 1 itself,
   which is a legitimate continuous relaxation of the discrete choice and
   lets gradient-like information in the "soft" region near 0.5 be used
   before the hard threshold is reached) but discard its default x0 and reset
   with `es.ask()`'s distribution re-centered/manually seeded near the best
   Phase-1 point to avoid wasting evals re-finding what Sobol already found.
   If the pycma wrapper proves awkward to re-center cleanly in a way that
   stays consistent with its internal bookkeeping, fall back to seeding by
   simply including the Phase-1 best point as a hint only in choosing which
   fixed flag combination to carry into Phase 3, and let CMA-ES run from its
   own default start — this is a legitimate, declared fallback, not a silent
   deviation.
   Rationale for CMA-ES as the main workhorse: budget of ~50 evals in 15D is
   small but CMA-ES degrades gracefully, handles box constraints natively,
   and its population-based search is more robust than a single gradient
   step to the discontinuities at branch boundaries.

3. **Phase 3 — local polish, ~22 evals** (`bobyqa_polish` or `nelder_mead_polish`):
   Take the best point after Phase 2, round coordinates 9-14 to the nearest
   {0,1} (since the model only reads them as booleans, continuous values near
   but not exactly at an extreme add no information beyond the threshold —
   fixing them turns the remaining problem into a clean 9-D continuous local
   search) and run BOBYQA (derivative-free, trust-region, good at exploiting
   local curvature cheaply) on the 9 continuous dims only, bounds-restricted,
   for the remaining budget. If BOBYQA's trust region collapses or raises
   before using the full remaining budget, fall back to Nelder-Mead restarts
   from the same point for the remainder.

## Explicit deviation criteria
- If, after Phase 1 (28 evals), the single best observed y is **not** at
  least as good as the median of the Phase-1 sample (i.e. Sobol got lucky
  only at the edge), extend Phase 1 by up to 10 more evals before moving to
  Phase 2, stealing the 10 evals from Phase 2's allocation.
- If, during Phase 2, the running best has **not improved for 20 consecutive
  evaluations**, stop Phase 2 early and move directly into Phase 3 with
  whatever budget remains (don't burn the rest of the planned 50 chasing a
  plateau).
- If Phase 3's local optimizer (BOBYQA) raises an exception, stalls (reports
  trust-region radius below a tiny threshold) or otherwise exits before
  using its allocated evals, switch to Nelder-Mead restarts from the best
  point found so far for the remaining budget, logging this in the
  changelog.
- If at any point the running-best value looks like it is stuck in an
  obviously penalized/infeasible region (e.g. the best y across an entire
  phase barely varies and is far worse than other phases' best), treat that
  as evidence the current flag-combination/region is infeasible and
  re-sample a handful of fresh Sobol/random points with different flag
  combinations instead of continuing to refine there — logged as a
  deviation with the observed evidence.
- Any change to this sequencing, the phase boundaries above, or a
  hyperparameter committed here, will be logged as a changelog entry citing
  the evaluation index and the observed data that triggered it.

## What will NOT be done
- No inspection of `objective.py` or any harness source beyond the public
  `AgentTaskHarness` interface.
- No attempt to identify or reverse-engineer the underlying kinematic model.
- No exceeding the 100-call budget; no second harness instance to reset the
  counter.
