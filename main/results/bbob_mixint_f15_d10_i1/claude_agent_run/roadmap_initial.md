# Roadmap — sandbox_task026

## Problem summary
- dim = 10, minimize, deterministic (no noise), budget = 100 evaluate() calls.
- Bounds: x0∈[0,1], x1∈[0,1] (integer); x2∈[0,3], x3∈[0,3] (integer);
  x4∈[0,7], x5∈[0,7] (integer); x6∈[0,15], x7∈[0,15] (integer);
  x8∈[-5,5], x9∈[-5,5] (continuous).
- Mixed-integer, piecewise-constant along x0..x7. Sealed black box, identity unknown.

## Planned method sequence and budget split (100 total)

1. **Sobol quasi-random initial design — 20 evals** (`sobol_init`)
   Low-discrepancy coverage of the full 10-d mixed box. With only 100 evals
   total and 8 discrete axes (effective combinatorial size up to
   2·2·4·4·8·8·16·16 ≈ 2.1M cells), an initial design cannot hope to cover
   the discrete grid densely, but it gives a reasonable, unbiased read on
   where promising regions of the joint space are (both which integer
   settings and which continuous sub-region) before committing further
   budget. Integer coordinates are rounded to the nearest integer in-bounds
   before each evaluate() call (the harness would round internally anyway;
   rounding explicitly keeps the recorded x consistent with what is
   actually evaluated and avoids wasting distinct Sobol draws that collapse
   to the same integer cell).

2. **CMA-ES on the full continuous relaxation — 50 evals** (`cma_es`)
   CMA-ES is seeded at the box midpoint/quarter-width sigma (its own
   standard default) and run as the main global refinement step. At each
   `ask`, the raw continuous proposal is rounded on x0..x7 only for the
   purpose of calling evaluate(); the value returned is told back to
   CMA-ES against the *original* (unrounded) proposed point. This is the
   standard "evaluate the repaired point, tell the sampled point" trick for
   mixed-integer CMA-ES: it lets the covariance/step-size adaptation keep
   operating on a smooth continuous distribution (so it can still detect
   gradient-like structure on x8/x9 and coarse trends on the integer axes)
   while every function value is still a real, validly-rounded evaluation.
   50 evals ≈ 4-5 pycma generations given the default population size for
   d=10 (popsize ≈ 4+3·ln(10) ≈ 11), enough for a few adaptation steps but
   not full convergence — acceptable given the tiny total budget.

3. **Local refinement around the incumbent best — 30 evals**
   (`local_coord_search`)
   A coordinate-descent / hill-climbing pass seeded at the best point found
   so far (from steps 1-2): for each of the 8 integer coordinates, try
   neighbors at ±1 (clipped to bounds) and keep any improvement
   immediately (greedy, first-improvement, one or more sweeps as budget
   allows); interleave small random/local perturbations on the two
   continuous coordinates (x8, x9) using a shrinking step size, since the
   piecewise-constant structure along the integer axes means CMA-ES's
   continuous step-size adaptation is a poor fit for final polishing there,
   while a direct discrete neighbor search is the natural tool for a flat,
   piecewise-constant landscape. Remaining budget (after the sweep) is
   spent on repeating sweeps / further continuous polishing around the
   current best.

## Reasoning
- No noise → no need to replicate points; every evaluation is pure
  information, so splitting budget across a global quasi-random phase, a
  global continuous optimizer, and a discrete local-search polish makes
  sense: broad coverage first, adaptive refinement second, exploit-the-best
  discrete structure last.
- 8 of 10 axes are integer-valued and the function is explicitly
  piecewise-constant along them: a pure continuous optimizer (CMA-ES,
  BOBYQA, L-BFGS-B, Nelder-Mead) run naively would waste evaluations
  probing points that round to the same cell, or fail to make progress
  once step sizes shrink below 1 on those axes. Combining a continuous
  global method (for the 2 genuinely continuous axes and coarse structure)
  with an explicit discrete coordinate search (for final exploitation of
  the integer axes) is the standard mixed-integer approach available here.
- Budget (100) is small relative to dimension and the discrete combinatorial
  size, so the plan favors broad exploration (Sobol + CMA-ES global search)
  over long local refinement, but still reserves a meaningful tail (30
  evals) for discrete exploitation once a good basin is identified, since
  piecewise-constant landscapes often have single-coordinate-flip
  improvements that a continuous optimizer alone would miss.

## Explicit deviation criteria
- If, after the 20-eval Sobol phase, the best value found has not improved
  at all relative to the single best Sobol point by evaluation 40 (i.e.
  20 evals into the CMA-ES phase), restart CMA-ES from the best point found
  so far (instead of the box midpoint) with a reduced sigma0, treating this
  as evidence the box-midpoint start was a poor basin.
- If CMA-ES's step size (sigma) collapses below 1e-3 before its 50-eval
  budget is exhausted (a sign of premature convergence), stop it early and
  shift the remaining evals from phase 2 into phase 3 (local search), since
  further CMA-ES evals would be wasted.
- If local coordinate search (phase 3) finds an improving move on its first
  full sweep over all 8 integer coordinates, it will keep sweeping
  (greedy coordinate descent) until either a full sweep produces no
  improvement or the budget for phase 3 is exhausted; if a sweep completes
  with no improving move before the phase-3 budget is used up, remaining
  evals in that phase are redirected to random local perturbations of
  (x8, x9) around the incumbent best instead of further integer sweeps.
- Any such deviation will be logged in changelog.md with the evaluation
  index at which it occurred and the evidence that triggered it.
