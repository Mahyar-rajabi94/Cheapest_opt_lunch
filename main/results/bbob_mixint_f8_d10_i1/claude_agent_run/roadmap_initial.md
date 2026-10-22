# Roadmap — sandbox_task025

## Problem summary
- dim = 10, minimize, deterministic (no noise), budget = 100 evaluate() calls.
- Mixed-integer: x0..x7 are integer-valued on stated per-coordinate bounds
  (x0,x1 in {0,1}; x2,x3 in {0,1,2,3}; x4,x5 in {0..7}; x6,x7 in {0..15});
  x8,x9 are continuous in [-5,5].
- The objective is piecewise-constant along the 8 integer axes (the harness
  rounds internally), so no local gradient information is available on those
  axes — only exact function evaluations at each integer value. The number of
  integer grid cells (2*2*4*4*8*8*16*16 ≈ 4.2M) is far larger than the
  budget, so brute force / full grid search is infeasible. x8,x9 have a
  normal continuous response and are the only axes where a continuous
  optimizer's gradient-free local model is meaningful.

## Method plan and budget split (100 evals total)

1. **Space-filling exploration (~20 evals, optimizer tag `sobol_init`).**
   Use the provided `sobol_search` wrapper over the full 10-D box (bounds as
   given), rounding x0..x7 to nearest integer before each evaluate() call.
   Goal: get a rough read of the landscape and a good starting point for
   local search, since we have no prior on this black box.

2. **Integer coordinate descent (~45 evals, optimizer tag `int_coord_descent`).**
   Starting from the best point found in phase 1, cycle through x0..x7 one
   coordinate at a time (randomized order each full pass). For each
   coordinate, evaluate a small set of candidate values (immediate neighbors
   of the current value, e.g. ±1, ±2 within bounds, capped to a handful of
   candidates to respect budget) while holding all other coordinates (both
   integer and the current continuous x8,x9) fixed, and greedily move to the
   best improving value found. Repeat passes until budget for this phase is
   used or a full pass yields no improvement. This directly handles the
   piecewise-constant/discrete nature of x0..x7 since it works by exact
   comparison of evaluated values rather than gradients.

3. **Continuous refinement of x8,x9 (~25 evals, optimizer tag
   `continuous_refine`).** With x0..x7 frozen at the best integer
   configuration found so far, run Nelder-Mead (`nelder_mead` wrapper)
   restricted to the 2-D subspace (x8, x9) only, starting from the current
   best (x8,x9), bounded to [-5,5]^2. Nelder-Mead is a good fit for a cheap
   2-D continuous local search with a tiny budget and no gradient
   requirement.

4. **Final polish (~10 evals, optimizer tag `final_polish`).** Spend any
   remaining budget alternating one more short pass of integer coordinate
   descent (in case the refined continuous coordinates change which integer
   neighbors look good) and/or a few more Nelder-Mead steps on (x8,x9),
   whichever phase showed more recent improvement. Reserve enough evals to
   stay within the 100-call budget; stop early if budget runs out.

## Reasoning
- With only 100 evaluations and 10 dimensions (8 of them discrete with large
  combinatorial range), neither a pure population-based continuous optimizer
  (CMA-ES/TuRBO/DE) nor a pure discrete brute-force search is a good match
  on its own: population optimizers waste evaluations treating discrete axes
  as if they were smooth, and brute force cannot cover the discrete grid.
  Splitting the problem — cheap global scan, then exact discrete coordinate
  descent, then continuous polish on the only genuinely continuous axes — is
  the most sample-efficient structure given the stated problem shape.
- Coordinate descent on the integer axes is chosen over e.g. a discrete
  CMA-ES variant because it needs no extra dependency, directly respects the
  piecewise-constant structure, and is cheap per step (one eval per
  candidate value tried).
- Nelder-Mead is chosen for the continuous sub-problem because it is simplex
  based (no gradient/surrogate needed), is well suited to very low dimension
  (2-D here) and small budgets, and the project's optimizer package already
  provides a wrapper for it.

## Explicit deviation criteria
- If phase 1 (Sobol, 20 evals) shows essentially no variation in y (e.g. the
  range of observed y values is below 1e-6), treat the function as
  effectively flat/insensitive in the explored region and switch remaining
  budget to a wider random search instead of local refinement (since local
  descent would have nothing to climb).
- If an entire pass of integer coordinate descent (one sweep over all 8
  coordinates) produces zero improvement in the best-so-far y, stop that
  phase immediately (even if its budget allocation isn't exhausted) and move
  on to continuous refinement early, logging the switch.
- If continuous refinement (Nelder-Mead) shows no improvement over its first
  10 evaluations, stop it early and spend the rest of that phase's budget on
  one more integer coordinate-descent pass instead.
- If at any point fewer than 5 evaluations remain, stop all search and keep
  the current best point rather than starting a new phase.
- Any such change will be logged in changelog.md with the evaluation index
  at which it occurred and the evidence observed.
