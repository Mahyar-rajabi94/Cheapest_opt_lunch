# Roadmap — sandbox_task027

## Problem summary
- dim = 10, minimize, deterministic (no noise), budget = 100 evaluate() calls.
- x0..x7 are INTEGER-valued (bounds: [0,1],[0,1],[0,3],[0,3],[0,7],[0,7],[0,15],[0,15]).
- x8, x9 are continuous in [-5,5].
- Objective is sealed; piecewise-constant along the 8 integer axes (no usable
  gradient there), smooth-ish (unknown) along x8/x9.

## Why a custom hybrid instead of an off-the-shelf continuous optimizer
A budget of only 100 evaluations for a 10-D mixed-integer problem is small
relative to what CMA-ES/TuRBO/BOBYQA etc. are tuned for, and all of them
assume a (locally) continuous response — they were not designed to reason
about 8 axes that are flat between integers. Rather than feed them rounded
inputs and hope the resulting discontinuities don't break their internal
models (step-size adaptation, trust regions, GP length-scales), I will use
transparent, budget-cheap custom logic for the integer block, and reserve a
small amount of simple local continuous search for x8/x9. This keeps every
evaluation interpretable and lets me log exactly why each deviation happens.

## Planned method sequence and budget split (target ~100 evals)
1. **Phase A — global scan (`sobol_scan`), ~20 evals.** Scrambled Sobol
   points over the full 10-D box (via `scipy.stats.qmc.Sobol`), with x0..x7
   rounded to nearest integer before evaluation. Purpose: get a
   space-filling look at the landscape and a reasonable starting incumbent,
   since with only 100 evals a purely local method risks starting in a bad
   basin.
2. **Phase B — integer coordinate descent (`intcoord_descent`), ~45 evals
   budget cap.** Starting from Phase A's best point, cycle through the 8
   integer coordinates in order; for each, try neighbors at ±1 (clipped to
   bounds) around the current incumbent value, keep the best found, and
   move to the next coordinate. Repeat full passes until a complete pass
   produces no improvement, or the phase budget cap is hit. This directly
   respects the piecewise-constant structure (exhaustive local
   neighbor-check rather than a gradient).
3. **Phase C — continuous compass/pattern search on x8,x9
   (`continuous_pattern_search`), ~20 evals budget cap.** From the current
   incumbent, try ±step along x8 and ±step along x9 (step starts at 1.25,
   i.e. 1/4 of the ±5 half-range), accept any improvement, halve the step
   on a failed pass, stop the phase when step < 0.05 or the cap is hit.
4. **Phase D — alternate B/C with remaining budget (~15 evals), then a
   final polish.** If both B and C have converged (full pass / step
   collapse) before budget exhausts, re-run a short B pass then a short C
   pass with whatever budget remains, since the two blocks can interact
   (a better continuous point may open up a better integer neighbor and
   vice versa). Any leftover evals (<5) are spent re-checking the single
   best-looking untried neighbor of the final incumbent.

## Explicit deviation criteria
- If Phase A's 20 samples already show one point dominating by a wide
  margin (e.g., best is better than the 2nd-best by more than the spread of
  the rest), I may shorten Phase A's remaining budget and move to Phase B
  early — logged in changelog.md with the observed margin.
- In Phase B, if a full 8-coordinate pass yields zero improvement, stop
  Phase B immediately (do not keep spending budget) and move to Phase C,
  even if under the ~45-eval cap.
- In Phase C, if the step size falls below 0.05 before the ~20-eval cap is
  reached, stop Phase C immediately and move to Phase D.
- If at any point remaining budget drops below 10, skip directly to a final
  best-neighbor check (last step of Phase D) rather than starting a new
  full phase.
- Any hyperparameter change (step sizes, neighbor radius, pass limits) made
  mid-run based on observed data will be logged as a changelog entry citing
  the evaluation count and the evidence.
