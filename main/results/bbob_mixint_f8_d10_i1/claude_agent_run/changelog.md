## Evaluation 65
**Change:** Stopped integer coordinate descent early (before its full 45-eval allocation) and moved on to continuous refinement
**Reason:** A full sweep over all 8 integer coordinates (step sizes 1 and 2) produced zero improvement on best-so-far y (still 1.83201), matching the roadmap's explicit stopping rule for this phase.

## Evaluation 65
**Change:** Used a bounded pattern/coordinate search on (x8, x9) instead of the roadmap's planned Nelder-Mead wrapper
**Reason:** The project's NelderMead wrapper always initializes at the midpoint of the full bounds box and searches all dim coordinates; it offers no way to fix x0..x7 at the integer-descent result while optimizing only x8,x9. A direct bounded pattern search on the 2-D continuous subspace achieves the same gradient-free local-refinement goal without that limitation.

## Evaluation 77
**Change:** Stopped continuous refinement early and fell back to an extra integer coordinate-descent pass for the rest of its budget
**Reason:** No improvement over best_y=1.80212 after the first 10 continuous-refine evaluations on (x8,x9), matching the roadmap's stopping rule for this phase.

