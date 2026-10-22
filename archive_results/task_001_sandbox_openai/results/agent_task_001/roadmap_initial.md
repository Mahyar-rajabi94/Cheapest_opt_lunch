# Initial optimization roadmap

## Planned sequence and budget

1. **SobolSearch: evaluations 1–256.** Use a deterministic low-discrepancy space-filling design to cover the ten-dimensional box uniformly and establish a broad baseline.
2. **CMA-ES: evaluations 257–1500.** Start the standard CMA-ES wrapper from the box midpoint with its default population and adaptation, using the Sobol results only as a diagnostic baseline rather than injecting external points into the optimizer. The large middle allocation is intended to let covariance adaptation exploit promising regions while retaining global search.
3. **BOBYQA: evaluations 1501–2000.** Use the remaining 500 evaluations for derivative-free trust-region refinement from its standard midpoint start. This is within the documented practical limit of the wrapper and provides an independent local-search phase at the end.

## Reasoning before observing data

The objective is deterministic, bounded, and ten-dimensional, but its geometry is unknown. A low-discrepancy initial design is a robust way to avoid placing all early evaluations in one random cluster. CMA-ES is appropriate for a smooth or moderately rugged continuous objective because it can learn scale and correlations without derivatives and can search globally at the start. A final trust-region method is useful if the objective has a locally smooth basin: it can make focused progress near its own starting region, and its 500-evaluation allocation stays below the wrapper's known slowdown range.

The planned order intentionally gives most of the budget to the adaptive population method, while reserving both broad coverage and a bounded local-refinement phase. All candidates will be clipped/managed within the stated box by the optimizer implementations, and every objective value will come directly from the harness.

## Pre-registered deviation criteria

- If any planned optimizer raises its documented exhaustion/stop condition before its allocation is complete, stop that phase and assign the unused evaluations to the next planned phase; record the change in `changelog.md`.
- During CMA-ES, if the running best has not improved by at least `1e-12 * max(1, abs(best))` over any consecutive 200 evaluations, restart CMA-ES once from the midpoint with a fresh seed and use the remaining CMA allocation; record the restart and evidence in `changelog.md`.
- During BOBYQA, if its ask/tell wrapper becomes unavailable or raises its documented exhaustion condition before evaluation 2000, stop and save the campaign; record the early stop and evidence in `changelog.md`.
- No other hyperparameter or method changes are authorized by this roadmap. Any such change would be a deviation and would be appended to the changelog before the changed evaluation is made.
