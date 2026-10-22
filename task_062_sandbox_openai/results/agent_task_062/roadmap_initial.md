# Initial roadmap — Task 062

This plan is recorded before any objective evaluation. The objective is sealed:
only the public harness dimension, bounds, direction, evaluation method, optimizer
label setter, and save method will be used. No objective implementation, task
implementation, external benchmark identity, or other campaign result will be read.

## Representation and allowed methods

Use one harness instance, with a total ceiling of 2000 actual evaluations.
Remove the 28 fixed coordinates, optimize the 160 free coordinates in the unit
box, and reconstruct all 188 coordinates using the public bounds for every call.
Fixed coordinates always equal their lower/upper bound. This affine scaling
prevents unequal coordinate widths from dictating the solver's step sizes.

Only BOBYQA, L-BFGS-B, and, conditionally, CMA-ES will be used. All three are
listed in src/optimizers/. BOBYQA and L-BFGS-B will use the same established
Py-BOBYQA and scipy implementations as those wrappers, driven synchronously
through a callback to harness.evaluate. This avoids the documented BOBYQA
thread/queue slowdown and permits refinement from the actual incumbent without
introducing any new optimizer. CMA-ES, if needed, uses the existing CMAES wrapper.
Every evaluation receives its current method label: bobyqa, lbfgsb, or cma_es.

## Nominal sequence and budget

1. **BOBYQA: evaluations 1–500.** Start at the free-box midpoint. Use
   Py-BOBYQA's standard model size and trust-region settings, with built-in
   bound scaling. The external campaign controls the evaluation ceiling.
2. **L-BFGS-B: the remaining approximately 1500 evaluations.** Start at the
   best actually evaluated BOBYQA point. Use scipy's standard finite-difference
   step and convergence settings. The external campaign caps all evaluations,
   including gradient and line-search calls, at 2000 in total.
3. **CMA-ES only as a conditional fallback**, using the existing wrapper's
   defaults on the normalized full free box and seed 62.

This is a high-dimensional problem with only 12.5 evaluations per free
coordinate. Broad independent random coverage and the listed differential
evolution's 1600-point initial population would leave little budget for learning.
BOBYQA can build a bounded local model without requiring derivatives, while the
objective's determinism allows L-BFGS-B to use finite differences if local
responses support refinement. At 160 free coordinates each finite-difference
gradient costs about 161 evaluations, so local refinement needs several hundred
calls before it can be judged. Smoothness and convexity are not assumed; CMA-ES
is reserved for unproductive local search. High-dimensional GP fitting is not
the initial choice given the modest objective budget and computational overhead.

## Quantitative adaptation rules

Let b(N) be the smallest real objective value through evaluation N, and let
S = max(1, abs(y(1))). All improvement statistics use actual evaluations.

- **Extend productive BOBYQA at 500:** if b(340) - b(500) is at least
  20% of max(1e-12, y(1) - b(500)) and at least 1e-5*S, continue the
  same solver state to evaluation 1000. This reallocates 500 calls from
  L-BFGS-B and will be logged before continuing.
- **One further BOBYQA extension at 1000:** if b(800) - b(1000) is at
  least 10% of max(1e-12, y(1) - b(1000)) and at least 1e-5*S, continue
  the same solver state to evaluation 1500. Otherwise hand off to L-BFGS-B.
  At 1500 hand off regardless, leaving approximately 500 refinement calls.
- **Local stagnation:** after at least 483 L-BFGS-B calls, if the running
  best improves by no more than 1e-8*S over its last 322 calls, switch to
  CMA-ES if at least 190 evaluations remain. Check at 100-call campaign
  checkpoints after the minimum local allocation. Otherwise continue to
  the budget limit.
- **Early solver termination or numerical failure:** save existing results
  and log the solver's reported status and actual best value. If BOBYQA
  terminates, move to L-BFGS-B from the incumbent. If L-BFGS-B terminates
  with at least 190 calls left, use CMA-ES for the balance. With fewer than
  190 calls left, also use CMA-ES to spend the remaining budget rather than
  restart a finite-difference gradient that cannot complete.
- No new optimizer, coordinate screening, objective identity inference, or
  unrecorded hyperparameter change is planned. Any additional departure
  requires a contemporaneous appended changelog entry citing observed
  numerical evidence. Record every actual method switch, extension, or restart,
  including planned handoffs, for a complete history.

## Integrity and completion

Every candidate value comes directly from harness.evaluate; no evaluations
are cached, synthesized, or written into the trace manually. Checkpoint with
harness.save(). Guard the total count before evaluation so the campaign never
attempts call 2001. Keep this roadmap unchanged. Append all subsequent campaign
changes to changelog.md. Finally verify the harness-generated trace.csv and
meta.json and confirm that the result directory contains exactly the required
four files.
