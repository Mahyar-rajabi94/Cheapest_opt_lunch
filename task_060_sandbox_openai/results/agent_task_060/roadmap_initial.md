# Task 060: initial optimization roadmap

This plan is written before the first objective evaluation and will not be edited.
The objective is a sealed, deterministic black box. The only prior information
used here is the stated 188-dimensional bounded domain, 160 free coordinates,
28 fixed coordinates, minimization direction, and total budget of 2,000 calls.

## Representation and permitted methods

Create exactly one `AgentTaskHarness(budget=2000)`. Obtain bounds through its
public API. Optimize the 160 free coordinates in a normalized unit box, restore
the original coordinate scales, and set all fixed coordinates to their public
lower bounds before every `harness.evaluate` call. Count every real evaluation,
including repeated points. Never inspect the harness implementation, model
assets, or task source, and never infer or search for the function's identity.

Use only optimizers already listed in `src/optimizers/`: SobolSearch, L-BFGS-B,
CMA-ES, and conditionally BOBYQA. Use the existing Sobol wrapper and the same
native SciPy, pycma, and Py-BOBYQA implementations used by the other wrappers.
Native calls permit starting at an observed incumbent and enforcing precise
phase caps without changing any optimizer source or creating a new optimizer.
Use the installed `/home/sartaaj/miniconda3/envs/osi26/bin/python` environment.

## Planned sequence and approximate allocation

1. **Sobol search: 64 evaluations.** Use the existing scrambled SobolSearch
   wrapper with seed 60 on the normalized free box. Retain its best observation.
   Define the fixed comparison scale `S = max(1, max(y)-min(y))` from these 64
   actual observations. This modest global sample supplies an incumbent and an
   observed value scale without spending most of the budget covering a very
   high-dimensional domain.
2. **L-BFGS-B: up to 483 evaluations initially.** Start at the best observed
   point, with normalized forward-difference `eps=1e-5` and standard SciPy
   convergence tolerances. A gradient site costs approximately 161 objective
   calls, so this allocation admits about three sites. Determinism makes a
   numerical-gradient trial reasonable, but smoothness is unknown. If the
   solver returns early, immediately move to CMA-ES. Do not judge progress in
   the middle of a finite-difference sweep.
3. **CMA-ES: the remaining budget, ordinarily at least 953 evaluations.**
   Initialize a single standard pycma CMAEvolutionStrategy at the best point
   observed across prior methods, normalized sigma 0.25, unit-box constraints,
   seed 60, and default population size/covariance adaptation. Evaluate and
   tell complete generations in order, apart from a final partial generation
   if necessary to use the exact remaining budget. CMA-ES provides a derivative
   free alternative if a local finite-difference model is unhelpful.
4. **Conditional final BOBYQA refinement: at most 500 evaluations.** This
   replaces part of the CMA-ES allocation only under the quantitative plateau
   criterion below. Start the same Py-BOBYQA solver used by the existing wrapper
   at the actual incumbent, in the unit box, using its standard initial radius
   (0.1), standard final radius (1e-8), and default interpolation size. Drive it
   synchronously to avoid the wrapper's documented scheduling limitation. If
   it returns early, resume the existing CMA-ES state until 2,000 calls.

High dimension and a tight budget favor a short initial sample and sustained
adaptive search. Differential Evolution's default 1,600-point initialization
would leave little adaptation budget. A full high-dimensional GP is costly;
Nelder-Mead also spends many evaluations forming a simplex. BOBYQA needs about
321 initialization calls here, so it is reserved for a final, evidence-triggered
refinement rather than added automatically.

## Quantitative adaptation criteria

- L-BFGS-B may extend from 483 to at most 805 calls if, after completing a
  gradient calculation, its observed best has improved by at least `1e-3*S`
  over the preceding approximately 322 local calls. Otherwise stop at the
  first completed gradient site at or after 483 calls. An objective callback
  enforces an absolute 805-call local ceiling and the global 2,000-call ceiling;
  these take priority over finishing a gradient calculation.
- At the first completed CMA-ES generation at or after total evaluation 1,500,
  provided at least 350 calls remain and CMA-ES has run for at least 380 calls,
  compare the running best now with the running best 380 calls earlier. If
  improvement is at most `1e-4*S`, switch to BOBYQA for at most the remaining
  500 calls. If improvement is larger, continue CMA-ES to the global cap.
- If CMA-ES meets its own stopping criteria with at least 350 calls left,
  switch to BOBYQA at the incumbent for at most 500 calls; if fewer remain,
  restart the same CMA-ES algorithm at the incumbent with sigma 0.1 and seed
  61, then use the remainder. If BOBYQA stops early, resume the prior CMA-ES
  state, or restart it with these settings if that state has stopped.
- If any method is unavailable or raises a numerical error, save all genuine
  evaluations, append the exact error and evaluation count to the changelog,
  and transfer the unused allocation to an available listed optimizer, with
  SobolSearch as the final available fallback. Do not reset the harness.
- Any departure not anticipated above requires an append-only changelog entry
  stating the observed numerical evidence before carrying it out. For clarity,
  log all method switches, budget extensions, and restarts even when planned.

## Records and completion

Set the active optimizer label before every method's evaluations. Write an
append-only `changelog.md`; never edit this roadmap. Call `harness.save()` at
phase boundaries and periodically during long phases. The harness alone writes
`trace.csv` and `meta.json`; never construct or edit them manually. At completion
verify exactly the four required files are present, the trace has at most 2,000
real evaluations, the reported best matches the observed minimum, every point
obeys the public bounds, and all 28 fixed coordinates retain their exact values.
