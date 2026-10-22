# Roadmap — Task 024 (dim=6, bounds=[0,1]^6, minimize, deterministic, budget=100)

## Problem read

- 6-dimensional, box-bounded in the unit cube, deterministic (no noise),
  minimize, 100 total `evaluate()` calls. No other structural information is
  available (and none should be sought) about the objective's identity,
  modality, or smoothness.
- Budget-to-dimension ratio is ~16.7x dim. This is squarely in the regime
  this project's own `gp_bo.py` docstring identifies as GP-BO's sweet spot
  ("expensive, continuous problems, budgets of order 10x dim"), and too
  small for population-based global methods (DE needs ~10^3-10^4 x dim;
  CMA-ES/TuRBO are more data-hungry than a 100-eval budget rewards well,
  though CMA-ES remains a reasonable fallback). A single local solver
  (BOBYQA/Nelder-Mead/L-BFGS-B) risks converging to a single basin with no
  way to tell if it's global, given an unknown, possibly multimodal
  function and no prior structural knowledge.

## Planned method sequence

1. **GP-BO** (`src/optimizers/gp_bo.py`, LogEI acquisition), driven
   sequentially one point at a time, as the primary method for the full
   budget. It already begins with its own internal Sobol' space-filling
   design (`n_init = 2*(dim+1) = 14` points) before switching to
   acquisition-driven points — so this single method already covers both
   an exploration phase and an exploitation phase without a separate
   hand-rolled Sobol stage.
2. **Contingency / deviation triggers** (quantitative, checked after every
   evaluation once the GP-BO acquisition phase has started, i.e. after the
   first 14 evaluations):
   - **Plateau switch to CMA-ES**: if the running best value has not
     improved over any stretch of the last **15 consecutive** GP-BO
     acquisition evaluations, stop GP-BO and switch to CMA-ES
     (`src/optimizers/cma_es.py`, standard midpoint/sigma0=box_width/4
     start) for the remaining budget. Rationale: a long improvement-free
     stretch under LogEI usually means the acquisition optimizer is stuck
     repeatedly exploiting the same local basin; a population-based
     global method is more likely to escape it than continuing GP-BO.
   - **Error fallback**: if GP-BO raises any exception (e.g. a GP-fitting
     numerical failure), switch immediately to CMA-ES for the remaining
     budget.
   - **Second plateau -> local polish**: if CMA-ES is active and itself
     shows no improvement over **20 consecutive** evaluations while at
     least 10 evaluations of budget remain, switch to a local polish with
     BOBYQA (`src/optimizers/bobyqa.py`) started at the current best point,
     for the remaining budget, to extract final precision around the best
     basin found so far.
   - If none of these triggers fire, GP-BO simply runs for the entire
     100-evaluation budget.
3. Every switch will be logged in `changelog.md` with the evaluation index
   and the concrete evidence (best-so-far history) that triggered it.

## Budget split (planned, not committed in advance beyond the triggers above)

- ~14 evaluations: GP-BO's internal Sobol initial design (exploration).
- Remaining ~86 evaluations: GP-BO LogEI acquisition (exploitation),
  unless a deviation trigger above fires, in which case the remainder is
  reallocated to CMA-ES and/or BOBYQA as described.

## Deviation criteria (restated explicitly, quantitative)

- No improvement in best-so-far for 15 consecutive GP-BO evaluations ->
  switch to CMA-ES.
- GP-BO raises an exception -> switch to CMA-ES immediately.
- No improvement in best-so-far for 20 consecutive CMA-ES evaluations,
  with >=10 evaluations of budget remaining -> switch to BOBYQA local
  polish from current best.
- Any other change to this plan not covered by the above will be logged
  in `changelog.md` with the evaluation index and the observed evidence
  that motivated it.
