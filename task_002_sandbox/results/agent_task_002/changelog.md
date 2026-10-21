
## Evaluation 737
**Change:** Restarted CMA-ES (restart #1, new seed) instead of continuing the previous run.
**Reason:** Best value improved by less than 1e-6*|best| over the last 200 evaluations (best stuck at 1065.07); pre-registered stagnation criterion fired.

## Evaluation 937
**Change:** Restarted CMA-ES (restart #2, new seed) instead of continuing the previous run.
**Reason:** Best value improved by less than 1e-6*|best| over the last 200 evaluations (best stuck at 1065.07); pre-registered stagnation criterion fired.

## Evaluation 1137
**Change:** Abandoning CMA-ES after max restarts; moving to local polish early with remaining global-search budget unused.
**Reason:** Used 2 restarts, still stagnating at best=1065.07; pre-registered cap on restarts reached (TuRBO fallback not needed since enough budget remains for local polish to still add value from the best basin found).

## Evaluation 1447
**Change:** BOBYQA converged (OptimizerExhausted raised); switching to L-BFGS-B for remaining local-polish budget.
**Reason:** BOBYQA signalled its own convergence after 310 evaluations in this phase, best=1005.97; per pre-registered criterion 3, fall back to L-BFGS-B rather than stop, since evaluation budget remains.

## Evaluation 1898
**Change:** Accepted local-polish result as new global best.
**Reason:** Local polish improved best from 1065.07 to 1005.97 via deterministic local refinement from the CMA-ES incumbent.
