# Initial decision: bounded CMA-ES exploration

## Problem and assumptions
Minimize the deterministic scalar objective in 20 real coordinates, each in [-5, 5], with no additional constraints. There are no prior observations. The hard total budget is 2000 evaluations, including the initial point and all probes. Objective identity, implementation, and optimum remain sealed. Determinism is supplied by PUBLIC_EVALUATOR.md; no smoothness, separability, or unimodality is assumed.

## First step only
Use the installed pycma 4.5.0 `CMAEvolutionStrategy` through a campaign-local adapter. Evaluate the box midpoint x = 0 once as a reference, then run up to 12 complete CMA-ES generations. Initialize the mean at x = 0, sigma = 2.0 (20% of the coordinate span), population size = 16, seed = 9530, and bounds = [-5, 5] per coordinate. Use full covariance from the start (`CMA_diagonal=0`) and active covariance updates (`CMA_active=True`). A population of 16 supplies eight selected parents per generation and balances direction learning against evaluation cost in 20 dimensions. Sigma 2.0 explores a substantial part of the box while retaining resolution around the initial mean.

Inherit pycma 4.5.0 recombination weights, covariance/step-size adaptation, BoundTransform boundary handling, and stopping defaults (including tolx=1e-11, tolfun=1e-11, tolfunhist=1e-12, and its dimension/population dependent stagnation criterion). Disable library logging (`verb_log=0`, `verbose=-9`); the harness owns evaluation records. Use the supported underlying API because the provided wrapper fixes initial sigma and population size. No dependency changes are needed.

## Limit and purpose
At most 193 additional objective calls: 1 midpoint plus 12 populations of 16. Pause at cumulative evaluation 193. Twelve generations allow initial mean and scale adaptation without committing a large share of the budget before observing behavior. Every proposed point is evaluated through the single persistent harness, with optimizer label `cma_es`; an independent adapter guard checks the block cap before each call. Retain optimizer state and all actual observations in the live process for review.

Inspect midpoint value, best value and evaluation index, generation minima/medians, improvement frequency, adapted sigma, coordinate standard deviations, covariance axis ratio, and boundary proximity. Pause sooner if pycma reports a stopping criterion, a nonfinite objective occurs, or an evaluator error occurs. An evaluator error ends the campaign without retrying the unknown evaluation. A library stop or lack of progress triggers review, without preselecting any subsequent method. Call harness.save() at the checkpoint.
