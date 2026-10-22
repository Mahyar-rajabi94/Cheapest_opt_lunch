# Initial decision: one bounded CMA-ES block

## Problem and assumptions
Minimize the deterministic scalar black box in 10 real coordinates, each in [-5, 5], with no additional constraints. The total hard budget is 2,000 objective calls, including initialization and repeated points. There are no prior observations. Smoothness, convexity, separability, and the optimum are unknown. The objective remains sealed; only the public evaluator interface and collected observations will be used. No replication is initially needed because the public interface states the objective is deterministic.

## First step only
Use the installed pycma 4.5.0 `CMAEvolutionStrategy` directly, permitting explicit population and scale choices beyond the supplied wrapper. Initialize the mean at the box center (ten zeros), with isotropic sigma0 = 2.0 in original units. Use population size 16, seed 20261022, bounds [-5, 5] in each coordinate, full covariance adaptation, and active covariance updates (`CMA_active=True`). Use pycma 4.5.0 inherited defaults for rank weights, learning rates, step-size adaptation, BoundTransform handling, and numerical stopping criteria. Set `verbose=-9` and `verb_log=0` to suppress library output files. These output settings do not change the search. Evaluate the center once as an observed reference, then execute up to 10 complete generations. Tag all evaluations `cma_es`.

CMA-ES uses objective rankings and learns directions without assuming derivatives or a particular functional form. A scale of 2.0 spans a substantial part of the 10-unit box, while population 16 provides eight positively weighted candidates per generation with a modest call cost. This is a prior-based choice, not an inference from unavailable observations.

## Step limit and checkpoint
At most 161 objective calls: 1 center call plus 10 generations of 16. Pause at cumulative evaluation 161. This uses 8.05% of the budget and allows initial directional and scale adaptation before assessing the trajectory; a shorter block would mainly expose initialization variation.

Every call is guarded against both the current block cap and total budget. Retain the same live harness, all observations, and the optimizer in a persistent process across checkpoints. Call `h.save()` at the checkpoint. Stop the block sooner if pycma reports a numerical/convergence stop, any objective value is nonfinite, or an evaluator error occurs; on an evaluator error, halt without retrying.

## Evidence to inspect
Review the center value, each generation's minimum and median, cumulative improvements, candidate spread, adapted step size, covariance axis ratio, proximity to bounds, and numerical stopping diagnostics. These will determine the next decision. This record makes no commitment to any later method or block.
