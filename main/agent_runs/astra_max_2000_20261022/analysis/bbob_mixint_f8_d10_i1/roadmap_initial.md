# First decision: normalized CMA-ES

Minimize a deterministic scalar black box in 10 real coordinates with bounds
`[[0,1],[0,1],[0,3],[0,3],[0,7],[0,7],[0,15],[0,15],[-5,5],[-5,5]]`.
There are no additional constraints, no noise, and no prior observations.
The hard campaign budget is 2,000 objective evaluations. The identity, smoothness,
and global optimum are unknown. Only the public evaluator interface will be used.
Every value will come from the same live harness's `evaluate()` method.

## Current block only

Use `cma.CMAEvolutionStrategy` from installed pycma **4.5.0**, a direct supported
library adapter because the provided wrapper fixes population size and initial
spread. Work in normalized coordinates `z=(x-lower)/(upper-lower)` so each physical
coordinate receives comparable initial search coverage.
Initialize at `z=[0.5]*10`, with isotropic `sigma0=0.30`, population size **16**,
seed **41**, bounds `[0,1]`, full covariance (`CMA_diagonal=0`), and active covariance
adaptation (`CMA_active=True`). Other learning rates, rank weights, step-size
adaptation, stopping tolerances, and boundary transformation inherit pycma 4.5.0
defaults. Disable console/file logging (`verbose=-9`, `verb_log=0`). Use the tag
`cma_es`. NumPy version is 2.4.6.

The broad initial spread and population of 16 suit unknown geometry and allow
covariance adaptation without presuming differentiability. There is no evidence
yet that this objective is smooth, rugged, separable, or easy.

The maximum for this block is **160 additional objective calls**, pausing at
**160 cumulative evaluations** (10 complete generations). Ten generations give
more informative directional and population trends than one or two generations,
while using only 8% of the total budget before the first evidence-based decision.
The runner guards every call against both the current block ceiling and the
harness budget, and returns control after this block. It retains the optimizer,
observations, and harness between commands.

At the checkpoint inspect running-best progress, per-generation minima and
medians, objective range, adapted mean/standard deviations, covariance axis ratio,
and any boundary concentration. These observations will determine the next
choice. Pause sooner on any library stopping condition, nonfinite objective,
or evaluator error; an evaluator error ends the run without retry. A short
plateau alone will not establish that the method is unsuitable. Save the harness
at the checkpoint. No subsequent method or budget allocation is committed here.
