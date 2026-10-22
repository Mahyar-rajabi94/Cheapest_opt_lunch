# First optimization step

Minimize the deterministic scalar objective over 10 coordinates, with bounds
`[[0,1],[0,1],[0,3],[0,3],[0,7],[0,7],[0,15],[0,15],[-5,5],[-5,5]]`.
There are no further constraints and no prior observations. The hard total
budget is 2,000 real harness evaluations. The objective remains a sealed black
box. No smoothness, separability, convexity, or known optimal value is assumed.
Determinism means this first step does not need replication.

Use the provided `src.optimizers.sobol_search.SobolSearch` wrapper, backed by
SciPy 1.17.1 `scipy.stats.qmc.Sobol`. Initialize a new 10-dimensional scrambled
Sobol sequence with seed 20261022. Explicit settings: `scramble=True` (inherited
wrapper default), 64 points; other sampler defaults are those of SciPy 1.17.1
(including no post-sampling optimization). Search coordinates are normalized
to [0,1], then mapped affinely to the public bounds for every evaluation.
This provides broad coverage without assuming local structure or privileging
coordinates with wider physical bounds. The optimizer label is `sobol_initial`.

The first block is at most 64 additional objective calls and pauses at
cumulative evaluation 64. A power of two preserves Sobol balance; 64 points
provide a modest initial picture in 10 dimensions while using only 3.2% of
the total budget. This is only the first-step decision; no subsequent method
or budget allocation is selected here.

At the checkpoint, inspect the observed range and quantiles, best locations,
progress through the sequence, and exploratory coordinate rank correlations.
Correlations from this small sample are uncertain and cannot establish
separability or absence of interactions. Stop sooner on any evaluator error
without retrying the unknown call, or on an invalid/nonfinite response; those
conditions require review. A persistent driver enforces both the block cap
and campaign counter before each call. Save harness artifacts at the checkpoint.
