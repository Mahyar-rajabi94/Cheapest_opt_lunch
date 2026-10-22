# Initial bounded decision

Minimize the deterministic scalar objective in five real coordinates with bounds
[78,102], [33,45], [27,45], [27,45], [27,45]. There are six black-box inequalities;
a point is feasible only when every returned constraint is <= 1e-10. The hard
budget is 2000 objective calls, including infeasible points and repeated calls.
There are no prior observations. The function remains sealed: use only the
public interface and collected data. No smoothness or feasible-volume assumption
is needed for this first block.

Use the provided `src.optimizers.sobol_search.SobolSearch` wrapper with seed
271828, the public box bounds, and 64 points. Its implementation uses
`scipy.stats.qmc.Sobol`, SciPy 1.17.1, scramble=True, and inherited default
bits=30 and optimization=None. NumPy is 2.4.6. The optimizer label is
`sobol_feasibility`. All constraints are retained separately; the wrapper's
unconstrained best is not used to select the campaign winner.

The first step has a maximum of 64 additional real objective evaluations and
pauses at cumulative evaluation 64. A power-of-two Sobol block gives balanced
coverage across five dimensions, while spending only 3.2% of the budget before
reviewing feasibility and scales. Initialization is the seeded scrambled sequence
across the complete box; there is no prior incumbent.

Inspect feasible count, objective quantiles, constraint ranges and violation
counts, and the locations and values of the best feasible observations. This
will reveal whether the design finds useful feasible starting points and which
constraints restrict good candidates. Stop sooner on an evaluator error or a
nonfinite output; do not retry an unknown evaluation. Every call is guarded by
both the current step limit and the single harness's remaining budget. Save at
the checkpoint, return control for a fresh decision, and retain observations
and the Sobol sampler state. No later method or block is selected here.
