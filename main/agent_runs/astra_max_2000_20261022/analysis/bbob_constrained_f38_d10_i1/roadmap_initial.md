# Initial optimization decision

Minimize a deterministic scalar objective on 10 coordinates in [-5, 5]. A point is feasible only when all three returned inequalities are <= 1e-10. The hard total budget is 2000 joint objective/constraint evaluations; infeasible points and repeats count. There are no prior observations. The objective and constraints are sealed black boxes; no smoothness, feasibility volume, or location of an optimum is assumed.

## First block only
Use the provided SobolSearch wrapper (src/optimizers/sobol_search.py), backed by SciPy 1.17.1 scipy.stats.qmc.Sobol. Evaluate the box centre first, followed by 64 scrambled Sobol points across the full bounds. Set seed=20261022 and scramble=True. Inherit SciPy 1.17.1 defaults bits=None (30 effective bits) and optimization=None. NumPy is 2.4.6. The wrapper retains its sampler and observations; constraint values are retained separately by the campaign driver, and feasible selection uses the actual returned inequalities.

The maximum is 65 additional objective calls, pausing at cumulative count 65. Sixty-four Sobol points form a balanced power-of-two design; together with the centre this gives a modest initial view of feasibility and objective scales in ten dimensions, without assuming a particular function structure. No replication is required because the interface states determinism.

At the checkpoint, inspect feasibility count, observed objective range and quantiles, best feasible observation, smallest constraint violations, and which constraints are restrictive. These observations will inform a fresh decision about search method and its settings. Pause sooner on an evaluator error or nonfinite output; do not retry an unknown evaluation. Every call will be tagged sobol_initial. No subsequent optimization block is selected here.
