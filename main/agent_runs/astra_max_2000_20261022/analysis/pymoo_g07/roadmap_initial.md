# First decision: structured initial design

Minimize a deterministic scalar objective in 10 dimensions, each in [-10, 10], with eight black-box inequalities. Feasibility requires every returned constraint <= 1e-10. The hard total budget is 2,000 calls, including infeasible points and any repeats. There are no prior observations. Objective and constraints remain sealed; smoothness, convexity, and the size or connectivity of the feasible region are unknown. All final results will be actual harness observations.

## Current step only
Use an initial space-filling and coordinate design, labeled `sobol_axial_design`: evaluate the center x=0, then both +/-2 displacements along each coordinate (20 points), then 32 scrambled Sobol points spanning the full bounds. Coordinate displacements are 10% of the box width; paired probes can expose local curvature and constraint sensitivity. The full-domain points give a modest initial assessment of feasible-region prevalence and objective variation without assuming a useful center.

Implementation: campaign adapter in work/campaign.py using the provided src.optimizers.sobol_search.SobolSearch wrapper and SciPy 1.17.1 qmc.Sobol. Seed=173, dimension=10, scramble=True, inherited bits=None (effective 30), optimization=None, float64 output. Exactly 32 Sobol samples preserve the power-of-two balance property. NumPy version 2.4.6. No additional dependency is needed. The same live harness, all observations, and sampler state will be retained.

## Evaluation limit and checkpoint
Maximum 53 additional objective calls; pause at cumulative evaluation 53. This is 2.65% of the total budget: 21 observations give paired local sensitivity information, and 32 cover the full domain while leaving most of the budget available for decisions informed by data. The call gate enforces both this block limit and the remaining harness budget. Pause sooner for a nonfinite returned value or an evaluator error; on evaluator error stop without retrying an unknown evaluation.

At the checkpoint inspect feasibility counts, the best feasible observation (if any), least constraint violation, objective and constraint ranges, paired coordinate responses, and whether local variation suggests usable smoothness. These observations will determine the next decision; no later method or budget allocation is selected now. Save the harness output at this checkpoint.
