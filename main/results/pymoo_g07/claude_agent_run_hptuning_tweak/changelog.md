## Decision after evaluation 16
**Budget:** 16 used / 100 total; 84 remaining.
**Evidence:** First Sobol' batch (16 points, seed 0) over the full `[-10, 10]^10` box gave merit
values ranging from 8281.1 (min) to 43669.0 (max), median 21220.5, all strictly positive and of
broadly similar order of magnitude (within roughly a 5x band, no single catastrophic outlier many
orders larger than the rest). This is consistent with a penalty term that is non-trivial over most
of the box (no point looked anywhere near a clean/feasible optimum) but still varies smoothly
enough in magnitude to be informative — i.e. the merit surface looks learnable rather than a flat
plateau with occasional cliffs. The best point found, x = [1.446, -6.724, -0.701, 1.190, 4.487,
2.137, 6.393, 6.007, 9.605, 5.680], gave merit 8281.1, clearly the best of the 16 and a useful
anchor for local refinement rather than restarting from the box center.
**Decision:** Switch to CMA-ES, restarted (fresh `cma.CMAEvolutionStrategy` instance — no prior
CMA-ES state to retain) and initialized at the best Sobol point found rather than the box midpoint.
The provided `CMAES` wrapper (`src/optimizers/cma_es.py`) always starts at the box midpoint and
exposes no way to override `x0`, so this step uses a campaign-local adapter that calls
`cma.CMAEvolutionStrategy` directly with the same box-constraint handling and verbosity settings
the wrapper uses, but with `x0` set to the best observed point. All 16 Sobol observations remain in
`harness`'s trace/counter; CMA-ES itself starts without prior told observations, since pycma's
ask/tell state cannot usefully absorb points that were not drawn from its own distribution.
**Hyperparameters:** `x0` = best Sobol point above (changed from the wrapper's midpoint default,
for the reason above). `sigma0` = 5.0 = (box width 20) / 4, following the same standard
box-constrained recommendation the wrapper itself uses (inherited heuristic, not tuned). `bounds`
= `[-10, 10]` per coordinate (from `harness.bounds`, unchanged). `seed` = 0, for reproducibility
(first use of this value in CMA-ES; distinct from the Sobol seed, no data dependency). Population
size and all other internal CMA-ES hyperparameters: pycma's own defaults (`popsize` = 10 for
dim=10, confirmed empirically), left untuned, matching the wrapper's documented policy.
**Reason:** CMA-ES is a reasonable choice given the evidence: the 16-point sample shows no reason
to believe the merit surface is simple/convex (so a single local optimizer from the center would
be risky), but also no reason to believe it needs more blind global coverage before exploiting the
best region found — CMA-ES's own step-size/covariance adaptation can continue exploring around the
best area while needing comparatively few evaluations per generation (10), which fits a budget of
84 remaining evaluations. It is also robust to the kind of non-smoothness a constraint-violation
penalty term can introduce, unlike a pure quadratic-model method.
**Next step limit:** 20 additional objective calls (2 full CMA-ES generations), at most 84
remaining; pause at cumulative evaluation 36.
**Why this length:** With popsize 10, two generations is the minimum needed to see whether
CMA-ES's adaptation is actually improving on the Sobol best (one generation alone updates the
covariance only once and is too little evidence to judge direction), while still leaving 64 of the
84 remaining evaluations uncommitted in case the method needs to change.
**Review:** At cumulative evaluation 36, inspect whether the best merit value has improved versus
the Sobol best (8281.1) and by how much per generation, whether CMA-ES's step size is shrinking
(indicating convergence) or still large (indicating continued exploration), and whether any
generation's points show renewed large merit values (indicating the search stepped into a more
heavily penalized region). No committed next method.

## Decision after evaluation 36
**Budget:** 36 used / 100 total; 64 remaining.
**Evidence:** Two CMA-ES generations (evals 17-26, 27-36) reduced the best merit from 8281.1
(Sobol) to 5622.2 (gen 1) to 3931.2 (gen 2) — a clear, monotonic, substantial improvement
(roughly 2.1x total) across just two generations. The step size `sigma` only shrank slightly
(5.0 -> 4.65 -> 4.40), meaning CMA-ES is still actively exploring/adapting rather than having
converged or stalled; the covariance has not yet collapsed. No generation produced values back up
near the original 8000-40000 range, suggesting the search is not bouncing between feasible and
heavily-penalized regions but is moving through a comparatively smoother descending region.
**Decision:** Continue CMA-ES unchanged — same `cma.CMAEvolutionStrategy` instance (`es`), carrying
forward its adapted mean, covariance, and step size from generation 2; no restart. Continuing is
preferred over switching because the evidence shows active, substantial, still-ongoing improvement
with no sign of stagnation, so there is no basis yet to believe another method would do better with
the remaining budget.
**Hyperparameters:** Unchanged from the previous decision (same `es` object: bounds `[-10, 10]^10`,
popsize 10, seed 0, current adapted sigma ≈ 4.40 and mean/covariance as left at the end of
generation 2). No new hyperparameters to set; CMA-ES's own internal adaptation is not treated as a
fresh hyperparameter decision per step.
**Reason:** A method that is still improving quickly and has not exhausted its own adaptation
(sigma still large relative to its starting value) does not yet warrant switching; doing so would
discard the adapted covariance information for no evidenced benefit.
**Next step limit:** 30 additional objective calls (3 generations), at most 64 remaining; pause at
cumulative evaluation 66.
**Why this length:** Three generations gives enough additional evidence to see whether the
improvement trend continues, flattens, or reverses, while still leaving 34 of the 100 total
evaluations uncommitted for a further adjustment, restart, or different method if CMA-ES plateaus.
**Review:** At cumulative evaluation 66, inspect the best merit per generation (continued
improvement vs. plateau), the trend in sigma (shrinking toward convergence vs. still large), and
whether remaining budget (34 at that point) is enough to justify either continuing CMA-ES to a
tighter local optimum or switching to a local polish method (e.g. BOBYQA/Nelder-Mead) once CMA-ES's
own progress slows. Pause sooner than 30 calls if three consecutive generations show no improvement
in best merit (a plateau signal, accounting for the fact the objective is deterministic so no
replication noise is involved).

## Decision after evaluation 66
**Budget:** 66 used / 100 total; 34 remaining.
**Evidence:** Three more CMA-ES generations (evals 37-46, 47-56, 57-66) gave per-generation bests
of 1159.4, 1485.3, 1177.8 — a large overall improvement versus the 3931.2 best at evaluation 36
(roughly 3.4x), but non-monotonic across individual generations (gen 4's best, 1485.3, was worse
than gen 3's 1159.4; gen 5 came back down to 1177.8, still short of gen 3's 1159.4). The running
best merit (1159.4, from generation 3, i.e. evaluation ~41-46) has not been improved upon in the
two generations since. Sigma is still shrinking only gradually (4.40 -> 4.40 -> 4.01), so CMA-ES
has not collapsed to a point and is still taking exploratory-sized steps; this, together with the
deterministic objective (no noise to explain the non-monotonicity), suggests CMA-ES is currently
circling a promising but structured (non-smooth or ridge-like) region rather than cleanly
converging on it.
**Decision:** Continue CMA-ES unchanged for one more block, same `es` instance (adapted mean,
covariance and sigma carried forward from generation 5), since two generations without a new best
is not yet a reliable plateau signal for this deterministic-but-apparently-rugged region and sigma
has not bottomed out — but shorten the step length and reserve a meaningful tail of the budget so
that if this block also fails to beat 1159.4, there is still enough budget left to either restart
CMA-ES with a smaller sigma (local polish around the current best) or switch methods, rather than
being forced to exhaust the remaining budget on a single choice.
**Hyperparameters:** Unchanged from the previous decision (same `es` object; bounds, popsize 10,
seed 0 carried over; current sigma ≈ 4.01 at the start of this block, adapted internally by CMA-ES).
**Reason:** Reducing the step length (rather than committing the rest of the budget) directly
reflects the weaker, less monotonic evidence from the last block: it buys one more look at whether
progress resumes while preserving the option to change course with a non-trivial remaining budget.
**Next step limit:** 20 additional objective calls (2 generations), at most 34 remaining; pause at
cumulative evaluation 86.
**Why this length:** Two generations is enough to tell whether the previous plateau was transient
(a new best appears) or is firming up (neither generation beats 1159.4), while leaving 14
evaluations in reserve regardless of outcome — enough for either a short local-polish restart or a
final small CMA-ES/alternative-method block.
**Review:** At cumulative evaluation 86, check whether the running best has improved past 1159.4,
how far sigma has shrunk, and how much budget (14, if this block runs to completion) remains; decide
between a further CMA-ES block, a sigma-reduced restart centered on the current best for a tighter
local polish, or a different local method, given what is left of the budget.

## Decision after evaluation 86
**Budget:** 86 used / 100 total; 14 remaining.
**Evidence:** Generation 6 (evals 67-76) found a new overall-best merit of 1056.8, improving on
1159.4 (first improvement in three generations), while generation 7 (evals 77-86) was clearly worse
(1983.7). Sigma continued its slow, steady shrink (4.01 -> 3.81 -> 3.52), still not near collapse.
The pattern across the last several generations (improve, plateau x2, improve, worsen) is
consistent with CMA-ES still actively searching a structured/rugged region rather than having
converged, and with only 14 evaluations left there is not enough remaining budget for another
full exploratory restart (e.g. a fresh global Sobol pass or a from-scratch CMA-ES run) to be
worthwhile — any alternative method would need to spend most of its budget on initialization alone
(e.g. BOBYQA's quadratic model for 10 dimensions needs on the order of 2*dim+1 = 21 points, more
than the entire remaining budget).
**Decision:** Continue CMA-ES unchanged, same `es` instance (mean/covariance/sigma carried over
from generation 7), using all remaining budget. This is the last step of the campaign: 14
evaluations is too small to justify switching to or initializing a different method, and CMA-ES has
just produced a new best three generations ago and is still taking meaningfully large
(non-converged) steps, so continuing it is the best use of the remaining calls.
**Hyperparameters:** Unchanged (same `es` object; bounds, popsize 10 (generation sizing), seed 0
carried over; sigma ≈ 3.52 adapted internally at the start of this block). Because only 14 calls
remain and the standard generation size is 10, this final step is split into one ordinary
generation of 10 (evals 87-96) followed by a smaller final partial ask/tell of 4 points
(evals 97-100), both drawn from and told back to the same `es` instance/distribution, to use the
entire remaining budget without exceeding it.
**Reason:** With 14 calls left, this is the final decision point; using them to continue the one
method that has driven essentially all of the improvement so far, from its current adapted state,
is lower-risk and better-justified than spending most of the remaining budget initializing an
untested alternative method.
**Next step limit:** 14 additional objective calls (all remaining budget); pause at cumulative
evaluation 100 (budget exhaustion).
**Why this length:** This is the last step; no budget remains to reserve for a further decision
point, so the full remainder is used here.
**Review:** Not applicable — this step ends the campaign (budget exhausted). The outcome (final
best merit and point) will be recorded in the stopping entry below.

## Final outcome and stop (evaluation 100)
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** The planned generation of 10 (evals 87-96) gave a best of 1438.7, not improving on
the running best of 1056.8 (from evaluation 76). The final partial batch of 4 points (evals 97-100)
was asked from and intended to be told back to the same CMA-ES instance, but `cma`'s `tell()`
raised `ValueError: not enough solutions passed to function tell (passed solutions=4 < mu=5)` — the
library requires at least `mu` (the number of parents used for the mean update, 5 here) solutions
per `tell` call, so a batch of 4 could not be registered as a CMA-ES update. This is a harmless
bookkeeping failure in the *optimizer's* internal state update, not in the harness: all 4 of those
points had already been produced as real, legitimate black-box queries via `harness.evaluate(x)`
before the `tell()` call was attempted, so they are correctly present as real rows in `trace.csv`
(evaluations 97-100) and correctly counted against the budget; only CMA-ES's own distribution
update for that last micro-batch did not occur, which is immaterial since the campaign ends here
regardless. None of the final 14 evaluations (87-100, best 1438.7) improved on the running best of
1056.8 found at evaluation 76.
**Decision:** Stop. The budget is exhausted (100/100 `evaluate()` calls used); no further
evaluations are permitted by the harness.
**Hyperparameters:** Not applicable (no further step).
**Reason:** Budget exhaustion is a hard stopping condition.
**Next step limit:** 0 (stopping).
**Why this length:** Not applicable.
**Review:** Not applicable.

**Final result:** best merit observed = **1056.7747727524052**, at evaluation 76 (method
`cma_es_custom_x0`), point x = [3.2519, -0.0809, 7.4915, 9.9918, -3.7084, 2.2189, 1.7624, 5.9125,
8.2618, 8.5899]. `harness.save()` had already been triggered automatically by the harness itself
once evaluation 100 was recorded (`evaluate()` calls `save()` internally upon reaching the budget);
`trace.csv` and `meta.json` in this results directory reflect all 100 real evaluations.

