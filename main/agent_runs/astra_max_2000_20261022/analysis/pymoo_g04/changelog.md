## Decision after evaluation 64
**Budget:** 64 used / 2000 total; 1936 remaining.
**Evidence:** Scrambled Sobol found 13/64 feasible points. The best feasible
observation (#28) is -28355.616825040095 at
[83.51335709542036,37.207859471440315,37.01385457254946,43.95979996584356,28.39388540200889].
Its largest constraint is -0.677456800059673. Objective values span
[-31231.684253302996,-23917.66142691179], including infeasible points.
Violation counts for constraints 1..6 are [0,42,0,0,17,5]. These data establish
usable feasible starts, but do not establish smoothness or global optimality.
A startup outdir spelling error was corrected before construction succeeded;
no objective was called by the failed construction. The live single harness
started at zero and is retained.
**Decision:** Switch to SLSQP constrained local optimization from observation
#28. Retain all real observations in an exact-point cache and retain the same
harness. Use a campaign-local pauseable adapter because the available generic
wrappers do not directly expose the six simultaneous hard constraints. A
solver thread can request points; only the main thread evaluates within the
current authorized block. Solver state stays alive across a checkpoint.
**Hyperparameters:** SciPy 1.17.1 `minimize(method='SLSQP')`, variables normalized
to [0,1], objective divided by 10000, constraints in their original units.
Require g <= -1e-7 inside the solver to provide a small feasibility cushion.
Joint second-order three-point derivatives use step 1e-5 in normalized units,
central differences in the interior and one-sided differences at the box.
One joint call returns objective and all constraints; exact cache hits do not
call the evaluator. ftol=1e-11, maxiter=200, disp=False; supply both objective
and constraint Jacobians, so SciPy finite-difference defaults are unused.
Other SLSQP internals inherit SciPy 1.17.1 defaults. There is no random seed
for this deterministic method.
**Reason:** Low dimension and deterministic values make a local smoothness
trial economical. Explicit constraints let local search improve feasible
points without depending on a penalty weight. Smoothness is a working
hypothesis to assess from actual progress, not prior knowledge of the function.
**Next step limit:** At most 120 additional calls; pause at cumulative 184.
**Why this length:** About ten full 11-point derivative sets plus some line
search work can show whether a meaningful feasible descent emerges; less than
7% of the remaining budget is committed before reconsideration.
**Review:** Inspect improvement, feasible fraction, active constraints, iterate
movement, and solver status. Pause sooner if SLSQP terminates; evaluator error
or nonfinite output ends evaluation without retry. Solver convergence would
be local evidence only. No subsequent method is preselected.

## Decision after evaluation 140
**Budget:** 140 used / 2000 total; 1860 remaining.
**Evidence:** SLSQP terminated successfully after 7 iterations and 76 new calls,
triggering an early checkpoint before the 120-call cap. It improved the feasible
best by 2309.921725475646 to -30665.53855051574 (#130), at
[78,33.000000000000036,29.995256505007852,45,36.77581240263093]. Constraints
2 and 5 are approximately -1e-7, with x1 and x2 at their lower bounds and x4
at its upper bound. The block had 13/76 feasible probes; many derivative probes
crossed an active constraint, as expected near the boundary. Smooth local
progress supports the derivative method, but one start cannot establish the
best basin.
**Decision:** Restart the same constrained local method from three deliberately
dispersed observed points, in a single bounded multistart block. Starts are
#21, #34, #63: first the lowest-objective initial observation (infeasible), then
two max-min distance choices in normalized coordinates, excluding the incumbent
and earlier chosen starts. Each run resets its SLSQP Hessian and iteration state
but reuses all actual observations via the exact cache. Preserve the completed
first run, all observations, and the single harness. Retain any unfinished new
run at the checkpoint. Do not launch any fourth start in this block.
**Hyperparameters:** Use the local adapter and SciPy 1.17.1 configuration recorded
at evaluation 64, changing constraint margin to 1e-9 and ftol to 1e-12. Keep
joint difference step 1e-5, objective scale 10000, original constraint units,
and maxiter=200 per solve. Tightening the margin reduces the systematic
objective loss of the feasibility cushion, while maintaining distance from the
1e-10 public feasibility tolerance. Max-min selection has no randomness.
**Reason:** The initial solver used only 76 calls, making independent basin and
feasibility-recovery checks inexpensive relative to 1860 remaining calls.
Diverse and initially infeasible starts give stronger evidence than repeatedly
polishing the same starting point.
**Next step limit:** At most 270 additional objective calls, pausing at
cumulative 410; stop sooner if all three selected solves terminate.
**Why this length:** Roughly three runs of the observed 76-call cost plus line
search allowance can compare endpoints while leaving most of the budget for a
fresh evidence-based choice. The cap is shared, not 270 per start.
**Review:** Compare feasible endpoints, objective values, active constraints,
constraint-recovery success, and any unfinished run. Termination of all three
runs is an early checkpoint, not an instruction to begin another method.
Evaluator errors or nonfinite results stop without retry.

## Decision after evaluation 401
**Budget:** 401 used / 2000 total; 1599 remaining.
**Evidence:** The three selected restarts all converged successfully in 7, 8,
and 9 iterations, using 261 calls in total (early checkpoint after all three).
Their feasible endpoint objectives differ by about 1e-11 and share the same
active box and black-box constraints. Best #206 is -30665.538670570626 at
[78,33,29.995256030474916,44.99999999999999,36.77581290075662], with maximum
g approximately -1e-9. The block had 35/261 feasible probes. A no-evaluation
response-surface diagnostic fitted generic affine and quadratic models to
only the first 32 Sobol observations. The affine model's objective holdout
error reached 392.75, while the quadratic model's maximum error was
6.19e-11 in objective and 1.99e-13 across constraints on the remaining 369
observations (32 global and 337 local). Training matrix condition number was
9.87. This is empirical predictive evidence, not identification of the private
function or a proof of its behavior everywhere.
**Decision:** Switch to an empirical quadratic-surrogate constrained search and
independent real validation. Fit a full quadratic to the original 64 globally
spread observations. Search this surrogate from 256 new scrambled Sobol starts,
all 32 box corners, and the incumbent. Evaluate the selected, polished candidate
once, then evaluate 64 next points from the retained original Sobol sequence.
Only harness observations count as results; predicted values are diagnostics.
Keep every observation, all local runs, and the same harness counter.
**Hyperparameters:** Campaign-local full total-degree-2 least-squares model,
21 basis terms, coordinates centered to [-1,1], response centering, no ridge
regularization; NumPy 2.4.6 `lstsq(rcond=None)`. Surrogate SLSQP uses SciPy 1.17.1,
analytic surrogate derivatives, objective scale 10000, ftol=1e-12, maxiter=200,
and unit-cube bounds. The 256 surrogate starts use SciPy Sobol scramble=True,
seed=314159, m=8; other starts are deterministic. Target g<=8e-11, leaving
2e-11 inside the public 1e-10 tolerance, substantially larger than observed
constraint model errors. Snap coordinates within 1e-7 of a box boundary, then
polish the inferred square active-constraint system with SciPy `root` (default
hybr, supplied Jacobian, xtol=1e-11). Other solver settings inherit SciPy 1.17.1.
The independent 64 real validation points continue the initial Sobol settings,
bringing its cumulative sample count to the balanced power of two 128.
**Reason:** Strong held-out predictive accuracy makes broad surrogate search
cheap and potentially useful. Independent fresh real points can falsify that
accuracy. More starts of the expensive local solver currently have less value
than testing the fitted response surface across the box.
**Next step limit:** At most 65 additional objective calls; pause at cumulative
466. Surrogate computations do not call the objective or populate the trace.
**Why this length:** One real candidate plus a balanced 64-point global
validation extension tests improvement and predictive reliability with 4.1%
of the remaining budget; it does not commit any later block.
**Review:** Inspect the actual candidate's feasibility and value, surrogate
endpoint agreement, and errors on the 64 fresh observations. Pause immediately
if any absolute prediction error exceeds 1e-5 in objective or 1e-8 in any
constraint. Evaluator errors/nonfinite outputs terminate without retry. The
finite validation set cannot prove global optimality of the sealed function.

## Decision after evaluation 466
**Budget:** 466 used / 2000 total; 1534 remaining.
**Evidence:** All 289 surrogate solves succeeded; their predicted endpoint
objectives span only 3.82e-9. The actual selected point (#402) improved the best
to -30665.538671880335 at [78,33,29.995256025298147,45,36.775812906190694],
with maximum constraint 8.000711204658728e-11, feasible under the supplied
1e-10 tolerance. The new block had 20/65 feasible points. All 65 fresh
observations matched the model: maximum objective error 7.28e-12 and maximum
constraint error 2.85e-14. The 64 independent global points included both
feasible and strongly infeasible observations. A further no-evaluation
Lagrangian diagnostic for the empirical quadratic produced a loose lower
bound 89.43 below the incumbent; it therefore supplies no tight global
certificate. Finite data and agreeing local searches do not prove the sealed
function's global optimum.
**Decision:** Continue using the retained empirical quadratic, with an
independent derivative-free global surrogate search and a real boundary audit.
Evaluate all 32 box corners to test response predictions at extreme inputs.
If the audit agrees, run constrained differential evolution on the surrogate,
polish its candidate with analytic surrogate derivatives, then use at most six
real calls to refine active constraints. Keep the fitted model, original data,
and the same harness. Differential evolution is a fresh solver with its own
population; previous optimizers and observations remain retained.
**Hyperparameters:** Corner audit uses no tuning. SciPy 1.17.1 differential
 evolution: strategy=best1bin, maxiter=500, popsize=20 (Sobol initialization
rounds to 128 individuals in 5D), mutation=(0.5,1.0), recombination=0.7,
tol=1e-10, atol=1e-12 on objective/10000, init=sobol,
rng=NumPy default_rng(1618033), updating=immediate, workers=1, polish=False.
NonlinearConstraint enforces all surrogate g<=1e-10. Other internal settings
inherit SciPy 1.17.1. Surrogate SLSQP uses the previous analytic configuration
with ftol=1e-13 and maxiter=200; root polish retains xtol=1e-11. Final constraint
target is 9.999e-11, within the public tolerance. Real residuals, not model
predictions, determine whether a candidate is feasible. Use the surrogate
active-constraint Jacobian to correct residuals, keeping active box coordinates
fixed. Never add surrogate outputs to the trace.
**Reason:** The model has strong independent support, so actual calls are more
informative at previously unaudited corners and at the exact feasibility
boundary. Differential evolution supplies an independent search mechanism.
The remaining objective gain from refining 8e-11 toward the public tolerance
is expected to be tiny, but a few real corrections can capture it without
claiming a global optimum.
**Next step limit:** At most 38 additional real objective calls, pausing at
cumulative 504: 32 corners plus no more than six candidate calls.
**Why this length:** Exhaustive corner coverage costs just 32 calls; up to six
corrections bound numerical polishing. Large further sampling blocks are not
justified before seeing this targeted evidence.
**Review:** Inspect corner prediction errors, the independent global-search
candidate, actual objective improvement, and feasibility. Pause on any model
error above 1e-5 in objective or 1e-8 in a constraint. Pause early when a feasible
polish has active residuals within 1.1e-14 of its target, or repeats the same
floating-point candidate. Evaluator errors stop without retry. Reassess whether
additional real evaluations have useful expected information at the checkpoint.

## Decision after evaluation 499
**Budget:** 499 used / 2000 total; 1501 remaining.
**Evidence:** All 32 real corner evaluations agreed with the retained surrogate:
maximum objective error 1.10e-11 and maximum constraint error 4.98e-14. The
independent surrogate differential evolution reached its 500-iteration cap
without meeting its strict stopping tolerance; its predicted objective was
-30665.53861866799, about 5.32e-5 above the incumbent, near the same point.
It made 10665 surrogate objective computations, none of which were real
objective evaluations or trace rows. Analytic local polishing then recovered
the same active set. One real candidate call reached the constraint target
accuracy and triggered the planned early checkpoint, so this block used 33
rather than 38 calls. It improved the observed feasible objective by
2.423621481284499e-8 to -30665.53867190457 at observation #499:
[78,33,29.99525602520233,45,36.77581290629128]. Its six constraints are
[-92.00000000009999,9.99875737761613e-11,-8.840500308982556,
-11.159499691017444,9.99911264898401e-11,-5.000000000099991].
The maximum is below the specified 1e-10 feasibility tolerance. There are
89 feasible observations among all 499 calls.
**Decision:** Stop the campaign early and save the single harness. Preserve the
immutable initial decision, append-only history, and harness-written trace and
metadata. No further objective calls are authorized by this decision.
**Hyperparameters:** Not applicable; no next optimizer block.
**Reason:** Four real constrained solves from distinct starts reached the same
solution. A full quadratic response surface was accurate on held-out existing
observations, 64 fresh global points, and every box corner; observed errors are
at floating-point scale. Its 289 multistart searches agreed, and independent
population search approached the same region before polishing. The final real
refinement reached the allowed constraint boundary to floating-point accuracy.
Together these observations make further sampling or repeated local solves
low expected value compared with preserving the 1501 unused calls. This is a
judgment about diminishing information and improvement, not a guarantee of the
sealed function's global optimum. The surrogate dual lower bound was loose,
and finite validation cannot exclude an unobserved feature.
**Next step limit:** 0 additional objective calls; stop at cumulative 499.
**Why this length:** The numerical refinement target is met and no concrete
unresolved discrepancy suggests a productive additional block.
**Review:** Not applicable; final save and read-only artifact consistency check.

## Decision after evaluation 499 — artifact reconciliation
**Budget:** 499 used / 2000 total; 1501 remaining. No additional evaluations
occurred during saving or verification.
**Evidence:** The final read-only consistency check exposed an evaluator-interface
discrepancy. PUBLIC_EVALUATOR.md states feasibility tolerance 1e-10. Observations
#402 and #499 satisfy that published tolerance, but trace.csv marks both
`feasible=False` and does not update running best for either. Every trace
feasibility flag instead agrees with strict g<=0 on the collected data. Thus
there are 87 harness-accepted feasible observations, versus 89 under the
published tolerance. The final metadata's official best is observation #206,
objective -30665.538670570626, at
[78,33,29.995256030474916,44.99999999999999,36.77581290075662], with constraints
[-91.999999999,-1.0000036354540498e-9,-8.84050030836994,
-11.15949969163006,-1.000000082740371e-9,-4.999999999].
This corrects the preceding decision's reported campaign winner: #499 was the
best under the published tolerance, but #206 is the harness-accepted result.
All objective and constraint values themselves remain actual observations.
**Decision:** Remain stopped. Report the harness-accepted best and the published
tolerance discrepancy. Preserve the already-saved harness artifacts exactly;
append this correction only to the decision history. Do not retry an evaluation,
construct another harness, or modify the trace or metadata.
**Hyperparameters:** Not applicable; no further optimizer block.
**Reason:** The optimization evidence still supports diminishing returns, but
artifact verification is authoritative about the recorded result. The feasibility
inconsistency must be disclosed, not silently repaired or hidden. The final
single-harness save completed before the read-only audit.
**Next step limit:** 0 additional objective calls; final count remains 499.
**Why this length:** Stop and report the evaluator-interface discrepancy rather
than issue further evaluations after the completed campaign.
**Review:** Read-only verification of all 499 rows against the harness's actual
feasibility flags, running best, metadata, budget, and exactly four required
files. No further optimization is scheduled.
