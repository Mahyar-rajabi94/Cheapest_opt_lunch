## Decision after evaluation 22
**Budget:** 22 used / 100 total; 78 remaining.
**Evidence:** Completed GP-BO's built-in 22-point scrambled Sobol initial
design (seed=0). Observed `y` spans a wide dynamic range: min=16267.5,
max=751501.4, median=231760.7, std=199155.6 (roughly on the same order as
the median, i.e. heavy spread rather than a tight cluster). Correlation
between `y` and squared distance from the box center (`||x||^2`) across
the 22 points is +0.68 — a moderate global trend exists (points near the
box center and a `||x||^2` of roughly 70-90 tend to score noticeably
better than points with `||x||^2` above ~100), but it is not a clean
monotonic relationship: one point with `||x||^2`=49.2 (closer to center
than most) scored y=103269, far worse than several points with larger
`||x||^2`. This indicates the function is not purely radially symmetric
around the box center — there is curvature/anisotropy a GP with its own
fitted lengthscales per dimension should be able to pick up, but a naive
"just move toward the center" heuristic would not reliably exploit it. No
errors or unexpected behavior in the harness or optimizer; GP fit/acqf
timing was fast (well under a second per evaluation), so GP-BO is cheap
enough at this scale to keep using for several more steps.
**Decision:** Continue with GP-BO (same `GPBO` object, same seed=0 run,
full history of the 22 told points retained — no restart). Move into its
acquisition-driven phase: now that `n_init=22` is exceeded, each `ask()`
call fits a fresh `SingleTaskGP` on all data told so far and proposes the
single point maximizing analytic LogEI.
**Hyperparameters:** Unchanged from the initial configuration recorded in
`roadmap_initial.md` (`src/optimizers/gp_bo.py`, unmodified defaults):
`SingleTaskGP` default kernel/priors with input `Normalize` and output
`Standardize` (the latter is particularly relevant given the wide `y`
range just observed), `LogExpectedImprovement` (noiseless, `best_f` = the
running minimum), `optimize_acqf` with `num_restarts=10`,
`raw_samples=512`. No tuning applied; these are the implementation's own
standard values, not problem-specific choices.
**Reason:** The initial design showed real, exploitable structure (a
moderate global trend plus apparent curvature) rather than either a flat,
unstructured landscape or an already-converged optimum, so there is
information for a surrogate model to use. GP-BO is explicitly suited to
budgets "of order 10x dim" (our full budget is exactly 10x the dimension
of 10), and with 78 evaluations remaining this step still fits comfortably
within that regime. Sequential GP-BO (n=1 per fit) is the most
sample-efficient option available here for exploiting a moderately
structured, expensive-feeling (in the sense of being budget-scarce)
10-dimensional landscape, compared to population-based methods
(CMA-ES, TuRBO) that are typically more sample-hungry per generation.
**Next step limit:** 20 additional objective calls, at most 78; pause at
cumulative evaluation 42.
**Why this length:** Large enough to let several rounds of GP
refitting/LogEI exploitation actually move the best-so-far value and
reveal whether the surrogate model is finding a good descent direction,
but small enough (roughly a quarter of the remaining budget) to check in
again before committing further evaluations to a method that may plateau
or start over-exploiting a bad local region.
**Review:** At evaluation 42, inspect: the trajectory of
`y_best_so_far` (strictly decreasing / rate of improvement), whether LogEI
proposals are clustering (exploitation) or still spread out (exploration),
and per-evaluation wall time (GP refit cost grows with data size). Will
reconsider continuing GP-BO as-is, switching to a local refiner (BOBYQA or
Nelder-Mead) from the current best point if GP-BO looks to have located a
promising basin, or restarting/hybridizing if progress stalls. No method
committed in advance.

## Decision after evaluation 42
**Budget:** 42 used / 100 total; 58 remaining.
**Evidence:** The 20-evaluation GP-BO acquisition block (evaluations
23-42) improved `y_best_so_far` from 16267.5 to 7363.0 — a >2x
improvement — with the improving point arriving at evaluation 38 (the
16th point of this block), not immediately. The 5 evaluations after that
(39-42: y = 31657.5, 32150.4, 61536.6, 182050.1) did not improve on it,
but they are not tightly clustered near the incumbent either (values
ranging over roughly an order of magnitude) — LogEI is still proposing
points with real spread, consistent with the acquisition function still
balancing exploration against exploitation rather than having collapsed
onto a single local optimum. Per-evaluation objective cost remains
negligible (`eval_seconds` ~0 each); the GP refit + `optimize_acqf` cost
(not separately logged, but reflected in total step wall time) stayed
practical for a dataset of size ~22-42. A single 5-evaluation stretch
without improvement is not, on its own, evidence the method has
stalled — LogEI's expected improvement is not monotonically decreasing
pointwise even when the search is working well, and the paragraph above
shows the preceding block had an 18-evaluation gap before its own
improving point.
**Decision:** Continue GP-BO unchanged (same `GPBO` object and full
42-point history retained; no restart).
**Hyperparameters:** Unchanged from the previous entry; referencing that
configuration (`SingleTaskGP` defaults, `LogExpectedImprovement`,
`optimize_acqf(num_restarts=10, raw_samples=512)`), not repeated here.
**Reason:** A single short non-improving stretch does not meet the bar
for switching methods given the explicit guidance to account for noise
and short plateaus; the method produced a real, large improvement within
this very block and continues to propose a non-degenerate spread of
candidate values. With 58 evaluations remaining (still well inside
GP-BO's own ~10x-dim intended regime for a 10-dim problem), there is
ample room to let it continue exploiting the structure identified so far
before reserving part of the budget for a possible local-refinement
hand-off near the end of the campaign.
**Next step limit:** 20 additional objective calls, at most 58; pause at
cumulative evaluation 62.
**Why this length:** Matches the previous step's length, which was long
enough to produce a clear signal (a >2x improvement) while still leaving
a meaningful fraction of the budget (38 evaluations) in reserve after this
block for either further GP-BO exploitation or a dedicated local-refiner
hand-off (e.g. BOBYQA/Nelder-Mead from the current best point) once the
surrogate's progress is clearer.
**Review:** At evaluation 62, inspect the updated `y_best_so_far`
trajectory, whether this block improved on 7363.0 and by how much, and
whether LogEI's proposed points are now visibly narrowing around a
region (a sign that a local refiner could cheaply polish the result) or
still broadly spread (a sign GP-BO is still doing useful global search).
Will decide then whether to continue GP-BO, hand off to a local solver,
or stop early if remaining budget plus observed plateau make further
evaluations unlikely to help. No method committed in advance.

## Decision after evaluation 62
**Budget:** 62 used / 100 total; 38 remaining.
**Evidence:** This 20-evaluation block (43-62) improved `y_best_so_far`
three more times: 7363.0 -> 3861.6 (eval 44) -> 3080.7 (eval 47) ->
2804.6 (eval 54), then no further improvement through eval 62. Inspecting
the actual proposed points shows LogEI has clearly identified a basin: a
cluster of points (44, 47, 51, 54, 58, 61) all sit close to
x ~ [-1, 0.3, -1.9, 2.0, -4.0, -3.0, -2.3, -2.7, -0.3, 2.6] with y values
in the 2800-9800 range, while other points in the same block (e.g. 48,
49, 50, 56, 60, 62) are far from that cluster with much worse y (up to
278262.3 at eval 62) -- i.e. LogEI is still interleaving exploitation of
the identified basin with genuine exploration elsewhere, not collapsed
into pure local refinement yet. The last 8 evaluations (55-62) did not
improve the incumbent, but 4 of them (57, 58, 59, 61) are near-cluster
points with similar-order y, consistent with the GP's own noise-free
uncertainty near a basin it has already sampled densely, not with the
method having nothing left to find.
**Decision:** Continue GP-BO unchanged for one more block (same object,
full 62-point history retained).
**Hyperparameters:** Unchanged; referencing the configuration recorded in
the evaluation-22 entry (not repeated here).
**Reason:** The within-block improvement trend (three strict improvements
in 20 evaluations) is still active, and the basin GP-BO has found is
being refined rather than abandoned. 38 evaluations remain, which is
enough for roughly one more exploitation/exploration block plus a
reserved tail; it is not yet clear whether that tail should go to more
GP-BO or to a local polish step (e.g. BOBYQA/Nelder-Mead seeded at the
current best), so that decision is deferred to the next checkpoint rather
than being locked in now.
**Next step limit:** 20 additional objective calls, at most 38; pause at
cumulative evaluation 82.
**Why this length:** Keeps the same block size used in the two prior
checkpoints (which has reliably surfaced a clear improve/plateau signal
each time) while deliberately leaving at least ~18 evaluations in
reserve afterward -- enough to run a short dedicated local-refinement
pass at the end of the campaign if the evidence at evaluation 82 favors
that over continuing GP-BO.
**Review:** At evaluation 82, inspect whether `y_best_so_far` improved
further and by how much, whether newly proposed points are now tightly
clustered around the current best (suggesting the remaining budget is
better spent on a cheap local refiner than on further expensive GP
refits), and remaining budget vs. the cost of one more GP-BO block. No
method committed in advance for the final stretch.


## Decision after evaluation 82
**Budget:** 82 used / 100 total; 18 remaining.
**Evidence:** This block (63-82) kept improving steadily and
monotonically in six separate steps: 2804.6 -> 2623.5 (eval ~68) ->
2394.9 (~72) -> 1989.4 (~75) -> 1476.3 (~78) -> 1192.8 (~79), i.e. roughly
one new improvement every 3-4 evaluations throughout the block, with no
extended non-improving stretch like the ones seen in earlier blocks. This
is the opposite of the "LogEI has nothing left to find" signal that would
favor handing off to a local refiner now: the acquisition function is
still finding better points inside (and apparently still shrinking
around) the basin identified earlier, at a rate at least as good as any
previous block.
**Decision:** Continue GP-BO unchanged through the remainder of the
budget (same object, full 82-point history retained; no restart).
**Hyperparameters:** Unchanged; referencing the configuration recorded in
the evaluation-22 entry (not repeated here).
**Reason:** With only 18 evaluations left, switching to a different local
solver now (e.g. BOBYQA/Nelder-Mead) would spend some of that budget on
that solver's own startup/trust-region-sizing overhead before it could
out-perform a method that is, on the evidence just observed, still
actively improving at a healthy rate within the same basin. There is no
plateau to justify the switch, and GP-BO's per-point cost at n~82-100 is
still practical (full blocks of 20 have been completing promptly
throughout the campaign). Finishing with the method already shown to be
working is the better use of a small remaining budget than paying a
switching cost now.
**Next step limit:** 18 additional objective calls, at most 18 (all
remaining budget); pause at cumulative evaluation 100 (budget
exhaustion).
**Why this length:** This is the entire remaining budget. Splitting it
into a smaller block and reserving some for yet another method would only
make sense if there were a concrete, evidence-based reason to expect a
different method to do better with fewer evaluations than GP-BO needs to
keep improving -- the evidence from this block shows the opposite is true
right now, so the full remainder is committed to one final GP-BO block
rather than held back without a reason to use it differently.
**Review:** This is the final step; at evaluation 100 (or budget
exhaustion) the outcome will be recorded and the campaign stopped. No
further decision point follows.

## Outcome at evaluation 100 (budget exhausted)
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** The final 18-evaluation GP-BO block produced one further
improvement, from 1192.8 to 1026.2 (at the 14th point of this block,
cumulative evaluation ~95), then no further improvement through
evaluation 100. Across the whole campaign, `y_best_so_far` fell
monotonically: 16267.5 (end of Sobol init, eval 22) -> 7363.0 (eval 42) ->
2804.6 (eval 62) -> 1192.8 (eval 82) -> 1026.2 (eval 100) -- a steady,
still-ongoing improvement trend right up to the last evaluation, with no
sign the method had exhausted what it could find in the available budget.
**Decision:** Stop. The evaluation budget (100/100) is exhausted; no
further `evaluate()` calls are permitted by the harness.
**Hyperparameters:** Not applicable (no further step).
**Reason:** Budget exhaustion is a hard stop per the harness and the
protocol; this is not a judgment call.
**Next step limit:** 0 (none remaining).
**Why this length:** Not applicable.
**Review:** Not applicable -- campaign complete. Final best observed:
y = 1026.1832616040374 at
x = [-1.3946255416194573, -0.6655218338941356, -1.2367329247028556,
1.4032135609063714, -4.50463719638259, -1.6747712751864985,
-2.6282902527460017, -1.3887630038319279, 0.7542512190920274,
4.2080096991692715], found using a single continuous GP-BO run
(LogEI acquisition, seed=0) across its full 22-point Sobol initial design
plus 78 acquisition-driven points, with no method switches during the
campaign (GP-BO was reconsidered, and explicitly re-chosen with reasons
recorded, at every checkpoint: evaluations 22, 42, 62, and 82).
