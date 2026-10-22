## Decision after evaluation 14
**Budget:** 14 used / 100 total; 86 remaining.
**Evidence:** Completed the full GP-BO initial Sobol design (14 points,
`seed=0`, bounds [0,1]^6). Observed y values ranged from 0.288533 to
0.693742 (mean 0.448743, std 0.136307) — a moderate, roughly 2.4x spread
with no extreme outliers or failures, suggesting the objective is bounded
and reasonably well-behaved over this box, not pathologically spiky at
this coarse sampling density. Best point so far:
y=0.288533 at x=[0.60524, 0.44838, 0.79001, 0.80003, 0.40476, 0.84379].
All 14 evaluations ran without error, confirming the harness/GPBO
integration works correctly.
**Decision:** Continue with the same method, GP-BO, now entering its
acquisition-driven phase (LogEI on a GP fit to all 14 points so far).
State reuse: continue the same `GPBO` instance (`X_`, `y_` from all 14
points retained) and the same harness/budget counter; no restart.
**Hyperparameters:** Inherited defaults from `src/optimizers/gp_bo.py`
(BoTorch SingleTaskGP, default kernel/priors, `Normalize`+`Standardize`
transforms, analytic LogEI with `maximize=False`), acquisition
optimization via `optimize_acqf` with `num_restarts=10`, `raw_samples=512`
(BoTorch tutorial standard pair) — none of these have been changed from
the previous configuration (the previous step did not exercise them at
all, since it was pure Sobol init). No manual tuning applied yet; these
are reasonable, well-tested defaults for a first GP-BO block and GP-BO
exposes no problem-specific setting that would warrant a different choice
without evidence yet of e.g. acquisition getting stuck.
**Reason:** With 14 points, a GP fit is meaningful (BoTorch's own
dimension-scaled lengthscale prior is designed to work with this much
data) and the LogEI acquisition can now start targeting the global
optimum based on actual observed structure rather than blind sampling.
Exact GP fitting cost is still trivial at this data size (~14-30 points),
so there is no efficiency reason to use large blocks; a moderate block
lets me check whether LogEI is converging toward a region or still
exploring broadly before committing much more of the 86 remaining calls.
**Next step limit:** 15 additional objective calls, at most 86; pause at
cumulative evaluation 29.
**Why this length:** Each GP-BO iteration costs one full GP refit, so 15
single-point acquisition steps is enough to see a clear local trend in
best-so-far (improvement rate, whether LogEI keeps proposing clustered or
scattered points) without spending a large fraction of the remaining
budget before the first real look at acquisition-driven behavior.
**Review:** At evaluation 29, check: (a) whether `y_best_so_far` improved
over 0.288533 and by how much, (b) whether proposed points are clustering
(exploitation) or still spread out (exploration), (c) any numerical issues
in GP fitting. No method change is preselected; continuing, adjusting
restarts/raw_samples, or switching will depend on this evidence.

## Decision after evaluation 29
**Budget:** 29 used / 100 total; 71 remaining.
**Evidence:** The 15 acquisition-driven points in this block had y in
[0.297200, 0.495770], mean 0.340586, std 0.046192 — markedly tighter than
the Sobol block's std of 0.136307, confirming LogEI has shifted from
broad exploration to exploiting the neighborhood of the current best
(most proposals landed in the 0.30-0.35 band, close to but not below the
incumbent 0.288533). However, `y_best_so_far` did **not** improve this
block (still 0.288533 from the init design); the closest attempt was
0.297200, within ~3% of the incumbent. No numerical/fitting errors
occurred. This is a short, single plateau (15 points) in a method whose
model is only 29 points deep in 6D — not sufficient evidence that GP-BO
is stuck, since 15 non-improving steps after switching from a diffuse to
exploiting regime is a normal pattern while the GP sharpens its estimate
of the optimum's exact location.
**Decision:** Continue with the same method, GP-BO, same `GPBO` instance
(all 29 points retained, no restart).
**Hyperparameters:** Unchanged from the previous block (BoTorch
SingleTaskGP defaults, LogEI, `num_restarts=10`, `raw_samples=512`). No
adjustment yet — the evidence (tightening exploitation, near-miss
candidates) suggests the acquisition is working as intended and simply
needs a few more iterations for the GP to resolve the optimum more
precisely, not that its configuration is miscalibrated.
**Reason:** A single non-improving block right after the exploration-to-
exploitation transition is expected behavior for EI-type acquisitions
refining a local estimate; abandoning GP-BO or changing its settings now
would discard information just as the model is converging on a
promising region, without evidence that a different method or setting
would do better here.
**Next step limit:** 15 additional objective calls, at most 71; pause at
cumulative evaluation 44.
**Why this length:** Same rationale as the previous block: small enough
to reassess soon given GP refits are fast, large enough to tell whether
the exploitation region continues to tighten/improve or whether GP-BO is
genuinely plateaued and a switch (e.g. to a local polish method, or
restarting exploration) is warranted.
**Review:** At evaluation 44, check whether `y_best_so_far` has improved
below 0.288533, how much tighter/looser the new block's y-spread is, and
whether proposed points keep concentrating near the same region (possible
local optimum) or move elsewhere (possible correction). Two consecutive
non-improving blocks with a tight, unchanging cluster would be stronger
evidence to consider switching/hybridizing (e.g. a local solver seeded at
the incumbent, or encouraging exploration) than one block alone.

## Decision after evaluation 44
**Budget:** 44 used / 100 total; 56 remaining.
**Evidence:** This block's y values were in [0.279918, 0.350528], mean
0.322149, std 0.023065 — tighter again than the previous block (std
0.046192), showing continued exploitation convergence. `y_best_so_far`
improved from 0.288533 to 0.279918, found at
x=[0.69472, 0.43898, 0.67424, 0.758, 0.27633, 0.0] — notably, the 6th
coordinate sits exactly at the lower bound (0), suggesting the optimum
may lie on or near that face of the box rather than in the interior. No
errors. Two consecutive blocks have now shown the acquisition tightening
around a region and the second one broke through the earlier plateau, so
GP-BO is actively making progress, not stuck.
**Decision:** Continue with the same method, GP-BO, same instance
(44 points retained, no restart).
**Hyperparameters:** Unchanged (`num_restarts=10`, `raw_samples=512`,
BoTorch SingleTaskGP/LogEI defaults as before) — the method is improving
and exploiting sensibly, including correctly identifying a boundary-
active coordinate, so there is no evidence yet to adjust the acquisition
optimizer's restart/sample budget.
**Reason:** Continued, if slow, improvement plus a tightening exploitation
cluster indicates the GP's model of the local optimum is still being
refined productively; the boundary-active coordinate is new, useful
structural information best exploited by continuing the same
surrogate-based search rather than switching away from it now.
**Next step limit:** 15 additional objective calls, at most 56; pause at
cumulative evaluation 59.
**Why this length:** Consistent with prior blocks — enough to observe
whether the recent improvement is an isolated step or part of a continuing
trend, without over-committing given 56 calls still remain for later
reconsideration (e.g. a local-polish step once GP-BO's gains clearly
taper off).
**Review:** At evaluation 59, check magnitude/rate of further improvement,
whether other coordinates also start pressing against bounds, and whether
the y-spread keeps tightening (healthy convergence) or stalls/widens
(would suggest reconsidering exploration vs. a local solver restart from
the incumbent).

## Decision after evaluation 59
**Budget:** 59 used / 100 total; 41 remaining.
**Evidence:** This block's y values were in [0.290842, 0.576447], mean
0.356765, std 0.071897 — wider than the previous block (std 0.023065),
including one notably high outlier (0.576447), indicating LogEI pushed
some exploration back out rather than purely exploiting. `y_best_so_far`
did not improve (still 0.279918); closest candidate was 0.290842. This is
the second non-improving block out of the last three (one improving block
at eval 44 in between), still within normal BO fluctuation given the GP
continues to reassess the acquisition landscape as more data arrives, but
the widening spread plus lack of improvement is worth tracking closely:
if the next block also fails to improve while the budget keeps shrinking,
that would be stronger evidence to shift some of the remaining budget to
a local solver exploiting the incumbent directly instead of continuing
global GP-BO indefinitely.
**Decision:** Continue with the same method, GP-BO, same instance
(59 points retained, no restart), for one more block before deciding
whether to keep going or hybridize with a local refinement step.
**Hyperparameters:** Unchanged (`num_restarts=10`, `raw_samples=512`,
BoTorch defaults as before). Not yet adjusting: the widened exploration
this block could plausibly resolve productively (e.g. discovering a
better region near the boundary-active coordinate found earlier), and
changing acquisition settings without clearer evidence of a specific
failure mode (e.g. GP fitting instability) would not be justified.
**Reason:** One non-improving, moderately wider-spread block does not by
itself indicate GP-BO has stopped being useful — EI-type acquisitions do
alternate between exploitative and more exploratory proposals as the
model's uncertainty estimate changes with new data. Remaining budget
(41) is still large enough to afford one more diagnostic block before
committing to a different strategy.
**Next step limit:** 15 additional objective calls, at most 41; pause at
cumulative evaluation 74.
**Why this length:** Keeps the same diagnostic block size used throughout
this campaign, balancing a clear enough read on the trend against not
overspending budget on a method that may need to be supplemented by a
local polish step with the remaining ~26 evaluations after this block.
**Review:** At evaluation 74, check whether `y_best_so_far` improves
below 0.279918, whether the spread tightens again (renewed exploitation)
or stays wide (possible genuine plateau), and explicitly weigh switching
some of the final ~26 evaluations to a local solver (e.g. BOBYQA or
L-BFGS-B, Py-BOBYQA/scipy, seeded at the current incumbent) to polish the
best region found so far if GP-BO shows no further net improvement across
two consecutive blocks.

## Decision after evaluation 74
**Budget:** 74 used / 100 total; 26 remaining.
**Evidence:** This block's y values were in [0.292635, 0.367528], mean
0.316227, std 0.017525 — tight again (similar to the eval-44 block), but
still no improvement over the incumbent 0.279918 (closest was 0.292635).
Across the last three 15-point blocks (29->44, 44->59, 59->74),
`y_best_so_far` improved only once (at eval 44, from 0.288533 to
0.279918) and has not moved in the two blocks since, despite the
exploitation spread repeatedly tightening into the 0.29-0.35 band around
the incumbent. This is a consistent pattern, not a single short plateau:
GP-BO keeps re-converging on essentially the same neighborhood without
beating its best point in it, which is the kind of evidence the roadmap
and prior decisions flagged as worth switching on. Remaining budget (26)
is now small enough that continued full GP refits (cost growing with
~75+ points) yield diminishing exploratory value per evaluation compared
to a solver that can move directly and efficiently within the already-
located basin.
**Decision:** Switch method for the remaining budget: a local polish
with BOBYQA (Py-BOBYQA, trust-region model-based solver), seeded at the
current GP-BO incumbent `x=[0.69472, 0.43898, 0.67424, 0.758, 0.27633,
0.0]` rather than the box midpoint. The project's packaged `BOBYQA`
wrapper (`src/optimizers/bobyqa.py`) hardcodes `x0 = box midpoint`
(`self.x0 = (lower + upper) / 2`), which would waste evaluations
re-finding a basin GP-BO already localized, and starting there also
ignores the coordinate-5 boundary activity (x5=0) GP-BO discovered. Per
the protocol's guidance to use the underlying library's own API via a
campaign-local adapter when a wrapper fixes a setting I need to control,
I call `pybobyqa.solve` directly (same background-thread/queue pattern as
the existing wrapper, same `scaling_within_bounds=True`, same
`maxfun=20000` convention so the harness's own budget counter remains the
only real stopping control) with `x0` set to the incumbent instead of the
midpoint. This is a restart in the sense of using a new optimizer object;
GP-BO's own `X_`/`y_` history and the harness's evaluation counter are
both preserved (no evaluations are discarded or redone), and the new
solver starts from the best observed point rather than from scratch.
**Hyperparameters:** `scaling_within_bounds=True`, `rhobeg`/`rhoend` left
at Py-BOBYQA's own defaults (unchanged from the packaged wrapper's
documented configuration: rhobeg=0.1 under scaling, rhoend=1e-8),
`maxfun=20000` (same as the wrapper, i.e. not the real stopping
condition). The only change from the packaged wrapper is `x0`: incumbent
point instead of the box midpoint, chosen specifically because 74
evaluations' worth of GP-BO evidence already localizes a better starting
region than the midpoint.
**Reason:** Two consecutive non-improving, tightly-clustered GP-BO blocks
after one earlier improving block indicate the GP's global model has
converged on this basin and is no longer finding better points within it
via LogEI; a dedicated local trust-region solver seeded at the incumbent
is better suited to efficiently refine within a basin using few
evaluations, which matches the small remaining budget (26).
**Next step limit:** 26 additional objective calls, at most 26 (all
remaining budget); pause earlier if BOBYQA's solver thread signals
convergence (raises `OptimizerExhausted`) before using all 26.
**Why this length:** This is the final block: all remaining budget is
reasonably allocated to a single local-refinement run, since there is no
further method-selection decision to defer evaluations for, and BOBYQA
may legitimately converge (and stop proposing points) before exhausting
the block, which would itself be informative.
**Review:** At the end of this block (26 calls or BOBYQA convergence,
whichever first), check whether `y_best_so_far` improved over 0.279918
and by how much, and confirm whether BOBYQA converged on its own or was
cut off by the budget. This determines whether to stop (if budget
exhausted or converged with no further useful options) or, if BOBYQA
converges early with budget still left, reconsider a further short step.

## Decision after evaluation 100 (final)
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** The BOBYQA-from-incumbent block ran all 26 of its allotted
evaluations (eval 75-100) without raising its own convergence signal
first -- i.e. the trust-region solver was still actively iterating when
the harness budget ran out, not finished on its own. It substantially
improved on the GP-BO incumbent: `y_best_so_far` dropped from 0.279918 to
a final 0.260655, with a clear, steady descent over the block
(0.279918 -> 0.261658 at eval 79 -> 0.261053 at eval 92 -> 0.260839 ->
0.260759 -> 0.260675 -> 0.260698 -> 0.260655 at eval 100), and the last
several values are converging tightly (within ~0.00005 of each other),
consistent with the trust-region radius shrinking toward `rhoend` right
as the budget was exhausted. Final best point:
`x = [0.69349, 0.448968, 0.68045, 0.870802, 0.273834, 0.017396]`,
`y = 0.260655`. Note coordinate 5 moved off the exact lower bound (0.0,
found by GP-BO) to 0.017396 under the local solver, consistent with the
true local optimum being near but not exactly on that face.
**Decision:** Stop. The evaluation budget (100/100) is fully exhausted;
no further `evaluate()` calls are permitted by the harness.
**Hyperparameters:** Not applicable (no further step planned).
**Reason:** Hard budget exhaustion, the only non-negotiable stopping
condition in the protocol. Separately, the evidence also supports that
this was a reasonable point to have switched to local refinement: BOBYQA
converted the last 26 evaluations into a further ~7% reduction in y
(0.279918 -> 0.260655) beyond what 60 additional GP-BO acquisition
evaluations alone had achieved, which it obtained efficiently by
exploiting the basin GP-BO had already localized rather than continuing
costly full-GP refits.
**Next step limit:** 0 (stopping).
**Why this length:** Not applicable.
**Review:** Not applicable; campaign complete.

### Operational note (process, not an optimization decision)
Across the campaign, each step was executed by a short-lived Python
process (via the `.venv` interpreter) that reconstructed the ongoing
`AgentTaskHarness` + optimizer state from a pickle file in a scratch
directory between steps, since each tool invocation starts a fresh
process; the harness and its accumulated `_rows`/evaluation counter were
preserved intact across all steps this way, and real `evaluate()` calls
only ever happened inside these scripts. On the final block, an attempt
to also pickle the BOBYQA adapter's background-thread/queue objects after
finishing the block failed (`_thread.lock` is not picklable) -- however,
this occurred only *after* the block's 26 real evaluations had already
completed and, since they brought the cumulative count to exactly 100,
the harness's own automatic `save()` (triggered internally by
`evaluate()` once `budget` is reached) had already written the final
`trace.csv`/`meta.json` to the results directory before the unrelated
pickling error occurred. No evaluation was lost, repeated, fabricated, or
hand-edited; the saved trace reflects exactly the 100 real
`harness.evaluate()` calls made over the course of the campaign. No
driver/adapter scripts were written into the sandbox or results
directory; all were kept in a separate scratch directory and are not
part of the deliverables.
