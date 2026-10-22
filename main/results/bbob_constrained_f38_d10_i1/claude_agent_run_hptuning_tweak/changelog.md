## Decision after evaluation 22
**Budget:** 22 used / 100 total; 78 remaining.
**Evidence:** The 22-point Sobol design (seed=0, over the full `[-5,5]^10`
box reported by `harness.bounds`) produced merit values ranging from
6551.4 (best) to 115449.1 (worst), median ≈ 34566. Every single point was
far outside any plausible "normal" objective range, consistent with the
stated behavior that extreme merit values reflect heavy constraint
infeasibility rather than an extreme underlying objective — i.e. the
feasible region (or near-feasible region) appears to occupy only a small
fraction of the box, and essentially the whole box is currently
infeasible. The spread of values (roughly 18x between best and worst) is
large but not astronomically so, and is not obviously discontinuous
(no isolated points many orders of magnitude apart from their neighbors
in rank), which is consistent with (but does not prove) a smoothly
varying penalty term that could support both surrogate modeling and local
descent. Best point so far: x ≈ [1.77, -1.09, 1.97, 2.68, -0.98, 4.65,
1.95, -4.20, -1.57, 1.71], y ≈ 6551.4.
**Decision:** Switch to GP-BO (`src/optimizers/gp_bo.py`, BoTorch
SingleTaskGP + analytic LogEI). Initialize it by `tell()`-ing it the 22
already-evaluated Sobol points directly, rather than discarding them and
re-running its own internal Sobol init — GP-BO's own initial-design rule
is `n_init = 2*(dim+1) = 22`, which exactly matches the number of points
already collected, so this reuses all existing data with zero waste and
the next `ask()` will immediately fit a GP and propose an
acquisition-driven point.
**Hyperparameters:** GP-BO's own inherited defaults (BoTorch
`SingleTaskGP`'s default kernel/priors, `Normalize`/`Standardize`
input/output transforms, analytic `LogExpectedImprovement`,
`optimize_acqf` with `num_restarts=10`, `raw_samples=512` — all as
documented in `src/optimizers/gp_bo.py`'s module docstring, no tuning
applied). `seed=0` for the (unused, since init is supplied externally)
internal Sobol fallback and acquisition-optimizer restarts, for
reproducibility. This choice is appropriate here because the task's own
documented regime for this implementation ("expensive, continuous
problems, budgets of order 10 x dim") matches this task almost exactly
(10 x dim = 100 = the entire budget), and because the evidence above
shows no sign the surface is so pathological (e.g. discontinuous) that a
GP surrogate would be unusable.
**Reason:** With 78 evaluations remaining and a budget-to-dimension ratio
that is in GP-BO's documented sweet spot, sequential model-based search
that uses every single evaluation to refine a global surrogate is more
sample-efficient here than a population-based method (which would spend
many evaluations per generation before adapting) or committing immediately
to a local solver seeded at the current best (which could get stuck if
that point is not in the basin of the true optimum — nothing yet rules
this out with only 22 points).
**Next step limit:** 20 additional objective calls (at most 78 remaining);
pause at cumulative evaluation 42.
**Why this length:** GP-BO proposes one point at a time, refitting the GP
each time; 20 sequential acquisition steps is enough to see whether LogEI
is making real progress (improving best-so-far, proposing points with
materially different/smaller merit than the Sobol set) without committing
more than about a quarter of the remaining budget before the next
checkpoint. Each GP refit at this data size (22-42 points, 10-D) is cheap,
so the limiting factor is information value, not compute cost.
**Review:** At evaluation 42, inspect: (1) whether best-so-far has
improved materially since evaluation 22, and by how much per evaluation;
(2) whether proposed points are concentrating in a region (suggesting the
GP has located a promising basin) or still scattered (suggesting the GP
is still exploring); (3) GP fit diagnostics if progress looks poor (e.g.
whether LogEI keeps proposing near-identical points, which could indicate
numerical issues). No commitment to a later method.

## Decision after evaluation 42
**Budget:** 42 used / 100 total; 58 remaining.
**Evidence:** Over the 20 sequential GP-BO/LogEI evaluations (23-42),
best-so-far improved from 6551.4 to 2341.8 (a ~64% reduction), with three
distinct improving steps spread across the block (evals 24, 32, 35, 41:
3114.6, 2911.7, 2341.8, and a near-tie at 2590.2 at eval 41) rather than
one early gain followed by a plateau — i.e. the surrogate is still
finding better points late in the block, not just at the start. At the
same time, most individual proposals (e.g. 78537, 31910, 74289, 27378)
are far worse than the running best and comparable in scale to the
original Sobol spread, so LogEI is still spending much of its budget
exploring high-uncertainty regions rather than purely exploiting a
located basin. This is consistent with a genuinely difficult/non-smooth
penalized surface and a GP that is still learning it, not with the method
having failed — a short plateau or noisy-looking sequence is expected
behavior for EI-driven search under heavy uncertainty, not evidence to
abandon it. The current best x has one coordinate pinned exactly at the
box edge (x5 = 5.0), which is worth continuing to track but is not yet
actionable with only one such observation.
**Decision:** Continue with GP-BO unchanged, reusing the same `gpbo`
object (all 42 told observations retained; no restart). Continuing is
justified because best-so-far is still improving within the most recent
block (including in its second half), the surrogate-based approach
matches this task's documented budget regime, and there is no evidence
yet (e.g. repeated near-duplicate proposals, stalled best-so-far for a
full block) that would justify switching methods now.
**Hyperparameters:** Unchanged from the previous decision (same `GPBO`
instance: BoTorch `SingleTaskGP` defaults, `LogExpectedImprovement`,
`optimize_acqf` with `num_restarts=10`, `raw_samples=512`, `seed=0`) — see
prior entry for full rationale; no changes indicated by current evidence.
**Reason:** The method is actively improving the best-so-far mid-block,
BoTorch's GP refit cost is still negligible at ~42-62 points, and nothing
in the data argues for a different method yet. Switching now would
discard the chance to exploit the surrogate model that has just started
to pay off.
**Next step limit:** 20 additional objective calls (at most 58 remaining);
pause at cumulative evaluation 62.
**Why this length:** Same rationale as the previous block (20 calls is
enough to see continued improvement or a genuine plateau without
committing a large fraction of the remaining budget); keeping the block
length consistent also makes the two blocks directly comparable for
judging whether the improvement rate is holding, slowing, or accelerating.
**Review:** At evaluation 62, compare the per-block improvement
(best-so-far reduction and number of distinct improving steps) against
this block's result; check whether proposals are starting to concentrate
spatially around the current best (a sign of convergence/exploitation)
versus remaining scattered; watch whether any coordinate is persistently
pinned at a bound, which would suggest the true optimum lies outside this
problem's box or at its edge. No commitment to a later method.

## Decision after evaluation 62
**Budget:** 62 used / 100 total; 38 remaining.
**Evidence:** This 20-evaluation GP-BO block (43-62) produced **no**
improvement in best-so-far (stayed at 2341.76), unlike each of the prior
two blocks. However, several proposals landed noticeably closer to the
current best in merit than the general Sobol-era spread (2802.8, 2840.6,
2857.7, 3084.9 — all within roughly +20-30% of the best, vs. a block
median of 6330.8 and a max of 60079.6), and the smallest observed
proposal-to-best Euclidean distance was 3.99 (vs. a 10-D box diagonal of
~31.6), with most distances in the 5-9 range — i.e. LogEI is still mixing
genuine local refinement attempts with broad exploration, but is no
longer able to find a strictly better point via its global acquisition
surface in a full block. Combined with the previous block's late-block
improvements, this looks like early-stage convergence of the global
surrogate search (diminishing returns) rather than total failure. One
coordinate (x5) in the current best remains at the upper bound (5.0)
across both improving blocks so far, which is worth a local method
re-checking explicitly (pushing against the boundary further is only
possible up to the box edge itself).
**Decision:** Switch to a local derivative-free solver — Py-BOBYQA — seeded
exactly at the current best point found (x ≈ [0.01, 1.30, -0.15, 1.42,
0.73, 5.00, 2.14, -4.35, -3.11, -1.45], y = 2341.76), rather than
continuing GP-BO's global search or using BOBYQA's own wrapper default
(box-midpoint start). The stock `src/optimizers/bobyqa.py` wrapper
hardcodes `x0 = box midpoint`, which would throw away the progress made
so far, so this uses `pybobyqa.solve` directly (same thread+queue
ask/tell adapter pattern as the wrapper, campaign-local, since the
wrapper does not expose a start-point argument) with `x0` set to the
current best. GP-BO's own all-told history (62 points) is preserved in
memory (not discarded) in case a later step wants to return to it or
compare; only the *active* search method changes.
**Hyperparameters:** `scaling_within_bounds=True` and `maxfun` set high
enough not to cut off the search early (inherited from the stock
wrapper's own rationale — Py-BOBYQA's default `maxfun` would otherwise be
far too small, and this project's external budget is the real limit, not
`maxfun`); bounds = `harness.bounds` (unchanged). `rhobeg` (initial
trust-region radius) is set to 1.0 rather than Py-BOBYQA's own default
behavior under `scaling_within_bounds` (0.1 of the rescaled unit cube) —
*increased* because the step is now starting from a point already deep in
infeasible territory with GP-BO's last two blocks showing live,
nearby-but-not-identical points still varying the merit by roughly 2-3x
within a 4-9 unit radius, suggesting the locally exploitable basin is not
extremely narrow; a too-small initial radius would spend many of the
remaining 38 evaluations on tiny, uninformative steps. `rhoend` left at
Py-BOBYQA's own default (1e-8) since over-tightening it is not expected to
help or hurt at this budget (same rationale as the stock wrapper).
**Reason:** GP-BO's global acquisition stalled for a full block while
still producing several near-best points, which is the signature of a
promising local basin that a trust-region local solver can exploit far
more cheaply per evaluation than continuing to pay for a full GP
refit + global multi-start acquisition optimization each step. With only
38 evaluations left, concentrating them on local refinement from the best
known point is a better use of a shrinking budget than continued global
search.
**Next step limit:** 25 additional objective calls (at most 38
remaining); pause at cumulative evaluation 87.
**Why this length:** BOBYQA is a local method expected to converge (and
stop proposing new points) on its own well before a large budget is
used; 25 calls is enough to see most or all of a typical convergence
trajectory from this starting point while preserving a reserve
(13 evaluations) to react if it converges early, stalls, or needs a
restart with adjusted hyperparameters.
**Review:** At evaluation 87 (or earlier, if BOBYQA's internal solver
thread exits / signals convergence before then, which is an explicit
earlier-pausing condition for this step): best-so-far improvement versus
2341.76; whether BOBYQA converged (exhausted its own stopping criteria)
or was still proposing meaningfully different points when the step limit
was reached; whether the x5=5.0 boundary coordinate moved off the bound
or stayed pinned (informing whether it reflects a true constraint on that
coordinate). No commitment to a later method.

## Correction before any evaluation in this step (still at evaluation 62)
**Budget:** 62 used / 100 total; 38 remaining (unchanged — this correction
consumed zero objective evaluations).
**Evidence:** Attempting to launch Py-BOBYQA with `rhobeg=1.0` under
`scaling_within_bounds=True` failed immediately with Py-BOBYQA's own
input-validation error ("gap between lower and upper must be at least
2*rhobeg") before calling the objective even once (confirmed separately:
0 calls to the objective function were made). Under
`scaling_within_bounds=True`, Py-BOBYQA rescales the problem to the unit
cube `[0,1]^d` internally, so `rhobeg` must satisfy `rhobeg <= 0.5` in
that scaled space, not the original `[-5,5]` units as I had incorrectly
assumed when picking 1.0. This is a corrected understanding of the
library's own documented interface, not new information about the
objective, and consumed no evaluations (the solver errors out
synchronously before the ask/tell loop starts).
**Decision:** Re-launch with the same seeding (`x0` = current best) and
all other settings unchanged, with `rhobeg=0.4` instead of 1.0 — the
largest round value still comfortably inside the valid `(0, 0.5)` range,
preserving the original intent (a larger-than-default initial trust
region, since Py-BOBYQA's own default under scaling is 0.1) without
hitting the validity boundary.
**Hyperparameters:** `rhobeg=0.4` (scaled units); all else unchanged from
the immediately preceding decision.
**Reason:** Keeps the same rationale (bigger first step than the library
default, given the local basin observed in the last two GP-BO blocks
spans several units in several directions) while respecting the library's
actual constraint.
**Next step limit:** 25 additional objective calls (unchanged from the
preceding decision; at most 38 remaining); pause at cumulative evaluation
87.
**Why this length:** Unchanged from the preceding decision.
**Review:** Unchanged from the preceding decision.

## Decision after evaluation 87
**Budget:** 87 used / 100 total; 13 remaining.
**Evidence:** Over this 25-evaluation BOBYQA block (63-87), best-so-far
improved only slightly, from 2341.76 to 2320.79 (~0.9%), reaching it only
on the very last evaluation of the block. The trajectory was not a clean
monotone local descent: several early and mid-block proposals were far
worse than the start (e.g. 8037.7, 17133.3, 27962.9), consistent with
Py-BOBYQA's own interpolation-set-building behavior (it evaluates points
around the current iterate to build its local quadratic model, not only
downhill steps), while the last four evaluations (85-87: 2385.8, 2370.1,
2320.8) show the trust region tightening onto a genuinely improving
sequence. The solver thread had not signaled convergence
(`converged_flag=False`) when the step limit was reached — it was still
actively running, not finished. This indicates the local basin around the
GP-BO best is real but shallow/hard to descend quickly, and BOBYQA is
still in progress, not stalled.
**Decision:** Continue BOBYQA unchanged for the remaining evaluations,
rather than switching methods again or stopping early. With only 13
evaluations left, there is not enough remaining budget to productively
set up and run a different method (e.g. GP-BO's GP refit + multi-start
acquisition optimization is comparatively expensive per useful step at
this budget size, and a from-scratch restart of any method would forfeit
the local progress just gained); continuing the one local solver that is
actively mid-convergence is the best use of a very small remaining
budget. (Operationally, this restarts the BOBYQA solver thread seeded at
the current overall best, x ≈ [...], y=2320.79 — the in-process solver
state itself is not resumable across process boundaries — but this is a
continuation of the same method and local search, not a change of
method or starting region.)
**Hyperparameters:** Unchanged: `rhobeg=0.4` (scaled units),
`scaling_within_bounds=True`, bounds = `harness.bounds`, `rhoend` =
Py-BOBYQA's own default. Seeded at the new current best (x with
y=2320.79) rather than the previous seed, since that is now the best
known point.
**Reason:** The evidence shows continued, if slow, improvement and an
unfinished solver run; with a 13-evaluation budget there is no realistic
alternative that could do better than letting the method already showing
a live improving trend keep going.
**Next step limit:** 13 additional objective calls (all remaining
budget); pause at cumulative evaluation 100 (budget exhaustion) or
earlier if the solver thread signals convergence first.
**Why this length:** Only 13 evaluations remain in the entire budget;
there is no reason to reserve any further, since no subsequent checkpoint
beyond budget exhaustion is possible.
**Review:** At evaluation 100 (or solver convergence, if earlier): final
best-so-far and whether BOBYQA was still improving or had converged by
budget exhaustion. This is the final step of the campaign; stop
afterward regardless of outcome, since no budget remains.

## Decision after evaluation 100 (final)
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** The final 13-evaluation block (88-100) produced no further
improvement in best-so-far (remained at 2320.7873 from the first
evaluation of the block onward); the nearest competing values were
2416.30 and 2469.25, with several much worse proposals (up to 15355.4)
again consistent with Py-BOBYQA's interpolation-point probing rather than
a monotone descent. The solver thread had not signaled convergence
(`converged_flag=False`) when the budget ran out — it was interrupted
mid-search, not finished on its own terms. The evaluation budget is now
fully exhausted; `harness.evaluate()` can no longer be called.
**Decision:** Stop. The harness enforces a hard 100-evaluation budget;
`harness.save()` has been called and `trace.csv`/`meta.json` are written.
No further evaluation is possible or permitted regardless of whether
BOBYQA looked like it still had room to improve.
**Hyperparameters:** Not applicable — no further step.
**Reason:** Budget exhaustion is a hard stopping condition per the
protocol; there is nothing left to decide beyond confirming the outcome.
**Next step limit:** 0 (stopping).
**Why this length:** Not applicable.
**Review:** Not applicable — campaign complete.

**Final outcome:** Best merit found: **2320.7873059238514**, at
x ≈ [0.062, 1.102, 0.035, 1.537, 0.670, 5.000, 2.137, -4.163, -3.290,
-1.486] (evaluation #88, carried over unchanged through the final block).
Total evaluations used: 100/100 (budget fully exhausted; this is why the
campaign stopped, not a voluntary early stop). Method sequence actually
used: Sobol space-filling (22 evals) -> GP-BO/LogEI (40 evals total,
across two blocks) -> BOBYQA local refinement seeded at the GP-BO best
(38 evals total, across two blocks, restarted once at the updated best
between blocks). Across the whole campaign, best merit improved from
6551.4 (best of the initial Sobol design) to 2320.8 (~65% reduction),
with most of the gain coming from GP-BO's global surrogate search and a
smaller additional gain from local refinement; x5 remained pinned at the
upper bound (5.0) throughout every improving point found after the
initial design, suggesting the true constrained optimum (or at least this
basin's optimum) lies on that face of the box.
