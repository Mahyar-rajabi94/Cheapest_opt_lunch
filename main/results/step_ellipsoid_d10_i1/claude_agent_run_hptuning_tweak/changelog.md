## Decision after evaluation 30
**Budget:** 30 used / 100 total; 70 remaining.
**Evidence:** The 22-point Sobol' initial design (GP-BO's `n_init`) found
values ranging from 275.28 (min) to 5206.29 (max), mean 1828.41, std
1277.12 — a very large dynamic range across the box, suggesting either a
steep/ill-conditioned landscape (large curvature anisotropy) or at least one
much-better basin than most of the box. The best value (275.28) was reached
at the 5th Sobol' point and never improved on by the subsequent Sobol'
points or by any of the first 8 GP-driven (LogEI-selected) points
(23-30: values 507.65, 619.85, 1186.31, 321.17, 714.01, 722.64, 1501.98,
563.59 — all worse than 275.28, and non-monotonic, i.e. still exploring
rather than converging). Running best has been flat at 275.2835 since
evaluation 5.
**Decision:** Continue GP-BO, same `GPBO` instance (preserving all 30 told
observations and GP/acquisition state-free refitting — GP-BO refits from
`self.X_`/`self.y_` on every `ask`, so no state needs to be explicitly
carried beyond keeping the same Python object), for a further block before
reconsidering.
**Hyperparameters:** Unchanged from `roadmap_initial.md` / `src/optimizers/gp_bo.py`
defaults (`n_init=22` already consumed, `num_restarts=10`, `raw_samples=512`,
analytic LogEI, `seed=0` carried in the same instance). No evidence yet
justifies deviating from these; 8 GP-driven points is too few to diagnose
whether LogEI itself is the problem versus simply needing more iterations to
narrow in given only ~8 non-init points so far to fit a 10-D ARD lengthscale
GP.
**Reason:** A short, 8-evaluation plateau after switching from a fixed
design to a model-driven acquisition does not by itself show GP-BO is
unsuited here (per protocol, noise/uncertainty must be accounted for, and a
GP fit to only 22-30 points in 10 dimensions has substantial lengthscale
uncertainty that more data should reduce). The large dynamic range in the
Sobol' sample makes it plausible the GP is still learning a reasonable
length-scale/anisotropy; abandoning the only surrogate-model-based global
method this early, with 70 evaluations still available, would discard
information for no concrete evidence of failure. If the next block still
shows no improvement, I will reconsider (e.g. restart the GP's hyperparameter
fit is already automatic per `ask`, or switch to a local trust-region solver
seeded at the current best, given the apparent large-scale structure).
**Next step limit:** 20 additional objective calls (cumulative evaluation
50 of 100).
**Why this length:** Large enough to let several more LogEI iterations
run (the GP refits on 31, 32, ..., 50 points, giving it a real chance to
sharpen its lengthscale estimates and start exploiting a basin if one is
identifiable) while still leaving half the total budget (50 calls)
uncommitted for a markedly different method if GP-BO continues to
underperform.
**Review:** At evaluation 50, check: (1) whether the running best improved
below 275.28, and by how much; (2) whether the per-point values among
evaluations 31-50 show a decreasing/concentrating trend (evidence the
acquisition is exploiting a basin) versus continued wide scatter (evidence
it is still purely exploring); (3) evaluation cost/time, in case GP fitting
cost is growing problematically. No committed next method.

## Decision after evaluation 50
**Budget:** 50 used / 100 total; 50 remaining.
**Evidence:** Block 31-50 (20 GP-driven LogEI evaluations) improved the
running best three times: 275.28 -> 260.09 (eval 31) -> 247.97 (eval 32) ->
226.18 (eval 37) -> 175.74 (eval 50, the final point of the block). Overall
best improved from 275.28 to 175.74, a ~36% reduction. Individual point
values in the block still range widely (175.74 to 885.71, median roughly
400-450), so the acquisition is not purely exploiting a single narrow basin
yet, but it is clearly finding better points over time rather than
plateauing (unlike the first 8-point sub-block reviewed at evaluation 30,
which showed zero improvement). This is positive evidence that GP-BO is
functioning and the earlier flat stretch was most likely GP/acquisition
"warm-up" with too little data, not a fundamental mismatch to this problem.
**Decision:** Continue GP-BO unchanged, same instance (all 50 told
observations retained, GP refit from scratch inside every `ask()` as
before).
**Hyperparameters:** Unchanged — `n_init=22` (already consumed), analytic
LogEI, `num_restarts=10`, `raw_samples=512`, same `GPBO` instance/seed state
as in the two prior steps (see roadmap_initial.md / decision after
evaluation 30 for the full inherited-default list). No new evidence argues
for changing them: the method is actively improving, so there is nothing to
fix via a hyperparameter change.
**Reason:** With active, non-trivial improvement happening and 50
evaluations still remaining, continuing the same working method is the
straightforward choice; switching now would discard a method that is
demonstrably still finding gains, without any concrete sign (e.g. renewed
plateau) that it has stopped being useful.
**Next step limit:** 25 additional objective calls (cumulative evaluation
75 of 100).
**Why this length:** Large enough to let the trend from the last block
(several improving LogEI points within a 20-call window) continue and
reveal whether it keeps finding gains or starts to plateau, while reserving
25 evaluations (25% of total budget) after this checkpoint for a different
approach (e.g. a local trust-region polish from the incumbent, such as
BOBYQA or CMA-ES seeded near the current best) if GP-BO's improvement rate
slows, rather than running GP-BO all the way to the budget limit by default.
**Review:** At evaluation 75, check: (1) size and count of further
improvements to the running best over this block vs. the previous block's
rate (275.28->175.74 across 20 calls); (2) whether recent points are
beginning to concentrate (smaller spread) around the current incumbent,
suggesting convergence/local exploitation, versus continued wide scatter;
(3) whether a plateau has reappeared and for how many consecutive calls.
No committed next method; a local-refinement switch for the final block is
a live option if evidence supports it, not a predetermined plan.

## Decision after evaluation 75
**Budget:** 75 used / 100 total; 25 remaining.
**Evidence:** Block 51-75 (25 GP-driven LogEI evaluations) improved the
running best twice more: 175.74 -> 164.40 (eval 51) -> 123.75 (eval 57) ->
121.9996 (eval 65). That is a further ~30% reduction on top of the previous
block's ~36% reduction — the improvement rate has not slowed down overall.
Several points late in the block cluster close to the incumbent (eval
64-70: 143.02, 121.9996, 198.33, 135.96, 147.04, 129.12 — five of six within
~20% of the running best), evidence the acquisition is exploiting a
promising basin, not just scattering randomly. The final five points of the
block (337.35, 273.11, 519.64, 345.07, 370.68) jumped back up, showing LogEI
is still also spending some evaluations on exploration elsewhere (expected
behavior of EI-style acquisitions, not a sign of malfunction) rather than
having fully converged.
**Decision:** Continue GP-BO unchanged, same instance (all 75 told
observations retained; GP refit from scratch inside every `ask()`, as
before).
**Hyperparameters:** Unchanged — see roadmap_initial.md / prior decisions
for the full inherited-default list (`n_init=22`, LogEI, `num_restarts=10`,
`raw_samples=512`, same seeded instance). The method continues to actively
improve the incumbent and shows a mix of exploitation near the current best
and legitimate continued exploration; there is no evidence (e.g. renewed
flat plateau, degenerate acquisition optimizer behavior) that would justify
a hyperparameter change or a restart.
**Reason:** Improvement has been sustained and comparable in magnitude
across both completed GP-driven blocks (-36%, then -30%), with clear signs
of basin exploitation alongside exploration — there is no indication GP-BO
has stalled. With 25 evaluations left, continuing the method that is still
working is preferred over switching to an untested local solver purely on a
budget-remaining heuristic; a switch would only be justified by evidence the
current method has stopped delivering, which has not happened.
**Next step limit:** 15 additional objective calls (cumulative evaluation
90 of 100).
**Why this length:** Shorter than the prior two blocks (20, 25) because the
remaining budget is now small (25) and I want one more checkpoint before
the final evaluations are committed, so that if GP-BO's improvement has
genuinely stopped by evaluation 90, there are still 10 evaluations left to
try a different, more locally exploitative method (e.g. a short BOBYQA or
CMA-ES run anchored near the current incumbent) rather than being forced to
spend the entire remaining budget on whatever this step's method is doing.
**Review:** At evaluation 90, check: (1) whether the running best improved
further past 121.9996, and the size of any improvement relative to this
block's exploitation-vs-exploration mix; (2) whether the proportion of
near-incumbent points in evaluations 76-90 is similar to, higher, or lower
than in 51-75 (more clustering would support continuing GP-BO for the final
10; a return to pure wide scatter with no improvement would support a
local-polish switch for the final 10). No committed next method.

## Decision after evaluation 90
**Budget:** 90 used / 100 total; 10 remaining.
**Evidence:** Block 76-90 (15 GP-driven LogEI evaluations) improved the
running best twice: 121.9996 -> 117.0982 (eval 76) -> 111.7467 (eval 85) ->
110.9472 (eval 87). The improvement magnitude this block (~9% reduction) is
smaller than the prior two blocks (-36%, then -30%), consistent with
convergence toward a local optimum rather than a stall: evaluations 84-87
(128.54, 118.99, 111.75, 110.95) form a tight, monotonically-improving
cluster right at the incumbent, i.e. strong exploitation of a narrow basin,
alongside a couple of clearly exploratory high-value points elsewhere in the
block (505.24, 856.53, 391.06, 289.17). This is the clearest exploitation
signal of any block so far.
**Decision:** Continue GP-BO unchanged for the final block, same instance
(all 90 told observations retained). I am not switching to a different
local solver for the final 10 evaluations: none of this package's other
closed-loop local solvers (BOBYQA, CMA-ES) accept a custom starting point —
both restart from the box midpoint by convention — so switching now would
spend part of the tiny remaining budget re-discovering what GP-BO already
knows (the current basin) rather than refining it, which would very likely
underperform letting GP-BO's LogEI continue to exploit the basin it has
already localized.
**Hyperparameters:** Unchanged — see roadmap_initial.md / prior decisions
(`n_init=22`, LogEI, `num_restarts=10`, `raw_samples=512`, same seeded
instance, all inherited `GPBO` defaults, no modification made at any
checkpoint in this campaign).
**Reason:** The method is still improving, now with a clear exploitation
signal right around the incumbent, and the only alternative methods
available in this package cannot be warm-started at the current best point,
making a switch strictly disadvantageous with only 10 evaluations left.
**Next step limit:** 10 additional objective calls (cumulative evaluation
100 of 100 — this exhausts the budget).
**Why this length:** This is the entire remaining budget. Given the
evidence of active exploitation in a narrowing basin and no viable
alternative method that could do better in so few calls, there is no reason
to reserve any further evaluations for reconsideration; this is the final
step of the campaign.
**Review:** Not applicable beyond this point — this step exhausts the
budget. The harness will auto-save at evaluation 100; final outcome will be
recorded as the stopping decision below.

## Final outcome and stopping decision
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** The final block (91-100) improved the running best once more,
from 110.9472 to 109.3136 at evaluation 92, then no further improvement
through evaluation 100 (remaining values in the block: 356.70, 541.78,
188.52, 256.46, 206.51, 184.82, 171.30, 324.34 — all worse than the
incumbent, a mix of moderate and clearly exploratory points). Across the
whole campaign the running best decreased monotonically each time GP-BO was
given a further block: 275.28 (eval 5, end of Sobol' init) -> 175.74 (eval
50) -> 121.9996 (eval 65) -> 110.9472 (eval 87) -> 109.3136 (eval 92), with
diminishing-but-still-positive returns per block, consistent with
convergence toward a local (and plausibly global, given the depth and
consistency of exploitation near this basin across three consecutive
blocks) optimum of the objective, not a stalled search.
**Decision:** Stop. The evaluation budget of 100 is exhausted; this is a
hard constraint per the harness and protocol, not a discretionary choice at
this point.
**Hyperparameters:** Not applicable — no further evaluations will be made.
**Reason:** Budget exhaustion is a hard stop by design (Section 5 of the
protocol): continuing is not possible, and `harness.evaluate()` would raise
`BudgetExhausted` regardless of my preference.
**Next step limit:** 0 (stopping).
**Why this length:** Not applicable (stopping).
**Review:** Not applicable (stopping).

**Summary:** Final best objective value: **109.31361705214506**, at
`x = [2.85274275, -0.07245486, 0.18820623, 5.0, 3.20941636, 0.67024484,
1.11695165, -0.2054536, 0.44746817, 0.31953076]` (evaluation 92 of 100).
The entire 100-evaluation budget was used; no stopping occurred before
budget exhaustion. A single method, GP-BO (`src/optimizers/gp_bo.py`),
was used for the whole campaign: a 22-point Sobol' initial design followed
by sequential LogEI-acquisition-driven points, with all hyperparameters
left at the module's own documented defaults throughout (no tuning was
found necessary or justified by the observed evidence at any checkpoint).
The method was never switched or hybridized, since at every checkpoint
(evaluations 30, 50, 75, 90) the evidence showed continued improvement
(after an initial 8-point warm-up with no improvement, which further data
resolved rather than confirming as a fundamental mismatch), and no
alternative method in the package could be warm-started near the discovered
basin to plausibly outperform continuing GP-BO given the small remaining
budgets at each later checkpoint.
