## Decision after evaluation 3
**Budget:** 3 used / 100 total; 97 remaining.
**Evidence:** 3 repeated `evaluate()` calls at the box midpoint
`x0 = [0,0,0,0,0]` all returned the *exact same* value, `y = 2476.207257355301`
(bit-identical across all 3 draws). This is surprising for a function
described as noisy/heavy-tailed: it suggests either (a) the noise is only
injected with some probability less than 1 per call (so 3 draws at one
point can plausibly all land on the "clean"/common outcome), (b) the
noise process is itself deterministic given `x` (e.g. seeded by a hash of
`x`, as some benchmark suites do for reproducibility), or (c) noise only
activates away from this particular point. 3 samples at a single `x`
cannot distinguish these; the "occasional large outlier" behavior
promised in the task description has not yet been observed at all.
**Decision:** Run one more small, targeted noise probe before committing
to a main optimization method: repeat evaluation at a *different* point
(not the midpoint, to rule out midpoint-specific special-casing) several
times, to check whether (i) repeats still agree exactly, and (ii) a rare
outlier appears when sampled a bit more times. This still directly serves
method choice (robust vs. standard global search, whether to budget
explicit re-evaluation of promising candidates later) and is cheap
relative to the 100-call budget.
**Hyperparameters:** Point chosen: a fixed quasi-random point inside the
bounds, `x1 = [2.5, -1.5, 3.0, -3.5, 1.0]` (an arbitrary off-center point
spanning several coordinates' signs/magnitudes, not selected based on any
knowledge of the objective). Number of repeats: 5 (enough to have a
reasonable chance of catching a "rare" outlier if the outlier rate is on
the order of 10-30%, while still being a small fraction of budget).
**Reason:** The stated noise characteristics materially affect which
method and settings are appropriate (e.g. whether to trust single
evaluations when ranking candidates, whether a GP's inferred observation
noise will be large or small, whether replication of promising points is
worth its evaluation cost later). Evidence so far does not confirm the
qualitative description, so it is worth one more cheap check before
locking in a method, rather than assuming heavy-tailed noise is active
everywhere based on the problem statement alone.
**Next step limit:** 5 additional objective calls; pause at cumulative
evaluation 8.
**Why this length:** 5 repeats at one new point is still under 10% of the
total budget combined with step 1, and is the natural unit for "does this
point's repeats disagree, and by how much" — a single additional repeat
would not be enough to see a distribution, while a much larger probe
would spend too much of a 100-call budget on characterization rather than
search.
**Review:** Whether the 5 values at `x1` agree exactly (as at `x0`), show
mild spread, or include a clear outlier; use this (together with the
`x0` result) to decide the method and any robustness provisions
(replication, median-of-repeats, trimmed acquisition, etc.) for the main
search step.

## Decision after evaluation 8
**Budget:** 8 used / 100 total; 92 remaining.
**Evidence:** At `x1 = [2.5, -1.5, 3.0, -3.5, 1.0]`, 5 repeats gave
`[79997.831..., 79997.831..., 79997.831..., 79997.140..., 79997.831...]`:
4 of 5 bit-identical, 1 differing by only ≈0.69 (≈0.00086% relative).
Combined with the 3 bit-identical draws at `x0`, the picture across both
probed points is: evaluations are *usually* exactly repeatable, with an
occasional small perturbation — not the large-magnitude outliers the task
description warns are possible, but consistent with "noisy, heavy-tailed,
*occasional* large outliers" being a tail event rare enough not to appear
in 8 draws. I will not spend further budget trying to directly observe a
large outlier (expensive relative to a 100-call budget and not guaranteed
to pay off); instead I will choose a main method that tolerates noise
structurally and add a verification step at the end rather than during
characterization.
**Decision:** Switch to GP-BO (`src/optimizers/gp_bo.py`, BoTorch
`SingleTaskGP` + analytic LogEI) as the main search method, started fresh
(no prior optimizer state to retain — the 8 characterization evaluations
used a fixed point, not this optimizer, so they are not fed into GP-BO's
training data; GP-BO starts its own Sobol initial design independent of
them). Rationale for GP-BO over the alternatives available in this
project:
- Closed-loop local solvers (BOBYQA, Nelder-Mead, L-BFGS-B) accept each
  raw `evaluate()` return as ground truth for trust-region/simplex/line-
  search decisions with no way to see repeats; a single bad read can
  misdirect them with no mechanism to recover other than restarting, and
  L-BFGS-B's finite-difference gradient is explicitly flagged in the task
  prompt as noise-sensitive.
- DE needs `pop_size = 10*dim = 50` just to complete one generation
  cycle's worth of context and is documented as needing far larger
  budgets (10^3-10^4 x dim) to show its strength; with 92 evaluations
  left it would get barely more than one generation of real
  selection pressure.
- CMA-ES is a reasonable global alternative (population size here
  ≈ 4+3·ln(5) ≈ 9, so ~10 generations fit in budget) and tolerates small
  per-sample noise reasonably well via rank-based selection across a
  population, but has no explicit model of observation noise and commits
  fully to population replacement each generation.
- GP-BO fits a `SingleTaskGP` with its own inferred Gaussian observation-
  noise term (not assumed noiseless internally, despite the module
  docstring's "noiseless" label referring to the analytic EI's plug-in
  `best_f`, not to the GP likelihood) — this lets small, frequent noise
  be absorbed into the posterior rather than chasing every fluctuation,
  and its documented regime ("budgets of order 10 x dim", i.e. ~50 here)
  fits comfortably within the 92 calls remaining (100 ≈ 20 x dim).
- Known residual risk: analytic LogEI's `best_f` is the raw minimum
  observed `y`, a plug-in point estimate that a lucky low noise draw
  could distort. Mitigation deferred to the final checkpoint: before
  reporting a final best, re-evaluate the then-current incumbent a few
  times to confirm it is not an outlier-driven false optimum (budget
  permitting).
**Hyperparameters:** Using `GPBO(bounds=harness.bounds, seed=0)` with its
own library defaults, none overridden: `n_init = 2*(dim+1) = 12` Sobol
points before any acquisition-based proposal; acquisition optimization
`num_restarts=10`, `raw_samples=512` (BoTorch tutorial-standard pair, per
the module's docstring); GP = BoTorch `SingleTaskGP` with its own default
kernel/priors, `Normalize` input transform to the task bounds, and
`Standardize` output transform; acquisition = analytic
`LogExpectedImprovement`, `maximize=False`. `seed=0` is set only for
reproducibility of the Sobol init and acquisition-restart sampling, not a
tuned choice. These are inherited defaults as documented in
`src/optimizers/gp_bo.py`; no values have been changed from the library's
own.
**Reason:** Given the evidence so far (small, mostly-absent noise with
occasional minor perturbation, and a budget at the small end of what a
5-D black box typically needs), a model-based method that (a) is
explicitly designed for this budget regime and (b) has a principled,
fitted notion of observation noise is the best structural fit, in
preference to closed-loop local solvers that are exposed to raw single
reads, or population methods tuned for far larger budgets (DE) or with no
explicit noise model (CMA-ES).
**Next step limit:** 20 additional objective calls; pause at cumulative
evaluation 28.
**Why this length:** This covers the full `n_init = 12` Sobol design plus
8 acquisition-driven (LogEI) proposals — enough acquisition steps after
the initial design to see whether the GP is finding real improvements
over the Sobol points, without committing more than ~1/5 of the total
budget before the next review. A shorter step would not get past the
init design to see the acquisition behave at all; a longer one would
delay reconsideration past a natural phase boundary.
**Review:** Compare `y_best_so_far` and its location after the Sobol
init (first 12) against after the 8 acquisition points; check whether
acquisition points are landing near previous good points (model trusted,
converging) or scattered (model still exploring) or repeatedly near one
already-tried point (possible outlier-chasing); check GP-BO's own
`X_, y_` for any point whose value looks inconsistent with its
neighbors. Decide whether to continue GP-BO unchanged, adjust it, or
switch.

## Decision after evaluation 28
**Budget:** 28 used / 100 total; 72 remaining.
**Evidence:** The 12 Sobol-init points (evals 9-20) ranged from
y≈2843 (eval 13, near origin: `[1.07, 0.55, 0.16, 2.67, -2.29]`) up to
y≈137145 at a far corner-ish point (eval 16), with every point farther
from the origin scoring clearly worse — consistent with a roughly
convex, origin-favoring landscape (large y growth with distance from
0, matching the stated-task midpoint probe value of y≈2476, which beat
all 12 Sobol points). The 8 acquisition (LogEI) points (evals 21-28) are
still fairly scattered in magnitude (4840 to 138539), but two of them
improved on everything seen so far: eval 25 (y≈1804.89, new best) and
eval 27 (y≈2121.10), both closer to the origin than most Sobol points,
and both fell after the model had enough data to start exploiting rather
than purely exploring. This is healthy early BO behavior (LogEI mixes
continued exploration — hence the occasional very high values like eval
26/28 — with exploitation near the emerging good region), not scattering
that looks like noise-chasing: no acquisition point has revisited
anywhere near an earlier point only to get a wildly different value
(no evidence yet of an outlier read misleading the model). No sign of
large-outlier noise distorting the trace so far — the 8 points' spread
tracks distance-from-origin, not erratic repeats.
**Decision:** Continue GP-BO unchanged — same running optimizer object,
same fitted-fresh-each-call GP/acquisition settings, all data retained
(`X_, y_` carried over internally). No evidence yet justifies switching
or adjusting: the acquisition only just started (8 points) and is already
producing improvements, and the budget regime (now 72 of ~ "10 x dim" =
50 remaining-equivalent) remains within GP-BO's documented comfort zone.
**Hyperparameters:** Unchanged from the previous entry (same `GPBO`
instance: `seed=0`, `num_restarts=10`, `raw_samples=512`, default
`SingleTaskGP`/`LogExpectedImprovement`); no relevant setting has changed.
**Reason:** The method is behaving exactly as expected for this stage
(exploration mixed with early exploitation, improving incumbent), the
noise probed earlier has not visibly disrupted it, and there is no
budget-efficiency reason to abandon a working model-based search this
early. Switching now would discard the already-fitted GP's accumulated
information for no observed benefit.
**Next step limit:** 20 additional objective calls; pause at cumulative
evaluation 48.
**Why this length:** Large enough to let the GP refit several more times
on a growing dataset and show a clearer trend (converging vs. still
scattered) than the last 8-point acquisition block did, while still
leaving at least half the total budget (52 calls) for a further
adjustment, a local-refinement switch, or final-answer verification
evaluations if needed.
**Review:** Whether `y_best_so_far` keeps improving and acquisition
points cluster increasingly tightly around the best region (converging,
continue), continues mixing wide exploration with occasional improvement
(continue, working as intended), stalls with no improvement despite tight
clustering (consider switching to a local refiner for the remaining
budget), or shows a suspicious single very-good value not corroborated by
nearby points (treat as possible outlier; consider re-evaluating that
point before trusting it as incumbent).

## Decision after evaluation 48
**Budget:** 48 used / 100 total; 52 remaining.
**Evidence:** Continued improvement: new incumbents at eval 38
(y≈1553.43, `[-0.58,-0.97,-2.15,2.15,-0.57]`), after the eval-25 best of
1804.89. Evaluations 29-48 are still a mix of exploration (several points
score 6000-157000, including one very high value at eval 37 where three
coordinates sit at the -5 boundary) and exploitation near the emerging
good region (evals 31/32/36/38/46 all score under ~2300, clustering
around moderate, non-boundary coordinate values with x4 often negative).
A notable pattern: many explored points push one or more coordinates to
the ±5 boundary (e.g. evals 33/34/35/36/37/39/40/41/42/43/44/47/48 all
have at least one coordinate at ±5), yet the best points found so far do
*not* sit at any boundary — this looks like the GP/EI still testing
whether the boundary helps (standard LogEI behavior under box constraints
with limited data) rather than the optimum being boundary-seeking;
continuing to let it resolve this is informative. No evidence of a
noise-driven outlier corrupting the trend: the improving points are
corroborated by several nearby points of similar, not wildly different,
magnitude (e.g. eval 38's 1553 sits among neighbors like eval 46's 1773
and eval 31's 2178 in a similar region), unlike what a single spurious
low outlier would look like (an isolated low value with no similar
nearby support).
**Decision:** Continue GP-BO unchanged. The method keeps improving the
incumbent at a reasonable cadence (roughly every 10-13 evaluations) and
there is no stall or noise-corruption evidence that would justify
switching to a local refiner yet. Same running optimizer instance and
data (`X_, y_` from all 48 evaluations retained internally).
**Hyperparameters:** Unchanged (`seed=0`, `num_restarts=10`,
`raw_samples=512`, default `SingleTaskGP` / analytic `LogExpectedImprovement`
as in the previous two entries).
**Reason:** Same reasoning as the last checkpoint applies with stronger
support now: improvement continues, the exploration/exploitation balance
looks healthy, and reserving budget for a later local-refinement or
verification phase does not require abandoning GP-BO now while it is
still finding gains.
**Next step limit:** 20 additional objective calls; pause at cumulative
evaluation 68.
**Why this length:** Keeps the review cadence similar to the last two
blocks (one GP refit cycle of meaningful size) while preserving at least
32 evaluations after this checkpoint — enough to either run a further
GP-BO block, switch to a local refiner to polish the best region, or
spend some calls on end-of-campaign verification (re-evaluating the
incumbent) regardless of what this block shows.
**Review:** Same criteria as before — incumbent improvement rate,
whether acquisition points are tightening around a region (suggesting a
later local-refinement step would pay off) or still scattering broadly
(suggesting continued global GP-BO is still worthwhile), and any sign of
an isolated, uncorroborated low value that should be treated as a
possible outlier rather than trusted at face value.

## Decision after evaluation 68
**Budget:** 68 used / 100 total; 32 remaining.
**Evidence:** Incumbent kept improving but with clearly diminishing
margins: 1804.89 (eval 25) -> 1553.43 (eval 38, -14%) -> 1507.75
(eval 55, -3%) -> 1480.94 (eval 67, -1.8%). The best region has
stabilized around `x0≈-1.3..-2.5, x1≈-1.7..-2.5, x2≈-1.0..-2.0,
x3≈0..1, x4` trending toward the `-5` boundary (several of the best
recent points, including the current incumbent at eval 67, have
`x4 = -5.0` exactly). Several nearby points (eval 49: 1761, eval 56:
1688, eval 60: 1849, eval 62: 1688) corroborate this region rather than
showing an isolated spurious dip, so the incumbent does not look like a
noise artifact. Exploration continues in parallel (several points still
score >10000, including boundary-heavy combinations), which is expected
LogEI behavior with remaining uncertainty, not malfunction. The shrinking
improvement margins suggest GP-BO is approaching a local optimum of this
region; whether it's the global optimum is unknown, but continuing to
let BO decide where to sample (global method, still proposing some
exploratory points) is more defensible with 32 calls left than either
prematurely switching to a purely local refiner (which would commit to
this region without further global checking) or abandoning GP-BO while
it is still producing (smaller but real) gains.
**Decision:** Continue GP-BO unchanged for one more, shorter block, then
deliberately reserve a final slice of budget for (a) a short local
refinement check and/or (b) noise-verification re-evaluation of the
final incumbent before reporting it (per the mitigation noted in the
eval-8 decision). Same running instance/state as before.
**Hyperparameters:** Unchanged (`seed=0`, `num_restarts=10`,
`raw_samples=512`, default `SingleTaskGP` / analytic `LogExpectedImprovement`).
**Reason:** Diminishing-but-still-real improvement does not yet justify
abandoning the current method, but the shrinking margins mean the value
of *more* global BO steps is itself diminishing relative to the value of
(a) confirming the final answer is not a noise artifact and (b) possibly
polishing it locally — both of which need to happen before the budget
runs out, so budget must now be deliberately reserved for them rather
than spent entirely on further GP-BO.
**Next step limit:** 15 additional objective calls; pause at cumulative
evaluation 83.
**Why this length:** Smaller than the previous two 20-call blocks,
proportionate to the smaller marginal gains being observed, while still
large enough for a few more GP refits to either find further improvement
or confirm a plateau. Leaves 17 evaluations deliberately unspent at this
checkpoint for a final decision between local refinement and/or
incumbent-verification re-evaluations.
**Review:** Whether the incumbent improves further (continue GP-BO,
possibly for one more short block) or plateaus (stop global search and
spend the remaining calls on local refinement around the incumbent
and/or repeated verification evaluations of it); re-examine whether
`x4` truly wants the `-5` boundary (if the best points keep pinning it
there, that is a real feature of the landscape in this region, not an
error).

## Execution note / deviation (before the next decision entry)
The driver script for the previous block still had `N_STEP = 20` hard-coded
from the eval-28 decision; an attempt to edit it to 15 (for the 15-call
block decided above, cumulative pause at 83) failed silently (a shell
`sed` command did not match and made no change), and this was not
verified before re-running the script. The block therefore ran 20 real
`evaluate()` calls (evals 69-88) instead of the decided 15, overrunning
the intended pause point of 83 by 5 evaluations. This is recorded
transparently as a step-length execution error, not a deliberate choice:
the step-limit discipline is an agent-side control on when to stop and
reassess, not a harness-enforced limit, and it was not correctly applied
here. All 20 evaluations are real, budget-counted `evaluate()` calls and
are kept in the trace as required (no fabrication/removal of real data).
No budget was exceeded (88 of 100 total, well within the hard 100-call
ceiling, which remains enforced by the harness regardless of this
script-level slip). The remaining-budget accounting below reflects the
true state (88 used / 12 remaining), and the evidence reviewed below
covers the full 20-evaluation block actually executed, not just the
first 15, since all 20 are genuine evidence.

## Decision after evaluation 88
**Budget:** 88 used / 100 total; 12 remaining.
**Evidence:** Improvement continued through the (overrun) block: 1480.94
-> 1475.07 (eval 70) -> 1381.42 (eval 72) -> 1224.68 (eval 85, current
incumbent, `x = [-1.196, -1.659, -0.522, 0.583, -4.157]`). The
improvement trend has not flattened as much as the previous checkpoint's
margins suggested (eval 72 and 85 each gave >6% gains), so GP-BO is still
finding real improvement, not purely plateaued. The best region continues
to center on small-magnitude `x0..x3` with `x4` solidly negative
(`-4.0` to `-5.0` across the best recent points: evals 70/72/83/85/88 all
have `x4` in `[-4.5, -4.15]` or exactly `-5`), reinforcing that this is a
real feature of the landscape (not a one-off boundary artifact), while
`x2` seems less tightly pinned (ranges `-0.22` to `-3.4` among good
points), suggesting the surface may be fairly flat along that coordinate
near the optimum. Nearby points continue to corroborate the incumbent
(eval 88: 1359, close to eval 85's region and value), so there is still
no sign of a noise-driven false incumbent.
**Decision:** With only 12 evaluations left, this is the final phase.
Given the active budget constraint, I will not switch methods (a
local-refiner restart would itself cost setup evaluations with a thread
based optimizer and no guarantee of beating continued GP-BO, which is
still producing gains). Plan for the remaining 12: run GP-BO for up to 9
more acquisition-driven evaluations to keep exploiting the still-improving
trend, then spend the final 3 evaluations re-evaluating the then-current
incumbent `x` to directly check (per the noise-verification mitigation
flagged at eval 8 and eval 68) whether its very low value is corroborated
or was a lucky noise draw, since the stated noise is heavy-tailed and the
final reported number should not rest on a single read when a few spare
evaluations can check it cheaply. Same running GP-BO instance/state.
**Hyperparameters:** Unchanged (`seed=0`, `num_restarts=10`,
`raw_samples=512`, default `SingleTaskGP` / analytic `LogExpectedImprovement`)
for the GP-BO portion; the verification portion uses plain repeated
`evaluate()` calls at a fixed `x` (no optimizer, as in the eval-3/eval-8
noise probes), not a new "method" requiring its own hyperparameters.
**Reason:** Continued real improvement makes it worth spending most of
the remaining budget on more GP-BO rather than stopping immediately;
but ending the campaign on an unverified single low reading, in a task
explicitly flagged for heavy-tailed noise, would risk reporting an
outlier as the result. Splitting the last 12 calls (9 search + 3 verify)
balances both concerns given how little budget is left. This split (not
a round fraction) is chosen because 3 repeats is the same minimum used
in the very first noise probe to see a spread/outlier, applied now to
the actual answer being reported rather than an arbitrary point.
**Next step limit:** 9 additional GP-BO objective calls; pause at
cumulative evaluation 97, strictly before the final 3-call verification
block (evaluation 98-100), which is itself the last step of the
campaign.
**Why this length:** 9 keeps a meaningful GP-BO block going (not a token
1-2 call gesture) while guaranteeing the 3 verification calls the
decision above commits to, within the 12 calls left.
**Review:** Final incumbent `x` and `y_best_so_far` after this block;
then proceed directly to the 3 verification evaluations at that `x` and
stop (budget will be exhausted: 97 + 3 = 100).

## Decision after evaluation 97 (final GP-BO block and closing decision)
**Budget:** 97 used / 100 total; 3 remaining.
**Evidence:** The 9-call GP-BO block (evals 89-97) found no further
improvement: values ranged 1279.38 to 15836.39, all at or above the
eval-85 incumbent of 1224.68 (`x = [-1.196, -1.659, -0.522, 0.583,
-4.157]`). The closest challenger (eval 90, y≈1279.38) is in the same
neighborhood as the incumbent, consistent with a local optimum having
been reached in this region rather than the incumbent being a fluke the
model simply hasn't revisited. This matches the plan made at evaluation
88: spend the last calls confirming the incumbent rather than continuing
to search with a near-exhausted budget.
**Decision:** Stop searching. Execute the planned 3-call verification:
re-evaluate the incumbent `x` exactly (no optimizer involved) to check it
is not an outlier-driven false optimum, then stop the campaign
(budget will be fully exhausted: 97 + 3 = 100).
**Hyperparameters:** Not applicable (repeated evaluation at a fixed
point, as in the eval-3/eval-8 noise probes).
**Reason:** This is the final opportunity to catch a noise-driven false
result before reporting it, directly addressing the task's explicit
heavy-tailed-noise warning; 3 calls is all that remains and is consistent
with the minimum repeat count used earlier in the campaign to detect a
spread or outlier.
**Next step limit:** 3 additional objective calls; pause (campaign end)
at cumulative evaluation 100 (full budget).
**Why this length:** Exactly the remaining budget; no further decisions
are possible after this.
**Review:** Whether the 3 verification reads agree with the original
1224.68 reading (supporting it as the reported result) or reveal that the
original was itself an outlier (in which case the verified, more typical
value among the 3 new reads, or their median, would be the more honest
figure to report instead). This is the last checkpoint; no further step
follows regardless of the outcome, since the budget is then exhausted.

## Final outcome
**Budget:** 100 used / 100 total; 0 remaining. Budget exhausted exactly
as planned (the harness auto-called `save()` on the 100th evaluation).
**Verification result:** All 3 re-evaluations of the incumbent `x =
[-1.19611596, -1.65936889, -0.52178876, 0.58334149, -4.15671915]`
returned the identical value `y = 1224.6756420920467` as the original
reading (evals 98, 99, 100 all bit-identical to eval 85). This matches
the pattern seen in the very first noise probes (section "Decision after
evaluation 8"): evaluations at a fixed point are usually exactly
repeatable, with only rare, typically small perturbations observed
elsewhere in the campaign (e.g. the eval-4 deviation during the `x1`
probe, and the 4-of-5 agreement with one ~0.0009%-different draw). No
evidence was found of the large-magnitude outliers the task description
warned were possible; the realized noise level over this campaign's 100
draws was mostly nil with occasional very minor jitter. The reported
incumbent is corroborated, not a lucky single read.
**Final best value:** `y = 1224.6756420920467` (minimize), at
`x = [-1.19611596, -1.65936889, -0.52178876, 0.58334149, -4.15671915]`.
**Evaluations used:** 100 / 100 (campaign ran to full budget, as the plan
from evaluation 88 intended: 9 more search calls + 3 verification calls
after the eval-88 checkpoint).
**Stopping reason:** Budget exhausted by design — the final 3 calls were
a deliberate verification step, not search, so there was no more budget
left (and no further decision to make) once they completed.
**Summary of method path:** (1) direct repeated-evaluation noise
probes at two fixed points (evals 1-8) to characterize noise before
committing to a method; (2) GP-BO (BoTorch `SingleTaskGP` + analytic
LogEI, library defaults, `seed=0`) as the sole search method from eval 9
through eval 97, continued unchanged across 5 checkpoints because it kept
producing real, corroborated improvement (2476 -> 1804.89 -> 1553.43 ->
1507.75 -> 1480.94 -> 1475.07 -> 1381.42 -> 1224.68) with diminishing but
non-zero margins and no evidence of noise-driven corruption, so no method
switch was ever triggered; (3) a final 3-call direct verification of the
incumbent (evals 98-100) before reporting it, given the task's explicit
heavy-tailed-noise warning. One execution deviation occurred and is
recorded above (after evaluation 68): a step-length edit to the driver
script silently failed, causing a planned 15-call block to run as 20
calls (evals 69-88 instead of 69-83); this was caught and disclosed at
the next checkpoint, all 20 evaluations are genuine and retained, and the
total budget was never exceeded (100/100 exactly, enforced by the
harness throughout).
