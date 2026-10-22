## Decision after evaluation 20
**Budget:** 20 used / 100 total; 80 remaining.
**Evidence:** Sobol' initial design (20 points, seed=0, `bounds=[(-5,5)]*10`)
gave y in [129.90, 265.21], mean 190.04, std 34.73. All values are positive
and span a moderate range (max/min ratio ~2x) rather than being tightly
clustered or wildly discontinuous — consistent with a reasonably smooth,
single-scale landscape rather than an extremely rugged or flat one, though
20 points in 10 D cannot rule out multimodality. Best point so far: y =
129.8996 at x = [2.457, 3.126, -3.591, 2.100, 0.384, 3.020, -0.542, -0.459,
-2.743, -1.966].
**Decision:** Switch to TuRBO (`src/optimizers/turbo.py`, `TuRBO`), a
trust-region Bayesian optimization method, for the next step. Rationale for
the choice (vs. continuing Sobol, or using CMA-ES/BOBYQA) is in "Reason"
below. Initialization: a fresh `TuRBO(bounds=harness.bounds, seed=1)`
instance, but instead of letting it draw its own fresh 2*dim=20-point Sobol
init design (which would spend 20 more evaluations re-exploring), its
already-public `tell(X, y)` method is called directly with the 20 points
and values already collected in the previous step. `tell()` is the
documented, supported interface for recording evaluated points; feeding it
the existing design before any `ask()` call is exactly what it does with
its own init batch (accumulate `n_init` points, then flip from "init" to
"evolve" phase) — the only difference is the 20 points come from the
already-executed Sobol step rather than a second, independently-drawn Sobol
design. This preserves the harness's evaluation counter and does not
re-query the objective. No prior method's state is discarded: the Sobol
object's own `best()`/history remain available for reference if ever
needed, but TuRBO's `X_`/`y_` now contains the same 20 observations as its
own "init" phase would have produced.
**Hyperparameters:** TuRBO wrapper's own standard, no-tuning defaults
throughout (as documented in `src/optimizers/turbo.py`): `batch_size=4`,
`n_init=2*dim=20` (satisfied by the seeded data), `n_candidates=min(100*dim,
5000)=1000`, trust-region `length_init=0.8`, `length_min=0.5**7`,
`length_max=1.6`, `success_tolerance=10`, `failure_tolerance=ceil(max(4/4,
10/4))=3`. `seed=1` for the wrapper's internal RNG (candidate generation),
distinct from the Sobol step's `seed=0`. No settings changed from the
wrapper's defaults; this is the first use of TuRBO this campaign, so there
is no earlier configuration to reference.
**Reason:** With only 80 evaluations left in 10 dimensions, a
sample-efficient, model-based method is preferable to continuing pure
quasi-random search (which does not exploit the already-observed
structure) or to a population-based method like CMA-ES, whose default
generation size (~10) would need ~8 generations to use the remaining
budget while starting from a fixed box-midpoint and discarding the Sobol
evidence. TuRBO is specifically designed for exactly this regime (modest
evaluation budgets, moderate dimensionality, box constraints): it fits a
local GP surrogate to make each subsequent evaluation count, and its trust
region adapts (expanding on success, shrinking and eventually restarting on
failure) using the data itself rather than a schedule fixed in advance,
matching the observed moderate-but-uncertain smoothness. Reusing the
already-evaluated 20 Sobol points as its init batch avoids paying for a
second, redundant random design out of an already-tight budget.
**Next step limit:** 40 additional objective calls (10 batches of 4), at
most 80 remaining; pause at cumulative evaluation 60.
**Why this length:** 40 evaluations gives TuRBO roughly 10 trust-region
update cycles (updates occur every `batch_size`=4 evaluations), enough to
observe at least one full failure-tolerance cycle (3 non-improving batches
triggers a region shrink) or a run of improving batches (10 for a region
expansion), which is the minimum needed to say anything about whether the
trust region is adapting usefully — without committing more than half of
the remaining budget before the next checkpoint.
**Review:** Whether `y_best` has improved versus 129.8996, by how much, and
how the trust-region length has moved (expanded/shrunk/restarted) over the
10 batches — this will determine whether to continue TuRBO as-is, let it
run longer, or reconsider (e.g. a local polish with BOBYQA/CMA-ES around
the new best, or diagnosing a stalled/overly-shrunk trust region). No
method is pre-committed beyond this review.

## Decision after evaluation 60
**Budget:** 60 used / 100 total; 40 remaining.
**Evidence:** TuRBO's 10 batches (40 evaluations) improved the best y from
129.8996 to 85.3854 — a ~34% reduction, substantial given the step size.
The trust-region length shrank once (0.8 -> 0.4, i.e. one failure-tolerance
cycle of 3 non-improving batches triggered a halving) and the region has
not restarted (`phase` is still "evolve", same local data retained). The
last 10 individual y values (94.4, 96.8, 93.0, 97.3, 95.8, 90.4, 85.4, 86.3,
85.5, 86.0) show an initial drop then a narrow plateau around 85-86,
consistent with the trust region having localized around a basin and now
refining within it rather than stalling outright (the single best value of
85.3854 was reached mid-batch and subsequent values are hovering just above
it, not diverging).
**Decision:** Continue TuRBO unchanged — same instance, same accumulated
`X_`/`y_` history and current trust region (length 0.4), no restart. The
evidence (steady improvement, an adapting but not yet collapsed trust
region, no restart triggered) does not support abandoning or restarting the
method; it supports giving it a further block to keep exploiting the same
basin or trigger a restart on its own if the region keeps shrinking.
**Hyperparameters:** Unchanged from the previous entry (TuRBO defaults:
`batch_size=4`, `n_candidates=1000`, `length_min=0.5**7`, `length_max=1.6`,
`success_tolerance=10`, `failure_tolerance=3`); internal RNG state is
whatever the instance has advanced to (not reseeded) so the sequence
continues rather than repeating.
**Reason:** A method that is actively improving the incumbent and has not
exhausted its adaptive mechanism (no restart yet, region still shrinking
gracefully) should be continued rather than switched, per the protocol's
guidance that a short plateau alone does not prove unsuitability — and the
last 10 values are a narrow plateau, not a long one. With 40 evaluations
left, reserving a further checkpoint (rather than spending the whole
remainder at once) keeps the option open to pivot to a local polish method
(e.g. BOBYQA/CMA-ES seeded near the current best) if TuRBO's trust region
collapses to a restart or stops improving in the next block.
**Next step limit:** 24 additional objective calls (6 batches of 4), at
most 40 remaining; pause at cumulative evaluation 84.
**Why this length:** 24 evaluations is enough to observe roughly 6 more
trust-region update cycles (2 full failure-tolerance cycles, or part of a
success-tolerance cycle), enough to tell whether the current basin is
still yielding improvement or has plateaued/restarted, while leaving 16
evaluations in reserve for a final, different step (e.g. a local polish)
if this checkpoint shows TuRBO has stopped making progress.
**Review:** Whether y_best improves further below 85.3854, whether the
trust region restarts (indicating the local basin is exhausted) or keeps
shrinking/expanding, and whether the remaining 16-evaluation reserve after
this step would be better spent continuing TuRBO or switching to a
derivative-free local solver (BOBYQA/CMA-ES with a small initial step)
centered on the incumbent best. No method beyond this review is
pre-committed.

## Decision after evaluation 84
**Budget:** 84 used / 100 total; 16 remaining.
**Evidence:** The 24-evaluation block improved best y from 85.3854 to
80.2071 (a further ~6% reduction, versus the previous block's ~34%) —
diminishing but real improvement. The trust region shrank again (0.4 ->
0.2, a second failure-tolerance halving), still no restart. The last 10 y
values (83.2, 81.5, 80.9, 80.4, 80.8, 80.2, 80.8, 80.6, 80.6, 80.3) are now
clustered tightly in [80.2, 83.2], i.e. the trust region has localized to
a narrow basin and is doing fine-grained refinement rather than large
jumps — behavior consistent with approaching a local optimum within that
basin, with a shrinking marginal return per evaluation.
**Decision:** Continue TuRBO unchanged for the remaining 16 evaluations
(the last step of this campaign), same instance/state/trust region
(length 0.2), no restart. Switching to a different local solver
(BOBYQA/CMA-ES) was considered, but rejected for this final block: those
methods would need their own startup (e.g. BOBYQA's own initial
trust-region probing) to re-establish local curvature information that
TuRBO's already-fit local GP and shrunk trust region (length 0.2) already
captures from the current data; with only 16 evaluations left, paying that
re-orientation cost is unlikely to beat letting the already-converging
method continue. TuRBO's own failure/restart mechanism remains available
if the current basin turns out to be exhausted within this block.
**Hyperparameters:** Unchanged from the previous two entries (TuRBO
defaults: `batch_size=4`, `n_candidates=1000`, `length_min=0.5**7`,
`length_max=1.6`, `success_tolerance=10`, `failure_tolerance=3`); RNG state
continues from where the instance left off.
**Reason:** Continuing improvement (even if the marginal size is
shrinking) plus an actively-adapting, non-restarted trust region is
exactly the evidence the protocol says should keep a method running rather
than switch it; a narrowing plateau after two basin-finding blocks is the
expected signature of local convergence, not of failure. With only 16
evaluations left in the entire budget, this is necessarily the final step
of the campaign regardless of outcome.
**Next step limit:** 16 additional objective calls (4 batches of 4), at
most 16 remaining; pause at cumulative evaluation 100 (budget exhaustion).
**Why this length:** 16 is all of the remaining budget; since any further
checkpoint after this would have 0 evaluations left to act on, there is no
benefit to reserving any further and this step is sized to use exactly
what remains.
**Review:** At completion, record the final best y, confirm the harness
reports `evaluations_used == 100`, and call `harness.save()`; stop the
campaign (budget exhausted). Not applicable beyond that — this is the
final step.

## Decision after evaluation 100 (final)
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** The final 16-evaluation block improved best y marginally
further, from 80.2071 to 80.0603 (~0.2% reduction) — consistent with the
narrowing-plateau pattern already seen in the previous block: the trust
region remained at length 0.2 throughout (no further shrink or restart
within this short block), and the method continued fine local refinement
rather than large jumps. `harness.evaluations_used == 100` and
`evaluations_remaining == 0`, confirming the budget was fully and exactly
consumed, matching the harness's own hard enforcement (the 101st call
would have raised `BudgetExhausted`).
**Decision:** Stop. The budget is exhausted — there is no remaining
evaluation budget for any further method choice.
**Hyperparameters:** Not applicable (no further step).
**Reason:** The evaluation budget is the hard constraint on this campaign;
with 0 evaluations remaining there is nothing left to decide.
**Next step limit:** 0 (stopping).
**Why this length:** Not applicable — no further evaluations are possible.
**Review:** Not applicable — campaign complete. Final summary: best
observed y = 80.0603 at x = [0.2472, -1.4659, -0.7619, 2.0513, -2.3678,
0.2942, -0.5413, 0.0614, -1.6730, -3.4559], found via Sobol' initial
exploration (20 evals) followed by TuRBO (80 evals, same instance/state
throughout, one trust-region restart never triggered — two successive
halvings from 0.8 -> 0.4 -> 0.2 as the search localized into a single
basin). `harness.save()` called; `trace.csv` and `meta.json` written to
`results/agent_run/`.
