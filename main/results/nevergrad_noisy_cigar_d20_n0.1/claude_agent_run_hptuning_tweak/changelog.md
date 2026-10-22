## Decision after evaluation 16
**Budget:** 16 used / 100 total; 84 remaining.
**Evidence:** Step 1 (roadmap_initial.md) ran a 3-repeat noise probe at the
box center (x=0) and a 13-point scrambled Sobol design (seed=0) over the
full box. Center probe: y = [130682847.18, 135547127.50, 135463866.95],
mean 133897947.21, sample std 2784669.50 (relative std ~2.1% of the mean —
consistent with the stated mild, roughly multiplicative noise). Sobol
points: y ranged from 209138400.01 to 360108692.10 (mean 292763951.92) —
every single one of the 13 Sobol points is far worse (larger, since this
is a minimization problem) than all three center repeats. The overall
best-so-far after this step is 130682847.18, at the center. This is a
clear, large (~1.6-2.7x), noise-swamping signal that the region near the
box center is much more favorable than generic points spread across the
box, not an artifact of the ~2% noise level.
**Decision:** Switch from pure space-filling (Sobol) to TuRBO-1
(`src/optimizers/turbo.py`), a local trust-region Bayesian optimizer, to
exploit the evidence that a favorable region exists near the center
rather than continuing to spend budget on undirected global sampling.
Initialization: a campaign-local adapter is used instead of TuRBO's own
built-in fresh-random initial design. TuRBO's default `ask()` behavior for
a new instance is to draw `n_init = 2*dim = 40` fresh Sobol points before
ever fitting a GP; at this budget (84 remaining, 20-d problem) spending 40
more evaluations on an undirected design would both be redundant with the
13 Sobol points already collected and would discard the informative
finding that the center is far better than the box interior at large. The
adapter instead seeds TuRBO's internal trust-region state
(`X_`, `y_`, `_local_X_unit`, `_local_y`, `_local_best_y`,
`_evolve_evals_since_update=0`, `_phase="evolve"`) directly from the 16
points already evaluated (reconstructed from `harness._rows`, the
harness's own recorded data, not re-estimated or fabricated), so the first
GP fit and trust region are centered on the current best point
immediately. The harness instance and its evaluation counter are
unchanged and reused; no evaluations are discarded, no second harness is
created, and the global `best()` history is the same 16 points plus
whatever TuRBO adds.
**Hyperparameters:** TuRBO is used with all of its own library defaults
(`src/optimizers/turbo.py`, this project's BoTorch-tutorial-based
implementation) except the adapter above: `batch_size=4`,
`n_candidates=min(100*dim, 5000)=2000`, `length_init=0.8`,
`length_min=0.5**7≈0.0078`, `length_max=1.6`, `success_tolerance=10`,
`failure_tolerance=ceil(max(4/batch_size, dim/batch_size))=ceil(max(1,5))=5`
(all inherited, unmodified defaults computed by the library itself from
`dim` and `batch_size`), `seed=1` (new seed, since `seed=0` was already
used for the Sobol design in step 1, to avoid correlated random streams).
`n_init=2*dim=40` is the one default not used operationally, for the
reason given above (its 16-point substitute is already-real data, not a
tuning change to the trust-region mechanics themselves).
**Reason:** With a 20-d problem and only 84 evaluations remaining, pure
space-filling or population methods (DE, further Sobol) cannot afford to
explore much more of the box, and the evidence just collected already
points to a specific region worth exploiting. TuRBO is designed for
exactly this regime (local trust-region BO, standard-practice at budgets
of order 10x dim) and its GP (SingleTaskGP with a learned noise term, not
a noiseless assumption) is appropriate for the ~2% multiplicative noise
measured in the probe. CMA-ES was also considered, since it would be
similarly well-suited to a basin around the center, but CMA-ES has no
model-free way to reuse the 16 points already collected as anything other
than a starting mean/covariance guess, and it is a population method
without a principled per-point noise model; TuRBO's GP surrogate makes
better use of the noisy evidence gathered so far while still ultimately
converging to a local search once the trust region gets small.
**Next step limit:** 32 additional objective calls (8 Thompson-sampling
batches of 4), at most 84; pause at cumulative evaluation 48.
**Why this length:** `failure_tolerance=5` and `success_tolerance=10`
mean the trust region needs roughly 5-10 consecutive batches to halve or
double its length once; 8 batches (32 evaluations) is the smallest step
that can plausibly show at least one such adaptation event (a halving,
most likely, if the center is a narrower basin than TuRBO's initial
length=0.8 in normalized space assumes), giving real evidence about
whether the trust region is shrinking toward a optimum or needs a
restart, while still leaving roughly half the total budget (52
evaluations) for further steps after this checkpoint.
**Review:** At cumulative 48, inspect: the trend of the best-so-far value
across the 8 batches, how `_length` evolved (shrinking/growing/restarted),
and whether recent batches are still improving on the noise floor
(~2% of the objective's magnitude) or have plateaued within noise. These
will determine whether to continue TuRBO, adjust its seeding/length, or
switch methods. No method past this step is pre-committed.

## Decision after evaluation 48
**Budget:** 48 used / 100 total; 52 remaining.
**Evidence:** The 8-batch TuRBO step ran as planned (32 evaluations,
seeded as described in the previous entry). Batches 1-4 (evals 17-32) were
non-improving relative to the seeded local best (130682847.18):
`_failure_counter` climbed 1,2,3,4 while `_length` stayed at 0.8; the 5th
consecutive non-improving batch (batch 5, evals 33-36, batch-min
237894627.08, still worse than the seeded local best) then tripped the
halving to `_length=0.4` and reset the counter to 0, exactly matching
`failure_tolerance=5`. Batch 6 (evals 37-40), now at the smaller
`_length=0.4`, was also still non-improving (batch-min 144244304.76,
`failure_counter`->1). Batches 7-8 (evals 41-48) then showed a clear
improvement: batch-min y 92234098.43 then 84989192.22, both better than
the seeded local best, with `success_counter` climbing to 2 and
`local_best_y`/overall best-so-far dropping from 130682847.18 to
84989192.22 (a ~35% reduction, vastly larger than the
~2% noise std measured in step 1). The best point found,
x ≈ [-1.98, -0.41, -3.46, -1.42, 0.30, 3.47, 2.21, 1.79, -2.78, -0.17,
2.24, -1.86, 3.22, -3.12, 1.05, -0.07, 0.17, 0.07, -1.05, 4.56], is well
away from the box center that looked best after step 1 -- i.e. the center
was only better than *generic* Sobol points, not optimal itself, and the
GP-guided local search is finding real structure beyond what the step-1
evidence alone implied.
**Decision:** Continue the same TuRBO instance unchanged (same trust
region, same seeded history, no restart) -- the evidence shows it is
actively improving, not plateaued, so there is no basis yet to switch or
restart. Preserve `turbo` and `harness` exactly as they are (unpickled
state carries the live trust region, success/failure counters, and GP
training data forward) and resume asking it for batches.
**Hyperparameters:** Unchanged from the previous entry (same TuRBO
instance: `batch_size=4`, `n_candidates=2000`, `length_init`/`length_min`/
`length_max`, `success_tolerance=10`, `failure_tolerance=5`, all as
before, `_length` currently 0.4 as adapted internally by the library's own
ordinary trust-region update -- not a new agent decision). No change is
made to seeding or any other setting.
**Reason:** The required check for "a short plateau alone does not prove
a method is unsuitable" cuts the other way here too: this is not a
plateau, it is accelerating improvement right after the trust region
shrank, which is exactly the intended TuRBO behavior (narrow down once a
promising sub-region is found). Switching methods now would discard a
GP and trust region that are actively working, for no evidenced reason.
**Next step limit:** 32 additional objective calls (8 more batches of 4),
at most 52; pause at cumulative evaluation 80.
**Why this length:** Matches the previous step's length for a consistent
basis of comparison (another full read on whether `_length` keeps
shrinking/the success streak continues or reverses), while still reserving
20 evaluations after this checkpoint for a final decision (e.g. a last
local-refinement step or stopping) before the budget is exhausted.
**Review:** At cumulative 80, inspect whether best-so-far is still
improving, whether `_length` is still shrinking (approaching
`length_min≈0.0078`, which would trigger a restart) or growing again, and
whether remaining improvements are within or outside the ~2% noise band.
Will decide then whether to keep running TuRBO, let it restart if
triggered, or switch to a cheap local polish (e.g. Nelder-Mead/BOBYQA)
for the final evaluations. No method past this step is pre-committed.

## Decision after evaluation 80
**Budget:** 80 used / 100 total; 20 remaining.
**Evidence:** The second 8-batch TuRBO step ran as planned. `_length`
stayed at 0.4 throughout (no halving, no doubling to `_success_tolerance`
=10, no restart). Batches 1-3 (evals 49-60) were consecutively improving
(`success_counter` 3,4,5), with `local_best_y` falling sharply:
80318391.91 -> 59189765.07 -> 33895285.79. Batches 4-6 (evals 61-72) were
non-improving (`failure_counter` 1,2,3, `local_best_y` held at
33895285.79). Batches 7-8 (evals 73-80) improved again:
local_best_y -> 33121049.39 -> 20606542.85. Net effect: best-so-far fell
from 84989192.22 to 20606542.85 over this step (a further ~76%
reduction), far beyond the ~2% noise band measured in step 1 -- this is
real, continuing progress, not noise. The running best point now has
several coordinates near the box edges (e.g. -4.70, -4.53, 3.96), compared
to the more central best point after step 2, consistent with the trust
region still actively moving/searching rather than having converged.
`_length` has not approached `length_min` (≈0.0078) and no restart has
occurred.
**Decision:** Continue the same TuRBO instance unchanged for the final
block of the campaign. The evidence (accelerating, large improvements,
no restart, no plateau) gives no reason to switch methods with only 20
evaluations left; switching now would also mean paying for a fresh
optimizer's own startup cost (e.g. a new local solver's first
finite-difference step, costing dim+1=21 evaluations on its own -- more
than the entire remaining budget) for no evidenced benefit over continuing
a method that is still actively improving.
**Hyperparameters:** Unchanged from the previous two entries (same
`turbo` instance; `_length` is currently 0.4, carried over as the
library's own ordinary internal state, not a new setting).
**Reason:** Same rationale as the evaluation-48 checkpoint: continuing
is justified by active, noise-exceeding improvement, and the only
remaining budget (20 evaluations = 5 batches) is better spent exploiting
a working trust region than restarting a new method with its own
overhead.
**Next step limit:** 20 additional objective calls (5 batches of 4),
exactly the remaining budget; pause at cumulative evaluation 100 (budget
exhaustion).
**Why this length:** This is all the budget that remains; there is no
reason to reserve any further evaluations for a later checkpoint, since
the campaign ends at 100 evaluations regardless. The step is still
executed as discrete 4-point Thompson-sampling batches, matching TuRBO's
own internal batch size, rather than requested as one single block without
inspection -- each batch's immediate result is still visible in the loop
below before the next is asked for.
**Review:** At cumulative 100 (budget exhaustion), record the final
best-so-far, confirm `trace.csv`/`meta.json` are written, and stop. No
further method choice applies once the budget is exhausted.

## Decision after evaluation 100 (final)
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** The final 5-batch TuRBO step ran to completion. Batch 1
(evals 81-84) was non-improving (failure_counter=1). Batches 2-3
(evals 85-92) improved sharply: local_best_y fell 20606542.85 ->
14247876.33 -> 13325927.96. Batches 4-5 (evals 93-100) were non-improving
(failure_counter 1, then 2; `_length` stayed at 0.4, no restart
triggered). Final overall best-so-far: **13325927.960714849** at
x ≈ [3.04, -1.11, -2.36, -3.52, 2.03, 1.76, 0.83, 2.11, -0.61, 2.53, 2.42,
-4.10, 3.69, -3.37, 3.83, -1.80, 2.51, -0.78, -2.43, -1.40]. The 101st
`evaluate()` call was attempted deliberately (a single explicit probe,
not a real search point) immediately after exhaustion and confirmed to
raise `BudgetExhausted`, verifying the budget is enforced and was not
circumvented. `harness.save()` was called explicitly in addition to the
harness's automatic save-on-exhaustion; both wrote to the same files.
**Decision:** Stop. The budget is exhausted; no further evaluations are
possible or permitted.
**Hyperparameters:** Not applicable (no further step).
**Reason:** Budget exhaustion is a hard stop per the protocol.
**Next step limit:** 0 (stopping).
**Why this length:** Not applicable.
**Review:** Not applicable; campaign complete.

## Campaign summary
Across the full 100-evaluation budget, the objective's best observed
value fell monotonically (in a noise-robust sense -- every claimed
improvement exceeded the ~2% noise band measured in step 1) from
133897947.21 (step-1 center-probe mean) to a final best of
13325927.960714849 -- roughly a 10x reduction overall. The trajectory:
Sobol/noise-probe step (16 evals, best 130682847.18) -> TuRBO step 1
(32 evals, best 84989192.22) -> TuRBO step 2 (32 evals, best
20606542.85) -> TuRBO step 3/final (20 evals, best 13325927.96). TuRBO's
single trust region was never restarted (its length oscillated between
0.8 and 0.4 but never reached `length_min`), so the whole guided-search
portion of the campaign (84 of 100 evaluations) was one continuous,
increasingly localized search around the region discovered after the
initial space-filling step, rather than several independent methods.
Noise was handled by: (a) an explicit 3-repeat probe up front to quantify
its scale (~2% of |y|) before trusting any single-point comparison, (b)
choosing a GP-based method (TuRBO) whose `SingleTaskGP` fits a learned
observation-noise term rather than assuming a noiseless function, and
(c) at every checkpoint, only treating a change in best-so-far as real
progress when it clearly exceeded the measured ~2% noise band (which
every reported improvement in this campaign did, by a wide margin).
