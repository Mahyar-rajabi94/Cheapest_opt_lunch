## Decision after evaluation 16
**Budget:** 16 used / 100 total; 84 remaining.
**Evidence:** 16 scrambled-Sobol points across the full `[-5,5]^40` box
(seed 0). Observed y ranged from 694,755.6 to 2,738,250.9 (factor ~3.9x),
mean 1,452,870, median 1,490,560 — large magnitude but a single coherent
order of magnitude (no sign bit flips, no extreme outliers by orders of
magnitude), consistent with a smooth, roughly unimodal-at-this-scale
landscape rather than a wildly discontinuous or highly multimodal one.
With only 16 points spread over a 40-D box, point-to-point distances are
large, so this sample cannot rule out local multimodality — it only
supports ruling out pathological scale behavior (e.g. need for log-space
modeling) and confirms there is a non-trivial gradient of y across the box
worth exploiting (best-so-far dropped from the first point's 1,571,859 to
694,756 within these 16 draws).
**Decision:** Switch to GP-BO (`src/optimizers/gp_bo.py`, BoTorch
SingleTaskGP + analytic LogEI), seeded with all 16 Sobol observations
collected so far (via `tell`) instead of drawing a fresh Sobol init.
**Hyperparameters:** `n_init` overridden from the wrapper's default
`2*(dim+1) = 82` down to `16` (the number of initial points already on
hand) via the documented public attribute on the `GPBO` instance — a
campaign-local adaptation, not a change to the underlying BoTorch
acquisition logic. Rationale: at this task's budget (100, i.e. 2.5x dim),
spending 82 evaluations on a pure-Sobol init (as the packaged wrapper
would do by default) would leave only 18 calls for any acquisition-guided
exploitation, defeating the purpose of using a surrogate at all. Reusing
the 16 Sobol points already collected avoids wasting evaluations
re-deriving an initial design. All other GPBO hyperparameters are kept at
the wrapper's inherited defaults (BoTorch `SingleTaskGP` default kernel/
priors, `Normalize`+`Standardize` transforms, analytic `LogExpectedImprovement`,
`optimize_acqf` with `num_restarts=10`, `raw_samples=512` — BoTorch
0.x tutorial defaults per the module docstring): no evidence yet justifies
changing these, and this is a first trial of the method.
**Reason:** The problem's deterministic, apparently-smooth, large-but-
single-scale response surface is the regime this GP-BO implementation is
documented for (expensive, continuous, budgets of order 10x dim — closer
to this task's 2.5x dim than CMA-ES's regime, which typically needs many
more generations than ~5-6 to usefully adapt its covariance at 40-D). A
model-based acquisition can also use the already-collected 16 points
immediately, rather than discarding them.
**Next step limit:** 20 additional objective calls, at most 84 remaining;
pause at cumulative evaluation 36.
**Why this length:** Each GP-BO step costs a GP refit + acquisition
optimization in 40-D from a small, growing dataset — 20 points is enough
to see whether LogEI is finding genuine improvements over the Sobol
best (694,756) and whether a 40-D GP fit on ~16-36 points is behaving
sensibly (e.g. not degenerating to near-random proposals), without
committing the majority of the remaining 84-call budget to an unproven
surrogate.
**Review:** At evaluation 36, inspect: whether best-so-far improved
materially versus 694,756; whether proposed points cluster (converging)
or continue spreading broadly (surrogate not yet informative); and
per-call wall time, to confirm the method remains tractable within the
remaining budget. No commitment to a later method.

## Decision after evaluation 36
**Budget:** 36 used / 100 total; 64 remaining.
**Evidence:** The 20 GP-BO (LogEI) points drove best-so-far from 694,756
down to 219,440 — a 3.2x improvement over the entire prior Sobol stage,
achieved in fewer evaluations. The within-step y values (min 219,440,
max 908,676, median 393,647) are noticeably lower on average than the
Sobol step's (median 1,490,560), showing the acquisition function is
steering toward better regions rather than proposing near-random points —
i.e. the 40-D GP fit on a growing, still-small dataset (16→35 points) is
informative enough to be useful, addressing the main risk flagged at the
last checkpoint. Best-so-far improved partway through the block (at the
11th point of this step) and then held for the remaining 9 — consistent
with normal LogEI behavior (most calls explore/refine around the current
incumbent; not evidence of stalling given it is a single step and improving
GPs often plateau briefly between incumbent updates). Wall time: ~10s for
20 evaluations (~0.5s/call) — negligible relative to budget, no tractability
concern.
**Decision:** Continue GP-BO unchanged (same `GPBO` instance, same
accumulated data/state — no restart).
**Hyperparameters:** Unchanged from the previous entry (`n_init=16`
already consumed/irrelevant now that `len(y_) > n_init`; `num_restarts=10`,
`raw_samples=512`, BoTorch default `SingleTaskGP` kernel/priors and
`Normalize`+`Standardize` transforms, analytic `LogExpectedImprovement`).
No evidence yet suggests these need revision.
**Reason:** The method is actively improving the incumbent and remains
computationally cheap; there is no indication (stalling, degenerate
proposals, excessive runtime) that would justify switching or retuning.
Continuing is the straightforward choice when a method is working and
budget remains.
**Next step limit:** 30 additional objective calls, at most 64 remaining;
pause at cumulative evaluation 66.
**Why this length:** Large enough to see whether the improvement trend
continues over a longer horizon (distinguishing genuine convergence from a
lucky early block), while leaving a deliberate reserve (34 calls) for a
possible final local-refinement stage once the GP-driven search's marginal
gains are better understood — not committing the entire remaining budget
to a single further block.
**Review:** At evaluation 66, inspect whether best-so-far continues to
drop, by how much relative to this block's drop, and whether proposed
points are beginning to cluster tightly around an incumbent (suggesting
local convergence, where a dedicated local refiner or reduced
exploration might be more efficient per remaining evaluation) versus
still moving substantially between points (suggesting continued global
GP-BO search remains worthwhile).

## Decision after evaluation 66
**Budget:** 66 used / 100 total; 34 remaining.
**Evidence:** This 30-point GP-BO block improved best-so-far from 219,440
to 114,723 (another ~1.9x), continuing the clear downward trend from the
previous block (3.2x). The best-so-far trajectory shows five distinct
improving updates spread fairly evenly through the block (at calls
2, 4, 11, 20, 25 of 30), not a single early drop followed by a long flat
tail — i.e. LogEI is still finding new incumbents regularly, with no sign
of convergence/plateau yet. Within-step y values still range widely
(114,723 to 439,767), showing the acquisition function is still balancing
exploration against exploitation rather than collapsing to one small
region. Wall time rose to ~1.17s/call (35s/30, vs ~0.5s/call at n=16-35
data), consistent with the expected cubic cost of exact GP fitting as the
dataset grows (now 66 points) — still cheap relative to the remaining
budget.
**Decision:** Continue GP-BO unchanged (same instance/state).
**Hyperparameters:** Unchanged — see prior entries (`num_restarts=10`,
`raw_samples=512`, BoTorch default GP kernel/priors and transforms,
analytic LogEI). Still no evidence to justify retuning; the method is
improving and tractable.
**Reason:** Continued, regular incumbent improvements and no plateau or
runtime concern both favor continuing the same method rather than
switching while it is still clearly working.
**Next step limit:** 20 additional objective calls, at most 34 remaining;
pause at cumulative evaluation 86.
**Why this length:** Keeps a deliberate reserve of 14 calls after this
checkpoint for a final decision (e.g. a short dedicated local-polish step
near the very end, or continuing GP-BO to completion), rather than
pre-committing the entire remaining budget to one more GP-BO block; 20 is
large enough to judge whether the improving trend is holding up as the GP
is fit on an increasingly larger, costlier dataset.
**Review:** At evaluation 86, inspect the magnitude of improvement in this
block versus the last two (is the rate of improvement per 20-30 calls
decaying, suggesting diminishing returns from further global GP-BO
exploration, in which case switching the final ~14 calls to pure local
exploitation near the current incumbent — e.g. a tight CMA-ES restart or
a tight Gaussian local perturbation around the best point — could be more
efficient than continuing broad LogEI search).

## Decision after evaluation 86
**Budget:** 86 used / 100 total; 14 remaining.
**Evidence:** This 20-point block improved best-so-far from 114,723 to
79,723 (~1.44x), continuing to improve but at a visibly decaying rate
versus the prior two blocks (3.2x, then 1.9x, then 1.44x per block of
comparable size) — consistent with gradually diminishing returns as the
surrogate narrows in, but still a genuine, non-trivial improvement (two
distinct incumbent updates within the block, at calls 14 and 17 of 20),
not a flat plateau. Within-step values still span a wide range (79,724 to
567,815), so LogEI is still exploring broadly rather than having fully
collapsed to pure local exploitation on its own. Wall time ~1.85s/call
(37s/20) at 66-86 accumulated points.
**Decision:** Continue GP-BO unchanged (same instance/state) for the
remaining budget, rather than switching methods or restarting.
**Hyperparameters:** Unchanged — see prior entries (`num_restarts=10`,
`raw_samples=512`, BoTorch default GP kernel/priors and transforms,
analytic LogEI).
**Reason:** With only 14 evaluations left, switching to an untested method
now (e.g. CMA-ES or a hand-rolled local perturbation) would spend a
meaningful fraction of the remaining budget re-establishing basic
information the GP already encodes (86 observations' worth of structure),
with no evidence that local-only exploitation would outperform LogEI's own
built-in exploitation behavior (LogEI already trades off exploration vs.
exploitation per point; it is not a purely global/undirected search that
needs an external local-refinement phase to close out). The method is
still improving every block, just by a shrinking amount, which is the
expected and acceptable behavior of Bayesian optimization approaching a
budget's end, not a failure signal. Continuing is therefore the better
use of a very small remaining budget than a cold-start switch.
**Next step limit:** 14 additional objective calls (all of the remaining
budget); pause at cumulative evaluation 100 (budget exhaustion).
**Why this length:** This is the entire remaining budget; since GP-BO is
still producing real improvements and no viable alternative is better
positioned to use the final 14 calls, there is no reason to reserve any
further holdout block — a smaller block followed by another checkpoint
would only cost one more GP refit's worth of overhead without changing
the decision (continue GP-BO) and would shrink an already-small final
block for no benefit.
**Review:** Not applicable beyond final reporting — this exhausts the
budget. No further method is pre-committed; this is simply the last block
of the already-chosen, still-working method.

## Outcome (campaign end)
**Budget:** 100 used / 100 total; 0 remaining. Budget exhausted exactly as
planned; the harness auto-saved `trace.csv`/`meta.json` on the 100th call.
**Final block evidence:** The last 14 GP-BO calls improved best-so-far from
79,723.6 to 50,230.3 (~1.59x), with three distinct incumbent updates
(calls 2, 4, 9 of 14) — the method was still finding genuine improvements
on the very last block, confirming the evaluation 86 decision to continue
GP-BO rather than switch was reasonable; there was no observed plateau at
any point in the campaign.
**Final result:** Best observed value y = 50,230.302 (minimize), found at
cumulative evaluation 95 and held through evaluation 100. Overall
trajectory across the full campaign: 16 Sobol points (best 694,756) → GP-BO
to evaluation 36 (219,440) → eval 66 (114,723) → eval 86 (79,724) → eval
100 (50,230) — a steady, monotonically-improving sequence of per-block
best values with a decaying but still-positive improvement rate, and no
method switch after the initial Sobol-to-GP-BO transition at evaluation
16.
**Decision:** Stop. Budget exhausted; no further evaluations possible or
needed.
