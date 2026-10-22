## Decision after evaluation 16
**Budget:** 16 used / 100 total; 84 remaining.
**Evidence:** Sobol' design (`seed=0`, `scramble=True`), 16 points. Observed
y values: min 174635.28, max 858003.46, mean 522449.15, std 213340.30
(max/min ratio ~4.9). All values are positive, of the same order of
magnitude (1e5), and vary smoothly in the sense that there is no
orders-of-magnitude outlier or degenerate/NaN value — i.e. no sign of a
broken wrapper or pathological scaling. With only 16 points spread across
a 20-dimensional box, this sample is far too sparse to determine
unimodality/multimodality with any confidence; it only rules out gross
problems and gives a first sense of scale.
**Decision:** Continue the same method (Sobol') from the same `SobolSearch`
instance (continuing its internal low-discrepancy sequence rather than
restarting a new one, which preserves its balance properties), extending
the design from 16 to 42 cumulative points. 42 = 2*(dim+1) is GP-BO's own
documented standard initial-design size
(`src/optimizers/gp_bo.py`: `n_init = 2*(dim+1)`). Building out to exactly
that many Sobol points lets these same, already-real evaluations be reused
as GP-BO's initial design in the next step (by seeding a freshly
constructed `GPBO`'s `X_`/`y_` with them) instead of discarding them and
paying for a second, redundant Sobol phase inside GP-BO itself.
**Hyperparameters:** Unchanged from the initial step's `SobolSearch(bounds,
seed=0)` (scipy `qmc.Sobol`, `scramble=True`, scipy's own defaults, no
tuning) — same instance, so no reconfiguration occurs.
**Reason:** The evidence so far says nothing that would favor an immediate
switch to a local or model-based method yet (not enough points to trust a
local model or a GP fit), and a GP-based acquisition strategy explicitly
suited to this budget regime (~10x dim; this task is 5x dim, GP-BO's
documented regime is of this same order) is the leading candidate for the
remaining budget. Reaching its conventional init size efficiently, using
points already committed to being Sobol-distributed, is the most
budget-efficient way to get there.
**Next step limit:** 26 additional objective calls (of 84 remaining);
pause at cumulative evaluation 42.
**Why this length:** 26 is exactly what is needed to bring the Sobol
design from 16 to GP-BO's documented standard initial-design size of
2*(dim+1) = 42, no more and no less — avoids both under-sampling the
initial design (which would make the first GP fit unreliable) and
over-spending on pure exploration before any model-guided (acquisition)
point is tried, which matters given only 100 evaluations total.
**Review:** At cumulative 42, inspect the full 42-point sample (range,
std, any visible structure such as a cluster of distinctly better points)
and fit quality considerations, then decide whether to proceed into
GP-BO's acquisition-driven phase as planned, extend the space-filling
phase further, or pivot to a different method if the evidence suggests
otherwise. No method beyond this Sobol extension is committed to yet.

## Decision after evaluation 42
**Budget:** 42 used / 100 total; 58 remaining.
**Evidence:** Full 42-point Sobol' sample: min 174635.28, max 1,042,552.12,
mean 517235.13, std 214871.77 — essentially the same scale, spread, and
best value as after the first 16 points (best-so-far has not moved since
evaluation 5). No new, markedly-better region turned up in the additional
26 points, and no sign of extreme ruggedness/outliers (max/min ratio ~6,
similar order to the 16-point sample). This is consistent with (not
proof of) a broadly single-basin, moderately-scaled landscape over this
box, which is more informative for a smooth-surrogate model than for
assuming many competing local optima.
**Decision:** Switch to GP-BO (`src/optimizers/gp_bo.py`, BoTorch
`SingleTaskGP` + analytic LogEI). Construct a fresh `GPBO` instance and
seed its internal `X_`/`y_` directly with the 42 already-collected,
real Sobol evaluations (same points/values just produced — not estimated),
via a campaign-local adapter that writes into the optimizer's own data
arrays rather than replaying them through `tell()` one at a time (no
difference in effect for this optimizer: `tell()` only appends to
`X_`/`y_`, which is exactly what the adapter does). Since
`len(self.y_) >= self.n_init` (42 >= 42) immediately after seeding, GP-BO's
`ask()` will skip its own internal Sobol phase and propose its first
acquisition-driven (LogEI) point right away — avoiding a second, redundant
round of pure space-filling.
**Hyperparameters:** GP-BO's own standard, no-tuning defaults (per its
docstring): `SingleTaskGP` default kernel/priors with `Normalize` input
transform to the task bounds and `Standardize` output transform;
analytic `LogExpectedImprovement` (noiseless, matching the stated
deterministic objective); acquisition optimized via `optimize_acqf` with
`num_restarts=10`, `raw_samples=512` (BoTorch's own tutorial values);
`n_init = 2*(dim+1) = 42` (already satisfied by the seeded data).
`seed=0` for the acquisition optimizer's internal random restarts, for
reproducibility — not a tuned choice. These are all inherited library/
wrapper defaults; nothing is being hand-tuned for this switch.
**Reason:** With 58 evaluations left and a GP surrogate that can exploit
all 42 real observations at once (rather than only the most recent
local neighborhood, as a trust-region or evolutionary method would),
GP-BO's sequential, sample-efficient acquisition loop is the best fit for
squeezing improvement out of a small remaining budget — this is exactly
the regime (budgets of order 10x dim; here 5x dim total, with 58 of it
left for acquisition) the wrapper's own docstring identifies as GP-BO's
intended use case, and the Sobol evidence gives no reason yet to prefer a
purely local solver (BOBYQA) or a population method (CMA-ES) that would
need more generations than this budget affords to adapt meaningfully in
20 dimensions.
**Next step limit:** 20 additional objective calls (of 58 remaining);
pause at cumulative evaluation 62.
**Why this length:** Each GP-BO `ask()` refits an exact GP (cost grows
with data size) and re-optimizes LogEI from scratch, so a batch of 20
one-at-a-time acquisition steps is enough to see whether the surrogate is
finding genuinely better points than the Sobol baseline (i.e. whether
best-so-far keeps dropping, and by how much per step) before committing
more of the remaining 58 evaluations to the same approach. It leaves 38
evaluations in reserve for either continuing GP-BO or switching to local
refinement afterward.
**Review:** At cumulative 62, inspect how much best-so-far improved over
the Sobol baseline (174635.28), the trajectory of the 20 acquisition
points (steadily improving vs. plateaued), and remaining budget, then
decide whether to continue GP-BO, switch to a local polish (e.g. BOBYQA
or CMA-ES seeded near the new best point) for the final evaluations, or
some other adjustment. No later method is committed to yet.

## Decision after evaluation 62
**Budget:** 62 used / 100 total; 38 remaining.
**Evidence:** GP-BO's 20 acquisition-driven points (evaluations 43-62)
dropped best-so-far from the Sobol baseline of 174635.28 to 8240.08 — a
~21x improvement, found by evaluation 48 (the 6th acquisition point).
However, none of the subsequent 14 acquisition points (evaluations 49-62)
improved on 8240.08; their y values ranged roughly 41,785-158,122, i.e.
LogEI kept proposing points that are still far worse than the current
best, consistent with the acquisition function spending this stretch
exploring uncertain regions of the 20-D space rather than refining close
around the current optimum (expected analytic-EI behavior with a GP fit
on scattered, globally-distributed data, not evidence the surrogate is
broken).
**Decision:** Switch to a local refinement method: BOBYQA
(`pybobyqa.solve`, as in `src/optimizers/bobyqa.py`), but via a
campaign-local adapter that starts the trust-region search from the
current best point (8240.08) instead of the wrapper's hardcoded box
midpoint — the wrapper's `BOBYQA` class does not expose a custom `x0`, so
this calls `pybobyqa.solve` directly with the same options the wrapper
uses otherwise (`scaling_within_bounds=True`, `maxfun=20000`,
`do_logging=False`), run in the same background-thread/queue pattern so it
can still be driven one point at a time against the live harness.
GP-BO's own `X_`/`y_` history is retained in-process (not discarded) in
case it is useful again, but is not fed into BOBYQA, which builds its own
local quadratic model from scratch around the new starting point; BOBYQA
has no mechanism to ingest arbitrary external points into its model.
**Hyperparameters:** `x0` = current best point (8240.08's location),
changed from the wrapper's default (box midpoint) — a deliberate,
evidence-based choice given 38 evaluations is too little for a
box-midpoint-started local search to both find and refine a good region
in 20-D. `rhobeg` left at Py-BOBYQA's own default (0.1, under
`scaling_within_bounds=True`, i.e. 10% of the rescaled unit-cube radius)
rather than shrinking it further: the new start is already in a good
region found by a global method, but its local scale/curvature is
unknown, so the library's standard default trust-region radius is kept
rather than guessing a smaller one. `rhoend` left at the library default
(1e-8). `maxfun=20000` (unchanged rationale from the wrapper's own
docstring: only the external budget should bind, not this internal cap).
**Reason:** With only 38 evaluations left and a GP-BO run that has clearly
plateaued for 14 consecutive points, a model-based local solver that
builds a quadratic trust-region model directly around the best known
point is better suited to squeezing further improvement from a small
remaining budget than either continuing the same GP-BO acquisition
(which just spent 14/20 steps without improving) or restarting a
population method like CMA-ES (which would need more generations than 38
evaluations affords to meaningfully adapt in 20 dimensions from a cold
start).
**Next step limit:** 25 additional objective calls (of 38 remaining);
pause at cumulative evaluation 87.
**Why this length:** BOBYQA's own default initial model-building phase for
an unconstrained quadratic model in d=20 dimensions uses on the order of
2*d+1 = 41 points before it is fully determined, which is more than the
entire remaining budget (38) — so full convergence is not guaranteed
either way. Checking in at 25 (leaving 13 in reserve) gives enough calls
to see whether the trust-region search is making real progress from the
new starting point before committing the rest of the budget to it, while
preserving some reserve for a different allocation if it is not.
**Review:** At cumulative 87, inspect whether best-so-far has improved
beyond 8240.08 and by how much, whether BOBYQA's internal solver thread
has already exhausted itself (raises `OptimizerExhausted`, a legitimate
early-stop condition) before using all 25 calls, then decide whether to
spend the final reserve continuing BOBYQA or stop early. No later method
is committed to yet.

**Implementation note (evaluations 63-87):** the harness/optimizer state
lives in one long-running Python process, driven across shell calls via a
small file-based RPC (two FIFOs) so that the *same* `AgentTaskHarness`
instance (and its real, in-memory budget counter) persists for the whole
campaign — no second harness instance is ever constructed, consistent
with the hard constraint against working around budget enforcement.
BOBYQA's own wrapper (`src/optimizers/bobyqa.py`) hardcodes its starting
point at the box midpoint with no way to override it, so starting from the
current best point (as decided above) was implemented as a campaign-local
adapter: a separate short-lived script calls `pybobyqa.solve` directly
with `x0` = the current best point, using the same options as the wrapper
(`scaling_within_bounds=True`, `maxfun=20000`, `do_logging=False`) and
routes every real function evaluation through the one persistent harness
process via the existing RPC, so the budget counter and trace are exactly
as if this were called inline. This changes only how the call is wired,
not the decision already recorded above. Result: best-so-far improved
from 8240.08 to 6527.96 over these 25 calls (not yet converged — the
local step limit was hit before BOBYQA's own internal stopping criterion).

## Decision after evaluation 87
**Budget:** 87 used / 100 total; 13 remaining.
**Evidence:** The 25-call BOBYQA block (evaluations 63-87, `x0` = the
8240.08 point) improved best-so-far to 6527.96. The trajectory within the
block was non-monotonic (individual trial points ranged ~6528-16287,
since a trust-region solver proposes exploratory/model-building points
that are not all improvements) but the running best dropped repeatedly
throughout the block, including on the very last call, with no sign of
having flattened out yet — BOBYQA was stopped by this block's own 25-call
limit, not by its internal convergence criterion (`rhoend`), so there is
still genuine local structure left to exploit.
**Decision:** Continue local refinement with a fresh BOBYQA run (same
campaign-local `pybobyqa.solve` adapter as above), restarted from the new
best point (6527.96's location). This is a restart, not a resumption: the
previous run's internal trust-region state (its local quadratic model)
is not retained — `pybobyqa.solve` is a single blocking call with no
save/resume API — so the only state carried forward is the new, better
`x0`, which is the information that matters most given how little budget
is left. All real observations collected so far remain in the trace
regardless.
**Hyperparameters:** Unchanged from the evaluation-62 decision
(`scaling_within_bounds=True`, `maxfun=20000`, `rhobeg`/`rhoend` at
Py-BOBYQA's own defaults), only `x0` is updated to the new best point.
**Reason:** With only 13 evaluations left, this is necessarily the final
step of the campaign regardless of outcome. Restarting BOBYQA from the
improved point is the best-justified use of so small a reserve: the
previous block showed continued, non-plateaued local progress, and a
trust-region method converges fast per-evaluation near a good point,
which suits a final short polish better than switching to a global method
(no budget left to explore broadly) or than GP-BO (fitting+optimizing a
GP over 87+ points per step is comparatively expensive for only 13 calls).
**Next step limit:** 13 additional objective calls (all of the remaining
budget); pause at cumulative evaluation 100.
**Why this length:** This is the entire remaining budget; since BOBYQA
was still improving and this is the last step of the campaign regardless
of the outcome, there is no reason to reserve any further evaluations
behind this final block.
**Review:** Not applicable beyond confirming the final best value and
that the budget is exhausted — this is the last step; no further method
choice follows it.

