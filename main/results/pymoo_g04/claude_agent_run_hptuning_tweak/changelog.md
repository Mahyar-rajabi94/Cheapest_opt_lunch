## Decision after evaluation 15
**Budget:** 15 used / 100 total; 85 remaining.
**Evidence:** 15 Sobol' points across the full 5-D box (seed 0) all
produced finite, broadly similar-magnitude merit values, ranging from
-28622.38 (best, at x ≈ [79.55, 39.86, 32.55, 43.54, 28.90]) to -15100.50
(worst), median -25016.80. No non-finite values, and no extreme
high-penalty outliers relative to the rest of the sample (the spread is
less than 2x between best and worst) — this suggests the sampled region is
largely feasible or close to it, with no obvious "cliff" dominating the
Sobol sample. The best point is an interior point, not clearly pinned to a
bound, which is mildly encouraging for a smooth local structure, but 15
points in 5-D is sparse and this is not enough to characterize smoothness
or multimodality with confidence.
**Decision:** Switch to GP-BO (`src/optimizers/gp_bo.py`, BoTorch
SingleTaskGP + analytic LogEI). Restart with a fresh optimizer instance
(GP-BO's own convention is to start from its own Sobol initial design, not
an arbitrary external one) but reuse all 15 already-collected observations
as the GP's training data via `tell`, so no evaluations are wasted; the
harness instance and its evaluation counter are unchanged.
**Hyperparameters:** GP-BO's own defaults, inherited from
`src/optimizers/gp_bo.py` (BoTorch SingleTaskGP default kernel/priors,
input normalization to `harness.bounds`, output standardization, analytic
LogExpectedImprovement, `optimize_acqf` with num_restarts=10,
raw_samples=512 — the pair BoTorch's own tutorials use). The wrapper's own
initial-design convention (2*(dim+1)=12 Sobol points) is superseded here
by feeding it the 15 points already collected, which exceeds that count,
so GP-BO can fit its first GP and propose an acquisition-driven point
immediately rather than drawing a fresh initial design. Seed=0 for the
wrapper's own internal RNG (candidate generation tie-breaks), not a tuned
choice.
**Reason:** Dim=5 and 85 evaluations remaining falls squarely in GP-BO's
documented practical regime (~10-20x dim, i.e. ~50-100). The Sobol
sample's finite, moderately-varying values with no visible blow-ups
suggest the merit surface is tractable for a smooth GP surrogate rather
than needing a cliff-tolerant method. GP-BO also makes the best use of a
tight 100-call budget by actively choosing each next point from a fitted
model, rather than spending more calls on an undirected design.
**Next step limit:** 20 additional objective calls, at most 85; pause at
cumulative evaluation 35.
**Why this length:** 20 calls (4 GP refits at a conservative batch-of-5
internal cadence, or up to 20 single-point acquisitions) is enough to see
whether LogEI-driven points measurably improve on the Sobol best, while
keeping most of the budget (65 calls after this block) uncommitted in case
GP-BO is not converging usefully and a different method should take over.
**Review:** At evaluation 35, check whether best-so-far improved versus
the Sobol best (-28622.38), whether successive GP-BO proposals cluster
(suggesting convergence to a local optimum) or keep exploring broadly, and
watch per-evaluation wall time (GP refit cost) relative to the 65
remaining calls. No committed next method.

## Decision after evaluation 35
**Budget:** 35 used / 100 total; 65 remaining.
**Evidence:** The 20 GP-BO/LogEI proposals improved best-so-far from
-28622.38 (Sobol) to -29904.90, a ~4.5% improvement, with the new best at
x ≈ [78.00, 35.57, 33.48, 45.00, 30.28] — notably sitting on two of the
box's bounds (x0 at its lower bound 78, x3 at its upper bound 45), unlike
the Sobol best which was interior. Individual proposals within the block
still varied substantially (from -18613.70 up to -29904.90), i.e. LogEI is
still alternating between exploitative moves near the current best and
exploratory probes elsewhere, not yet collapsed into a tight local
plateau. Wall time was negligible (~3.1s for 20 GP refits + acquisitions),
so compute cost is not a constraint on step length. No non-finite or
wildly divergent merit values appeared.
**Decision:** Continue GP-BO unchanged, same running instance (same `X_`,
`y_` history retained, no restart) — the method is actively improving and
has not plateaued, and the identified boundary-seeking behavior (two
coordinates pinned to bounds) is exactly the kind of structure a
continuing LogEI search over the same box can keep exploiting.
**Hyperparameters:** Unchanged from the previous block (see decision after
evaluation 15: BoTorch SingleTaskGP defaults, Normalize/Standardize
transforms, analytic LogEI, optimize_acqf with num_restarts=10,
raw_samples=512, seed=0). Continuing the same `GPBO` object means its
internal RNG state also carries over rather than resetting.
**Reason:** Evidence directly supports continuation: best-so-far is still
moving, the within-block spread shows the acquisition function is still
discriminating between promising and unpromising regions (not just noise
around a converged point), and nothing about the last block (cost,
degenerate proposals, stagnation) suggests a different method would do
better with the remaining budget.
**Next step limit:** 25 additional objective calls, at most 65; pause at
cumulative evaluation 60.
**Why this length:** Large enough to let the GP refit on a meaningfully
bigger dataset (36→60 points) and show whether improvement continues,
flattens, or the boundary-seeking pattern sharpens further, while still
leaving 40 evaluations (a substantial fraction of the total budget) for a
further GP-BO block, a local polish step (e.g. BOBYQA/L-BFGS-B from the
current best), or a method switch if this block stagnates.
**Review:** At evaluation 60, compare best-so-far against -29904.90;
check whether the rate of improvement is slowing (diminishing returns
suggesting a local-solver polish or stop) or still substantial (suggesting
continuing GP-BO); check whether proposed points continue to press against
the same bounds (x0 low, x3 high) or move off them.

## Decision after evaluation 60
**Budget:** 60 used / 100 total; 40 remaining.
**Evidence:** Best-so-far improved only marginally this block, from
-29904.90 to -29991.26 (~0.3%), at x ≈ [78.00, 33.00, 31.47, 34.46, 37.29]
— now with *two* coordinates pinned to bounds (x0 at lower bound 78, x1 at
lower bound 33), versus one previously. Several of the last ~8 proposals
clustered tightly in the -29959 to -29974 range (e.g. evaluations 53, 54,
56, 57, 58, 60), consistent with LogEI narrowing around a local optimum,
interspersed with a few much worse exploratory probes (44: -26000.8, 49:
-27052.5, 50: -25123.6, 59: -19806.9) that did not pay off. This pattern
(small gains, a tight cluster near the current best, continued
bound-pinning) suggests GP-BO has located the neighborhood of a local
(likely boundary-constrained) optimum and is now spending a good share of
each block's calls on global exploration that is not improving the
incumbent much further.
**Decision:** Switch to a BOBYQA local-polish step, restarted fresh but
seeded at the current incumbent rather than the box midpoint. The packaged
`BOBYQA` wrapper (`src/optimizers/bobyqa.py`) hardcodes
`x0 = midpoint of bounds`, which would discard the progress already made,
so I am using a small campaign-local adapter that calls the same
underlying `pybobyqa.solve` with identical settings but
`x0 = GP-BO's current best point` — this is exactly the "campaign-local
adapter" case anticipated by the protocol when a wrapper fixes a setting I
need to control. This is a local trust-region model-based solver, well
suited to refining within the neighborhood of an already-identified
near-optimum, especially since two coordinates are pinned to bounds
(BOBYQA handles box constraints directly and natively, rather than relying
on surrogate-model LogEI to keep rediscovering that boundary). The GP-BO
history (`X_`, `y_`, 60 points) is preserved in memory in case BOBYQA
stalls and I decide to return to GP-BO; the harness and its evaluation
counter are unchanged.
**Hyperparameters:** Same as the packaged `BOBYQA` wrapper's own defaults
(Py-BOBYQA / `pybobyqa.solve`, `scaling_within_bounds=True`,
`maxfun=20000`, `rhoend` at the library's own default 1e-8), except
`x0` set to the current incumbent [78.00, 33.00, 31.47228347, 34.45849542,
37.28530667] instead of the box midpoint, and `rhobeg` left at
Py-BOBYQA's own default for `scaling_within_bounds=True` (0.1 in the
rescaled unit-ish space) — unchanged from the packaged wrapper's choice,
since the trust region should still start reasonably sized relative to the
box, not pre-shrunk, given I have no local curvature estimate yet.
**Reason:** Diminishing per-block improvement plus repeated bound-pinning
points to being in the basin of a (possibly boundary) local optimum where
a dedicated local solver should out-perform continued global
acquisition-driven search on a per-evaluation basis. 40 evaluations remain
(enough for BOBYQA's trust-region iterations in 5-D) and this is an
information-cheap thing to try before committing the rest of the budget.
**Next step limit:** 20 additional objective calls, at most 40; pause at
cumulative evaluation 80.
**Why this length:** 20 calls is enough for several BOBYQA trust-region
shrink/expand cycles in 5 dimensions to show convergence behavior, while
reserving at least 20 evaluations afterward either to let BOBYQA finish
converging, to resume GP-BO if the polish underperforms, or to stop early
if it has clearly converged.
**Review:** At evaluation 80, check whether BOBYQA's incumbent improved on
-29991.26, whether it has converged (small/no further movement,
trust-region radius collapsing) or is still making progress, and whether
it moved off the two currently-active bounds (which would suggest the true
optimum is interior and the GP-BO-identified boundary was only a local
artifact) or confirmed them.

## Decision after evaluation 80
**Budget:** 80 used / 100 total; 20 remaining.
**Evidence:** BOBYQA's 20-call block improved best-so-far from -29991.26
to -30232.87 (~0.8%), at x ≈ [78.26, 33.18, 31.15, 36.91, 37.34] — now
interior on all 5 coordinates (no longer pinned to the bounds GP-BO had
been pressing against), which answers one of this checkpoint's review
questions: the GP-BO boundary behavior looks like a local-search artifact
rather than the true optimum sitting on the box edge. The trajectory
within the block was not monotonic (e.g. a dip to -27530.2 at evaluation
69, -28776.6 at 71) reflecting BOBYQA's trust-region model-improvement
probes, but the final several evaluations (76-80: -30103.5, -30194.95,
-30222.9, -30219.0, -30232.9) show a clear, still-active improving trend
with no sign of convergence (trust-region collapse) yet — Py-BOBYQA's
solver thread had not returned/exhausted by evaluation 80.
**Decision:** Restart BOBYQA with a fresh solver instance, seeded at the
new incumbent [78.26248254, 33.18276222, 31.14820972, 36.90691873,
37.34322905] (rather than the box midpoint), for the final block. This is
a restart rather than a continuation for a practical reason specific to
this wrapper's implementation: Py-BOBYQA's solver runs in a background
daemon thread that is local to the Python process that created it (see
`src/optimizers/bobyqa.py`'s `queue`/`threading` plumbing); each of my
execution blocks in this campaign runs as a separate process, so the
previous block's live solver thread (and its internal trust-region
state/model) does not persist across blocks the way an in-memory
array-based optimizer (CMA-ES, DE, GP-BO) would. Restarting at the
best-known point is the correct way to continue local refinement given
that constraint: it resets the trust-region radius to BOBYQA's own default
starting size rather than whatever (unobservable) radius the previous
thread had reached, but starts from the best point found so far rather
than from scratch, so no evaluation progress is lost even though internal
solver state is. The harness and its evaluation counter remain unchanged
and are the sole source of truth for the trace.
**Hyperparameters:** Identical settings to the previous BOBYQA block
(Py-BOBYQA `scaling_within_bounds=True`, `maxfun=20000`, library-default
`rhobeg`/`rhoend`), with only `x0` updated to the new incumbent above.
**Reason:** The previous block was still actively improving with no
convergence signal, and the move to an interior solution removed the one
open question (boundary vs. true optimum) that might have motivated
reconsidering the method. With only 20 evaluations left, restarting the
same well-matched local solver from the best point found is the most
direct way to use the remaining budget; there is not enough budget left to
productively explore a different global method from scratch, and BOBYQA
has so far produced the best results per evaluation of any method tried
in this campaign.
**Next step limit:** 20 additional objective calls, at most 20 (all
remaining budget); pause at cumulative evaluation 100 (budget exhaustion).
**Why this length:** This is the entire remaining budget. Splitting it
into a smaller block and reserving some would only be useful if another
method were likely to be tried afterward, but no evaluations would remain
for that; using the full remaining budget on the method that has been
improving fastest per-evaluation is the better use of it.
**Review:** Not applicable beyond this block — budget will be exhausted at
evaluation 100. The outcome (final best-so-far, whether BOBYQA converged
or was still improving when the budget ran out) will be recorded as the
closing entry.

## Decision after evaluation 100 (final)
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** The final 20-call BOBYQA block improved best-so-far from
-30232.87 to -30253.63, at x ≈ [78.00, 33.30, 31.14, 36.91, 37.33]
(evaluation 96). As in the previous block, individual evaluations were not
monotonic (model-improvement/trust-region probes dipped as low-quality as
-27835.70 at evaluation 89), but the best values within the block clustered
tightly (-30232.87, -30250.85, -30234.49, -30253.63, -30247.60, -30243.81
at evaluations 81/83/85/87/94/96/97/99) with a span of under 0.1% among the
top values, indicating the search has converged into a narrow local
neighborhood. The solver thread had not emitted its "converged" sentinel
when the budget ran out (no early-exhaustion signal was seen), so this is
budget exhaustion, not solver convergence, that ends the campaign.
**Decision:** Stop. The budget of 100 evaluations is exhausted;
`harness.evaluate()` can no longer be called.
**Hyperparameters:** Not applicable (no further evaluations).
**Reason:** Hard budget constraint reached — there is no remaining budget
to execute any further decision, and the harness enforces this directly
(`BudgetExhausted` would be raised on any further `evaluate()` call).
**Next step limit:** K = 0 (stopping).
**Why this length:** Not applicable — stopping.
**Review:** Not applicable — stopping.

## Final outcome
- **Global best merit found:** **-30253.63293456146**, at
  x ≈ [78.0, 33.30236973, 31.14247280, 36.90728142, 37.32986669],
  observed at evaluation 96 under the `bobyqa_polish` tag (the last BOBYQA
  restart block, seeded at the previous best).
- **Evaluations used:** 100 / 100 (budget fully exhausted; not stopped
  early).
- **Campaign summary:** Sobol' space-filling (15 calls) → GP-BO/LogEI,
  warm-started on the Sobol data (20 + 25 = 45 calls) → BOBYQA local
  polish from the GP-BO incumbent, restarted once partway through for an
  implementation reason (background-thread solver state does not persist
  across separate process-level execution blocks), seeded each time at the
  best point found so far (20 + 20 = 40 calls). Best-so-far improved
  monotonically across the whole campaign: -28622.38 (Sobol) →
  -29904.90 → -29991.26 (GP-BO) → -30232.87 → -30253.63 (BOBYQA).
- `harness.save()` was called explicitly after the final evaluation (the
  harness's own automatic save on budget exhaustion had already written
  the same files); `results/agent_run/trace.csv` (100 rows) and
  `results/agent_run/meta.json` were confirmed present and consistent with
  this account.
