## Decision after evaluation 16
**Budget:** 16 used / 100 total; 84 remaining.
**Evidence:** Sobol' initial design (seed=0), 16 points, `y` ranged from
16267.5 (best) to 751501.4 (worst), median ≈ 231761 — a large (≈46×)
spread. Correlation between `y` and squared distance from the box center
(`||x||^2`) across the 16 points is +0.63: a real but imperfect
relationship, i.e. points closer to the center tend to score lower but
not uniformly, consistent with a smooth, anisotropic bowl-like landscape
(some directions much steeper than others) rather than either a flat
region or scattered, uncorrelated local optima. No single-point evidence
of multimodal traps: all 16 values are strictly positive, continuously
graded, and the best two points share a somewhat similar mix of
coordinate signs. Evaluation time was negligible (sub-millisecond), so
per-iteration solver overhead is not a constraint on method choice.
**Decision:** Switch from Sobol' to BOBYQA
(`src/optimizers/bobyqa.py`, Py-BOBYQA), a sequential, model-based
trust-region local solver. This step restarts rather than continues an
optimizer instance (Sobol' has no further role once we commit to local
refinement of the apparent bowl); the Sobol' observations themselves are
preserved in the trace (tagged `sobol_init`) but are not fed into
BOBYQA's model, since Py-BOBYQA only accepts points through its own
closed-loop solver starting from its own initial interpolation set — it
has no supported API to seed extra, externally-obtained points into that
set.
**Hyperparameters:** Per the wrapper's own documented, no-tuning
defaults: starting point = box midpoint (0,...,0) (not the best Sobol'
point — BOBYQA's wrapper hardcodes the midpoint as its start and exposes
no parameter to change it; the midpoint is itself a reasonable start
given the moderate center-ward correlation just observed), bounds
`[[-5,5]]*10`, `scaling_within_bounds=True`, `rhobeg` left at the
library's own default (0.1 under the internal unit-cube scaling),
`rhoend=1e-8` (library default), `maxfun=20000` (the wrapper's fixed,
large ceiling so the external budget is what actually limits the run).
`seed` is accepted but unused by BOBYQA (deterministic given start point
and bounds).
**Reason:** BOBYQA builds and refines a quadratic trust-region model of
the objective from the points it visits, which is exactly suited to a
smooth, anisotropic, bowl-shaped landscape like the one the Sobol'
evidence suggests, and is normally far more sample-efficient than
population-based methods (CMA-ES, DE) on such landscapes, especially
important with only 84 evaluations left for a 10-dimensional problem.
CMA-ES was considered but set aside for now: its default population size
here (~10) would allow only ~8 generations in the remaining budget,
likely too few generations to meaningfully adapt its covariance matrix to
the anisotropy already visible in the data, whereas BOBYQA's local
quadratic model can exploit that anisotropy directly from its
interpolation set.
**Next step limit:** 30 additional objective calls, at most 84; pause at
cumulative evaluation 46.
**Why this length:** Large enough to let BOBYQA build its initial
quadratic interpolation set (roughly `2*dim+1 = 21` points for an
under-determined model) and take several trust-region update steps
beyond that, so the trend in `y` is informative rather than just
reflecting initial model construction; small enough to leave most of the
budget (54 evaluations) uncommitted in case BOBYQA stalls, converges to a
poor local point, or the evidence instead favors a different method.
**Review:** Whether `y_best_so_far` is still improving meaningfully over
this block (vs. plateauing early, which could indicate a shallow local
trap or premature trust-region shrinkage) and how it compares with the
magnitude of improvement already seen; whether BOBYQA's internal solver
thread has already returned (converged/exhausted) before using the full
30 calls. No commitment to a later method.

## Decision after evaluation 46
**Budget:** 46 used / 100 total; 54 remaining.
**Evidence:** BOBYQA's 30-call block (same `bobyqa1` instance, continuing
its solver thread/state) took the running best from 16267.5 down to
2101.4 — a >7x reduction. The trajectory within the block is not
monotonic (e.g. a spike to 61670.4 at call 23 of the block, interpolation
probes that land far from the model's current trust region are expected
during active model updates), but the running best fell steadily and the
block's final 9 values (3417.9, 2350.4, 2965.0, 3154.9, 3668.9, 2125.1,
2360.2, 2101.4, with the very last call producing the block's lowest
value) show the trust region has clearly moved into a much better, still
actively improving region, not a plateau. The solver thread has not
exhausted itself (no `None` sentinel / `OptimizerExhausted` raised) —
it has more proposals queued.
**Decision:** Continue the same method and the same BOBYQA instance
(`bobyqa1`), preserving its solver thread, interpolation model, and
trust-region state exactly as left off; no restart.
**Hyperparameters:** Unchanged — same configuration as the previous
BOBYQA step (see entry above): no exposed hyperparameters need
reconsideration mid-run (`rhobeg`/`rhoend`/`maxfun`/`scaling_within_bounds`
all still as documented there; BOBYQA has no population size, step size,
or acquisition setting to retune), and the instance is not restarted so
there is nothing new to initialize.
**Reason:** The evidence (still falling running best, no plateau, no
exhausted-solver signal, consistent with expected noisy-looking but
improving trust-region behavior for a model-based local solver) gives no
reason to switch or adjust; BOBYQA is actively exploiting the
anisotropic-bowl structure identified earlier more effectively than
expected at the last checkpoint.
**Next step limit:** 30 additional objective calls, at most 54; pause at
cumulative evaluation 76.
**Why this length:** Same rationale as the previous BOBYQA block: long
enough to let several more trust-region updates play out so a genuine
trend (continued improvement vs. plateau vs. exhaustion) is visible,
while preserving a meaningful remaining reserve (24 evaluations) for a
further continuation, a restart with different interpolation geometry,
or a switch, depending on what this block shows.
**Review:** Whether the running best keeps falling by a comparable or
larger margin, whether it plateaus (several consecutive calls with no
improvement, which would suggest the trust region has converged near a
local optimum — not necessarily the global one, given only one basin has
been explored so far), or whether the solver thread exhausts itself
(returns its sentinel) before using the full 30 calls, which would itself
be the trigger to stop this method early regardless of the nominal step
limit.

## Decision after evaluation 76
**Budget:** 76 used / 100 total; 24 remaining.
**Evidence:** The running best fell from 2101.4 to 555.7 (another >3.7x
reduction) over this block. Unlike the previous block, this one is
almost monotonically decreasing call-by-call (2156→2052→2137→2023→1874→
1648→1623→1637→1589→1592→1515→1505→1458→1482→1316→1285→1133→1134→1053→
1006→956→909→864→878→761→701→619→600→556, with one uptick to 687 on the
very last call) — a smooth, sustained descent with no plateau and no
solver-exhaustion signal, indicating BOBYQA's trust region is still
actively and efficiently shrinking toward a local optimum of the basin it
is in.
**Decision:** Continue the same BOBYQA instance (`bobyqa1`) unchanged,
using the entire remaining budget (24 evaluations) as the final step of
this campaign.
**Hyperparameters:** Unchanged, same as previous two BOBYQA entries; not
applicable to revise further (no new configuration decision; instance and
solver-thread state carried over as-is).
**Reason:** With only 24 evaluations left and a 10-dimensional problem,
there is not enough remaining budget to productively fund a second,
independent global-search method (e.g. restarting Sobol'/CMA-ES/DE from
scratch) and still leave room for a meaningful local refinement after it
— any switch now would likely strand both methods half-finished. The
current method is demonstrably still converging fast and has not
plateaued or exhausted its proposals, which is the condition under which
the protocol calls for continuing rather than switching. Using the full
remaining budget on it is therefore the best expected use of what is
left, and this is also expected to be the final step of the campaign
(remaining budget reaches 0 at its end).
**Next step limit:** 24 additional objective calls, at most 24; pause at
cumulative evaluation 100 (budget exhaustion).
**Why this length:** This consumes the entire remaining budget rather
than reserving a further checkpoint, because reserving any fraction now
would only shorten this already-converging local refinement without a
credible alternative use for the held-back calls (no new method could be
meaningfully started and finished in fewer than ~24 evaluations in 10
dimensions at this point). The checkpoint after this block is therefore
expected to be the final one, triggered by budget exhaustion itself
rather than by a fixed, pre-chosen length.
**Review:** Final running best and whether BOBYQA's trust region
continued to improve, plateaued, or had its solver thread return
(exhausted) before the full 24 calls were used; this is the stopping
checkpoint, so the main thing to record is the campaign's final outcome,
not a further method choice.
