## Decision after evaluation 20
**Budget:** 20 used / 100 total; 80 remaining.
**Evidence:** Sobol' initial design (20 points, seed 0, x0..x7 rounded to
nearest integer before evaluation). Observed y ranged from 105.13 to
3275.92 (mean 1311.9, std 884.4) — a wide spread, confirming the
landscape is far from flat and that location matters a lot. Linear
correlation of each raw coordinate with y (rough signal only, 20 points):
x4 +0.71 (strongest), x7 +0.41, x8 +0.32, x6 +0.24, x5 +0.20, x2 +0.17,
x0 +0.16, x3 -0.14, x9 -0.09, x1 +0.003 (negligible). This suggests lower
x4, lower x7/x6, and more negative x8 tend toward lower y, with x1 and x9
showing little linear effect over this sample (though 20 points cannot
rule out non-monotonic/interaction effects). The current incumbent is
eval #10: x = [0,0,0,3,2,1,8,3,-0.952,-0.512], y = 105.126 — notably not
an extreme point on every correlated axis, so interactions plausibly
matter and a coordinate-wise local search around it is a sensible next
step rather than trusting single-axis correlations to extrapolate far.
**Decision:** Switch from the global space-filling design to integer
coordinate descent on x0..x7, starting from the current incumbent
(eval #10). For each integer coordinate, evaluate its two unit
neighbors (x_i-1 and x_i+1, clipped to bounds, skipped if out of bounds),
and move to the best of {current, neighbor-} / {neighbor+} before
proceeding to the next coordinate (classic greedy/best-improvement
coordinate descent, step size 1). x8, x9 are held fixed at the
incumbent's values during this pass — this step is deliberately scoped to
learning the shape of the integer lattice around the incumbent; continuous
refinement is deferred to a later step once an integer local optimum (or
at least one full pass of evidence) is available. This is a custom
campaign-local method (no existing wrapper fits in-place coordinate
descent on a mixed lattice), built directly on `harness.evaluate`.
**Hyperparameters:** Coordinate order: x0, x1, ..., x7 (simple ascending
index order; no prior reason from the correlation data to reorder, since
interactions are already suspected). Step size: 1 (finest resolution, to
avoid skipping over a narrow optimum on axes with small ranges like
x0,x1 in {0,1}). Acceptance: best-improvement among {current, x_i-1,
x_i+1} (not first-improvement), so each coordinate's local neighborhood is
fully resolved before moving on. Not applicable: no learning-rate/step-decay
parameters exist for this hand-written method.
**Reason:** The Sobol step confirmed meaningful structure but gave no
basis yet for trusting correlations beyond the sampled region; a bounded,
cheap (<=2 evals/coordinate) local search directly tests the neighborhood
of the best point found so far on the axes we know we must treat as
piecewise-constant/discrete, which neither CMA-ES/DE (population-heavy,
designed for continuous landscapes) nor a plain continuous local solver
could do correctly on these 8 axes.
**Next step limit:** 16 additional objective calls (2 per coordinate x
8 coordinates, the maximum possible; fewer will be used if a boundary
coordinate only has one valid neighbor), at most 80 remaining; pause at
cumulative evaluation 36.
**Why this length:** 16 is exactly one full cyclic pass over all 8
integer coordinates at unit step size — the smallest unit of evidence
that tells us whether the incumbent is already a coordinate-wise local
optimum or whether (and where) it can still be improved. Stopping
mid-pass would leave some coordinates untested and bias the next
decision.
**Review:** Whether the pass found any improving moves (and how many/how
large), whether the new incumbent's coordinates still look close to the
Sobol-implied low-y region, and how much of the 80 remaining budget a
further pass (or continuous refinement on x8/x9) would consume. No
commitment yet to how many further coordinate-descent passes will follow.

## Decision after evaluation 32
**Budget:** 32 used / 100 total; 68 remaining.
**Evidence:** First coordinate-descent pass used only 12 of the
budgeted 16 evaluations (x0, x1, x3 each had only one valid unit
neighbor at the time, being at/near a bound), and improved the incumbent
substantially and repeatedly: y went 105.126 -> 83.033 (x2: 0->1) ->
34.753 (x4: 2->1) -> 26.555 (x5: 1->2) -> 16.551 (x6: 8->7) -> 14.889
(x7: 3->4). Every coordinate touched except x0, x1, x3 produced an
improving move, and several of the largest-correlation coordinates from
the Sobol step (x4, x6, x7) indeed moved toward lower y as expected. Since
each coordinate was only evaluated relative to the *incumbent at the time
it was visited*, earlier coordinates (x0..x3) were evaluated against a
now-stale incumbent and have not been re-tested against the updated
context (e.g. x2, x4, x5, x6, x7 all changed after x0/x1/x3 were visited).
This is a standard limitation of a single coordinate-descent cyclic pass:
it is not yet confirmed to be at even a coordinate-wise local optimum.
**Decision:** Continue the same method (integer coordinate descent,
best-improvement, unit step) for a second full cyclic pass over x0..x7,
re-starting from the new incumbent found in pass 1. x8, x9 remain fixed
at their Sobol-inherited values; continuous refinement is still deferred
until the integer block looks closer to a local optimum, since further
large integer improvements are still occurring and would otherwise
invalidate any continuous tuning done around a stale integer point.
**Hyperparameters:** Unchanged from the previous entry (coordinate order
x0..x7 ascending, step size 1, best-improvement over
{current, x_i-1, x_i+1}).
**Reason:** Pass 1 produced improvements on 5 of 8 coordinates and ended
with the last-visited coordinate (x7) still improving, which is evidence
the local neighborhood has not been exhausted — continuing the same cheap,
well-performing method is justified over switching, and a second full pass
is the natural unit of evidence for whether we are approaching a coordinate-wise
local optimum (diminishing/no improvements) or still moving substantially.
**Next step limit:** 16 additional objective calls (one more full cyclic
pass, same upper bound logic as before; likely fewer will be used given
boundary effects), at most 68 remaining; pause at cumulative evaluation 48.
**Why this length:** Matches the size of one full cyclic pass, the same
granularity of evidence used in the previous decision, so pass-to-pass
improvement can be compared on a like-for-like basis.
**Review:** Total/size of improvements in pass 2 versus pass 1 (expect
diminishing returns if near a local optimum); whether any coordinate still
shows a large improving move (signal to keep iterating) or all 8 report no
improvement (signal to stop integer refinement and move to continuous
refinement on x8/x9, or to consider a restart/different starting region
if the found local optimum seems weak relative to remaining budget).

## Decision after evaluation 45
**Budget:** 45 used / 100 total; 55 remaining.
**Evidence:** Pass 2 used 13 of 16 budgeted evaluations (x0, x1 again
had only one valid neighbor and did not improve; x3 did not improve) and
again produced large, repeated improvements: y went 14.889 -> 11.894
(x2: 1->2) -> 4.502 (x4: 1->0) -> 1.832 (x6: 7->6). x7 was re-tested
(3 and 5) and neither improved on the current incumbent's x7=4, and x5
likewise showed no improving move this pass. So out of 8 coordinates,
4 improved (x2, x4, x6, and transitively the chain of re-centering), and
improvements are getting smaller in absolute terms (105->83->35->27->17->15
in pass 1, i.e. ~90 total improvement, vs 15->12->4.5->1.8, i.e. ~13 total
improvement in pass 2) — consistent with approaching, but not yet at, a
coordinate-wise local optimum on the integer block. x0, x1, x3, x7 showed
no improvement in pass 2 specifically; x5 also showed no improvement this
pass (it had improved in pass 1).
**Decision:** Continue the same method (integer coordinate descent,
best-improvement, unit step) for a third pass over x0..x7. The diminishing
but still-nonzero improvement (pass 2 still found 3 improving moves and
roughly halved the incumbent's y) means the integer neighborhood is not
yet exhausted. x8, x9 remain fixed; continuous refinement is still
deferred by the same reasoning as before — the integer point is still
moving enough that tuning x8/x9 now risks being thrown away.
**Hyperparameters:** Unchanged (coordinate order x0..x7, step size 1,
best-improvement over {current, x_i-1, x_i+1}).
**Reason:** Returns are diminishing but real (pass 2 still cut y by
~71% relative to its start); stopping the integer phase now would leave
an easy, nearly-free improvement on the table relative to its evaluation
cost (13 evals for a >10x reduction in y). A third pass is cheap evidence
for whether we've reached a fixed point.
**Next step limit:** 16 additional objective calls (one more full cyclic
pass), at most 55 remaining; pause at cumulative evaluation 61.
**Why this length:** Same one-full-pass unit of evidence as the last two
decisions, for a like-for-like comparison of improvement size across
passes.
**Review:** Whether pass 3 finds zero, few, or many improving moves. Zero
or near-zero improving moves across all 8 coordinates signals a
coordinate-wise local optimum has been reached and that remaining budget
(55 after this step, or more precisely 55-K) should shift toward
continuous refinement of x8/x9 at the fixed integer point, and/or a
broader integer restart if remaining budget comfortably allows testing a
different basin.

## Decision after evaluation 57
**Budget:** 57 used / 100 total; 43 remaining.
**Evidence:** Pass 3 (12 evaluations, all 8 integer coordinates
revisited) found **zero** improving moves: every unit-neighbor tried was
worse than or equal to the incumbent y=1.8320 at
x=[0,0,2,3,0,2,6,4,-0.952,-0.512] (e.g. x2:1->4.827 worse, x3:3.. already
boundary-matching 4.827 for a different neighbor, x6:5->1.896 and 7->4.502
both worse than 1.832, x7:3->2.400 and 5->2.594 both worse). This is a
clean signal that the incumbent is a coordinate-wise local optimum on the
integer lattice at unit-step resolution. x8, x9 have not been touched
since the Sobol initial design (still at their Sobol-drawn values
-0.952, -0.512) despite x8 showing a non-trivial correlation with y in
the initial sample, so they are very likely not locally optimal and are
the clearest remaining source of cheap improvement.
**Decision:** Switch to continuous local refinement of x8, x9 only,
holding the integer coordinates fixed at the current incumbent
[0,0,2,3,0,2,6,4]. Use scipy's Nelder-Mead simplex
(`scipy.optimize.minimize`, the same underlying method and version as
`src/optimizers/nelder_mead.py`) called directly rather than through that
wrapper, because the wrapper always initializes `x0` at the box midpoint
with no way to override it through its public interface — here I need
`x0 = [-0.952, -0.512]` (the current incumbent's continuous coordinates),
not the midpoint, to refine from the best point found so far instead of
restarting continuous search from scratch. This is "using the underlying
library's supported API" as permitted when a wrapper fixes a setting I
need to control.
**Hyperparameters:** `method="Nelder-Mead"`, `bounds=[[-5,5],[-5,5]]`,
`x0=[-0.95184465, -0.51232345]` (current incumbent, not the midpoint),
`options={"adaptive": True, "maxfev": 20}` — `adaptive=True` matches the
wrapper's own documented standard recommendation (Gao & Han 2012 scaling)
for a non-trivial dimension; `maxfev=20` is the hard cap on function
evaluations for this step (see step limit below), substituting for the
wrapper's effectively-unlimited `maxfev=10**9` since here the step budget
itself is the intended stopping condition, not solver convergence.
scipy's other Nelder-Mead tolerances (`xatol`, `fatol`) are left at their
library defaults (1e-4 each) — inherited default, not tuned.
**Reason:** The integer block is at a local optimum and further
unit-step passes are known to yield zero improvement right now, so
continuing that method would waste evaluations. x8/x9 have had no
local refinement at all yet and showed a real (if modest) correlation
with y, so switching to a continuous local method for exactly 2
dimensions is the next-cheapest source of likely improvement; Nelder-Mead
is an appropriate choice for a small (2-d), likely-smooth local refinement
once a good starting point is known (per the wrapper's own documented role
as "a local refiner, not a global search method").
**Next step limit:** 20 additional objective calls (`maxfev=20`, a hard
cap enforced both by scipy's option and manually since the harness budget
must not be exceeded), at most 43 remaining; pause at cumulative
evaluation 77.
**Why this length:** 20 evaluations is a generous-but-bounded budget for
2-d Nelder-Mead to converge from a reasonable starting point (2-d simplex
local search typically converges in well under this many evaluations),
while preserving at least 23 evaluations afterward for a further decision
(e.g. a second integer pass to re-check whether the now-updated x8/x9
values change which integer unit-moves improve, given this is not
provably a jointly separable problem).
**Review:** Whether Nelder-Mead converges (stops proposing points) before
using all 20 evaluations, the resulting y versus 1.832, and whether it is
worth re-running the integer coordinate descent once more at the new
x8/x9 values (since the local optimum found in passes 1-3 was conditioned
on the old x8/x9 and joint interactions have not been ruled out).

## Decision after evaluation 77
**Budget:** 77 used / 100 total; 23 remaining.
**Evidence:** The capped Nelder-Mead run used its full `maxfev=20` budget
(scipy's own stopping message: "Maximum number of function evaluations
has been exceeded", i.e. it had not yet converged to its own tolerance)
and only improved y from 1.8320 to 1.7222 (x8,x9 moved from
[-0.952,-0.512] to [-0.634,-0.456]). This is a much smaller return per
evaluation than every integer coordinate-descent pass so far (pass 1: ~90
improvement/12 evals; pass 2: ~13/13; pass 3: 0/12; this NM step:
~0.11/20) — evidence that the objective is only weakly sensitive to
x8/x9 in this neighborhood (consistent with their modest Sobol
correlations, 0.32 and -0.09), not that continuous refinement is broken.
The integer local optimum found in passes 1-3 was conditioned on the old
x8/x9; since x8/x9 have now moved, it is still open whether any integer
unit-move newly improves at this slightly different continuous point.
**Decision:** Run one more integer coordinate-descent pass (pass 4),
same method as passes 1-3, over x0..x7 at the current incumbent
(x8,x9 now fixed at the NM result). This checks for any interaction-driven
improving move unlocked by the x8/x9 change, at low cost, before
committing the remaining budget. Continuous refinement is not continued
further immediately because its demonstrated marginal return here is low
relative to its per-evaluation cost compared to the integer passes.
**Hyperparameters:** Unchanged coordinate-descent settings (order x0..x7,
step size 1, best-improvement over {current, x_i-1, x_i+1}).
**Reason:** Given only 23 evaluations remain, re-checking the integer
block (historically far higher-yield per evaluation) for newly-unlocked
moves is a better use of a bounded block than continuing to press
Nelder-Mead past its demonstrated low marginal return in this
neighborhood; a short, bounded pass preserves the option to spend
whatever remains afterward on whichever method the evidence then favors.
**Next step limit:** 16 additional objective calls (one more full cyclic
pass, bounded as before; likely fewer used due to boundary effects, e.g.
x0,x1,x3 have had only one free neighbor throughout this run), at most 23
remaining; pause at cumulative evaluation 93.
**Why this length:** Same one-full-pass unit as passes 1-3, enabling
direct comparison of this pass's improvement size against the prior
(1.832 -> ... from pass 3's confirmed zero), while deliberately leaving a
small reserve (at least 7 evaluations) after this pass for a final
decision rather than exhausting the budget on an unexamined assumption
that this pass will find large gains.
**Review:** Whether pass 4 finds any improving moves at all now that
x8/x9 differ from passes 1-3 (evidence of a real interaction effect) or
again finds zero (confirms the integer local optimum is robust to the
small x8/x9 change). Either result determines how the final ~7+
evaluations are spent (further continuous polish vs. stopping).

## Decision after evaluation 89
**Budget:** 89 used / 100 total; 11 remaining.
**Evidence:** Pass 4 (12 evaluations) again found **zero** improving
moves across all 8 integer coordinates at the updated x8/x9
(e.g. x6: 5->1.787 and 7->4.392, both worse than the incumbent's 1.722;
x7: 3->2.466 and 5->2.027, both worse). This confirms the integer local
optimum x=[0,0,2,3,0,2,6,4] is robust to the small x8/x9 shift from the
previous step — no interaction effect was strong enough to unlock a new
move. The incumbent remains y=1.7222 at
[0,0,2,3,0,2,6,4,-0.634,-0.456]. Only 11 evaluations remain.
**Decision:** Spend the remaining 11 evaluations finishing the continuous
refinement of x8/x9 (resume Nelder-Mead from where it was cut off at
`maxfev=20` in the earlier step, which had not converged by its own
tolerance). The integer block has now shown zero improving unit-moves
across two consecutive passes at two different x8/x9 values, which is
strong evidence it is a stable local optimum not worth further budget;
x8/x9 refinement had nonzero (if modest) improvement and was stopped only
by its evaluation cap, not by convergence, so it is the better remaining
use of a small final budget than, e.g., a fresh random restart (which
would need far more than 11 evaluations to explore a 10-d mixed space
and could not even complete one Sobol/coordinate-descent cycle).
**Hyperparameters:** Continue `scipy.optimize.minimize` Nelder-Mead,
restarting the simplex from the current best continuous point
`x0=[-0.63431522,-0.45558763]` (Nelder-Mead's internal simplex state from
the previous call was not preserved by the thread-free direct-call
approach, so this is a fresh simplex at the improved point rather than a
resumed one — a known, accepted limitation given scipy's synchronous
`minimize` does not expose incremental simplex state through its public
API). `adaptive=True` (unchanged), `bounds=[[-5,5],[-5,5]]` (unchanged),
`maxfev=11` (the entire remaining budget, since this is the final step).
**Reason:** This is the last block of budget; committing all of it to the
one method with demonstrated (if slow) nonzero marginal improvement, and
no reason to reserve further evaluations for a future decision, is
appropriate for a terminal step.
**Next step limit:** 11 additional objective calls (all remaining
budget); pause at cumulative evaluation 100 (budget exhaustion).
**Why this length:** This is the entire remaining budget; there is no
future step to preserve evaluations for.
**Review:** Final best y and point, for the closing report. Not
applicable beyond this — campaign ends at budget exhaustion.

## Decision after evaluation 100 (final)
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** The final Nelder-Mead block used its full 11-evaluation
cap (again ending on "Maximum number of function evaluations has been
exceeded" rather than its own convergence tolerance) and improved y
only marginally further, from 1.7222 to 1.7205
(x8,x9: [-0.634,-0.456] -> [-0.630,-0.401]). This is consistent with the
earlier evidence that the objective is only weakly, smoothly sensitive to
x8/x9 near this integer cell — small, slow, diminishing-but-real gains,
not a sign of a bug or a missed large improvement. Final best observed:
**y = 1.7205226924599348** at
**x = [0, 0, 2, 3, 0, 2, 6, 4, -0.63035075, -0.4014866]**
(eval index 100, `optimizer` tag `nelder_mead_cont_x8x9_final` in
trace.csv).
**Decision:** Stop. The evaluation budget of 100 is exhausted; the
harness raises `BudgetExhausted` on any further call, and the protocol's
hard constraint forbids working around that.
**Hyperparameters:** Not applicable — no further evaluations.
**Reason:** Budget exhaustion is a hard stop, not a judgment call; there
is nothing left to decide beyond recording the outcome.
**Next step limit:** 0 (stopping).
**Why this length:** Not applicable.
**Review:** Not applicable — campaign complete.

## Summary

Campaign trajectory (best-so-far y at each stage):
- Sobol' initial design (20 evals): 105.126
- Integer coordinate descent pass 1 (12 evals, cum. 32): 14.889
- Integer coordinate descent pass 2 (13 evals, cum. 45): 1.832
- Integer coordinate descent pass 3 (12 evals, cum. 57): 1.832 (no improvement — local optimum reached)
- Continuous Nelder-Mead on x8,x9 (20 evals, cum. 77): 1.722
- Integer coordinate descent pass 4 (12 evals, cum. 89): 1.722 (no improvement — local optimum confirmed robust)
- Continuous Nelder-Mead on x8,x9, final block (11 evals, cum. 100): 1.7205

Final result: **y = 1.7205226924599348**, x = [0, 0, 2, 3, 0, 2, 6, 4,
-0.63035075, -0.4014866]. All 100 evaluations used; stopped only because
the budget was exhausted.
