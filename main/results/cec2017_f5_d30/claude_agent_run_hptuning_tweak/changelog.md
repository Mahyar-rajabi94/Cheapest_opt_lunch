## Decision after evaluation 16
**Budget:** 16 used / 100 total; 84 remaining.
**Evidence:** Sobol scan (n=16, seed=0) across the full [-100,100]^30 box
gave y in [1353.76, 1839.00], median 1573.49, std 153.35 (a fairly narrow
relative spread, ~11% of the median, with no near-zero or wildly outlying
values and no evaluation errors/non-finite values). Sorted y-values show a
fairly smooth, roughly continuous spread with no single dramatic gap,
which with only 16 points in 30-D is not enough to determine
unimodality/multimodality or local smoothness — it only rules out gross
pathologies (errors, huge outliers, degenerate flat regions). The best
point (y=1353.76) sits at distance ~315 from the origin (box diagonal
~548), i.e. well inside the box but off-center, with no other strong
structure visible from so few points.
**Decision:** Switch to CMA-ES for this step, since the Sobol scan's main
useful output is "here is one comparatively good region of a 30-D box
with uncertain structure," and CMA-ES is a robust default for exactly
this situation — it needs no gradient or smoothness assumption, adapts
its search distribution online, and tolerates mild multimodality far
better than a pure local trust-region model would if the landscape turns
out to be rugged. The wrapper (`src/optimizers/cma_es.py`) always starts
`x0` at the box midpoint, which would discard the one piece of
information step 1 produced (a better-than-random region), so I use a
small campaign-local adapter that calls `cma.CMAEvolutionStrategy`
directly (same library, same convention) with `x0` set to the best Sobol
point instead of the midpoint. All other settings follow the wrapper's
own stated "standard library defaults only" convention. Sobol's own
(`opt`) and the harness's observations/evaluation counter are preserved;
nothing told to Sobol is discarded, it is simply not the active method
for new evaluations.
**Hyperparameters:**
- `x0`: best point found by the Sobol scan (`y=1353.76`) — changed from
  the wrapper's default (box midpoint), justified above.
- `sigma0`: 50.0 = one quarter of the box's per-dimension width (200),
  i.e. the same convention the CMAES wrapper itself uses for an
  uninformed box-constrained start (`src/optimizers/cma_es.py`
  docstring). Kept at this (not shrunk) despite centering on a "good"
  point, because that point is only the best of 16 widely scattered
  samples in 30-D — weak evidence about local scale — so the default,
  broader step size is kept rather than assuming we are already close to
  a basin worth exploiting tightly.
- `popsize`: left at pycma's own default, `4 + floor(3*ln(dim))` =
  `4 + floor(3*ln(30))` = 14. Not tuned, per the wrapper's own stated
  convention and because no evidence yet favors a non-default population
  size.
- `bounds`: the task's box `[-100,100]^30`, enforced via pycma's own
  `bounds` option (same as the wrapper).
- `seed=0`: fixed for reproducibility, consistent with step 1's seeding.
- pycma version: whatever is installed in this project's `.venv` (not
  pinned/changed by me); all other internal CMA-ES hyperparameters
  (learning rates, damping, recombination weights, etc.) are left at that
  version's library defaults — no tuning, matching the wrapper's stated
  policy.
**Reason:** With 84 evaluations remaining and dim=30, CMA-ES's default
population size of 14 permits 6 full generations if run to budget
exhaustion. Starting from the best known region rather than the box
midpoint uses the one informative signal available without overcommitting
to it (by keeping the default, non-shrunk step size). This is a better
use of a very tight budget than continuing pure space-filling (which
cannot meaningfully cover 30-D with any realistic remaining sample count)
or jumping to a purely local quadratic-model solver before there is any
evidence the landscape is smooth enough to reward that assumption.
**Next step limit:** 28 additional objective calls (2 full CMA-ES
generations of 14), at most 84 remaining; pause at cumulative evaluation
16 + 28 = 44.
**Why this length:** Two generations is enough for CMA-ES's step-size and
covariance adaptation to show an initial trend (whether the best-so-far
value is improving, and whether the mean/step size are moving sensibly)
without committing the majority of the remaining budget before the first
reassessment. One generation alone would be too noisy to judge adaptation
behavior; committing all 6 possible generations now would remove the
chance to adjust (e.g. restart, change sigma, or switch methods) if early
evidence suggests CMA-ES is not well suited here.
**Review:** At cumulative evaluation 44, inspect: whether y_best_so_far
has improved versus 1353.76, the trend of each generation's best/mean y,
and pycma's reported step size (`sigma`) to see whether it is
shrinking (converging) or still large (still searching broadly). Pause
sooner only if `evaluate()` errors, returns non-finite values, or the
harness reports budget exhaustion — none expected. No commitment to a
particular method after this checkpoint.

## Decision after evaluation 44
**Budget:** 44 used / 100 total; 56 remaining.
**Evidence:** Two CMA-ES generations (popsize 14) run from `x0` = best
Sobol point: generation 1 best y=1156.51 (overall best, improving from the
Sobol-stage best of 1353.76), generation 2 best y=1132.74 (overall best
again). Step size `sigma` declined only modestly, from 50.0 to 46.60 to
45.13 — i.e. CMA-ES is still searching at close to its initial scale, not
collapsing toward a point, while nonetheless producing a consistent
generation-over-generation improvement in the best value (1353.76 ->
1156.51 -> 1132.74, roughly 16% total reduction so far). This is evidence
the method is working productively in this region (steady improvement,
no stagnation) without yet showing convergence (sigma still large), so
there is no reason to restart, change hyperparameters, or switch methods.
**Decision:** Continue CMA-ES unchanged — same `es` object (its internal
mean, covariance, step size, and evolution paths are preserved across
this checkpoint exactly as pycma maintains them internally; only the
harness's active-optimizer label and my own decision point pause between
generations). No restart: pycma's adaptive state is exactly the kind of
"ordinary internal adaptation" the protocol says does not need a fresh
agent decision for every internal update, and restarting now would throw
away two generations' worth of learned covariance/step-size information
for no evidenced benefit.
**Hyperparameters:** Unchanged from the previous entry (bounds, seed=0,
popsize=14 from pycma's own default formula, all other internal pycma
settings at library defaults). Not applicable: no new hyperparameters are
being introduced.
**Reason:** Steady, non-stagnant improvement with a step size still at
~90% of its initial value is exactly the situation in which continuing
the same method/state is preferable to switching: there is no evidence of
a plateau or misfit method to react to, and CMA-ES's own state already
encodes everything learned so far about this region's local geometry.
**Next step limit:** 42 additional objective calls (3 full generations of
14), at most 56 remaining; pause at cumulative evaluation 44 + 42 = 86.
This leaves 14 (one more generation's worth) of budget unallocated after
this step for a final checkpoint, rather than spending the entire
remaining budget now.
**Why this length:** Three generations gives enough further evidence to
see whether the improving trend continues, flattens, or reverses, while
deliberately preserving at least one more full generation of budget (14
evals) after this block so a genuine decision (continue the last
generation, restart with injected diversity if stagnated, or stop early)
remains possible rather than mechanically spending to budget exhaustion
in this step.
**Review:** At cumulative evaluation 86, inspect the per-generation best
y trend and `sigma` trend over these 3 generations (continued
improvement vs. plateau; sigma still shrinking gradually vs. collapsed or
still essentially flat). Pause sooner only on an evaluation
error/non-finite value or budget exhaustion (not expected, since 42 is
exactly divisible by popsize 14). No commitment to a method for the final
step yet.

## Decision after evaluation 86
**Budget:** 86 used / 100 total; 14 remaining.
**Evidence:** Three further CMA-ES generations: gen3 best=1107.99 (new
overall best), gen4 best=1008.45 (new overall best, a large ~9% drop from
1107.99), gen5 best=1018.23 (slightly worse than gen4's best, so the
overall best stayed at 1008.45 — ordinary non-monotonic CMA-ES behavior,
not evidence of a plateau by itself). Sigma stayed essentially flat across
these generations (44.65 -> 44.79 -> 45.81), i.e. CMA-ES has not entered a
local convergence/shrinking phase; it is still exploring at roughly its
original scale after 5 generations. Overall trajectory across all 5
generations so far: 1353.76 (Sobol) -> 1156.51 -> 1132.74 -> 1107.99 ->
1008.45 -> (1018.23, non-improving) — net ~25% reduction from the Sobol
best, with the two largest single-generation gains (gen2->gen3 style
drops) occurring in generations 2 and 4, so there is no multi-generation
stagnation to react to (only one non-improving generation just occurred).
**Decision:** Continue CMA-ES unchanged for one final generation, using
exactly the 14 evaluations remaining. Same `es` object/state as before
(mean, covariance, step size, evolution paths carried over; only the
harness's optimizer label is reasserted). This is both because the
evidence (flat, non-collapsed sigma plus a recent large-gain generation)
gives no reason to prefer a restart or a different method over the final
stretch, and because no other method could be meaningfully initialized,
run, and compared within only 14 evaluations as reliably as simply
finishing the population-based step already in progress — e.g. switching
to a new local solver now would spend most or all of the remaining budget
just on its own startup (BOBYQA needs ~dim+2=32 points to build its first
model; even a single CMA-ES generation here is a well-matched, already
"warmed up" unit of work for exactly 14 evaluations, since popsize=14).
**Hyperparameters:** Unchanged (bounds, seed=0, popsize=14, all other
pycma internals at library defaults, as in the previous two entries). Not
applicable: no new hyperparameters introduced.
**Reason:** With only 14 evaluations left — exactly one more CMA-ES
generation and too few to responsibly initialize an alternative method
from scratch — continuing the current, still-active (non-converged,
still-improving-on-net) method is the best use of the remaining budget.
**Next step limit:** 14 additional objective calls (1 final generation of
14), at most 14 remaining; pause at cumulative evaluation 86 + 14 = 100
(full budget exhaustion).
**Why this length:** This is the entire remaining budget; it is used in a
single final generation (rather than, say, stopping at some smaller
number) because partial generations are not meaningful for a
population-based method expecting exactly `popsize` points per
generation, and because there is no remaining budget left over after this
step to warrant preserving for a future reconsideration point.
**Review:** Not applicable beyond this step — this is the final step; it
ends at budget exhaustion. The only check is the resulting best y and
that the harness's automatic `save()` on budget exhaustion completes
successfully, writing `trace.csv`/`meta.json`.

## Decision after evaluation 100 (final)
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** The final CMA-ES generation (generation 6 overall) produced
a within-generation best of 1063.07, which did not improve on the
existing overall best of 1008.45 from generation 4. The overall best
observed across the full campaign therefore remains **y = 1008.4480853582129**,
found during the CMA-ES stage seeded at the best Sobol point. The harness
reports `evaluations_used = 100`, `evaluations_remaining = 0`, confirming
the full budget was consumed and none was exceeded.
**Decision:** Stop. The evaluation budget of 100 is fully exhausted (the
harness would raise `BudgetExhausted` on any further `evaluate()` call),
so no further step is possible regardless of what the evidence might
otherwise suggest.
**Hyperparameters:** Not applicable — no further evaluations will be
made.
**Reason:** Hard budget constraint reached; stopping is the only
available action.
**Next step limit:** K = 0 (stopping).
**Why this length:** Not applicable — no further evaluations remain in
the budget.
**Review:** Not applicable — campaign complete.

## Final summary
- Total evaluations used: 100 / 100 (budget fully exhausted; no early
  stop).
- Campaign trajectory (overall best y after each stage): Sobol scan
  (n=16) -> 1353.76; CMA-ES gen 1 -> 1156.51; gen 2 -> 1132.74; gen 3 ->
  1107.99; gen 4 -> **1008.4480853582129** (final best); gen 5 -> no
  improvement (1018.23); gen 6 (final) -> no improvement (1063.07).
- Final best objective value: **1008.4480853582129** (minimize), found by
  the `cma_es_seeded` method (CMA-ES from pycma, `x0` seeded at the best
  point found by an initial Sobol scan, `sigma0` = one quarter of the box
  width, `popsize=14` from pycma's own default formula, all other
  internal pycma hyperparameters at library defaults).
- `harness.save()` was called automatically by the harness itself upon
  reaching `evaluations_used == budget == 100`; it was also called
  explicitly at the end of the final script for certainty. `trace.csv`
  and `meta.json` were written to this results directory.
