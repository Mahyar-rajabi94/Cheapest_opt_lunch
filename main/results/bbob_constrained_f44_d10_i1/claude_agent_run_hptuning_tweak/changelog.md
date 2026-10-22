## Decision after evaluation 16
**Budget:** 16 used / 100 total; 84 remaining.
**Evidence:** The 16-point Sobol' batch (seed=0) produced merit values ranging
from -563.20 to 171846.43, median 35304.98. Fifteen of the sixteen values are
large positive numbers (11507 to 171846), consistent with heavy constraint
penalty across most of the box. One point stands out sharply: x =
[-3.475, -1.892, -1.102, -4.460, 1.166, -1.246, 0.574, -2.647, -0.558, 3.862]
with merit = -563.20, roughly 4 orders of magnitude better than the next-best
sampled point and the only negative value observed. Since the objective is
deterministic, this is a reliable (not noisy) signal that this region is
near- or fully feasible with a genuinely small/negative underlying objective,
while most of the box is heavily infeasible. This is exactly one step's
worth of evidence (no earlier information to compare against), so I am not
over-reading a short plateau — this is the full first look at the landscape.
**Decision:** Switch from pure exploration (Sobol) to a local/adaptive
refinement method centered on the best point found: CMA-ES, initialized at
that point rather than the box midpoint. The standard `CMAES` wrapper
(src/optimizers/cma_es.py) hardcodes initialization at the box midpoint with
sigma0 = quarter mean box width, which is not what I want here (I have
evidence the midpoint region is poor and want to exploit the located
region) — a relevant hyperparameter the wrapper does not expose. Per the
protocol's allowance to use the underlying library directly when a wrapper
fixes a setting I want to control, I will call `cma.CMAEvolutionStrategy`
directly with a campaign-local adapter that otherwise mirrors the wrapper's
ask/tell behavior and bounds handling. No prior CMA-ES state exists yet, so
this is a fresh initialization, not a restart.
**Hyperparameters:** `x0` = the best Sobol point found above (not the
wrapper's default midpoint). `sigma0 = 1.5`: smaller than the wrapper's
default box-derived value (2.5, a quarter of the box width 10) because I now
have evidence pointing to a specific good region rather than no information
at all, so a tighter initial search radius is appropriate to exploit it
while still allowing CMA-ES to adapt outward if needed. `bounds` = the same
`harness.bounds` ([-5,5] per coordinate), enforced via pycma's own `bounds`
option, same as the wrapper. Population size and all other internal
hyperparameters left at pycma's own defaults (popsize = 4 + floor(3*ln(10))
= 10 for dim=10), matching the wrapper's convention of leaving these
untuned. `seed=0` for reproducibility. Implementation: `cma` (pycma)
package, as already used by the project's CMAES wrapper.
**Reason:** The sharp, deterministic contrast between one near-feasible
point and fifteen heavily-penalized points suggests a localized feasible
region rather than a smooth global landscape, so further undirected
sampling (more Sobol) is unlikely to be as efficient as concentrating the
remaining budget around the one confirmed good region. CMA-ES's
covariance adaptation can refine both the location and local shape of the
optimum there, and is more robust to the non-smooth kinks a penalty-based
merit function can introduce than a quadratic-model method like BOBYQA
would be. Given the very tight remaining budget (84 evaluations), committing
to adaptive local refinement now is preferable to spending more on pure
exploration.
**Next step limit:** 30 additional objective calls (3 full CMA-ES
generations at popsize 10), at most 84 remaining; pause at cumulative
evaluation 46.
**Why this length:** CMA-ES updates its search distribution once per full
generation (popsize 10), so a single generation is too little to judge
whether the step size/covariance adaptation is working. Three generations
(30 evals) gives enough update cycles to see a visible trend (improving,
stagnant, or diverging) while leaving a majority of the budget (54 evals)
uncommitted for further adjustment or a different method if this one
underperforms.
**Review:** At cumulative 46, I will check whether the running best merit
has improved substantially beyond -563.20, whether CMA-ES's step size
(sigma) is shrinking (indicating convergence) or growing (indicating it is
still searching), and whether the per-generation best values are trending
down. I will pause sooner only if an early generation produces a
catastrophic result suggesting a bug (e.g. evaluations erroring or all
generations identical to the start), which has not happened so far. No
later method is preselected.

## Decision after evaluation 46
**Budget:** 46 used / 100 total; 54 remaining.
**Evidence:** Three CMA-ES generations (popsize 10, x0 = best Sobol point,
sigma0 = 1.5) produced per-generation bests of -660.29, -777.19, and
-1197.95, each generation improving on the previous best (-563.20 going
in). Sigma decreased mildly each generation (1.331 -> 1.238 -> 1.215),
indicating the search is still adapting rather than having collapsed/
stalled. Per-generation means are noisy and sometimes large and positive
(1303.17, 96.02, 2317.71), showing the population still frequently samples
into heavily-penalized (infeasible) territory even as the best member of
each generation keeps improving — consistent with a narrow feasible region
that CMA-ES is still in the process of locating/shaping its covariance
around. Current overall best: merit = -1197.95 at x = [-0.747, 0.751,
-2.470, -2.336, 0.053, -3.210, -0.003, -2.965, -0.708, 1.850].
**Decision:** Continue the same CMA-ES run (same `es` object, state and
observations fully preserved — no restart) for another block of
generations. Progress is still monotonic and sigma has not collapsed, so
there is no evidence yet that this method has stalled.
**Hyperparameters:** Unchanged from the previous entry (x0 was only the
initialization and is no longer relevant to a continuing run; popsize=10,
bounds, seed=0 inherited at construction and still in effect as part of the
live `es` object's internal state). No reconsideration of hyperparameters
is applicable mid-run beyond CMA-ES's own internal adaptation, which is
part of executing the method, not a separate agent decision.
**Reason:** Continuing lets CMA-ES keep adapting its covariance/step size
using all information gathered so far; restarting would discard three
generations of useful adaptation for no evident benefit, since the
improvement trend has not plateaued.
**Next step limit:** 30 additional objective calls (3 more generations at
popsize 10), at most 54 remaining; pause at cumulative evaluation 76.
**Why this length:** Matches the previous block's length for a comparable,
readable trend signal (3 more generation-level data points), while
deliberately leaving a meaningful reserve (24 evaluations) after this block
in case a switch to a local polish method (e.g. BOBYQA from the CMA-ES
best) is warranted once/if CMA-ES's improvement rate slows.
**Review:** At cumulative 76, check whether per-generation best values are
still improving or have plateaued, and whether sigma is shrinking toward
convergence. If improvement has clearly slowed relative to this block,
consider switching the remaining budget to a local polish method seeded at
the current best; if still improving at a similar rate, consider continuing
CMA-ES with the remaining budget instead. No method is pre-committed.

## Decision after evaluation 76
**Budget:** 76 used / 100 total; 24 remaining.
**Evidence:** Six generations of CMA-ES total. Per-generation bests:
-660.29, -777.19, -1197.95, -1228.56, -955.82, -1025.93. The overall best
improved steadily through generation 3 (eval 56, best -1228.56) but the
last two generations (eval 66, eval 76) both failed to beat it
(-955.82 and -1025.93), i.e. no improvement in the most recent 20
evaluations. Sigma is still shrinking but slowly (1.210 -> 1.170 -> 1.130
over these three generations, down from 1.331 initially), and per-generation
means remain highly variable (603.85, 1471.65, 23.76), showing the
population is still landing partly in heavily-penalized territory. This is
a genuine plateau in the best-so-far trend (two full generations with no
improvement), not just a single noisy point — and since the function is
deterministic, this is a reliable signal, not a fluctuation to discount.
Current overall best: merit = -1228.5607 at x = [-2.385, -2.151, -4.611,
-3.136, 0.454, -2.834, -0.541, -3.541, -2.587, 2.420].
**Decision:** Switch from CMA-ES to a local, model-based polish: Py-BOBYQA
(the same underlying library as the project's `BOBYQA` wrapper,
src/optimizers/bobyqa.py), seeded at the current best point rather than the
wrapper's hardcoded box-midpoint start. The standard wrapper only supports
starting at the midpoint, which is not useful here (midpoint is far worse
than the region already found) — a relevant setting the wrapper does not
expose — so, per the protocol's allowance, I will call `pybobyqa.solve`
directly in a campaign-local adapter, calling `harness.evaluate` directly
from its objective callback (no background thread/queue needed here since
`pybobyqa.solve` is itself the top-level call and can block synchronously
in this step, unlike the wrapper's ask/tell design needed for interleaving
with other optimizers mid-run).
**Hyperparameters:** `x0` = current CMA-ES best (not the wrapper's
midpoint). `bounds` = harness.bounds, with `scaling_within_bounds=True`
(same as the wrapper, Py-BOBYQA's own standard convenience for box
constraints, letting `rhobeg` default to the library's standard 0.1 in
scaled units). `rhobeg` and `rhoend` left at Py-BOBYQA's own library
defaults (unchanged from the wrapper's convention of not tuning them).
`maxfun=24`, set to exactly the remaining budget so the harness's hard
budget limit is respected by construction (BOBYQA is closed-loop and does
not take an external step-count argument the way ask/tell methods do).
Implementation: Py-BOBYQA (`pybobyqa`), same version already used by this
project's BOBYQA wrapper.
**Reason:** CMA-ES's best-so-far has plateaued for two full generations
while its step size is still only shrinking slowly, suggesting it would
spend much of the small remaining budget (24 evals, barely 2 more
generations) continuing to explore broadly rather than tightening precisely
onto the local optimum. A model-based trust-region local solver initialized
exactly at the current best uses the remaining budget to refine that single
point's neighborhood directly, which is a better match for a small
remaining budget once a promising basin has already been identified.
**Next step limit:** 24 additional objective calls, at most 24 remaining
(the entire rest of the budget); pause at cumulative evaluation 100 (budget
exhaustion) or earlier if Py-BOBYQA converges/exits first (it is a
closed-loop solver and may terminate before using the full `maxfun`).
**Why this length:** This is the last planned step: only 24 evaluations
remain in the budget, and BOBYQA's closed-loop nature means it should be
allowed to run until either it converges on its own or the budget is
exhausted, rather than being interrupted at an arbitrary intermediate count
with a local solver's trust region mid-collapse.
**Review:** At completion (budget exhaustion or BOBYQA's own convergence),
record the final best merit, total evaluations used, and whether BOBYQA
improved on the CMA-ES result. This is expected to be the final checkpoint
of the campaign unless BOBYQA exits with evaluations still remaining, in
which case I will decide what, if anything, to do with any leftover budget.

## Decision after evaluation 100
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** Py-BOBYQA ran to its `maxfun=24` limit (exit flag 1, "max
evals: objective has been called MAXFUN times") without an earlier
convergence signal, using all 24 remaining evaluations. It improved the
overall best from -1228.5607 (CMA-ES's result) to **-1352.0878** at
x = [-2.3849, -2.1514, -4.6108, -2.1358, 0.4538, -2.8345, -0.5406, -3.5411,
-2.5873, 2.4203] (note x3 moved from -3.1358 to -2.1358 relative to the
seed point; other coordinates changed only slightly, consistent with a
local trust-region refinement rather than a jump to a different basin).
This is a genuine, further improvement (-123.5 relative to the CMA-ES
best) from the same deterministic evaluator, confirming the local-polish
step was worthwhile with the remaining budget. The harness's evaluation
counter reached exactly 100, auto-triggering `harness.save()`.
**Decision:** Stop. The evaluation budget (100/100) is fully exhausted;
no further `evaluate()` calls are possible or permitted.
**Hyperparameters:** Not applicable — no further step follows.
**Reason:** Budget exhaustion is a hard stop per the harness and protocol;
there is no remaining budget to decide anything further with.
**Next step limit:** 0 (stopping).
**Why this length:** Not applicable — 0 further evaluations are available.
**Review:** Not applicable — campaign complete. Final result: best observed
merit = -1352.0878024965991, using all 100 of 100 permitted evaluations,
reached via Sobol' exploration (16 evals) -> CMA-ES from the best Sobol
point (60 evals, 6 generations) -> Py-BOBYQA local polish from the CMA-ES
best (24 evals, terminated at its own evaluation limit without full
convergence to its internal trust-region tolerance). `harness.save()` was
called explicitly (redundant with the automatic save at budget exhaustion,
but confirms `trace.csv`/`meta.json` reflect the final state); both files
are present in `results/agent_run/`, alongside this changelog and
`roadmap_initial.md`, with nothing else in that directory.
