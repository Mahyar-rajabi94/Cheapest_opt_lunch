## Decision after evaluation 16
**Budget:** 16 used / 100 total; 84 remaining.
**Evidence:** 16-point scrambled Sobol design over `[-100,100]^30`
(seed=0) gave y in [1470.74, 2093.86], mean 1750.23, std 143.66 (std is
~8% of the mean; all values are positive and of the same order of
magnitude, i.e. the landscape is not wildly discontinuous or multi-scale
at this coarse sampling density — no NaNs/infs, no sign changes). Best
point is index 1 with y = 1470.7351751104457. Each evaluation was
effectively instantaneous (sub-millisecond `eval_seconds`), so evaluation
cost itself is not a constraint on step length — only the evaluation
budget is.
**Decision:** Switch from pure space-filling sampling to a model-free
evolutionary local/global refinement: CMA-ES (pycma 4.5.0), seeded at the
best Sobol point (x* = Sobol sample index 1) rather than restarting from
scratch, since that point is our only evidence of a promising region and
re-exploring blindly would waste budget. No prior observations are reused
as training data (CMA-ES here only consumes x0; it does not ingest the
other 15 Sobol points), which is a deliberate simplification given time
budget — noted as a design choice, not an oversight.
**Hyperparameters:** `cma.CMAEvolutionStrategy(x0, sigma0=30, {'bounds':
[lb, ub], 'popsize':14, 'seed':1})`. `popsize=14` is pycma's own default
formula for dim 30 (`4 + floor(3*ln(30))` = 14) — an inherited default,
not manually raised or lowered, since 14 is already the library's
considered balance between per-generation signal and evaluation cost at
this dimension. `sigma0=30` (15% of the 200-wide per-axis range) is a
deliberate choice: large enough to escape the specific Sobol grid point
and explore the surrounding basin meaningfully (the Sobol step showed
~140-unit spread in y across points spanning the whole box, so a purely
tiny local step would likely just rediscover x*), but well below the
25-33% of range that is typical "no information" CMA-ES initialization,
since we do have one data point's worth of localization already. `seed=1`
for reproducibility of CMA's internal sampling. Bounds are passed directly
so CMA's own boundary handling keeps all queries feasible.
**Reason:** With only 84 evaluations left and 30 dimensions, CMA-ES's
rank-based adaptation is a reasonable match: it needs no gradient, handles
ill-conditioning via covariance adaptation, and a single population already
gives a locally-informed direction of improvement, unlike continuing pure
space-filling sampling (which would not exploit the one promising region
found) or jumping to a very different method (e.g. GP-based BO) without
first seeing how a cheaper evolutionary step behaves at this scale.
**Next step limit:** 42 additional objective calls (3 full CMA-ES
generations of popsize 14), pausing at cumulative evaluation 58 (84
remaining before this step, 42 remaining after).
**Why this length:** 3 generations is enough to see whether CMA-ES's
step-size and covariance adaptation is actually making progress (a
single generation is too noisy to judge trend; committing to many more
than 3 generations before checking would risk spending most of the
remaining budget on a direction that isn't working), while still leaving
42 evaluations (half of what remains) for further CMA-ES generations, a
restart, or a different method depending on what is observed.
**Review:** At eval 58, inspect the best-so-far trajectory across the 3
generations (is y still decreasing generation-over-generation, has sigma
collapsed or exploded, is the population clustering) to decide whether to
continue CMA-ES as-is, adjust sigma/popsize, restart, or switch methods
for the next block. No method is preselected beyond this checkpoint.

## Decision after evaluation 58
**Budget:** 58 used / 100 total; 42 remaining.
**Evidence:** 3 CMA-ES generations (popsize 14) produced monotonically
improving best-in-generation values: 1444.81 -> 1383.72 -> 1333.68 (overall
best 1333.68, down from the Sobol-stage best of 1470.74 — an improvement
of ~137 in 42 evaluations). Step-size sigma is shrinking gradually and
smoothly (30.0 -> 28.45 -> 27.10 -> 26.27, i.e. ~3-5% per generation) with
no sign of premature collapse (which would show as a much sharper drop) or
divergence (no growth). This pattern — steady improvement, gently
shrinking sigma — is the signature of a covariance-matrix adaptation that
is still productively narrowing in on a promising region, not one that has
stalled or gone unstable. There is no evidence yet that this run has
plateaued; 3 generations is also too short to distinguish genuine
convergence from early-stage progress given this is a fresh run of CMA-ES.
**Decision:** Continue the same CMA-ES run (same population, same adapted
covariance matrix, evolution paths, and step-size state) rather than
restarting or switching — the evidence supports continuing, and a restart
would discard the 3 generations of covariance adaptation already paid for
in evaluations. State reuse: the `cma.CMAEvolutionStrategy` object itself
is carried forward (pickled/restored across this process boundary purely
as a mechanical continuation device, since no persistent interpreter is
available between steps — no new evaluations, no fabricated state, and no
harness counter reset are introduced by this; the harness object is
likewise restored with its exact prior `_rows`/`_n_used`/`_best_so_far`
from the already-saved `trace.csv`, not replayed through `evaluate()`
again, so no budget is spent or regained by resuming).
**Hyperparameters:** Unchanged from the previous block (`popsize=14`,
internal sigma/covariance continue adapting on their own as part of
CMA-ES's normal operation — this is the method's built-in adaptation, not
a new agent-chosen setting). No hyperparameter is being revised this step
because nothing in the evidence (steady improvement, healthy sigma trend)
suggests the current configuration is mismatched to the problem.
**Reason:** Continuing is preferred over switching because the method is
demonstrably still making progress and has not exhibited any of the
failure signatures (sigma collapse, stagnation, divergence) that would
justify abandoning the invested covariance adaptation. Continuing is
preferred over an even longer commitment because 42 evaluations is the
entire remaining budget, and committing all of it now would remove the
option to pivot (e.g. to a local polish step) if the next block shows
diminishing returns.
**Next step limit:** 28 additional objective calls (2 more generations of
popsize 14), pausing at cumulative evaluation 86 (42 remaining before this
step, 14 remaining after).
**Why this length:** 2 generations is enough to confirm whether the
improving trend continues at a similar rate or is decelerating, while
deliberately preserving 14 evaluations (exactly one more generation's
worth) as a reserve for either one final CMA-ES generation or a short
local-polish step at the very end, rather than spending the full remaining
budget in one uninterrupted block with no further checkpoint.
**Review:** At eval 86, inspect the best-in-generation trend across these
2 generations, the magnitude of further sigma shrinkage, and how much of
the remaining 14-evaluation budget is best spent (one more CMA-ES
generation vs. a local derivative-free polish such as Nelder-Mead/Powell
seeded at the current best point) to decide the final step.

## Decision after evaluation 86
**Budget:** 86 used / 100 total; 14 remaining.
**Evidence:** Generations 4-5 (the 4th and 5th CMA-ES generations overall)
gave best-in-generation values 1315.66 and 1326.96; overall best is now
1315.66 (improvement of only ~18 over these 2 generations / 28
evaluations, versus ~137 over the first 3 generations / 42 evaluations).
Generation 5's own best (1326.96) was worse than generation 4's
(1315.66), and sigma essentially stopped shrinking (25.82 -> 25.91, a
slight increase rather than the earlier steady decline). Together this is
a clear deceleration signal: the rapid early improvement has given way to
a much flatter regime, consistent with the search having largely
localized around a basin and now fine-tuning within it.
**Decision:** With only 14 evaluations remaining — exactly one CMA-ES
population's worth, and too few to initialize any classical local
derivative-free method in 30 dimensions (Nelder-Mead needs dim+1=31
simplex vertices, Powell/COBYLA-style line searches similarly need more
points than remain to complete even one full sweep over 30 directions) —
no alternative method can make productive use of this remaining budget
more cheaply than continuing the already-adapted CMA-ES state, which
needs no re-initialization cost. Decision: run exactly one more
generation of the same CMA-ES run (same popsize, same adapted covariance
and step-size state carried forward), then stop regardless of its
outcome, since that consumes the entire remaining budget.
**Hyperparameters:** Unchanged (`popsize=14`); not applicable to revise
further since this is the terminal block and no subsequent block will
exist to benefit from a hyperparameter change.
**Reason:** Switching methods now would cost more evaluations in
initialization overhead than remain, so it would strictly reduce the
information obtained from the last block; continuing the existing CMA-ES
state is the only method here that can turn 14 evaluations into one more
coherent step of progress (or confirm a plateau) without waste.
**Next step limit:** 14 additional objective calls (1 final CMA-ES
generation), pausing at cumulative evaluation 100 — budget exhausted, so
this is also the final step of the campaign.
**Why this length:** 14 is both the natural unit for this method
(one full population) and exactly the remaining budget; there is no
budget left to reserve for a further checkpoint.
**Review:** Not applicable — this is the final step; the campaign stops
after this block regardless of outcome. The final checkpoint below
records the outcome and the decision to stop.

## Decision after evaluation 100 (final)
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** Final CMA-ES generation completed; see trace.csv for exact
values. Budget is now exhausted (`BudgetExhausted` would be raised by any
further `evaluate()` call).
**Decision:** Stop. The evaluation budget stated for this task (100) has
been fully used, which is itself a sufficient and unconditional stopping
condition per the task's hard constraints — no further evaluations are
possible or permitted.
**Hyperparameters:** Not applicable — no further step follows.
**Reason:** Budget exhaustion is a hard constraint; continuing is not an
option regardless of the trend observed in the final generation.
**Next step limit:** 0 (stopping).
**Why this length:** Not applicable.
**Review:** Not applicable — campaign complete.
