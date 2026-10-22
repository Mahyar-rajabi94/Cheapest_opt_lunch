## Decision after evaluation 16
**Budget:** 16 used / 100 total; 84 remaining.
**Evidence:** 16 Sobol' points (seed=0) spread across the full [-5,5]^10 box
produced `y` values ranging from 1216.24 (eval 7,
x=[-1.72,2.73,0.53,0.78,3.48,-4.04,-1.86,-2.03,-4.61,0.81]) to 3057.81 (eval
16), with most values clustered in the 1300-2300 range. All values are
large and positive (hundreds to low thousands) with no sign change and no
single wild outlier — the max/min ratio across this sample is only about
2.5x, which, given the points are spread across the whole box and not
clustered, is consistent with a reasonably smooth, continuously varying
surface rather than one with sharp spikes or deep narrow wells at this
scale. 16 points in 10-d is too few to confirm separability, multimodality,
or conditioning, so this is a tentative read, not a conclusion.
**Decision:** Switch to CMA-ES (`src/optimizers/cma_es.py`, wrapping
`cma.CMAEvolutionStrategy`) for the next block. CMA-ES is a strong default
for box-constrained continuous black-box optimization at this dimension
(10) and is far more sample-efficient per generation than restarting a
pure space-filling or single-point local search would be here, given the
smooth-looking evidence so far and the small remaining budget (84). It is
initialized fresh (its own standard box convention: x0 = box midpoint,
sigma0 = quarter of mean box width), not seeded from the Sobol' points,
since the wrapper's interface does not support injecting prior
observations and this is a plain restart of a different method, not a
continuation of Sobol'; the Sobol' evaluations remain in the trace and
inform this choice but are not fed into CMA-ES's own adaptation.
**Hyperparameters:** `bounds=[[-5,5]]*10` (from harness); `seed=0`. All
other settings are pycma's own defaults (no tuning), per the wrapper's
docstring: in particular `popsize` (lambda) is pycma's standard formula,
4 + floor(3*ln(10)) = 10, so each generation costs 10 evaluations.
**Reason:** With no evidence yet of sharp multimodality and a landscape
that looks smooth so far, an adaptive covariance method should locate a
good descent direction faster than continuing uniform coverage, while
remaining robust if the surface turns out to be mildly anisotropic or
ill-conditioned (CMA-ES adapts its covariance to that automatically).
**Next step limit:** 40 additional objective calls, at most 84; pause at
cumulative evaluation 56.
**Why this length:** 40 calls = exactly 4 full CMA-ES generations (popsize
10), enough to let the covariance matrix adapt past its initial isotropic
guess and show a real trend in the running best, without committing more
than half of the remaining budget to a single untested method.
**Review:** At eval 56, inspect whether `y_best_so_far` improved
meaningfully past 1216.24 and whether the trend across the 4 generations is
still improving, stalled, or oscillating; also check the evolving step
size/spread implied by the proposed points. Pause earlier only if CMA-ES
raises an error or the population collapses/diverges visibly before 40
calls are used.

## Decision after evaluation 56
**Budget:** 56 used / 100 total; 44 remaining.
**Evidence:** An execution error, not a methodological one, is reported
here first because it changes what this block's data actually represents.
The CMA-ES wrapper's documented contract requires `ask(n)`/`tell(X, y)` to
be called once per generation, in lockstep (population size here is 10).
The driver command used for this step requested all 40 points in a single
`ask(n=40)` before any `tell`, which silently called the underlying
`cma.CMAEvolutionStrategy.ask()` four times with no intervening `tell()`.
As a result only the first 10 points (one real generation) were ever
properly registered with CMA-ES's internal adaptation via `tell()`; the
`tell()` call for the second chunk of 10 then raised
`RuntimeError: tell can currently only be called once per iteration`,
which stopped further internal updates. All 40 points had, by that point,
already been sent through `harness.evaluate()` (labeled `"cma_es"`) --
every one of them is a real, valid evaluation and is correctly present in
`trace.csv` (evals 17-56) -- but 30 of the 40 do not reflect genuine
multi-generation CMA-ES search; they are, in effect, repeated draws from
the algorithm's initial (gen-1) sampling distribution, not an adapted one.
Consistent with this, the block's values (min 1289.84, max 2463.85 across
all 40) show no improvement over the Sobol' incumbent (1216.24, eval 7)
and no visible downward trend. This is treated as inconclusive evidence
about CMA-ES itself (it was not actually driven correctly for 3 of its 4
intended generations), not as evidence that adaptive search cannot help.
The driver bug is fixed going forward (population-based `ask`/`tell` calls
will be issued exactly one generation at a time, matching each method's
own batch-size contract) via a corrected IPC command; the harness's budget
counter and all 56 real evaluations were preserved exactly (rehydrated
from the already-saved `trace.csv` into the same running process's
bookkeeping) while fixing this, so no evaluation was lost, duplicated, or
fabricated.
**Decision:** Switch to BOBYQA (trust-region local solver,
`src/optimizers/bobyqa.py`'s own library call, Py-BOBYQA /
`pybobyqa.solve`), restarted from the best point observed so far (eval 7,
y=1216.241) rather than the wrapper's hardcoded box-midpoint start. The
stock wrapper always starts at the midpoint, so this uses a small
campaign-local adapter (same thread+queue bridge, same library call and
defaults, only `x0` is exposed as a parameter) per the protocol's allowance
for wrapping a library directly when a wrapper fixes a setting I need to
control. Rationale for the method: with 56 evaluations spent and no
evidence yet of a better basin than the one around eval 7, and the
landscape looking smooth rather than sharply multimodal (Section 1's
read, still only tentatively supported), a sample-efficient local
trust-region refiner should make rapid progress descending from the best
known point, at roughly 1 evaluation per internal step -- far cheaper per
step than continuing any population-based or GP-based global method with
only 44 calls left.
**Hyperparameters:** `bounds=[[-5,5]]*10` (from harness); `x0` = eval 7's
point (given above) instead of the library wrapper's default box
midpoint; `scaling_within_bounds=True`, `maxfun=20000`, `rhoend`=library
default (1e-8) -- all inherited unchanged from `src/optimizers/bobyqa.py`'s
own documented defaults (Py-BOBYQA, Cartis/Fiala/Marteau/Roberts 2019).
`rhobeg` is Py-BOBYQA's own default for `scaling_within_bounds=True` (0.1
in the rescaled unit-ish cube), not hand-tuned.
**Reason:** BOBYQA builds a local quadratic model and trust region around
its start; starting it at the best point found so far, rather than at the
uninformative box midpoint, lets it spend its budget exploiting the most
promising region already identified instead of re-discovering it.
**Next step limit:** 25 additional objective calls, at most 44; pause at
cumulative evaluation 81.
**Why this length:** BOBYQA is a closed-loop sequential solver (1 point
per internal step) that may converge (raise `OptimizerExhausted`) well
before using a full allotment, which is normal, expected behavior for it,
not a failure; 25 calls leaves 19 in reserve to either let it continue, or
switch to a different method/region if it converges early or stalls,
while still giving it a meaningful number of trust-region updates to work
with (10-d models typically need on the order of dim+1 to a few times
dim points to refine the local quadratic model).
**Review:** At eval 81 (or sooner if BOBYQA raises `OptimizerExhausted`),
check whether `y_best_so_far` improved below 1216.241, by how much, and
whether BOBYQA's own reported trust-region radius / result status
suggests it is still making progress or has effectively converged.

## Decision after evaluation 81
**Budget:** 81 used / 100 total; 19 remaining.
**Evidence:** BOBYQA, restarted at eval 7's point, improved the incumbent
from 1216.241 to 1174.927 (eval 60, the 4th BOBYQA call), a roughly 3.4%
relative drop. The next 21 BOBYQA-proposed points (evals 61-81) stayed in
a 1174.9-1285.6 band with no further improvement on the running best;
several of these are higher than the eval-60 point, which is expected and
not evidence of divergence -- a trust-region method samples around its
current model to refine the local quadratic approximation and does not
propose only strictly-improving points. `bobyqa_run_block` reports
`exhausted: false`, so the solver has not converged/stopped; it would
keep proposing points if given more budget. The flattening of
`y_best_so_far` over the last ~21 calls, though, suggests this particular
basin/trust region may be nearly exhausted of easy gains, or is
refining at a finer scale than 21 evaluations can show clearly yet.
**Decision:** Continue the same method (BOBYQA, same live optimizer
instance `bob1`, same trust-region state carried over -- no restart) for
the remaining budget. With only 19 evaluations left, no alternative
method is viable to initialize properly from scratch (a CMA-ES generation
alone costs 10, TuRBO's own initial design costs 2*dim=20 which exceeds
what remains, and a fresh BOBYQA/Nelder-Mead/L-BFGS-B restart would throw
away the trust-region progress already made without any guarantee of
reaching a better basin in so few calls). Continuing lets BOBYQA either
converge naturally (stop proposing points via `OptimizerExhausted`, which
is normal, expected closed-loop behavior for this method, not a failure)
or make further incremental progress within its current model.
**Hyperparameters:** Unchanged -- same `bob1` instance, same bounds,
`scaling_within_bounds=True`, `maxfun=20000`, library-default `rhobeg`
(0.1)/`rhoend` (1e-8), as configured in the eval-56 decision above.
**Reason:** Given the remaining budget is too small to properly initialize
any other method in this package, and BOBYQA has not converged or
diverged, using the rest of the budget to let the current closed-loop
local solver either converge or continue refining is the best use of the
final calls.
**Next step limit:** 19 additional objective calls, at most 19; pause at
cumulative evaluation 100 (budget exhaustion) or sooner if
`OptimizerExhausted` is raised.
**Why this length:** 19 is all the remaining budget; since no further
method switch is viable with less than this, and this is the final step of
the campaign, there is no reason to reserve any further evaluations.
**Review:** Not applicable beyond this step -- this is intended to be the
final block. At its end, record the final best value, evaluations used,
and whether the stop was due to budget exhaustion or BOBYQA converging
first.

## Decision after evaluation 100 (final)
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** BOBYQA continued refining over the final 19 calls, with
`y_best_so_far` dropping steadily rather than flattening again: 1174.927
(eval 81) -> 1160.286 (85) -> 1151.305 (87) -> 1139.906 (88) ->
1137.888 (91) -> 1136.067 (92) -> 1127.952 (95) -> 1127.029 (eval 100,
the final call). `bobyqa_run_block` reports `exhausted: false` -- the
solver was still proposing new points and had not converged when the
budget ran out; `n_used` reached exactly 100, matching
`meta.json`'s `evaluations_used` and `trace.csv`'s row count.
**Decision:** Stop. The harness budget is exhausted
(`evaluations_remaining == 0`); no further `evaluate()` calls are
permitted.
**Hyperparameters:** Not applicable (stopping).
**Reason:** The stated budget of 100 real evaluations has been fully and
exactly used; stopping is mandatory, not a judgment call. Final campaign
result: best observed `y = 1127.0292826404702` at eval 100 (point
`x = [-1.418270, 2.731702, 1.463643, 0.947372, 3.508261, -4.132495,
-1.688404, -2.125885, -4.685883, 0.766311]`, from the BOBYQA block
restarted at the incumbent found by Sobol' sampling), direction minimize.
**Next step limit:** 0 (stopping).
**Why this length:** Not applicable.
**Review:** Not applicable -- campaign complete. If further budget were
ever made available, the natural next step would be to continue this same
BOBYQA run (it was still actively improving, not converged) rather than
switch methods, since the evidence at this final checkpoint shows
uninterrupted descent.
