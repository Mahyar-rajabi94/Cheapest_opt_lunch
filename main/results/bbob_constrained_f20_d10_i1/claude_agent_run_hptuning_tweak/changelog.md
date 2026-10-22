## Decision after evaluation 16
**Budget:** 16 used / 100 total; 84 remaining.
**Evidence:** Sobol' init step (seed=0, 16 points) produced merit values
ranging from 8251.25 (best) to 290444.56 (worst), mean ~70159, with
substantial spread across the box (no two points close in value). This
is consistent with a merit landscape where large regions of the 10-D box
are penalized (infeasible or far from optimum) and only some regions are
markedly better; the best point found, x ≈ [-1.72, 2.73, 0.53, 0.78,
3.48, -4.04, -1.86, -2.03, -4.61, 0.81], is a reasonable seed for a local
search but there is no evidence yet of multiple distinct promising basins
(too few points to tell) nor evidence that this is the global basin.
**Decision:** Switch to CMA-ES, initialized at the best Sobol point found
rather than the box midpoint, to exploit the directional information from
step 1 while retaining a fairly wide initial step size for continued
exploration around it. The standard wrapper (`src/optimizers/cma_es.py`)
hardcodes x0 = box midpoint, so I use a thin campaign-local adapter that
calls `cma.CMAEvolutionStrategy` directly with the same ask/tell contract,
keeping all other pycma internals (population size, learning rates, etc.)
at library defaults, per section 3 of the protocol (wrapper's fixed
setting overridden via the underlying library's supported API).
**Hyperparameters:** x0 = best Sobol point (overridden, rationale above);
sigma0 = 2.0 (chosen by me: smaller than the wrapper's default quarter-box
width of 2.5, since the search is now centered on a point already known to
be decent rather than the uninformative midpoint, but still wide enough to
cover a large fraction of the box and avoid premature collapse given only
16 data points so far); `bounds` = harness.bounds (inherited, same box
constraint handling as the wrapper); `seed=1` (new seed, since this is a
different sampler than Sobol; pycma's own default for popsize, weights,
learning rates — version: `cma` as installed in this sandbox's .venv).
This optimizer instance starts fresh (no prior CMA-ES state to retain);
the harness instance and its evaluation counter are preserved unchanged
from step 1.
**Reason:** Given only 84 evaluations remain for a 10-D problem, continuing
pure space-filling exploration has a low expected return; CMA-ES is a
strong general-purpose choice for exploiting a promising region found so
far while still adapting its covariance/step size as new data arrives,
and it is better suited than a pure local method (e.g. BOBYQA) at this
stage because the merit surface's local shape around the best point is
still unknown and the step is cheap to walk back from if it stalls.
**Next step limit:** 40 additional objective calls (cumulative 56); at
10 evaluations/generation this covers 4 CMA-ES generations, enough to see
whether the step-size/mean are adapting toward better values before
reassessing.
**Why this length:** Four generations is enough for CMA-ES's covariance
and step-size adaptation to become visible (a single generation is too
noisy to judge trend), while still leaving roughly half the remaining
budget (44 of 84) for further refinement, restart, or a different method
if this step does not show improvement.
**Review:** At evaluation 56, check whether the running best merit has
improved versus the step-1 best (8251.25), whether CMA-ES's internal best
is still improving across generations or has plateaued, and whether the
population's spread of values suggests the step size is still shrinking
usefully or has stalled; this will inform whether to continue CMA-ES,
adjust sigma, or switch methods (e.g. to BOBYQA/TuRBO for a more local
finish) for the next step. No method beyond this step is preselected.

## Decision after evaluation 56
**Budget:** 56 used / 100 total; 44 remaining.
**Evidence:** Over 4 CMA-ES generations (10 evals each), the per-generation
best improved 10975.86 → 13880.64 → 4412.47 → 3159.58 (overall best now
3159.58, down from 8251.25 at the end of step 1), a clear net improvement
of ~62% despite one noisy generation (gen 2 best 13880.64, worse than gen
1's 10975.86 — expected variance while CMA-ES's covariance is still
adapting, not a sign the method has failed). Within-generation spread
remains large and did not shrink monotonically (gen 4 contained both the
new best, 3159.58, and several very poor points, e.g. 145543.10,
68919.13, 56700.78, alongside other good points like 3963.71, 4501.32,
4517.15). This pattern — tight cluster of good values plus occasional very
large ones in the same generation — is consistent with the search region
straddling a constraint boundary (recall: large merit likely reflects
infeasibility/penalty, not an extreme true objective), which CMA-ES's
covariance adaptation should progressively learn to avoid rather than
requiring a method change.
**Decision:** Continue CMA-ES, same `es` object and internal state
(mean, covariance, step size, evolution paths all retained — this is
the optimizer's own ordinary internal adaptation, not a restart), same
harness instance/counter.
**Hyperparameters:** No changes. Reference step-2 configuration
(x0 = Sobol best, sigma0 = 2.0, bounds = harness.bounds, seed = 1,
popsize = 10 from pycma's own default for dim=10, verbose=-9); sigma and
the covariance matrix continue to adapt internally per pycma's standard
algorithm, which is execution of the chosen method, not a new decision.
**Reason:** The method is still actively improving the best-so-far value
at a non-trivial rate and has not plateaued; there is no evidence yet that
CMA-ES has converged or stalled (only 4 generations completed), so
switching now would discard adaptation progress without cause. Continuing
is the evidence-driven choice; a fixed reserve of the remaining budget
will still be kept for a possible local-polish or alternative step after
the next checkpoint.
**Next step limit:** 30 additional objective calls (cumulative 86); at
popsize 10 this is 3 more generations. This leaves 14 evaluations in
reserve after the next checkpoint, enough for a short local polish
(e.g. BOBYQA/Nelder-Mead from the best point) or a few more CMA-ES
generations, whichever the evidence then supports.
**Why this length:** Three more generations is enough to see whether the
recent improvement (last two generations) continues, stalls, or reverses,
while deliberately preserving a non-trivial reserve (14 evals) rather than
committing the full remaining budget to one method, consistent with not
pre-allocating the whole campaign.
**Review:** At evaluation 86, check the trend of per-generation best and
mean values, whether the overall best continues to improve meaningfully
beyond 3159.58, and whether within-generation spread is narrowing (sign
of convergence) or still wide (sign more generations could help). This
will determine whether to continue CMA-ES with the remaining reserve,
stop CMA-ES and spend the reserve on a local polish method from the best
point found, or another adjustment. No later method is preselected.

## Decision after evaluation 86
**Budget:** 86 used / 100 total; 14 remaining.
**Evidence:** Over the last 3 generations, per-generation best values were
2917.18 (new overall best, found early in this step), then 4910.54, then
8895.27 — not monotonically improving, and the overall best improved only
marginally versus step 2 (3159.58 → 2917.18, ~8%). Per-generation means
did trend down (37655.79 → 27575.63 → 16279.64), but the step-size
diagnostic is the clearer signal: `es.sigma` moved from its initial 2.0 to
2.41 over 7 generations (i.e. it grew slightly rather than shrinking),
indicating CMA-ES has not located a tight local basin to converge into —
it is still sampling at roughly its initial scale. Combined with the
within-generation spread seen since step 2 (clusters of good values next
to occasional very large ones), this is consistent with a rugged,
boundary-heavy merit surface where this run of CMA-ES is not converging
within the remaining budget.
**Decision:** Stop CMA-ES (it is not converging and only 14 evaluations
remain — not enough for further generations to plausibly shrink sigma
meaningfully) and switch to a local polish: BOBYQA (Py-BOBYQA), a
model-based trust-region local solver, seeded at the best point found so
far (x from CMA-ES, y=2917.18) rather than the box midpoint. The standard
wrapper (`src/optimizers/bobyqa.py`) hardcodes x0 = box midpoint and the
interpolation-set size `npt` at Py-BOBYQA's own default, so I use a
campaign-local adapter (same background-thread/queue pattern as the
wrapper, calling `pybobyqa.solve` directly) to set x0 = best point and a
minimal `npt`, since only 14 evaluations remain and BOBYQA needs at least
`dim+1` points to build its first local model.
**Hyperparameters:** x0 = best CMA-ES point (overridden, rationale above);
`npt = dim + 1 = 11` (overridden down from Py-BOBYQA's richer default of
~2*dim+1, the minimum valid interpolation-set size, chosen because only 14
evaluations remain and a full default model would leave almost no budget
for actual trust-region steps); `rhobeg = 0.5` (overridden down from the
wrapper's effective ~0.1-of-unit-cube-scale default, chosen because the
search is now local around an already-reasonable point rather than
spanning the box, so a smaller initial trust-region radius is appropriate
for a short, local polish); `scaling_within_bounds=True`, `rhoend` left at
Py-BOBYQA's own default (1e-8), `bounds = harness.bounds` (all inherited,
unchanged from the wrapper's own defaults, implementation: Py-BOBYQA as
installed in this sandbox's .venv). This is a fresh optimizer instance
(BOBYQA carries no state from CMA-ES by construction); only its starting
point is informed by CMA-ES's result. The harness instance/counter is
preserved unchanged.
**Reason:** With only 14 evaluations left and clear evidence that CMA-ES's
global step size is not shrinking (no sign of convergence), a model-based
local solver seeded at the current best is a better use of the small
remaining budget than either continuing an unconverged CMA-ES or
restarting a new global search from scratch; BOBYQA is specifically
designed to make efficient use of very few evaluations once a reasonable
starting point is available, unlike population-based methods.
**Next step limit:** 14 additional objective calls (cumulative 100) — all
remaining budget, since this is the final step. Pause (stop) at 100.
**Why this length:** This uses the entire remaining budget because it is
the last step in the campaign (100-call hard limit) and there is no
further budget to hold in reserve past this point; 14 evaluations split as
11 for BOBYQA's minimal initial model plus 3 for subsequent trust-region
steps is the most BOBYQA can be given here.
**Review:** Final checkpoint — report final best merit and evaluations
used; not applicable beyond this (stopping).

## Decision after evaluation 100 (final)
**Budget:** 100 used / 100 total; 0 remaining.
**Evidence:** BOBYQA's 14 evaluations (11 for the minimal initial
interpolation model, 3 trust-region/model-update steps) produced a small
further improvement: best merit went from 2917.18 (seed point) to
2825.39, found at evaluation 93 (the 7th BOBYQA call). Many of the other
13 points in this step were markedly worse (e.g. 160442.49, 33899.78,
17858.45), again consistent with a boundary-heavy local landscape even at
this smaller radius (rhobeg=0.5) — BOBYQA's model-building/trust-region
steps necessarily sample some points that turn out infeasible/penalized
as it maps the local curvature, which is expected behavior, not a sign of
failure. The harness's evaluation counter reached 100 and raised/auto-
triggered `harness.save()` on the final call, exhausting the stated
budget.
**Decision:** Stop. The budget is exhausted — no further `evaluate()`
calls are possible (this is a hard constraint, and the harness itself
would raise `BudgetExhausted` on any further attempt).
**Hyperparameters:** Not applicable (no further step).
**Reason:** Budget exhaustion is a hard stop per the protocol; the
campaign ends here regardless of whether further improvement might have
been possible with more evaluations.
**Next step limit:** 0 (stopping; 0 evaluations remain).
**Why this length:** Not applicable — no budget remains.
**Review:** Not applicable — campaign complete. Final best merit
observed: 2825.391996673463, found at evaluation 93 during the BOBYQA
polish step. `harness.save()` was triggered automatically by the harness
upon reaching the budget limit, writing `trace.csv` (100 rows) and
`meta.json` to this results directory.
