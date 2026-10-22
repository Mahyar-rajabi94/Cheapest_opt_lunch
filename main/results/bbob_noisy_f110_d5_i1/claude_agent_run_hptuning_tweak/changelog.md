## Decision after evaluation 12
**Budget:** 12 used / 100 total; 88 remaining.
**Evidence:** Step 1 (Sobol' initial design, n=12, seed=0, via `SobolSearch`)
completed. Observed `y` values: min=1880.19, max=147298.60, mean=48200.64,
std=42214.96 (minimize). The spread across the 12 points (~145,000) is far
larger than any noise magnitude plausible at this scale would need to be to
still leave a clear signal, and the best point found
(`x ≈ [1.074, 0.545, 0.164, 2.668, -2.288]`, well inside the box, not at a
corner) is far from the worst points (several of which sit at larger
Euclidean distance from this region) — this is consistent with a smooth,
unimodal-ish bowl-like landscape with usable structure, not a flat or
noise-dominated response surface. 12 points is too few to be certain of
global shape, but there is clearly exploitable signal relative to apparent
noise.
**Decision:** Switch to GP-BO (`src/optimizers/gp_bo.py`, BoTorch
`SingleTaskGP` + analytic `LogExpectedImprovement`). I will instantiate a
fresh `GPBO(bounds=harness.bounds, seed=0)` and immediately feed it the 12
already-evaluated Sobol points via `tell(X, y)`, so its internal
`n_init = 2*(dim+1) = 12` initial-design requirement is already satisfied
by the data just collected (no evaluations are repeated or wasted
re-deriving an initial design). Its `ask()` will therefore fit a GP and
propose an acquisition-driven point on the very next call. No other state
needs to be reset or carried over (the Sobol sampler itself is not reused
further this run).
**Hyperparameters:** `GPBO` defaults, inherited from
`src/optimizers/gp_bo.py` (BoTorch, current environment version) and not
changed yet: `n_init = 2*(dim+1) = 12` (already satisfied), `num_restarts =
10`, `raw_samples = 512` for `optimize_acqf`; `SingleTaskGP` with its
library-default kernel/lengthscale prior, `Normalize` input transform to
`harness.bounds`, `Standardize` outcome transform, and its own fitted
(homoskedastic) observation-noise term — this last point is the main reason
GP-BO is a reasonable choice under noise (it does not treat a single
observation as exact truth). `seed=0` for the acquisition optimizer's random
restarts, for reproducibility. I have not yet seen evidence that any of
these need tuning; I will reconsider after observing how GP-BO's proposed
points behave here specifically (e.g. whether the fitted noise term looks
reasonable, whether acquisition points cluster tightly or jump around).
**Reason:** The visible structure in Step 1's data makes a model-based,
sample-efficient method worthwhile rather than continuing pure space-filling
search; GP-BO is the best-suited method already available for this budget
regime (its own docs recommend ~10x dim, we are already at 12 points i.e.
~2.4x dim with 88 of 100 remaining, i.e. ~17.6x dim remaining) and it
explicitly accounts for noise via a fitted likelihood term, unlike
e.g. plain CMA-ES or Nelder-Mead which treat each observation as exact.
**Next step limit:** 8 additional objective calls, at most 88 remaining;
pause at cumulative evaluation 20.
**Why this length:** Each GP-BO `ask()` refits a GP from scratch on all data
so far, so the model and acquisition landscape can shift meaningfully after
even a few new points; a short block (8, i.e. roughly dim+3) gives enough
new information to judge whether the acquisition is converging toward a
promising region or behaving erratically (e.g. due to noise misleading
`best_f`), without committing a large fraction of the remaining 88 to one
untested configuration.
**Review:** At cumulative evaluation 20, inspect: whether `y_best_so_far`
has improved beyond Step 1's 1880.19 and by how much; whether proposed
points are converging into a local region or still exploring broadly;
whether any individual acquisition point's observed `y` looks like an
outlier relative to its neighbors (possible noise signal); and whether GP
fit/acquisition timing is acceptable. No preselected next method; continuing
GP-BO unchanged is as valid as adjusting `num_restarts`/`raw_samples` or
pausing to take a repeated-point noise estimate, depending on what this
block shows.

## Decision after evaluation 20
**Budget:** 20 used / 100 total; 80 remaining.
**Evidence:** GP-BO's first 8 acquisition-driven points (fed the 12 Sobol
points first via `tell`, then run one-at-a-time) gave `y` values ranging
from 4082.19 to 97464.59 (eval 13-20: 5172.08, 80348.48, 53212.96,
97464.59, 4082.19, 9193.61, 6249.84, 13582.36). None improved on the
Step 1 incumbent `y_best_so_far = 1880.19` at
`x ≈ [1.074, 0.545, 0.164, 2.667, -2.288]`. Several proposed points sit at
or near the box edges (e.g. eval 14: two coords at -5/+5; eval 20: two
coords at -5/+5), which is consistent with LogEI still exploring
high-uncertainty regions this early (only 12-19 training points for a 5-d
GP) rather than evidence that GP-BO is misbehaving. The one point closest
in character to the incumbent (eval 17, small values in the first 3
coords) gave 4082.19 — noticeably worse than 1880.19 but on the same
order of magnitude, plausible under either noise or a genuinely steep
local landscape between the two points. 8 points is too few after only 12
initial points to expect exploitation to have taken over yet; a short
plateau at this stage does not indicate the method is unsuitable.
**Decision:** Continue GP-BO unchanged, same live `GPBO` instance (all 20
points told so far retained, no restart) — there is no evidence yet that
the model or acquisition is misbehaving, only that it is still in an
exploration-heavy phase expected this early with few training points
relative to a 5-d input space.
**Hyperparameters:** Unchanged from the previous entry (`n_init=12`
already satisfied, `num_restarts=10`, `raw_samples=512`, BoTorch
`SingleTaskGP`/`LogExpectedImprovement` defaults, `seed=0`). Not revising
yet since no evidence points to a specific setting being the bottleneck.
**Reason:** Switching methods or hyperparameters now would be reacting to
noise/early-exploration variance rather than to a demonstrated weakness;
the GP has very little data per dimension so far (20 points in 5-d, i.e.
4 points/dim) and LogEI's exploration behavior so far (probing box edges)
is exactly what is expected at this stage, not a sign of malfunction.
**Next step limit:** 10 additional objective calls, at most 80 remaining;
pause at cumulative evaluation 30.
**Why this length:** Slightly longer than the previous block (10 vs 8) to
give the GP more data before the next review, while still being small
enough (10% of total budget) to catch a persistent lack of progress early;
30/100 will have used almost a third of the budget on model-building if
acquisition still has not found an improvement, which is enough data to
meaningfully judge the method.
**Review:** At cumulative evaluation 30, inspect whether `y_best_so_far`
has improved at all from 1880.19, whether acquisition points are
narrowing toward a region (e.g. near the current incumbent) rather than
still probing box edges, and whether the GP's implied noise level (via
its fitted likelihood, if easily inspected) looks reasonable relative to
the scale of `y`. No preselected next method.

## Decision after evaluation 30
**Budget:** 30 used / 100 total; 70 remaining.
**Evidence:** The second GP-BO block (evals 21-30) again failed to beat
the Step 1 incumbent (`y=1880.19`). Values ranged from 3398.83 up to
221537.37 — the dynamic range has grown even larger (now ~118x between
best and worst observed, vs ~78x after Step 1), and proposed points
continue to sit heavily on box edges/corners (e.g. eval 22:
`[5, 0.207, -0.558, -5, -0.957]`, eval 23: three coords at ±5, eval 28:
one coord at -5). Across 18 GP-BO points total, none has approached the
incumbent's region (coords roughly in `[-2.3, 2.7]`, none at a bound), and
the single closest earlier attempt (eval 17) was still 2.2x worse. This
pattern — acquisition persistently favoring box edges over refining the
one promising region already found — combined with the very large and
still-growing dynamic range of `y`, suggests the GP's output
standardization (mean/std over all told points, dominated by large-value
outliers at the edges) is compressing the interesting low-value region
into a small part of the standardized output range, making LogEI
comparatively insensitive to real improvements there. This is a genuine
mismatch between this problem's output scale and analytic LogEI's
noiseless-at-best_f assumption, not an early-exploration artifact —
18 points with zero movement toward the known good region is enough
evidence.
**Decision:** Switch method: restart with CMA-ES (`cma` library, pycma
reference implementation, the same one `src/optimizers/cma_es.py` wraps),
used via a campaign-local adapter rather than that wrapper, because the
wrapper fixes `x0` to the box midpoint and I want to initialize at the
best point found so far rather than the box center. This is a restart
(no state carried over from GP-BO); the 30 evaluations already spent
remain in the harness's own record and contribute to the campaign's
evaluation count, but CMA-ES will build its own search distribution from
scratch. Rationale for CMA-ES specifically: it is population-based, so
each generation's update is informed by multiple simultaneous samples
rather than a single point, giving it a measure of robustness to
per-observation noise that a single-point local method (e.g. L-BFGS-B on
a finite-difference gradient) would not have, while still being far more
local/exploitative than continuing to let GP-BO re-explore box edges.
**Hyperparameters:** Using `cma.CMAEvolutionStrategy` directly (pycma,
same version as `src/optimizers/cma_es.py`'s dependency) with:
`x0 = [1.07353212, 0.54487804, 0.16378565, 2.66753089, -2.28759103]`
(the current incumbent, from Step 1's Sobol design) instead of the box
midpoint, because 30 evaluations already identify this as a much better
starting region than the box center; `sigma0 = 2.0` (vs. the wrapper's
own default rule of box-width/4 = 2.5 for this box) — slightly reduced
because the search should start more local around a point already known
to be good, while remaining large enough to escape this point's immediate
neighborhood if it is not in fact a good region at larger scale (deviation
from the wrapper's default rule, justified by having a concrete incumbent
to center on rather than an uninformed midpoint); `bounds` enforced via
pycma's own `bounds` option (same mechanism as the wrapper), covering
`[-5, 5]` per coordinate; `seed=0` for reproducibility; population size,
learning rates, and all other internal CMA-ES parameters left at pycma's
own defaults (no evidence yet to justify tuning them, and CMA-ES's default
population sizing is itself a well-studied, dimension-dependent rule, not
an arbitrary default).
**Reason:** Directly follows from the evidence: GP-BO has shown it cannot
exploit the known good region at this problem's output scale within a
reasonable number of evaluations, while 30 evaluations have reliably
triangulated one region as much better than the rest of the box. A
population-based local/semi-global method seeded there is the best match
for the remaining budget and the noise characteristic.
**Next step limit:** 24 additional objective calls, at most 70 remaining;
pause at cumulative evaluation 54. With dim=5, pycma's default population
size is `lambda = 4 + floor(3*ln(5)) = 8`, so 24 calls covers 3 full
generations.
**Why this length:** A single generation (8 evals) is too little to judge
whether CMA-ES's covariance adaptation is moving in a useful direction
under noise; 3 generations gives the step-size/covariance adaptation a
real chance to show a trend (improving, stagnant, or diverging) while
still leaving 46 evaluations (nearly half the total budget) for further
generations or a different adjustment afterward.
**Review:** At cumulative evaluation 54, inspect whether the best-so-far
value has improved below 1880.19, whether successive generations' mean/
best values show a consistent improving trend (accounting for noise) or
have stagnated, and whether `sigma` has shrunk sharply (possible premature
convergence) or stayed large (still searching broadly). No preselected
next method; continuing CMA-ES, shrinking/growing sigma via a restart, or
switching again are all open.

## Decision after evaluation 54
**Budget:** 54 used / 100 total; 46 remaining.
**Evidence:** CMA-ES ran 3 generations (popsize 8, evals 31-54) seeded at
the previous incumbent with `sigma0=2.0`. Generation 1 (evals 31-38):
range 6831.93-156874.94, no improvement. Generation 2 (evals 39-46): range
7402.71-142581.79, no improvement. Generation 3 (evals 47-54): produced a
clear improvement at eval 50,
`x=[0.126, -0.342, 0.933, 2.779, -1.317]`, `y=490.708` — about 3.8x better
than the Step 1/GP-BO incumbent of 1880.19, and far below every other
point evaluated this campaign. `sigma` has decreased only modestly
(1.8924 -> 1.8815 -> 1.5377), i.e. the search is still exploring at a
fairly broad scale, not prematurely converged, and a real improvement
appeared only after the mean had a couple of generations to adapt —
consistent with CMA-ES still learning the local covariance structure
rather than having stagnated. This is one new best point so far (not yet
confirmed against noise by repeated evaluation), but it stands out clearly
against ~50 other observations in this run, none of which come close.
**Decision:** Continue CMA-ES unchanged, same live `es` object (covariance
matrix, step size, and evolution path all retained from generations 1-3;
no restart).
**Hyperparameters:** Unchanged from the previous entry: `x0` was only used
to initialize generation 1 and is no longer relevant (the mean has since
moved, now implicitly tracking the eval-50 region); `sigma0=2.0` was only
the initial step size and `sigma` is now internally adapted to 1.5377;
`bounds=[-5,5]^5` and `seed=0` (already consumed, governs the RNG stream
going forward) unchanged; popsize=8 and all other pycma internals
unchanged. No evidence yet justifies a manual override of the adapted
`sigma` or any other internal parameter — this is exactly the kind of
"ordinary internal adaptation" the protocol treats as part of executing
the method, not a separate tuning decision.
**Reason:** The method just produced its first real improvement after 2
quiet generations; stopping or switching now, right after the first sign
of progress and with sigma still fairly large (i.e. still able to search
a meaningful neighborhood), would discard exactly the information that
justifies continuing. 46 evaluations remain, enough for several more
generations plus budget for an end-of-campaign noise check.
**Next step limit:** 24 additional objective calls, at most 46 remaining;
pause at cumulative evaluation 78. This covers 3 more generations
(popsize 8).
**Why this length:** Same reasoning as the previous CMA-ES block: a single
generation is too little to see whether the adaptation around the new
best region is productive, while 3 generations leaves roughly 22
evaluations afterward for either further refinement or a dedicated
noise-confirmation step (repeated evaluations at the best point) before
the budget runs out.
**Review:** At cumulative evaluation 78, inspect whether `y_best_so_far`
has improved further below 490.71, whether `sigma` is shrinking
(convergence toward a local optimum) or still large, and whether enough
budget will remain afterward to re-evaluate the final incumbent a couple
of times to sanity-check it against noise before the campaign ends. No
preselected next method.

## Decision after evaluation 78
**Budget:** 78 used / 100 total; 22 remaining.
**Evidence:** 3 more CMA-ES generations (evals 55-78) did not beat the
eval-50 incumbent (`y=490.708`), but several points landed close to it in
both `x`-space and `y`-value: eval 63 (`x=[-0.094, 0.443, 0.494, 2.810,
-1.187]`, `y=579.640`), eval 68 (`y=1104.222`), eval 62 (`y=1228.716`),
eval 61 (`y=1514.645`) — all near the incumbent's neighborhood
(small first/second/third coords, 4th coord around 2.7-3.3, 5th coord
negative), unlike the much larger values (5,000-177,000+) still produced
by points further away. This is consistent with a real local basin around
the incumbent rather than the eval-50 value being a lucky noisy outlier:
multiple nearby points independently landed in the same low range.
`sigma` continues to shrink gradually (1.4331 -> 1.4204 -> 1.3002),
i.e. the search is converging but not yet fully collapsed.
**Decision:** Run one more short CMA-ES block (same live `es` object,
state retained) to use most of the remaining search budget, then stop
exploring and spend a small reserved amount on repeated evaluations at
the final incumbent to sanity-check it against noise before the campaign
ends — since the task is explicitly noisy and a single observation
(eval 50's 490.708) should not be reported as "the" best value without
some confirmation that it is not an unusually low noisy draw.
**Hyperparameters:** CMA-ES settings unchanged (state carried over;
popsize=8, sigma internally at ~1.30 and continuing to adapt, bounds and
seed unchanged) — same rationale as the previous entry: this is ordinary
internal adaptation of a method still showing a plausible local basin,
not a point requiring a fresh tuning decision.
**Reason:** With only 22 evaluations left, committing all of them to
further CMA-ES generations risks ending the campaign on an unconfirmed
single noisy reading; reserving some for repeated measurement directly
addresses the stated noise characteristic of this task. 16 further search
evaluations (2 generations) is enough to let the still-shrinking `sigma`
either tighten the estimate further or reveal that this generation's
points are no longer improving, while leaving a meaningful block (6
evaluations) for confirmation.
**Next step limit:** 16 additional objective calls, at most 22 remaining;
pause at cumulative evaluation 94.
**Why this length:** 2 more generations (16 evals, popsize 8) is the
largest block that still leaves at least 6 evaluations in reserve for an
end-of-campaign noise-confirmation step, which I judge more valuable at
this stage than spending the full remaining budget on search alone, given
`sigma` is already fairly small (~1.3, down from 2.0 at restart) and
further gains are likely incremental.
**Review:** At cumulative evaluation 94, inspect whether a new best has
emerged, whether `sigma` has continued shrinking (supporting convergence),
and then move to a final confirmation step: repeated evaluations at
whichever point is the best-so-far at that time, to check its value is
not an outlier before reporting it and stopping.

## Decision after evaluation 94
**Budget:** 94 used / 100 total; 6 remaining.
**Evidence:** 2 further CMA-ES generations (evals 79-94) again failed to
beat `y=490.708` (incumbent since eval 50,
`x=[0.126, -0.342, 0.933, 2.779, -1.317]`); best values this block were
2051.51 and 2466.68, both still well above the incumbent. `sigma`
continued shrinking (1.1824 -> 1.0304), consistent with the search
tightening around the current basin without finding anything better,
i.e. diminishing returns from continued CMA-ES generations at this
budget. Only 6 evaluations remain — too few for another full generation
(popsize 8) and, more importantly, not enough to productively keep
searching versus directly addressing the task's stated noise
characteristic before reporting a final answer.
**Decision:** Stop optimizer search. Use the remaining 6 evaluations to
repeatedly evaluate the current incumbent `x=[0.126, -0.342, 0.933,
2.779, -1.317]` to check whether `y=490.708` (a single noisy observation)
reflects this point's typical value or was an unusually low draw, since
the task explicitly states repeat evaluations can differ and this is the
last chance to distinguish signal from noise before the budget is
exhausted.
**Hyperparameters:** Not applicable — no optimizer is active for this
step, only repeated calls to `harness.evaluate()` at a fixed `x`.
**Reason:** With 6 evaluations left, neither continuing CMA-ES (too few
for a meaningful next generation and already showing diminishing returns)
nor switching to a new method (no time to make a global method pay off)
is as valuable as directly verifying the reliability of the number that
will be reported as the campaign's result, given the task is noisy by
design.
**Next step limit:** 6 additional objective calls, at most 6 remaining;
pause at cumulative evaluation 100 (budget exhausted).
**Why this length:** This is the entire remaining budget; 6 repeated
readings give a usable (if small) sample to judge the point's mean and
spread without any being wasted, since the budget is exhausted exactly
at the end of this step regardless.
**Review:** Not applicable — this is the final step. After it, call
`harness.save()`, confirm `trace.csv`/`meta.json` exist, and report the
mean/spread of the 6 repeated readings alongside the single best observed
value, both from the harness's own `y_best_so_far` tracking.
