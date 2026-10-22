## Decision after evaluation 12

**Budget:** 12 used / 100 total; 88 remaining.

**Evidence:** The initial 12-point scrambled-Sobol design (GP-BO's own
`n_init = 2*(dim+1) = 12` rule) produced y values:
`[445996.25, 39389.47, 11484853.87, 3759544.85, 243601.67, 501334.28,
209864.73, 1625920.25, 401543.99, 18634.24, 338108.59, 2351.75]`.
Min 2351.75, max 11,484,853.87, mean ~1.59e6, std ~3.15e6 (std exceeds the
mean, confirming a heavily right-skewed/heavy-tailed distribution driven by
a few very large values, e.g. the 11.48M and 3.76M points). All 12 values
are strictly positive. The current incumbent is x=[0.133, -1.589, -0.859,
0.693, 3.985], y=2351.75. No point has been replicated yet, so I cannot yet
separate observation noise from landscape shape at this incumbent, but the
smallest observed values (2351.75, 18634.24, 39389.47) are far enough below
the bulk of the distribution that this is very unlikely to be pure noise
around a much larger true value — worth exploiting via acquisition next.
Space-filling, independent points carry no mid-batch feedback, so there was
no condition to stop this step earlier than planned.

**Decision:** Continue with GP-BO's acquisition-driven phase, but fit the GP
on log(y) instead of raw y, via a campaign-local wrapper
(`_work/log_gpbo.py: LogGPBO`) that delegates `ask()`/the acquisition
machinery unchanged to the existing `GPBO` class and only transforms the
value passed to its internal `tell()`. State transfer: the same harness
instance and its budget counter are kept untouched (12/100 used, carried
forward in-process); a new `LogGPBO` is constructed and immediately given
all 12 already-collected (X, y) pairs via one `tell()` call, so no
observations are discarded, and the fresh object just starts life informed
by the same history the raw `GPBO` already had — this is a deliberate
restart of the *optimizer wrapper object* (to change how it treats y), not
of the search.

**Hyperparameters:**
- GP: `SingleTaskGP`, BoTorch's own default kernel/priors (inherited
  default, unchanged from `gp_bo.py`'s own configuration, v. as installed in
  this project's venv).
- Acquisition: analytic `LogExpectedImprovement`, `maximize=False`
  (inherited default from `gp_bo.py`, now operating on log-space y).
- Acquisition optimizer: `optimize_acqf` with `num_restarts=10`,
  `raw_samples=512` (inherited defaults from `gp_bo.py`, unchanged).
- New: `y -> log(max(y, 1e-9))` transform before every `tell()` to the
  underlying GP (new, campaign-local; rationale above). `eps=1e-9` is a
  floor only relevant if a future y is <= 0, which has not occurred yet in
  any of the 12 samples; monotonic so it never changes which point is best.
- `n_init` is no longer relevant going forward (already satisfied); from
  here `GPBO.ask()` returns acquisition-optimized points.

**Reason:** The GP surrogate underlying LogEI assumes roughly
well-behaved, non-heavy-tailed residuals; the observed 4-orders-of-magnitude,
right-skewed spread (a couple of points over 1e6 against a minimum near
2.3e3) risks letting a few extreme values dominate the fitted length-scales
and noise term, degrading the model's resolution in the more interesting,
lower-value region where the next acquisition points should concentrate.
A monotonic log transform is a standard, low-risk way to stabilize this
(no information is discarded, ranking is preserved, no change to which x is
currently best), and all 12 observed values are positive so the transform is
well-defined. Given the small total budget (100), I would rather make this
adjustment once, deliberately, right after seeing this evidence rather than
later discover the GP's acquisitions are being misled by the scale of the
tail.

**Next step limit:** 20 additional objective calls (at most 88 remaining);
pause at cumulative evaluation 32.

**Why this length:** Each GP-BO acquisition step requires a fresh GP fit
and an `optimize_acqf` call (num_restarts=10, raw_samples=512) and only
proposes one point at a time (GPBO does not support batches), so a step of
20 gives the acquisition loop enough sequential refinements to meaningfully
move off the Sobol baseline and show a real trend (improving, plateauing, or
not), while still leaving 68 evaluations (most of the remaining budget) for
further adaptation after reviewing the outcome. Given the budget is small
overall, I don't want to commit a much larger fraction before the next
checkpoint.

**Review:** At evaluation 32, I will check: whether the running best has
improved materially below 2351.75 and by how much; whether consecutive
acquisition points are clustering (model converging on a region) or still
scattered (model still exploring); and whether the log-transform appears to
be helping (e.g. acquisition points landing in plausible, non-extreme
regions rather than repeating the Sobol step's wild swings). No later method
is pre-selected; I may continue GP-BO, revisit the transform, add
replication at the incumbent to check for noise, or switch method entirely,
depending on what this step shows.

## Decision after evaluation 32

**Budget:** 32 used / 100 total; 68 remaining.

**Evidence:** Over the 20 log-space GP-BO acquisition points (evals 13-32),
the running best **did not improve at all**: it stayed at 2351.75 (set by
the Sobol initial design). This step's y values ranged 6520.17 to
13,565,752.98 (mean ~1.40e6) -- essentially the same heavy-tailed character
as the init design, despite the log transform. More tellingly, the later
acquisition points (evals 27-32) were driven to the box boundary
(coordinates at exactly +-5 in most dimensions), indicating the acquisition
optimizer was choosing high-*uncertainty* exploration over exploitation
near the incumbent -- the opposite of what I wanted LogEI to do once enough
data was available. Taken together with the still-wild y range, the log
transform did not fix the underlying issue: this objective looks
multi-modal/non-smooth enough on a global scale that a single global GP is
spending its evaluations testing corners rather than refining the best
region.

Looking at the 10 best points among all 32 by y, most form a geometrically
coherent cluster (x2 roughly in [-3.6,-0.8], x3 in [-3.9,-0.2], x4 in
[0.6,3.68], x5 in [3.5,5.0], x1 fairly free in [-2.5,2.5]) with y ranging
2351.75-50707.09: evals with y={2351.75, 6520.17, 6966.59, 7735.25,
39076.11, 41352.96, 43987.29, 50707.09}. Three additional low-ish values
(18634.24, 39389.47, 39620.36) sit far outside this cluster (near opposite
corners / large |x1|); given the function is noisy and the overall y-scale
swings by orders of magnitude even between nearby points elsewhere, I treat
these three as either noise or a separate, unconfirmed region -- not strong
enough evidence to split the remaining (tight) budget chasing them right
now, but noted in case later evidence revives interest.

**Decision:** Switch method to **TuRBO** (`src/optimizers/turbo.py`), a
local trust-region BO method, specifically to stop wasting evaluations on
global boundary exploration and instead refine the one clearly-better
cluster found so far. Rather than spending a fresh `n_init = 2*dim = 10`
Sobol evaluations (as TuRBO normally would), I am seeding its trust region
directly from the 8 already-collected, geometrically coherent cluster points
above -- no new evaluations are spent on re-initialization, which matters
given the small remaining budget. This is a deliberate restart of the
optimizer object (TuRBO keeps no history of its own yet), implemented via a
campaign-local adapter: after constructing `TuRBO(bounds=harness.bounds,
seed=2)`, I directly set its internal local-trust-region state
(`_local_X_unit`, `_local_y`, `_local_best_y`, `_phase="evolve"`,
`_length=0.8` (its own `_length_init`), fresh success/failure counters) from
those 8 points, instead of letting it run its own Sobol init phase. The
harness instance and its budget counter (32/100 used) are untouched and
carried forward in-process. Observations from the discarded GP-BO run are
not lost -- the harness's own trace/best-so-far already reflect them, and
they remain in `state.pkl`'s history for context -- only the *local* GP used
by the new trust-region loop starts from the seeded 8 points, not the full
32 (by design -- that's what makes it a local method).

**Hyperparameters:** TuRBO's own standard, no-tuning defaults, all inherited
from `src/optimizers/turbo.py` (documented in its module docstring):
batch_size=4, n_candidates=min(100*dim,5000)=500, length_init=0.8,
length_min=0.5**7, length_max=1.6, success_tolerance=10,
failure_tolerance=ceil(max(4/4, 5/4))=2, local GP = SingleTaskGP fit on
-y (since MaxPosteriorSampling maximizes) with Normalize/Standardize
transforms. No changes to these; they are not being newly tuned, just
adopted as implemented. The only non-default element is the seeding
described above (not a hyperparameter of TuRBO itself, but how its state is
initialized for this run).

**Reason:** The evidence from the last two steps is specific: global GP-BO
(raw and log-space) is choosing boundary/global exploration while a clear,
coherent best region already exists in the data. A trust-region method is
designed exactly for this situation -- concentrate the search budget in a
box around the current local best, with an adaptive radius, rather than
letting a global acquisition function keep probing corners. Seeding from
existing cluster data (vs. a fresh 10-point Sobol init) preserves the small
remaining budget for actual trust-region refinement instead of paying for
information we already have.

**Next step limit:** 24 additional objective calls (at most 68 remaining);
pause at cumulative evaluation 56.

**Why this length:** TuRBO proposes/updates in batches of 4 and only
re-evaluates its trust-region length after each full batch; with
failure_tolerance=2 and success_tolerance=10, 24 evaluations = 6 batches is
enough to see at least one full length-adjustment cycle (e.g. 2 consecutive
failing batches would already halve the trust region once) while leaving
44 evaluations (most of the remaining budget) for further adaptation
depending on what this block shows. I am avoiding a longer commitment given
how multi-modal the function has already shown itself to be.

**Review:** At evaluation 56 I will check: whether the running best
improved below 2351.75 and by how much; how the trust-region length
`_length` evolved (expanded/contracted/restarted) as a sign of whether the
local region is fruitful; and whether batches are landing inside the box
(progressing) or repeatedly failing (signal to reconsider -- e.g. revisit
the three distant low-value outliers, try a different local method, or
allocate a small amount of budget to confirm/deny noise at the incumbent).
No later method is pre-committed.

## Decision after evaluation 56

**Budget:** 56 used / 100 total; 44 remaining.

**Evidence:** Over the 24-evaluation TuRBO block (seeded trust region, evals
33-56), the running best did **not** improve below the incumbent 2351.75,
but the trajectory is informative: early batches (evals 33-46) still landed
on very large values (up to 14.8M) and the trust region shrank steadily
(length 0.8 -> 0.4 -> 0.2 -> 0.1 across repeated failing batches, consistent
with `failure_tolerance=2`), showing the seeded box was initially still too
big / not centered tightly enough on the true local structure. But the last
few evaluations (54-56) landed close to the incumbent in both x-space
(x~[0.14-0.17, -1.4 to -2.1, 1.2-3.6, 1.9-2.1, 3.6-4.4], i.e. near
[0.133,-1.589,-0.859,0.693,3.985]-ish, modulo some dims still drifting) and
in y: 442127.98 (eval54, an outlier that doesn't fit the pattern -- evidence
the landscape is not fully smooth even in this refined box), then 2727.37
(eval55) and 3121.83 (eval56) -- both within ~15-33% of the incumbent
2351.75. This is the first time multiple points this close to the incumbent
value have been produced, which is useful evidence about **noise
magnitude** at this scale: three observations (2351.75, 2727.37, 3121.83) at
similar-but-not-identical x cluster within roughly a 770-unit range,
suggesting the combination of residual x-sensitivity and observation noise
at this scale is on the order of a few hundred to ~1000 units -- i.e. the
incumbent's 2351.75 is not wildly more special than nearby points, and a
single-point "best" here should be trusted only loosely.

**Decision:** Continue TuRBO unchanged, same object/state (trust region
currently length=0.1, phase="evolve", carried over in-process with the same
harness and its 56/100 budget counter -- no restart). The method is doing
what it's supposed to: shrinking toward a smaller, increasingly specific
region as it fails to find improvements at the current scale, and the most
recent evaluations are the closest to the incumbent seen since the original
Sobol design. I want to let it continue this refinement rather than switch
again, since switching before seeing whether this trust region's narrowing
converges or needs a restart would throw away the adaptation already in
progress.

**Hyperparameters:** Unchanged from the previous TuRBO configuration
(batch_size=4, length_max=1.6, length_min=0.5**7, success_tolerance=10,
failure_tolerance=2, n_candidates=500) -- see decision after evaluation 32
for the full inherited-default list. No new tuning; this is the same
running optimizer object, so "continuing" here literally means more
ask/tell calls on it, with its own internal length-adaptation (ordinary
internal adaptation, not a new agent decision).

**Reason:** The step's second half showed the clearest convergence signal
of the whole campaign so far (three low, mutually-close y values near the
global incumbent). Given the overall evidence of strong multi-modality /
heavy-tailed response elsewhere in the domain, and that the trust region is
actively and automatically protecting against wasting budget on bad
batches (via its own shrink/restart logic), continuing is the action most
likely to either confirm convergence to a good local optimum or trigger an
informative restart if the region is exhausted (length < length_min =
0.0078125, auto-handled by TuRBO itself without a separate decision).

**Next step limit:** 20 additional objective calls (at most 44 remaining);
pause at cumulative evaluation 76.

**Why this length:** 20 = 5 more batches of 4, enough to either see the
trust region bottom out / restart (length_min is close, at 0.1 only ~4 more
halvings away) or see continued convergence close to the incumbent, while
preserving a meaningful remainder (24 evaluations) for a final
confirmation/refinement step afterward, including possibly replicating the
incumbent a few times near the very end to get a cleaner noise-adjusted
estimate of the true best value.

**Review:** At evaluation 76 I will check: whether the running best improved
below 2351.75; how many times the trust region restarted and whether
restarts keep landing in the same neighborhood (reinforcing confidence this
is a real optimum) or scatter elsewhere (suggesting the earlier cluster was
partly a noise artifact); and whether enough of the recent batch is tightly
clustered near the incumbent to justify spending some of the final budget on
pure replication of the incumbent for a noise-robust estimate rather than
further exploration. No later method is pre-committed.

## Decision after evaluation 76

**Budget:** 76 used / 100 total; 24 remaining.

**Evidence:** Over the last 20 TuRBO evaluations (57-76), the trust region
kept shrinking without ever being reset by a restart (0.1 -> 0.05 -> 0.025,
all via ordinary failure-driven halving; `_phase` stayed "evolve" the whole
time). It is now centered tightly on x~[0.11-0.16, -1.0 to -1.9, -0.68 to
-0.9, 0.25-0.8, 3.7-4.6] -- clearly in the same neighborhood as the
incumbent [0.133,-1.589,-0.859,0.693,3.985]. But within this now-small box
(length 0.025 of the unit cube, i.e. roughly +-0.1 absolute per dimension
around the center after lengthscale weighting), y still swung from 2604.36
(eval68, very close to the incumbent) up to 5,086,455.95 (eval59) and
2,948,803.81 (eval70) -- a >1000x range **inside a box that has already
shrunk to a small fraction of the domain**. This is importantly different
from ordinary per-point observation noise: noise alone would not explain
1000x swings between evaluations whose x-vectors differ by only a few
hundredths to a few tenths per coordinate. The more consistent explanation
is that this objective has extremely high local sensitivity / curvature in
at least one direction (an ill-conditioned or narrow-valley-like local
structure), so even tiny coordinate changes away from a precise ridge can
blow the value up by orders of magnitude -- on top of whatever genuine
observation noise is also present. The running best is still 2351.75,
unchanged since eval 11/12 of the initial Sobol design, despite 65
additional evaluations concentrated in its neighborhood by two different
methods (log-GP-BO near it, then TuRBO directly centered there). The
clustering of several of the lowest values seen in this whole campaign
(2351.75, 2604.36, 2727.37, 2948.80 [point], 3121.83, 3503.64) all in this
same small neighborhood is consistent evidence that this region does
contain a true local optimum; the large excursions are consistent with high
sensitivity along some other direction(s) that the isotropic/GP-driven
Thompson sampling keeps occasionally hitting.

**Decision:** Stop further exploratory search and switch to a short,
**verification + fine local exploitation** step with the remaining budget:
(1) first, re-evaluate the current incumbent x=[0.133,-1.589,-0.859,0.693,
3.985] itself 5 times (new, independent `evaluate()` calls at the exact same
x) to get a noise-robust estimate of its true value, since with a noisy
objective a single observation (2351.75) is not reliable enough to be
reported as "the" answer; (2) use the remaining ~19 evaluations for a small
local random-perturbation search (uniform jitter of a few small fixed radii
along each coordinate independently, centered on the incumbent, clipped to
bounds) to probe immediately around it for any further improvement, given
the evidence above that this neighborhood is both promising and locally
very sensitive, so a tiny, targeted probe is more informative here than
continuing TuRBO's broader (and already-shrunk) trust-region sampling. This
is a deliberate method switch away from TuRBO (which has had 44
evaluations in this neighborhood without improving past the incumbent) and
does not reuse TuRBO's state; it is a fresh, simple procedure I will drive
directly (no ask/tell optimizer object needed). The harness and its 76/100
counter are carried forward unchanged.

**Hyperparameters:** Replication count = 5 (new): enough to get a usable
sample mean/spread at the incumbent without spending a large share of the
remaining 24 calls, chosen given the noise/sensitivity magnitude observed
above (hundreds-to-thousands scale) -- 5 samples gives a standard error
roughly sqrt(5)~2.2x smaller than a single sample, a meaningful improvement
for this budget cost. Local perturbation radii (new): three small relative
step sizes per coordinate -- 0.02, 0.05, and 0.15 (absolute units, versus a
domain half-width of 5), reflecting the observed sensitivity scale (TuRBO's
final box had length 0.025 of the unit cube = 0.25 absolute full-width, so
0.02-0.15 probes both inside and somewhat beyond that most-recent box);
one perturbed point per coordinate direction per radius, i.e. up to
5 dims x 2 directions x up to 2 radii, capped by the remaining budget (19
calls after replication), so not every combination will be tried --
coordinates/radii will be prioritized by remaining budget as the step
executes, and this allocation will be recorded in the final summary rather
than fixed in advance since it depends on exactly how many calls remain
after replication.

**Reason:** With only 24 evaluations left and strong evidence that (a) a
promising neighborhood has been found, (b) the function is noisy enough
that the single best observed value could be a lucky draw, and (c) the
local landscape is sensitive enough that broad trust-region sampling keeps
overshooting, the most useful use of the remaining budget is direct noise
characterization at the incumbent plus a cheap, targeted local probe --
rather than continuing an optimizer (TuRBO) that has already spent 44
evaluations in this exact neighborhood without net improvement, or
restarting a new global method that could not converge meaningfully in <24
evaluations in 5D given the evidence of strong multi-modality elsewhere.

**Next step limit:** 24 additional objective calls (all of the remaining
budget); pause at cumulative evaluation 100 (budget exhaustion).

**Why this length:** This is the final step: 24 is all that remains, and
given the budget is almost gone, there is no benefit to reserving further
evaluations for a subsequent checkpoint -- the plan already splits this
block internally (5 replication + up to 19 local perturbation) and will be
executed as such, but since nothing remains after it, this is recorded as
the final step.

**Review:** Not applicable for a further method decision (this is the final
step). I will inspect the 5 replication values and the local-perturbation
results to report the final best value, its noise-adjusted context, and
whether the perturbation step found anything better than the original
incumbent, then call `harness.save()` and confirm `trace.csv`/`meta.json`.

## Final outcome (evaluation 100, budget exhausted)

**Budget:** 100 used / 100 total; 0 remaining.

**Replication result (evals 77-81, 5 calls at the exact incumbent x
[0.1328, -1.5890, -0.8585, 0.6934, 3.9854]):** y = {35206.42, 12609.79,
4634939.18, 1183240.27, 476390.60}; mean ~1,268,477, std ~1,735,851, min
12609.79, max 4,634,939.18. This is the single most important finding of
the whole campaign: evaluating the *identical* point 5 times in a row
produced values spanning more than two orders of magnitude, with none of
the 5 replicates coming anywhere near the previously recorded 2351.75 at
that same x. This rules out my earlier (evaluation-76) hypothesis that the
huge value swings seen inside TuRBO's shrunk trust region were mainly due
to extreme local curvature/sensitivity to x -- the evidence now shows most
of that variance is **observation noise itself**, and it is large,
heavy-tailed, and not small/additive: a given x can return anything from
~1e4 to several million on different calls. This also means the original
incumbent value of 2351.75 was very likely an unusually lucky low noise
draw rather than a reliable estimate of this x's true expected value, and
by the same logic, the final reported minimum below is best read as "the
lowest single noisy draw observed in 100 calls," not a noise-free optimum.

**Local perturbation result (evals 82-100, 19 calls, small coordinate-wise
jitters of radius 0.02/0.05/0.15 around the incumbent):** despite the
now-confirmed heavy noise, the running best improved twice during this
block: eval 88 (x4 perturbed by -0.05) gave y=2344.08, and eval 100 (x0
perturbed by -0.02) gave y=1857.42, the final campaign best. Most other
perturbed points, like the replicates, landed anywhere from ~1900 to ~3.0e6,
consistent with the same heavy noise rather than a clean local gradient.
Given the replication evidence, I do not read these two "improvements" as
proof that x0=0.113 or x4=0.643 are truly better than the original
incumbent's coordinates -- they are at least partly, possibly mostly, more
lucky low noise draws. With 0 evaluations remaining there is no budget left
to disambiguate further (e.g. by replicating these new points too).

**Decision:** Stop -- budget exhausted (100/100). `harness.save()` was
called (also triggered automatically on the final `evaluate()`); `trace.csv`
(100 rows) and `meta.json` confirmed present in `results/agent_run/`.

**Final best observed value:** y = 1857.4239589114316, at
x = [0.1128, -1.5890, -0.8585, 0.6934, 3.9854] (eval 100). Given the
confirmed noise magnitude, this should be reported with the caveat that it
is a single noisy draw in a region that, across ~15 nearby/replicate
observations collected over the campaign, has repeatedly shown values
anywhere from ~1.9e3 to several million; the region itself (roughly
x0 in [0.1, 0.18], x1 in [-1.9, -1.0], x2 in [-1.0, -0.7], x3 in [0.5, 0.8],
x4 in [3.7, 4.8]) is reasonably well-evidenced as better than the rest of
the explored domain (where typical values were 1e5-1e7), even though the
exact minimum within it could not be pinned down more precisely with the
noise level observed and the remaining budget.

