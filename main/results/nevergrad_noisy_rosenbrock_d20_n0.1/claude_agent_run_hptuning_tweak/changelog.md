## Decision after evaluation 19
**Budget:** 19 used / 100 total; 81 remaining.
**Evidence:** 16 Sobol points gave y in [649303, 2158539] (mean 1.218e6, std
4.279e5, i.e. a ~35% relative spread across the box) — a clear, usable
signal rather than noise-dominated flatness. The best of the 16 was
point index 10, y=649303. The dedicated 4-draw noise probe at a fixed x
gave values {2158539, 2134317, 2156743, 2167204}: mean 2.1542e6, sample
std 14021, i.e. a ~0.65% relative noise level — roughly 50x smaller than
the across-space signal spread. This confirms the "mild" noise
characterization: individual evaluate() calls are informative on their
own, no heavy averaging/replication is needed per candidate, but naive
finite-difference gradients using very small step sizes could still be
swamped by this noise floor.
**Decision:** Switch to CMA-ES, restarted fresh (no prior optimizer state
to preserve — Sobol's internal sequence state is not reused). Rather than
the project's CMAES wrapper's default initialization (box midpoint), I
use a campaign-local adapter that calls pycma's
`cma.CMAEvolutionStrategy` directly with the same options the wrapper
would use (box bounds constraint, verbose=-9) but with `x0` set to the
best Sobol point found (index 10) instead of the box midpoint, since we
now have evidence this region is comparatively good and warm-starting
there should be a strictly better use of the very limited remaining
budget. This is a supported use of the underlying library's own
constructor API (the wrapper exposes no x0 parameter, so per protocol
item 3 I go to the underlying library directly) and does not touch the
harness, its budget counter, or problem constraints.
**Hyperparameters:**
- `x0` = best Sobol point (coordinates from eval index corresponding to
  the y=649303 point) — changed from the wrapper's box-midpoint default,
  justified above.
- `sigma0` = 2.5 (one quarter of the box width 10, matching the project
  CMAES wrapper's own standard box-constrained recommendation) — kept at
  this inherited default since nothing yet suggests the useful length
  scale is smaller; shrinking it now would be guessing without evidence.
- `popsize` (lambda) = pycma's own default for dim=20, i.e.
  `4 + floor(3*ln(20))` = 13 — kept at the library default (pycma
  0.x, as pinned in requirements.txt/.venv) since no evidence yet argues
  for a non-default population size, and a larger population would cost
  more generations than this budget can afford.
- `bounds` = [-5,5]^20 (hard box constraint, enforced by pycma's own
  `bounds` option, matching the wrapper).
- `seed` = 0, for reproducibility (not a quality lever).
- No special noise handling (e.g. UH/noise-handling options) enabled:
  CMA-ES's rank-based selection is already reasonably robust to the
  ~0.65% relative noise level measured above, and turning on
  uncertainty-handling machinery would consume extra evaluations per
  decision that this budget cannot spare.
**Reason:** CMA-ES needs no large fixed up-front initial-design cost
(unlike GP-BO's 42-point or TuRBO's 40-point standard initial designs,
which alone would consume ~50% of the *entire* 100-evaluation budget,
let alone the 81 remaining) and adapts its search covariance using
ordinary population-based updates, which suits a tight budget (81
evaluations left) better at this dimension (20). Warm-starting from the
best Sobol point uses the evidence just collected rather than discarding
it.
**Next step limit:** 65 additional objective calls (at most 81
available); pause at cumulative evaluation 19 + 65 = 84.
**Why this length:** 65 evaluations at popsize=13 is 5 complete
generations, enough to observe a real adaptation trend (step-size and
covariance changes, whether the population mean is still improving) while
deliberately leaving ~16 evaluations in reserve for a further checkpoint
rather than committing the rest of the entire budget to one unreviewed
block.
**Review:** At the next checkpoint, inspect the per-generation best/mean y
trend, whether sigma is still shrinking/growing sensibly, and the overall
best y versus the 19-evaluation baseline (649303). Pause sooner only if
`BudgetExhausted` is hit (should not occur, 84 < 100) or if CMA-ES
diverges obviously (e.g. best y get worse than the Sobol baseline over a
full generation, suggesting a bug) — no such condition is expected but
is checked for. No method is pre-committed for after this step.

## Decision after evaluation 79
**Budget:** 79 used / 100 total; 21 remaining.
**Evidence:** 5 CMA-ES generations (60 evals, popsize 12 — pycma's own
default for dim=20, `4 + floor(3*ln(20)) = 12`, not 13 as I'd estimated
in the prior entry; corrected here) ran: gen bests
616413, 485472, 139587, 100847, 187983 (mean per generation
928451, 1.066e6, 730108, 432108, 404546). Overall best-so-far dropped
from 649303 (end of step 1) to 100847 — a >6x improvement — reached at
generation 4. Generation 5's best (187983) was worse than generation 4's,
but its mean (404546) kept falling, and sigma has stayed essentially
flat (2.34 -> 2.24 -> 2.26 -> 2.31 -> 2.32), i.e. the step size has not
started shrinking yet: CMA-ES is still in an active search/adaptation
phase, not a converged/plateaued one. A single worse generation-best is
expected stochastic variation (consistent with the mild measured noise
and the population's own spread) and does not indicate the method has
stopped being useful.
**Decision:** Continue the same CMA-ES run (same `es` object: same
adapted mean, covariance/step-size state, and evaluation history — no
restart), for exactly one more full generation. A full generation is
used (rather than a partial one) because the underlying `cma` library's
`tell()` expects the complete set of points it just `ask()`-ed; telling
a partial population would either block the update or require discarding
points, neither of which is a good use of scarce remaining evaluations.
**Hyperparameters:** Unchanged from the previous entry (same `es`
instance: popsize 12, bounds, seed/RNG state as already advanced,
sigma/covariance as currently adapted) — not applicable to restate, per
protocol, since nothing is being changed.
**Reason:** The still-flat sigma and still-falling generation means
indicate the population has not exhausted what it can learn about the
local landscape; one more generation is cheap (12 of the 21 remaining
evaluations) and directly continues a run that is still actively
improving, which is a better use of budget than abandoning it mid-adaptation.
**Next step limit:** 12 additional objective calls (at most 21
available); pause at cumulative evaluation 79 + 12 = 91.
**Why this length:** Exactly one more full generation — the minimum unit
CMA-ES can usefully consume and report back on with this implementation.
Taking more than one generation now (e.g. committing to 2) would leave no
room to reassess given only 21 evaluations remain in total; taking a
partial generation would not produce a usable `tell()` update.
**Review:** At the next checkpoint, inspect generation 6's best/mean
against the generation 1-5 trend and the current best-so-far (100847),
and whether sigma starts shrinking (indicating convergence) or not.
With only 9 evaluations left after this step, also decide then whether a
further full CMA-ES generation is affordable/justified or whether a
cheaper local exploitation move around the current best point is a
better use of the final budget. No method is pre-committed.

## Decision after evaluation 91
**Budget:** 91 used / 100 total; 9 remaining.
**Evidence:** Generation 6 (12 evals, same `es` state as before): best
181409, mean 435493, sigma 2.349 — essentially unchanged from generation
5 (sigma 2.32) and still worse than generation 4's best of 100847, which
remains the overall best-so-far. Generation bests across the whole
CMA-ES run: 616413, 485472, 139587, 100847, 187983, 181409 — two
consecutive generations (5 and 6) have now failed to beat the
generation-4 incumbent, and sigma has not begun shrinking, meaning the
population is still spread at roughly its starting scale without having
relocated to a better region in the last two generations. With only 9
evaluations left, a further CMA-ES generation is not even possible
(popsize 12 > 9 remaining), so continuing that method unmodified is not
an option regardless of this evidence.
**Decision:** Stop using population-based CMA-ES generations (budget
cannot support another full generation) and switch, for this final
block, to a direct local-exploitation probe around the best point found
so far (`es.result.xbest`, y=100847): evaluate 9 random Gaussian
perturbations of that point (not drawn through any packaged
optimizer/ask-tell object — a minimal campaign-local method, since no
packaged method here fits a 9-evaluation budget), and keep whichever
evaluated point (including the incumbent, implicitly, via
`y_best_so_far`) is best. This is the last step: all 9 remaining
evaluations are used here since no further checkpoint will follow.
**Hyperparameters:**
- Perturbation distribution: isotropic Gaussian, `x_candidate = x_best +
  N(0, s^2 I)`, clipped to the [-5,5] box.
- Step scale `s` = 0.5 — a new, explicitly chosen value (not inherited):
  set smaller than CMA-ES's current sigma (~2.35) because the last two
  generations' failure to improve at that scale suggests the immediate
  neighborhood of the incumbent, at a finer scale than CMA-ES was still
  sampling at, is the more promising place left to probe with so few
  evaluations; set larger than the measured noise floor (0.65% relative,
  i.e. far smaller than 0.5 in these coordinates) so candidate
  differences will not be swamped by noise.
- 9 independent draws, one evaluation each, no replication per candidate:
  with only 9 calls left, spending any of them on repeat-confirmation of
  a single candidate was judged a worse use of budget than covering more
  distinct nearby points, given the noise is small relative to the y
  differences seen so far (100847 vs. the 180-700k range of recent
  generations).
**Reason:** This directly follows from the evidence: CMA-ES's global
population search has stalled for two generations at its current scale,
the remaining budget is too small to run another full CMA-ES generation,
and a cheap local neighborhood search around the current best point is a
reasonable way to spend the last 9 evaluations looking for a finer-scale
improvement without any method-level overhead.
**Next step limit:** 9 additional objective calls (all of the remaining
budget); pause at cumulative evaluation 91 + 9 = 100 (full budget).
**Why this length:** This is the final step — all remaining budget is
used since no further decision checkpoint will follow it.
**Review:** Not applicable — this is the stopping step. After these 9
evaluations the budget is exhausted; `harness.save()` will be called (the
harness also calls it automatically at budget exhaustion) and the final
best-so-far reported.
