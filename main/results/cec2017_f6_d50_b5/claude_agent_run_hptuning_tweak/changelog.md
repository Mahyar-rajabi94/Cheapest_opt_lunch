## Decision after evaluation 16

**Budget:** 16 used / 100 total; 84 remaining.

**Evidence:** Sobol scan (seed=0, n=16) over the full `[-5,5]^50` box gave
`y` in `[621.15, 629.74]`, mean 624.73, std 2.57 (relative spread of
~0.4% of the mean). This is a strikingly narrow range for a 50-D box scan
and is consistent with concentration-of-measure for a landscape built from
many roughly-symmetric per-coordinate contributions (e.g. a quadratic-like
or sum-type objective), rather than a landscape with wildly different
basins across the box. A per-coordinate linear-correlation check against
`y` showed one coordinate at `|corr|≈0.76` and the rest much weaker (mean
`|corr|≈0.22`, noisy at n=16); not strong enough evidence of a dominant
linear term to commit a whole step to exploiting a single axis, but
consistent with (not contradicting) a quadratic-ish bowl where some
directions carry more curvature/weight than others. Best point so far:
`y_best = 621.15` at the Sobol point with coordinates including
`x0≈3.51, x1≈4.31` (near the box edges) and others more central.
No evidence of a highly rugged/discontinuous landscape at this sampling
density, and no reason yet to suspect multimodality severe enough to need
a multi-restart global strategy before trying a single local/population
search.

**Decision:** Switch to CMA-ES, built directly against the `cma` library
(not `src/optimizers/cma_es.CMAES`, whose wrapper hardcodes the box
midpoint as `x0`) so that the search can be seeded at the Sobol-identified
best point instead of the midpoint — a free improvement given data already
collected, with no extra evaluation cost. This is a fresh optimizer
instance; the Sobol points/values are retained in the overall trace and
inform this seeding choice but are not re-told to CMA-ES (CMA-ES's
update rule expects points it itself proposed per generation, not
arbitrary external points).

**Hyperparameters:** `x0 = x_best` from the Sobol step (coordinates noted
above) instead of the box midpoint (deliberate change, justified above).
`sigma0 = 2.5` = one quarter of the box width (`10/4`), the same standard
box-constrained convention documented in `src/optimizers/cma_es.py`
(inherited, unchanged). `bounds = [[-5]*50, [5]*50]` enforced via pycma's
own `bounds` option (inherited, unchanged). `popsize`: left at pycma's own
dimension-based default, `4 + floor(3*ln(50)) = 15` — not shrunk further,
because this formula is itself the library's tuned recommendation for
d=50, and reducing it below that without evidence would be an untested
change that risks destabilizing covariance adaptation rather than a
justified response to observed data. `seed = 0` for reproducibility.
Implementation/version: `cma` (pycma) as already pinned in this project's
venv/requirements.

**Reason:** The narrow, smoothly-varying `y` range across a full-box scan
suggests a landscape where a population-based method that adapts a search
distribution (mean + covariance) toward lower regions is a reasonable
next move, and is far more budget-efficient here than a quadratic-model
trust-region solver (BOBYQA) whose default model size (`2*dim+1=101`
points) would consume more than the entire remaining budget before taking
a single step. CMA-ES's population-based updates degrade gracefully with
few generations, unlike a solver that needs a complete initial model
before it can act at all.

**Next step limit:** 60 additional objective calls (4 full generations of
popsize 15), at most 84 remaining; pause at cumulative evaluation 76.

**Why this length:** 60 is an exact multiple of the population size so the
step ends on a clean generation boundary (no partially-told generation
left dangling across the checkpoint). 4 generations is enough to observe
whether the step-size/mean are moving meaningfully and whether the
population is converging, diverging, or stagnating, while deliberately
leaving 24 evaluations (not the full remaining budget) for a further
decision — continuing, restarting with adjusted sigma, or switching to a
local refinement step — informed by that evidence.

**Review:** At evaluation 76, inspect the generation-by-generation best/mean
objective values, whether `sigma` is shrinking (convergence) or growing
(divergence/exploration), and whether the running best has improved
materially versus the Sobol baseline (621.15). Pause sooner only if an
exception/instability occurs (e.g. numerical failure in the solver); no
other early-stop condition applies to a fixed generation count. No
commitment to a later method.

## Decision after evaluation 76

**Budget:** 76 used / 100 total; 24 remaining.

**Evidence:** Ran 4 CMA-ES generations (popsize 15, `x0` = Sobol best,
`sigma0=2.5`) as planned. Per-generation best `y`: 621.31, 622.09, 622.11,
618.91. Running best improved from 621.15 (Sobol baseline) to 618.91,
achieved in the 4th generation — real but modest progress (~0.36% of the
mean scale observed in the Sobol scan), consistent with the very low
generations-per-dimension ratio available under this budget (4 generations
at d=50 is far below what CMA-ES normally needs to converge). `sigma`
shrank steadily and monotonically across all 4 generations (2.500 → 2.375
→ 2.288 → 2.252 → 2.238), i.e. about 10% total reduction, indicating the
search is still contracting toward a region rather than stagnating or
diverging; generations 1–2 did not improve on generation 0's best, but
generation 3 did, so there is no persuasive evidence of a plateau (only 4
generations have run, and CMA-ES's per-generation best is expected to be
noisy even while the overall distribution improves).

**Decision:** Continue the same CMA-ES run (no restart), preserving its
full internal state (mean, covariance, sigma, evolution paths) — a restart
would discard the adaptation already built up over 4 generations for no
evidenced benefit, and switching to a different method now, with only 24
evaluations left, would spend a disproportionate share of the remainder on
a new method's own startup cost (e.g. any local solver's initial model, or
a fresh population's first generation) rather than on directed search.
Since 24 is not an exact multiple of the popsize (15), this final step
will ask for a partial batch of 24 points in one call
(`es.ask(number=24)`) rather than a full extra generation followed by a
wasted partial one; pycma's `ask`/`tell` support an explicit `number`
argument for exactly this case.

**Hyperparameters:** All unchanged from the previous step's configuration
(`sigma` and the mean/covariance are the evolved internal state, not
reconfigured settings); `bounds`, `seed=0` continuation, and `popsize=15`
(library default for d=50) are referenced as before, not restated.

**Reason:** Monotonic sigma shrinkage plus a new best in the most recent
generation together indicate the run is still making useful progress, and
with this little budget left, preserving momentum in an already-adapting
search outweighs the uncertain benefit of switching methods or restarting.

**Next step limit:** 24 additional objective calls (all of the remaining
budget), at most 24 remaining; pause at cumulative evaluation 100.

**Why this length:** This is deliberately the final step: 24 is small
enough that no realistic alternative split (e.g. reserving a few
evaluations for yet another method) would give a new method enough
evaluations to do anything useful (fewer than half a CMA-ES generation, or
a tiny fraction of a trust-region model's minimum size), so the full
remainder is committed to finishing out the current, still-improving run
rather than fragmenting it further.

**Review:** At evaluation 100 (budget exhaustion), inspect the final best
value versus 618.909604 and whether the partial-batch generation improved
on it, then stop (budget exhausted — no further evaluations possible
regardless of evidence). Not applicable beyond confirming the stopping
record, since no method choice follows this step.

## Decision after evaluation 100 (final)

**Budget:** 100 used / 100 total; 0 remaining.

**Evidence:** The final partial CMA-ES batch (24 points, same evolved
mean/covariance/sigma from the prior 4 generations, asked via
`es.ask(number=24)`) was evaluated. Its best value was 619.106287,
slightly worse than the 618.909604 recorded at evaluation 76 (generation
3), so the running best was not improved in this final batch; the overall
campaign best remains **618.909604**, found during generation 3 of the
CMA-ES run (cumulative evaluation ≤76). This is consistent with the
previous step's evidence of real but slow, noisy progress: not every
batch improves on the best, especially this late with `sigma` still
around 2.2 relative to a box half-width of 5 — the search had not fully
converged within the available budget, so some batch-to-batch variation
in the best observed value is expected.

**Decision:** Stop. The budget is exhausted — evaluation 100 was the last
permitted `evaluate()` call, and a further call would raise
`BudgetExhausted`. `harness.save()` was called explicitly (in addition to
the harness's own automatic save triggered by the budget-exhausting call)
to write the final `trace.csv` and `meta.json`.

**Hyperparameters:** Not applicable — no further evaluations.

**Reason:** Hard budget constraint; stopping is the required outcome, not
a discretionary choice, once evaluation 100 is reached.

**Next step limit:** 0 (budget exhausted; no further objective calls are
permitted).

**Why this length:** Not applicable.

**Review:** Not applicable — campaign complete. Final result: best
observed objective value 618.909604 (minimization), at 100/100
evaluations used (none held back; the run was not stopped early).
