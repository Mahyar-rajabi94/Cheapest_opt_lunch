# Roadmap (pre-registered, before any evaluate() call)

## Problem as stated
- dim = 20, bounds = [-5, 5]^20, direction = minimize
- noise: present, mild (roughly multiplicative-scale, not heavy-tailed) —
  a single evaluate() call is a noisy point estimate of the true value
- budget = 100 evaluate() calls total (hard, harness-enforced)
- sealed black box; no attempt will be made to identify the function

## Key constraint shaping this plan
100 evaluations for a 20-dimensional space is extremely tight (5x dim,
versus the ~2000-eval / ~200x-dim convention used elsewhere in this
project). This rules out anything that relies on many generations of
adaptation (e.g. plain CMA-ES, whose default population size at dim=20 is
~12-13, giving only ~7-8 generations — not enough for its covariance
matrix to meaningfully adapt) or on dense local probing (finite-difference
L-BFGS-B costs ~dim+1 = 21 evaluations per gradient step, i.e. at most
~4-5 steps total; BOBYQA/Nelder-Mead build a local model/simplex that
needs a comparable number of points before any real progress). All of
these also degrade in the presence of noise: finite-difference gradients
and small-simplex moves are both sensitive to evaluation noise swamping
small finite differences.

Model-based Bayesian optimization is the standard choice when evaluations
are this scarce, because it uses *all* data to build a global surrogate
instead of relying only on local comparisons, and a GP surrogate
explicitly models/averages out observation noise (it fits a noise
variance rather than treating each y as exact), which directly helps with
the stated mild noise. Plain GP-BO (this project's `gp_bo.py`) is a
reasonable baseline, but TuRBO (`turbo.py`) is the better-matched choice
here: it was specifically designed (Eriksson et al. 2019) for the regime
where vanilla GP-BO's global acquisition surface becomes unreliable as
dimension grows past roughly 10-20, by concentrating the GP model and
search inside a local trust region instead of fitting one GP over the
whole 20-D box with only ~100 points. This is a property of the method's
design (a GP's global surrogate quality degrades with many fewer points
than the ~exponentially-growing volume of a 20-D box as dimension grows),
not something inferred from running it on this task.

## Plan
**Single primary method: TuRBO-1, standard no-tuning defaults, for the
full 100-evaluation budget.**

- Initial design: TuRBO's own standard initial design, n_init = 2*dim =
  40 Sobol points (library default, not hand-tuned down), to seed the
  first trust region's GP.
- Main phase: remaining 60 evaluations as 15 Thompson-sampling batches of
  4 inside the adaptive trust region (library defaults: length_init=0.8,
  doubling after 10 consecutive successful batches, halving after
  failure, restart below length_min = 0.5**7).
- No replicate evaluations of the same x are planned: with only 100
  evaluations total and dim=20, spending budget on repeated samples of a
  single point (to directly measure noise) is not worth it when the GP's
  own fitted noise variance already accounts for mild observation noise
  across the whole dataset. If the trace later shows noise is large
  enough to be destabilizing the model (see deviation criteria), this
  will be revisited.
- `set_active_optimizer("turbo")` will be called once, before the first
  evaluate(), and will remain active unless a deviation below triggers a
  method switch (in which case the new optimizer's name will be set and
  logged in changelog.md).

## Explicit, quantitative deviation criteria
1. **Repeated trust-region collapse.** If TuRBO's trust region restarts
   (shrinks below length_min, discarding its local data) more than once
   before evaluation 70, that is evidence the 100-eval budget is too
   small for TuRBO's trust-region mechanism to get traction per restart
   (each restart re-spends ~2*dim=40 evals-worth of design logic, though
   handled internally as new local data, not a fresh 40-Sobol draw).
   In that case, switch the remaining budget to a direct local pattern
   search (coordinate-wise +/- step probes, step shrinking on failure)
   seeded at the global incumbent `harness` has seen so far, favoring
   pure exploitation over further trust-region restarts.
2. **Stagnation.** If `y_best_so_far` has not improved for 30 consecutive
   evaluations (tracked from the trace) after evaluation 50, switch the
   remaining budget to closed-loop Nelder-Mead (`nelder_mead.py`) seeded
   at the incumbent, to exploit the known-good region directly rather
   than continue a global/semi-global search that is not finding
   improvements.
3. **Hard stop.** Never exceed 100 evaluate() calls; never construct a
   second harness to reset the counter.

Any time one of these triggers (or any other actual deviation, e.g. a
library error forcing a fallback), an entry will be appended to
changelog.md with the evaluation count, what changed, and the specific
evidence observed.
