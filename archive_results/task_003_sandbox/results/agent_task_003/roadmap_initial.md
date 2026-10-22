# Roadmap — Task 003 (pre-registered, written before any `evaluate()` call)

## 1. What I know

- 10-dimensional, box-bounded `[-5, 5]^10`, deterministic, minimize.
- Budget: 2000 evaluations total. That is 200x the dimension — generous
  for a derivative-free global method, but not so large that I can afford
  to waste it on a single strategy if it turns out to be a poor fit (e.g.
  highly multimodal vs. unimodal, separable vs. not — I have no idea yet
  which this function is).
- No noise, so every evaluation is fully informative and repeat
  evaluations of the same point are never useful.
- I'm told nothing about modality, separability, smoothness, or
  conditioning. A 10-d deterministic black box could be smooth and convex
  (favoring a local/gradient method from a good start), could be
  multimodal with many local optima (favoring population-based global
  search), or could be deceptive/rugged (favoring broad exploration before
  committing).

## 2. Methods, sequence, and budget split

Three phases, run in sequence, with the split adjusted adaptively within
the stated rules in Section 3 (not arbitrarily):

1. **`sobol_scan`** — ~256 evaluations (Sobol' quasi-random, a power of 2
   for good coverage uniformity). Purpose: cheap, unbiased global coverage
   of the box, with zero assumptions about structure. This also gives a
   rough sense of the function's scale/range and of whether multiple
   distinct basins exist, before I commit real budget to any specific
   method.

2. **`cma_es`** (with restarts I drive manually, since the installed
   wrapper is plain CMA-ES with no built-in restart strategy) — up to
   ~1400 evaluations. CMA-ES is my default workhorse for this budget
   regime because: it needs no gradient, handles non-separable and
   moderately multimodal landscapes far better than coordinate-wise or
   purely local methods, self-adapts its step size/covariance online (so
   it doesn't need hand-tuning for a function I can't inspect), and is
   known to scale reasonably to d=10 within a budget of hundreds to low
   thousands of evaluations. I will seed its initial mean using the best
   point(s) found by the Sobol scan rather than always the box midpoint,
   since the scan is real, already-paid-for information about where the
   function is lower.

3. **Local polish** (`bobyqa` primarily, `nelder_mead` or `lbfgsb` as
   fallback) — remaining budget (~300-450 evaluations), applied to the
   best point(s) found so far. Purpose: CMA-ES's own convergence can
   stall slightly short of a precise local optimum; a trust-region local
   solver started right at CMA-ES's best point should be able to shave
   off the remaining gap cheaply. BOBYQA is preferred because it's
   model-based and sample-efficient; it is reserved for *short* runs here
   (well under the ~500-evaluation point where the project notes it can
   become slow when driven one point at a time), consistent with its use
   purely as a final-polish step rather than a primary search method.

Rough split: 256 / ~1400 / ~340, totaling 2000. The CMA-ES/polish
boundary is not fixed in advance — see the deviation rules below, which
determine exactly when phase 2 ends and phase 3 begins.

## 3. Reasoning for these choices (from problem structure alone, no data yet)

- Dimension 10 is low enough that a GP-based method (GP-BO, TuRBO) could
  in principle work, but those are most valuable in the *very* low budget
  regime (~10x dim, i.e., ~100 evaluations) where every evaluation is
  precious and exact GP cubic-cost fitting is cheap. At a budget of 2000,
  CMA-ES and DE-style evolutionary search are the better fit: they scale
  to far more evaluations without the GP's fitting-cost blowup, and they
  do not rely on a GP's smoothness assumptions holding globally.
- A purely local method (L-BFGS-B, Nelder-Mead, BOBYQA) run once from the
  box midpoint risks getting stuck in a local optimum if the function is
  multimodal — which I cannot rule out. So I avoid committing most of the
  budget to a single local run; I use local solvers only for final
  polishing once a promising basin has already been identified globally.
- Starting with a cheap, assumption-free Sobol scan before committing to
  CMA-ES costs little (13% of budget) and gives real evidence (not a
  guess) about where to seed CMA-ES's mean, which matters because
  CMA-ES's default (box midpoint) is an arbitrary choice with no
  particular reason to be good for an unknown function.
- Determinism means there's no benefit to re-evaluating any point or to
  methods designed for noisy objectives (e.g. noise-averaging); every
  evaluation should go to a genuinely new point.
- Hybridizing is explicitly encouraged, and chaining
  global-coverage -> adaptive global search -> local polish is a standard,
  well-justified pattern for exactly this kind of sealed black-box
  problem.

## 4. Explicit, quantitative deviation criteria (pre-registered)

I will deviate from the plan above, and log each deviation in
`changelog.md`, under exactly these conditions:

- **D1 (CMA-ES stagnation -> restart).** During the `cma_es` phase, if the
  running best value has not improved by a relative amount greater than
  1e-6 (i.e. `(best_old - best_new) / max(1, abs(best_old)) <= 1e-6`)
  over the most recent 300 evaluations, OR CMA-ES's own internal
  termination criteria fire (`es.stop()` non-empty, e.g. step size
  collapsed), I will restart CMA-ES as a fresh instance. Restarts
  alternate strategy: the 1st restart re-seeds at a new random point in
  the box with the default sigma0 (diversification, in case the current
  basin is suboptimal); the 2nd restart re-seeds at the best point found
  so far with a reduced sigma0 = 0.5x the default (intensification, in
  case the basin is right but convergence stalled); further restarts keep
  alternating diversify/intensify.
- **D2 (reserve for local polish).** Regardless of how `cma_es` is doing,
  once cumulative evaluations reach 1660 (budget 2000 minus a reserved
  floor of 340), I end the `cma_es` phase and move to local polish, so
  that a refinement step always runs on whatever the best point is at
  that time.
- **D3 (local polish exhaustion -> alternate local method).** If the
  active local solver (BOBYQA first) converges/stops proposing points
  (raises its own "exhausted" signal, or scipy's Nelder-Mead/L-BFGS-B
  wrapper does the same) before the local-polish budget is used, I will
  apply a different local method (trying BOBYQA -> Nelder-Mead -> L-BFGS-B
  in that order, skipping any already tried) from the current best point,
  and if all three converge with remaining budget still unused, spend any
  leftover budget on an additional CMA-ES restart (diversify) followed by
  a final BOBYQA polish of whatever that finds.
- **D4 (early phase-1/2 boundary).** If, during the Sobol scan, the best
  value found has not changed at all across the last 128 of the 256
  points (suggesting the scan is already uninformative / flat region
  dominance), I will end the Sobol phase early at the next power-of-2
  checkpoint (128) rather than spending the full 256, and shift the
  saved evaluations to the `cma_es` phase budget.

Every time one of D1-D4 actually fires, or I make any other unplanned
change (different hyperparameter, a hybridization not described above,
etc.), I will append a dated entry to `changelog.md` citing the specific
observed numbers that triggered it, per the task protocol. This file
(`roadmap_initial.md`) will not be edited again after this point.
