# Task 006 — Initial roadmap (pre-registered before any evaluation)

Written 2026-10-22, before the first call to `harness.evaluate()`. No data
from the objective has been seen. Everything below is reasoned from the
stated problem properties only: minimize a black box on [0, 1]^11, 100
evaluations hard budget, no random noise but **history-dependent** values
(re-evaluating the same `x` generally gives a different `y`).

## 1. Planned methods, sequence and budget split

| Phase | Evals | Count | Optimizer label | Method |
|---|---|---|---|---|
| A | 1–20 | 20 | `lhs_init` | Space-filling Latin hypercube (scipy `qmc.LatinHypercube`, centred-discrepancy optimized, seed 0) |
| B | 21–22 | 2 | `repeat_probe` | Two deliberate exact re-evaluations: the best LHS point and the median LHS point |
| C | 23–76 | 54 | `gp_bo_logei` or `gp_bo_tnei` (see rule D1) | Global GP Bayesian optimization, one point per iteration |
| D | 77–100 | 24 | `trbo_local` | Local trust-region BO (TuRBO-1 style, batch size 1) centred on the incumbent |

Implementation details committed to now:

- **Surrogate (C and D):** BoTorch `SingleTaskGP` with its default ARD RBF
  kernel and dimension-scaled lengthscale prior. The likelihood noise is
  **learned** (not fixed to ~0), because the history dependence means the
  same `x` can produce different `y` and the model must be allowed to treat
  that as inconsistency instead of interpolating it exactly. Outputs are
  standardized before fitting (plus optional warping, rule D2). The GP is
  refit from scratch on all data before every proposal.
- **Acquisition (C):** analytic LogEI on the full box (repeatable branch)
  or qLogNoisyEI (history-effect branch), maximized with BoTorch
  `optimize_acqf` (10 restarts, 1024 raw samples, `sample_around_best`
  initialization).
- **Duplicate guard (all GP phases):** a proposal within L2 distance 1e-3
  of any already-evaluated point is replaced by the next-best restart
  candidate. Spending budget on near-repeats is pointless when repeats are
  not informative about the same quantity. Only Phase B repeats on purpose.
- **Local phase (D):** trust-region box centred on the best observed point,
  side length L0 = 0.4 of the unit cube, rescaled per dimension by the GP's
  ARD lengthscales (normalized to geometric mean 1, as in TuRBO), clipped to
  [0, 1]. The same acquisition as Phase C is maximized inside the box.
  A *success* is a new value below the TR best by more than 1e-3·s (s is
  defined in D1). After 3 consecutive successes L doubles (max 0.8). After
  2 consecutive failures L halves. The centre always moves to the best
  point found inside the TR phase.

## 2. Reasoning (from problem properties only)

- **Budget ≈ 9·d.** 100 evaluations in 11 dimensions is a very
  sample-starved regime. Population methods are a poor fit at this size.
  CMA-ES with its default λ = 4+⌊3 ln 11⌋ = 11 gets ~9 generations. DE is
  similar or worse. Neither can adapt a covariance or converge. Finite-
  difference L-BFGS-B pays 12 evaluations per gradient (~8 steps total) and
  is fragile when values are not repeatable. Nelder–Mead needs 12 points
  just for its simplex and converges slowly in 11-D. BOBYQA needs 2d+1 = 23
  evaluations for its first quadratic model and is purely local. The
  method family designed for ~10·d budgets is surrogate-based Bayesian
  optimization: it reuses every evaluation, and it trades off global and
  local search explicitly.
- **Bounded unit box.** The box is already the natural GP input
  normalization. A space-filling LHS of ~2·d points (here 20, plus the 2
  probes ≈ 2(d+1)) is the standard minimum for a meaningful ARD GP fit. LHS
  also stratifies every 1-D projection, which helps lengthscale estimation.
- **History dependence.** This is the property that most threatens a
  surrogate model. A GP that interpolates exactly would be corrupted by
  inconsistent values. So (i) the noise variance is learned, (ii) the size
  of the effect is measured explicitly with 2 evaluations (2% of the
  budget), and (iii) if the effect is non-negligible, the evaluation index
  is added as a GP input. That lets the model represent drift over the
  campaign rather than only treating it as noise. Measuring first is far
  cheaper than letting an unknown inconsistency silently degrade 78 model-
  driven evaluations.
- **Global → local split.** In 11-D with <100 points, EI tends to keep
  exploring (often toward box faces/corners) long after a good basin has
  been found. TuRBO's results show that restricting the search to an
  adaptive trust region helps a lot in moderate/high dimensions. TuRBO from
  the start is tuned for budgets of hundreds to thousands (2·d init, batches
  of 4, success/failure tolerances of ~d batches). So it is used here only
  as the final exploitation phase, after the global GP has located a basin.

## 3. Pre-registered decision rules (branches of the plan, not deviations)

- **D1 — history-effect model (decided right after eval 22).** Let
  s = q90 − q10 of the 20 LHS values, Δ_i = y_repeat − y_original for the two
  probes, and r = max|Δ_i| / s.
  - r ≤ 0.02 → effectively repeatable: GP on x only, analytic LogEI with
    best_f = observed best (label `gp_bo_logei`).
  - r > 0.02 → GP on (x, t) with t = evaluation index / 100. The acquisition
    is qLogNoisyEI, evaluated at t fixed to the next index, with baseline =
    all evaluated x at that t (label `gp_bo_tnei`). The same model is used
    in Phase D.
- **D2 — output warping (decided after eval 20).** If |skewness| of the 20
  LHS values > 1.0, the standardized y are Yeo–Johnson transformed (MLE λ,
  refit each iteration) before GP fitting. Otherwise the y are only
  standardized.

## 4. Quantitative criteria for deviating from this plan

Each of these, if triggered, is a deviation and will be logged in
`changelog.md`.

- **X1 — global stagnation.** If 15 consecutive Phase-C evaluations
  produce no new best value, end Phase C immediately and give all
  remaining budget to Phase D.
- **X2 — uninformative surrogate.** At evals 40 and 60, compute the
  Spearman correlation between closed-form leave-one-out GP predictions
  and the observed (transformed) targets. If ρ_LOO < 0.3, the GP is not
  ranking points usefully. Replace the rest of Phase C with CMA-ES (pycma,
  default population, x0 = incumbent, σ0 = 0.2, box bounds), and replace
  Phase D with continued CMA-ES.
- **X3 — trust-region collapse / restart.** In Phase D, if the TR side
  length drops below 0.01 (12 consecutive failures from L0), restart the TR
  at the best observed point that is ≥ 0.3 (L2, unit cube) from the current
  centre and is in the best 25% of all observations. If no such point
  exists, spend the remaining budget on global GP-BO (Phase C settings).
- **X4 — numerical failure.** If the GP fit raises an error on 2
  consecutive iterations, switch to CMA-ES (x0 = incumbent, σ0 = 0.1) for
  the rest of the current phase.
- **X5 — large history effect.** If r > 0.5 (history effect at least half
  the robust spread of y), the raw observed minimum is an unreliable
  incumbent. Phase D then centres its TR on the evaluated point with the
  lowest GP posterior mean (at the next t), not on the raw best value.
- **X6 — surrogate mismatch in the local phase.** If during Phase D the
  last 6 observed values are all worse than the GP's own prediction at
  proposal time, by more than 0.1·s each, the model is systematically
  over-optimistic locally. Halve L immediately and log it. If it repeats a
  second time, fall back to X3's restart.

Any other change I make mid-campaign (hyperparameters, labels, hybrid
steps) will also be logged in `changelog.md` with the evaluation number and
the specific observed evidence.
