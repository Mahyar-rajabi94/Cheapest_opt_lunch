# Roadmap — initial plan

## Problem as given
- Dimension: 5, bounds [-5, 5] per coordinate (a reasonably small, bounded continuous box).
- Direction: minimize.
- Noise: evaluate() is explicitly noisy (repeated calls at the same x can return different
  values). Not deterministic.
- Budget: 100 evaluate() calls total — very small for dim=5 (20x dim). This rules out
  anything that relies on large sample counts (e.g. pure evolutionary strategies with
  large populations, dense grid/random search, or finite-difference gradients, which are
  additionally unreliable under noise since differencing two noisy evaluations amplifies
  the noise relative to the signal).

## Method sequence and budget split

1. **Sobol initial design — 12 evaluations** (`optimizer` label: `sobol_init`).
   Space-filling design (`2*(dim+1)` points, the convention already used by this
   repo's own GPBO wrapper) to get broad, unbiased coverage of the 5-d box before
   fitting any surrogate. With a noisy function and only 100 calls, we cannot afford
   a separate "exploration" phase using a weaker method (e.g. plain random search) —
   Sobol gives the best coverage per point and its output is consumed directly by the GP.

2. **GP-based Bayesian optimization — up to ~70 evaluations** (`optimizer` label: `gp_bo`).
   Using this repo's `GPBO` wrapper (SingleTaskGP + analytic LogEI, multi-start L-BFGS
   acquisition optimization). Rationale:
   - Budget (100) is small relative to dimension (5), which is exactly the regime GP-BO
     is designed for (sample-efficient, each point chosen to maximize expected
     information/improvement rather than spending evaluations on undirected search).
   - SingleTaskGP fits a (by default) homoskedastic noise term from the data itself,
     so it partially accounts for noise by not interpolating through every point
     exactly — unlike e.g. a local pattern-search method that would chase a single
     lucky noisy low reading.
   - Caveat (explicit, pre-registered): BoTorch's analytic LogEI used here is the
     noiseless-EI formula referencing the single best **observed** y. Under noise this
     can over-commit to a point whose observed value is a lucky low outlier rather than
     a truly good location. I mitigate this with phase 3 below (confirmation
     re-evaluation) rather than switching acquisition functions mid-run, to keep the
     GP-BO phase itself exactly as implemented/tested in this repo.

3. **Confirmation / refinement — remaining budget (~18 evaluations)** (`optimizer`
   label: `confirm_refine`).
   Once phase 2's budget is spent, take the current top 2-3 distinct x locations
   (by observed y) and re-evaluate each at least once more (replication) to check
   whether their apparent advantage survives a second noisy draw, averaging the
   repeats to estimate the true value. Any remaining budget after that is spent on
   small random perturbations (Gaussian jitter, shrinking scale) around whichever
   location has the best *averaged* value, as a cheap local refinement step that
   does not use finite differences (avoids noise-amplification from differencing).

Total planned: 12 + 70 + 18 = 100.

## Why not other candidates
- **BOBYQA / Nelder-Mead / L-BFGS-B**: local, trust-region/simplex methods that assume
  (near-)exact function values to decide accept/reject or build local models; with
  real observation noise they can mistake noise for signal and collapse/misstep.
  Could be considered as a late-stage local refiner, but GP-BO's acquisition already
  serves that purpose while explicitly modeling uncertainty, so I'm not pre-registering
  them as a separate phase.
- **CMA-ES / Differential Evolution**: population-based methods generally want
  populations of size ~4+dim per generation and multiple generations to be effective;
  100 evaluations at dim=5 only buys a handful of generations, too few to expect
  reliable convergence, and they do not explicitly model/average out noise the way a
  GP's fitted noise term does.
- **TuRBO**: reasonable alternative to GP-BO, but needs batches of 4 and its own
  10-point init, and is more naturally suited to somewhat larger budgets / higher
  dimension where trust-region restarts pay off. With only 100 calls total I prefer
  the fully sequential GP-BO so every single evaluation is individually chosen by the
  acquisition function rather than committed as part of a batch before seeing
  within-batch results.

## Explicit deviation criteria (quantitative, pre-registered)

- If, during phase 2, the best-so-far value has not improved for **20 consecutive
  GP-BO evaluations**, stop phase 2 early and move directly to phase 3
  (confirmation/refinement) with whatever budget remains — treat this as evidence
  that the GP-BO phase has plateaued (whether from genuine convergence or from the
  noiseless-EI caveat above over-committing to a stale incumbent).
- If GP-BO's internal GP fit or acquisition optimization raises an exception (e.g.
  numerical issues from near-duplicate points), fall back for the rest of that
  phase's remaining budget to Sobol sampling (`optimizer` label: `sobol_fallback`),
  logging this as a deviation.
- If, after telling the optimizer a batch, candidates returned by `ask()` are
  (numerically) duplicates of already-evaluated points, perturb them with small
  Gaussian noise before evaluating, to avoid wasting budget on a point whose value
  we can already estimate.
- Any such change will be logged in `changelog.md` with the triggering evaluation
  index and the observed evidence.
