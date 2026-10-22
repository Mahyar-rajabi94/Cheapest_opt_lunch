# Roadmap — Lookup-table molecular optimization (892 candidates, budget 294, minimize, deterministic)

## Problem shape

This is a discrete, finite-pool selection problem: 892 real molecules given as
SMILES strings, each with one fixed (deterministic) measured scalar property.
Each candidate may be evaluated at most once, and the harness enforces a hard
budget of 294 evaluations (~33% of the pool). There is no interpolation
structure given by the harness itself, but the SMILES strings let me compute
my own chemical-similarity structure (molecular fingerprints) and build a
surrogate over it — this turns the problem into something close to pool-based
Bayesian optimization over a fixed candidate set, which is a well-matched and
well-studied approach for exactly this situation (small fixed library,
expensive/one-shot "measurement", a real-valued chemical property to
minimize).

Chosen representation: ECFP4-style Morgan fingerprints (radius=2, 2048 bits)
computed with RDKit, compared with Tanimoto similarity. Tanimoto similarity
over binary fingerprints is a valid positive-semi-definite kernel, so it can
be used directly as a Gaussian Process covariance ("Tanimoto GP"), a standard
technique for molecular property optimization over fixed libraries.

## Planned method sequence

1. **Phase 1 — diversity-only initialization (40 evaluations, ~14% of budget).**
   Before any y-data exists, there is nothing to exploit, so the only
   sensible objective is maximum coverage of chemical space. I will use
   RDKit's MaxMin picker (a greedy farthest-point algorithm under Tanimoto
   distance on the Morgan fingerprints) to choose 40 mutually dissimilar
   molecules spanning the scaffold diversity of the full 892-candidate pool.
   Label: `diverse_init_maxmin`.

2. **Phase 2 — Tanimoto-kernel GP Bayesian optimization (230 evaluations,
   ~78% of budget, running total 270/294).**
   After phase 1 gives initial (fingerprint, y) pairs, fit a zero/constant-mean
   Gaussian Process whose kernel is the precomputed Tanimoto similarity matrix
   over all 892 candidates (standardized y, small jitter on the diagonal for
   numerical stability). At each step, compute the closed-form GP posterior
   mean/variance for every still-unevaluated candidate conditioned on all
   observations so far, and pick the candidate maximizing Expected Improvement
   (EI, minimization form) over the current best observed y. Evaluate it,
   add it to the observed set, and refit. This balances exploration
   (uncertain, dissimilar molecules) against exploitation (molecules similar
   to good ones found so far) automatically through the EI acquisition.
   Label: `gp_tanimoto_ei`.

3. **Phase 3 — local exploitation around the best-found molecule (remaining
   24 evaluations, running total 294/294).**
   Spend the last ~8% of budget purely exploiting: rank all still-unevaluated
   candidates by Tanimoto similarity to whichever molecule currently holds
   the best (minimum) observed y, and evaluate them in decreasing order of
   similarity. This is a final local search around the best basin found,
   on the assumption that structurally close analogues of a good molecule
   are likely to also be good (common in molecular property landscapes),
   without relying further on the (by then budget-starved) GP extrapolation.
   Label: `exploit_similarity_to_best`.

## Why this split

- 892 candidates is a large enough pool that pure random sampling at a 33%
  budget would likely miss both rare, dissimilar high-value scaffolds (hence
  diverse init) and fine local optima around a good hit (hence the final
  exploitation phase).
- A GP surrogate over a Tanimoto kernel is the workhorse of the budget range
  in between, since it can use actual chemical similarity to interpolate
  value across the *unevaluated* two-thirds of the pool, which pure
  nearest-neighbor heuristics cannot do once there are enough points to fit
  it well — the 40-point diverse init is meant to give the GP enough spread
  to make this calibration meaningful from the start of phase 2.
- 40/230/24 is a judgment call, not derived from data (none exists yet):
  40 is enough points to span diversity without spending too much of a tight
  budget on data that carries no exploitation signal; 24 reserved for a purely
  local mop-up phase is a small tax against the possibility the GP's global
  model is systematically miscalibrated near the optimum.

## Explicit deviation criteria

I will deviate from the plan above, and log it in changelog.md, if any of the
following are observed:

- **Plateau in phase 2:** if the running best y has not improved over 60
  consecutive evaluations within phase 2 (~26% of that phase's budget), I
  will shift acquisition from pure EI to a similarity-weighted variant that
  up-weights candidates near the current best (more exploitative), ending
  phase 2 early and moving the freed budget into phase 3's local search.
- **GP numerical instability:** if the Cholesky/solve step for the GP
  posterior fails or produces non-finite values even after jitter increases,
  I will fall back to a simpler k-nearest-neighbor-in-fingerprint-space
  predictor for the acquisition ranking for the rest of phase 2.
- **Degenerate diversity phase:** if MaxMin picking is unavailable or fails
  for any reason, I will fall back to random sampling without replacement
  for phase 1's 40 initial points.
- Any other ad hoc change made mid-run (e.g. adjusting the EI exploration
  parameter xi, changing phase boundary counts) will also get a changelog
  entry citing the specific observation that motivated it.
