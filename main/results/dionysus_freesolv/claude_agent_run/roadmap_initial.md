# Roadmap — task029 (lookup-table molecular optimization)

## Problem

- Finite pool of 509 real molecules, given only as SMILES strings.
- `harness.evaluate(idx)` returns a real, deterministic, experimentally
  measured scalar property for molecule `idx`. Direction: **minimize**.
- Each candidate may be evaluated **at most once** (picking a repeat raises
  `CandidateAlreadyEvaluated` without consuming budget, so repeats are just
  wasted logic, not wasted budget — but still to be avoided).
- Total budget: **275 evaluations** out of 509 candidates (~54% of the pool).
- No noise, so a single measurement per molecule is fully trustworthy — no
  need to spend budget on replication.
- There is no continuous/interpolation structure in the harness itself, but
  the SMILES strings carry real chemical structure. Molecules with similar
  structure (scaffold, functional groups, size, polarity) plausibly have
  correlated values of whatever property is being measured, exactly as
  spatial proximity correlates values in a continuous black-box. This
  motivates treating the problem as a **structure-based active-search /
  surrogate-model problem** over a fixed pool, rather than pure random
  sampling.
- RDKit is available in this environment (confirmed via `pip install rdkit`
  succeeding and `Chem.MolFromSmiles` parsing all 509 SMILES without
  errors), so I can compute real cheminformatics features (Morgan/ECFP
  fingerprints) rather than relying on raw strings.

## Method plan and budget split

**Phase 1 — Diverse initial sampling (50 evaluations, optimizer label
`diverse_init`).**
- Compute Morgan (ECFP4, radius=2, 2048-bit) fingerprints for all 509
  candidates via RDKit.
- Use a MaxMin-style farthest-point selection (via Tanimoto distance) to
  pick 50 molecules that are maximally spread out in fingerprint space,
  starting from a random seed molecule (fixed numpy seed for
  reproducibility).
- Rationale: with zero data so far, the best use of initial budget is
  broad coverage of the chemical space so that (a) we get an honest sample
  of the value range and which structural regions look promising, and
  (b) a downstream surrogate model has non-redundant training data. Pure
  uniform-random sampling would also give coverage but MaxMin explicitly
  avoids wasting evaluations on near-duplicate structures, which matters
  more here because the pool is small (509) and budget is a large fraction
  of it (54%), so redundant picks are proportionally expensive.
- 50 evaluations (~18% of budget) is enough points to fit a first
  regression surrogate (feature dimensionality will be reduced via PCA or
  bit-hashing to keep it well under n=50) while leaving the bulk of the
  budget (225 evals) for exploitation.

**Phase 2 — Surrogate-guided sequential selection (remaining budget, up to
225 evaluations, optimizer label `rf_surrogate` unless changed; batches of
25, retraining after each batch).**
- Fit a Random Forest regressor (scikit-learn, `n_estimators=300`,
  default other hyperparameters) on Morgan fingerprint bits (as binary
  features) → measured y, using all evaluations collected so far.
- For every remaining (unevaluated) candidate, predict the mean `mu` and
  the across-tree standard deviation `sigma` (used as an uncertainty
  proxy, analogous to a GP posterior std).
- Score candidates by a lower-confidence-bound acquisition
  `score = mu - k * sigma` (we are minimizing), with initial `k = 1.0`,
  and evaluate the **top 25** scoring unevaluated candidates per batch.
- Retrain the forest after every batch using all data collected so far
  (50 + 25*n), so later batches benefit from more training signal.
- Rationale: Random Forests handle high-dimensional sparse binary
  fingerprint features well without heavy tuning, require no distributional
  assumptions, and give a usable (if approximate) uncertainty estimate for
  free via inter-tree disagreement — a reasonable proxy for a GP when a
  proper Tanimoto-kernel GP library isn't already available in this
  environment. Batches of 25 (~9% of budget per batch) let the surrogate
  update on meaningfully more data each round while still retraining
  roughly 9 times over the remaining budget, which is frequent enough to
  adapt without excessive overhead.

## Explicit deviation criteria

I will deviate from the Phase 2 defaults above if any of the following are
observed, and will log the change in `changelog.md`:

1. **Plateau:** if the running best `y` does not improve for two
   consecutive batches (50 evaluations) during Phase 2, I will increase the
   exploration weight `k` (e.g. from 1.0 to 2.0) to push the acquisition
   toward higher-uncertainty regions, or temporarily switch a batch to pure
   diversity sampling (MaxMin among the remaining pool) if increasing `k`
   alone does not restore improvement within one further batch.
2. **Degenerate/overconfident surrogate:** if the Random Forest's
   out-of-bag R² (computed via `oob_score_=True`) is negative or the
   inter-tree `sigma` collapses to near-zero across the whole remaining
   pool (suggesting the model has nothing left to discriminate on), I will
   switch the acquisition to pure exploitation (`k = 0`, greedy on `mu`)
   since uncertainty is no longer informative, or fall back to random
   sampling among the current top-quartile-by-prediction if even the mean
   predictions look untrustworthy (OOB R² < 0).
3. **Batch size near budget exhaustion:** the final Phase 2 batch will be
   truncated to whatever evaluations remain (so the campaign never exceeds
   275 total), and if fewer than 25 remain I will simply evaluate the top-N
   scoring remaining candidates where N = evaluations remaining.
4. **Candidate pool exhaustion safeguard:** since budget (275) < pool size
   (509), running out of unused candidates is not expected, but if it ever
   happened I would stop early and call `harness.save()`.

Any other hyperparameter changes (e.g. adjusting `n_estimators`,
`max_depth`, fingerprint radius/bit size) made mid-campaign in response to
observed behavior will also be logged in `changelog.md` with the
evidence/reasoning, per protocol.
