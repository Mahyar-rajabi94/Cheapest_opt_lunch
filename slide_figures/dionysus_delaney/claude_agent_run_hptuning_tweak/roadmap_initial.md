# Roadmap — Initial Step

## Problem understood

- Lookup-table / fixed finite candidate pool: 892 real molecules, given as
  SMILES strings (`harness.candidates`), via
  `LookupTaskHarness` (`src/agent_harness/lookup_harness.py`).
- Each candidate index may be evaluated **at most once** — picking an
  already-used index raises `CandidateAlreadyEvaluated` without consuming
  budget, so repeats are just wasted re-decisions, not wasted budget.
- Objective: a real experimental molecular property, `direction = "min"`,
  deterministic (`noisy = False`, confirmed from `harness.noisy`).
- Budget: 294 `evaluate()` calls total, hard-enforced by the harness
  (`BudgetExhausted` once used up).
- I do not know what the property is, and will not try to find out beyond
  what the SMILES themselves suggest. I am not given a sealed black box in
  the usual sense — the SMILES strings are the public input representation,
  so I am explicitly allowed to use them: compute structural features
  (fingerprints), reason about scaffolds, and use chemistry intuition. I
  will not read `lookup_objective.py` or anything under `main/data/`.
- No optimizer library is provided for this task type. I will build my own
  lightweight selection logic using RDKit descriptors/fingerprints computed
  from the SMILES strings, since there is no continuous interpolation
  structure to hand to a standard continuous optimizer.

## Assumptions

- 294/892 ≈ 33% of the pool is evaluable. This is generous enough that a
  pure "explore broadly, then exploit around the best-found regions"
  strategy is reasonable: there is room for both a diversity-covering phase
  and a substantial exploitation phase afterward.
- Structurally similar molecules (by Morgan/ECFP fingerprint Tanimoto
  similarity) plausibly have correlated property values for *many* real
  molecular properties (this is the premise behind QSAR / similarity-based
  virtual screening). I will test this assumption empirically once I have
  initial observations, rather than assume it holds for whatever property
  this happens to be — if early similarity-based picks don't show
  correlation with the best-found points, I will reconsider and fall back
  toward broader/uninformed sampling for the remainder of the budget.
- No further domain knowledge about the specific property is required or
  available; I will reason purely from (SMILES features, observed y) pairs.

## Step 1 — diverse initial coverage

**Method:** Compute ECFP4 Morgan fingerprints (radius=2, 2048 bits, RDKit
default `GetMorganFingerprintAsBitVect`) for all 892 candidates. Use
RDKit's `MaxMinPicker` (an approximate max-min diversity picker operating
on Tanimoto distance between fingerprints) to select an initial batch that
is maximally structurally diverse, so the first observations span as much
of the chemical space / scaffold variety as possible before any exploitation
decision is made. This is a standard, well-justified choice for an unknown
discrete objective with no prior information: it avoids wasting early
budget on near-duplicate molecules and gives the best chance of finding at
least one promising structural neighborhood to exploit later.

**Hyperparameters:**
- Morgan fingerprint: radius=2, nBits=2048 (RDKit library defaults for
  ECFP4-equivalent fingerprints; widely used default, no tuning needed at
  this stage since I have no data yet to tune against).
- `MaxMinPicker.LazyBitVectorPick`: deterministic given a fixed random seed
  (seed=0) for the first-pick tie-break; no other parameters.
- No surrogate model yet — this step is pure unsupervised diversity
  selection, there is nothing to fit yet.

**Step length:** 25 evaluations (cumulative count 25 / 294 after this
step). Rationale: large enough to get a first read on the range and spread
of y across diverse scaffolds (a sample of this size, picked for
diversity rather than randomly, should reveal roughly where the best
observed values cluster structurally), while small enough (~8.5% of
budget, ~2.8% of the pool) that most of the budget is preserved for a
more targeted phase once I have real evidence to act on.

**What I want to learn / evidence at checkpoint:**
- The distribution (min, median, max, spread) of y across this diverse
  batch, to calibrate what counts as a "good" value for this task.
- Which of the 25 diverse molecules scored best/worst, and whether their
  structures (functional groups, scaffold, size) suggest any visible
  pattern (e.g. molecule size/complexity correlating with y, or a specific
  functional group standing out) worth exploiting.
- Whether simple computed descriptors (e.g. molecular weight, logP, number
  of rings, TPSA) show any visible correlation with y in this batch —
  this will inform whether a descriptor-based or fingerprint-similarity-based
  exploitation strategy is more promising next.

No early-pause condition within this fixed small step (25 calls on a
deterministic objective, no runaway-cost risk); I will evaluate all 25 then
stop and review before continuing.
