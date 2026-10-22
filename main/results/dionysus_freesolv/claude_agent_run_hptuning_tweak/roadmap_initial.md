# Roadmap — Initial Step

## Problem understood

- Fixed, finite candidate pool: 509 real molecules given as SMILES strings
  (`harness.candidates`). There is no interpolation/continuum between them —
  I can only pick one pool index per `evaluate()` call, each index usable
  exactly once (`CandidateAlreadyEvaluated` otherwise, no budget cost).
- Objective: a real experimental molecular property, `direction="min"`,
  `noisy=False` (deterministic lookup) — confirmed directly from the harness
  (`harness.direction`, `harness.noisy`). I do not know which property it is
  and am not trying to find out; I am told not to open
  `lookup_objective.py` or any file under `main/data/`, and have not done so.
- Total evaluation budget: 275 calls to `harness.evaluate(idx)`, hard-enforced
  by the harness (raises `BudgetExhausted` past the limit). 275/509 ≈ 54% of
  the pool, so there is room to be reasonably exploratory before needing to
  commit to pure exploitation.
- All 509 SMILES parsed successfully with RDKit (`Chem.MolFromSmiles`), so
  I can use cheminformatics features (fingerprints) for similarity/diversity
  reasoning without needing the hidden labels. This is explicitly permitted
  by the task (SMILES are the public input representation here, not a sealed
  identity).

## Assumptions

- Since nothing is known yet about which region of chemical space has low
  values of the (unknown) property, there is no principled way to guess a
  "promising" starting scaffold purely from chemical intuition without any
  data — a first batch chosen for **structural diversity** is the least
  wasteful way to spend early budget: it maximizes the chance of landing a
  reasonable candidate in whichever part of chemical space turns out to be
  good, and gives the broadest possible training signal for a subsequent
  similarity-/surrogate-based exploitation phase.
- With no information yet on how smooth or multimodal the property is with
  respect to structure, I will treat near-neighbor similarity as a reasonable
  (but unverified) prior for similar property values, and will check this
  assumption empirically as soon as I have early data (e.g. by comparing
  observed y variance among structurally close pairs, if any occur early).

## Step 1 plan

**Method:** Diversity-first pool sampling via RDKit Morgan fingerprints
(radius 2, 2048 bits, inherited default parameters from
`rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator`, RDKit version as
installed in this sandbox's `.venv`). I computed one fingerprint per
candidate (precomputed from the public SMILES list only — no evaluate()
calls consumed), then ran a greedy **MaxMin diversity picker**: start from a
fixed-seed (`numpy.random.RandomState(42)`) random candidate, then repeatedly
add the candidate maximizing the minimum Tanimoto distance to all already-picked
candidates. This is a deterministic, well-established way to spread an initial
batch across chemical space rather than clustering in one scaffold family.

**Hyperparameters:**
- Fingerprint: Morgan/ECFP-like, radius=2, nBits=2048 — standard default
  settings for ligand-based similarity screening; no tuning performed since
  this is a one-off feature computation, not a fitted model.
- MaxMin picker seed: 42 (arbitrary, fixed for reproducibility of this run).
- Batch size for step 1: **30** candidates (idxs precomputed:
  `[102, 2, 24, 68, 105, 437, 445, 209, 225, 28, 159, 350, 422, 409, 75, 212,
  343, 486, 0, 448, 431, 84, 218, 295, 475, 328, 278, 353, 425, 476]`).

**Why this step length (30 of 275):** Large enough to get an initial read on
the distribution of y across diverse chemical space (roughly ~6% of the pool,
~11% of budget) without committing more than a modest fraction of budget to
pure exploration before any exploitation. Small enough that ~245 evaluations
remain for subsequent similarity-guided / surrogate-guided exploitation once
I see which regions look promising. I am not committing to any particular
follow-up method yet.

**What I want to learn / evidence to inspect at the checkpoint:**
- The range, mean, and spread of observed y across these 30 diverse
  candidates — is there a clear best region or is the property roughly flat?
- Whether any structural pattern (e.g. functional group, ring system, size)
  visibly correlates with low y among the top few hits — this will inform
  whether a similarity-based greedy-exploitation step or a surrogate-model
  (e.g. k-NN or simple regression on fingerprints) step is more appropriate
  next.
- Running best value and which molecule attained it.

No early-pause condition is set for this first batch (it's a fixed, already
precomputed set of 30 indices with no adaptive stopping rule needed); I will
evaluate all 30 and then checkpoint at cumulative evaluation 30.
