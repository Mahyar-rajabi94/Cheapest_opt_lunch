# Task: Pre-process an OPV device dataset for a pool-based optimization benchmark

## Context

The dataset comes from Seifrid et al., "Beyond molecular structure: critically assessing machine learning for designing organic photovoltaic materials and devices" (*J. Mater. Chem. A*, 2024). It contains 558 literature-reported organic photovoltaic (OPV) devices.

It will be used in a **pool-based** benchmark:

- The **pool** is a fixed set of real devices (candidates). Each has 11 known features and a measured PCE.
- **Optimizers** choose which candidate to "test" next. Its PCE is then **looked up** from the dataset. There is no emulator and no model error.
- Optimizers that naturally choose from a list (Bayesian optimization, random search, greedy selection) work directly. Continuous optimizers (GA, CMA-ES, PSO) propose a point in feature space, which is mapped to the **nearest candidate not yet queried**.

The search space is the pool itself. There are no bounds, no constraints, and no snapping.

Structural descriptors (BigSMILES-derived, fingerprints, Mordred) and polymer molecular weights (Mn, Mw, PDI) are excluded. In tests, donor structure and molecular weights did not improve predictions, and Mn, Mw, and PDI are 63–80% missing.

**Your job is pre-processing only.** Do not run any optimizer.

## Input

- File: `pce_cleaned_dataset.csv` (identical to `datasets/Min_2020_n558/cleaned_dataset.csv` in the paper's GitHub repo).
- Expected shape: 558 rows × 45 columns.
- Requires Python with pandas and NumPy (SciPy optional, for the nearest-candidate search).
- Use the exact column names given below. Several contain spaces, units, and special characters.

## Ground rules

1. Do not impute missing values.
2. Do not alter any recorded value, even one that looks wrong. Flag it in the report instead.
3. Do not drop rows except those with missing features (Step 4).
4. Do not filter candidates by bounds or physical constraints. Every candidate is a real device.
5. Do not use device outcomes or post-fabrication measurements as features. This excludes `Voc (V)`, `Jsc (mA cm^-2)`, `FF (%)`, `PCE (%)`, all mobility columns, and all thickness columns.
6. **Never write PCE into the features file.** Optimizers must only be able to access PCE through `query()`.
7. Report the row count after every step and compare it against the "Sanity checks" section. If a count differs, stop and explain why before continuing.

## Steps

### Step 1: Load and validate

- Confirm the shape is 558 × 45.
- Confirm that `calculated PCE (%)` equals `Voc (V)` × `Jsc (mA cm^-2)` × `FF (%)` to within 1e-6 for every row.
  - Note: `FF (%)` is stored as a **fraction** (0.24–0.84) despite the "%" in its name. Do not divide it by 100.

### Step 2: Target

- Use `calculated PCE (%)` as the target. Rename it to `PCE` in the output.
- Do not use `PCE (%)` (the reported value). The paper's authors found errors in about 18% of reported values and used the calculated value instead.

### Step 3: Features (11)

| Short name | Exact column name | Unit |
|---|---|---|
| `HOMO_D` | `HOMO_D (eV)` | eV |
| `LUMO_D` | `LUMO_D (eV)` | eV |
| `Eg_D` | `Eg_D (eV)` | eV (optical gap) |
| `HOMO_A` | `HOMO_A (eV)` | eV |
| `LUMO_A` | `LUMO_A (eV)` | eV |
| `Eg_A` | `Eg_A (eV)` | eV (optical gap) |
| `DA_ratio` | `D:A ratio (m/m)` | mass ratio, donor/acceptor |
| `additive_conc` | `solvent additive conc. (% v/v)` | % v/v |
| `anneal_temp` | `temperature of thermal annealing` | °C (25 = not annealed) |
| `HTL_energy` | `HTL energy level (eV)` | eV |
| `ETL_energy` | `ETL energy level (eV)` | eV |

- All 11 are numeric features of each candidate. `HTL_energy` (3 distinct values) and `ETL_energy` (14 distinct values) are used as plain numbers, with no one-hot encoding and no snapping.
- Exclude `Ehl_D (eV)` and `Ehl_A (eV)`. They equal LUMO − HOMO exactly, so they are redundant. Confirm this numerically and report the maximum absolute difference.

### Step 4: Drop rows with missing features

- Drop any row with a missing value in any of the 11 features.
- Expected result: **541 rows**. No other rows are dropped at any later step.

### Step 5: Merge exact replicates into single candidates

This step is essential. Rows with identical features are indistinguishable to an optimizer, so each group must become one candidate.

- Group rows by the 11 features, rounded to 6 decimals.
- For each group, output:
  - `PCE`: mean of the target
  - `PCE_std`: standard deviation of the target (NaN for single rows)
  - `n_replicates`: group size
  - Metadata: the unique values in the group, joined with `"; "`
- Expected result: 13 replicate groups covering 28 rows, leaving **526 candidates**. All 526 feature vectors must be unique; verify this.
- Expected `PCE_std` across replicate groups: median 0.38, maximum 2.09. This serves as a noise estimate for optimizers that model noise.
- One group contains two different acceptors with identical energy values: PBDB-T with ITBTR-C4 and with ITBTR-C6. This is expected. Note it in the report.

### Step 6: Identifiers

- **`candidate_id`:** assign integers 0–525 after sorting candidates by `DOI`, then `Donor`, then `Acceptor`, then the 11 features. This keeps the IDs stable and reproducible.
- **`pair_id`:** one ID per unique (`Donor SMILES`, `Acceptor SMILES`) combination.
  - Expected: 474 distinct pairs. 33 pairs have more than one candidate, covering 85 candidates.
  - Candidates of the same pair share all six energy features and differ only in processing and interlayers. Querying one tells an optimizer a lot about the others, so results should also be reported as **distinct pairs found**.

### Step 7: Scaling

- Scale each feature to [0, 1] using its **min and max over the 526 candidates**: `x_unit = (x − min) / (max − min)`.
  - This is allowed because every candidate's features are known before any query. It uses no PCE values, so nothing leaks.
- Expected pool ranges:

| Feature | Min | Max |
|---|---|---|
| `HOMO_D` | −5.68 | −5.01 |
| `LUMO_D` | −3.75 | −2.78 |
| `Eg_D` | 1.48 | 2.12 |
| `HOMO_A` | −6.38 | −5.02 |
| `LUMO_A` | −4.23 | −2.83 |
| `Eg_A` | 1.10 | 2.43 |
| `DA_ratio` | 0.333 | 3.333 |
| `additive_conc` | 0.0 | 6.0 |
| `anneal_temp` | 25 | 175 |
| `HTL_energy` | −5.30 | −4.70 |
| `ETL_energy` | −4.65 | −2.14 |

- **Do not scale PCE** in any file. Each optimizer's surrogate must standardize PCE using only the candidates it has queried, refitting at every iteration. Add this as a note in the report for downstream use.

### Step 8: Functions for the benchmark

Write these in `preprocess.py` (or a small `pool.py` it creates). They must work only from the output files.

1. **`query(candidate_id)`**
   - Returns the candidate's PCE from `pool_labels.csv` and appends `(candidate_id, PCE, timestamp)` to a query log.
   - If a candidate is queried again, return the stored value, mark it as a repeat in the log, and do not count it as a new experiment.
2. **`nearest_unqueried(x_unit, queried_ids)`**, for continuous optimizers
   - Returns the `candidate_id` of the candidate closest to `x_unit`, by Euclidean distance on the scaled features, excluding `queried_ids`.
   - Break ties by the lowest `candidate_id`.
   - Accept points outside [0, 1]; just return the nearest candidate.
3. **`top_k_ids(fraction=0.05)`**
   - Returns the IDs of the top-performing candidates.
   - **For evaluation only.** It must never be called by an optimizer.

Add notes to the report:

- All 11 features count equally in the distance, so differences in processing variables matter as much as differences in energy levels.
- Many continuous proposals will map to the same nearby candidates. The query log should record both the proposed point and the candidate it was mapped to.

### Step 9: Benchmark reference values

Compute these and include them in the report and in `pool_spec.json`:

| Reference | Expected |
|---|---|
| Pool size | 526 candidates |
| Median PCE | 8.94% |
| Top 5% | 27 candidates with PCE ≥ 12.74%, from 23 distinct pairs |
| Best candidate | 15.71% (PBDB-TF:Y6) |
| Top 10 PCE values | 15.71, 14.43, 14.29, 14.19, 14.13, 13.65, 13.59, 13.52, 13.49, 13.41 |
| Random search, expected queries to the first top-5% candidate | (N + 1)/(K + 1) = 18.8 |
| Random search, expected queries to the best candidate | (N + 1)/2 = 263.5 |

### Step 10: Optional starting sets

Generate predefined starting sets so every optimizer begins from the same points:

- Default: 30 random seeds × 15 candidates each, drawn uniformly from the pool.
- Option: draw only from below-median candidates (PCE < 8.94%; 263 candidates). This makes the optimizer's climb more visible. It uses PCE to choose the starting points, so state this in the report.

Save as `initial_sets.json`, a mapping from seed to a list of `candidate_id`s.

### Step 11: Informational flags (not filters)

- Compute the physical-consistency checks below for each candidate and store them as boolean columns. They are **for information only; do not remove any candidate.**
  - `LUMO_offset`: LUMO_D − LUMO_A ≥ −0.15
  - `HOMO_offset`: HOMO_D − HOMO_A ≥ −0.15
  - `eff_gap_in_range`: 1.05 ≤ LUMO_A − HOMO_D ≤ 1.96
  - `gap_consistency_D`: Eg_D − (LUMO_D − HOMO_D) ≤ 0.15
  - `gap_consistency_A`: Eg_A − (LUMO_A − HOMO_A) ≤ 0.15
- Expected: 502 candidates pass all checks and 24 fail at least one.
- **Known flags to report.** List these for the user to verify against the source papers (DOIs are in the rows):
  - **PBDB-TF:Y6 (15.71%, the best candidate)** fails `gap_consistency_A`. Its acceptor optical gap is recorded as 1.81 eV, but Y6 is usually reported at about 1.3–1.4 eV. Its PCE is a real measurement, so it stays in the pool. However, its features are likely wrong, which may mislead optimizers' surrogate models.
  - **PBDTTz-N** fails `gap_consistency_D`. Its HOMO–LUMO gap is 1.71 eV, but its optical gap is recorded as 1.95 eV.
  - **PDINO** is recorded at −3.90 eV in 101 rows but at −2.18 eV in 1 row (ETL encoding inconsistency).

## Outputs

1. **`preprocess.py`**: one reproducible script that performs Steps 1–11 and defines the Step 8 functions.
2. **`pool_features.csv`**: 526 rows, with **no PCE column**. Columns:
   - `candidate_id`, `pair_id`
   - metadata: `Donor`, `Acceptor`, `DOI`, `Donor SMILES`, `Acceptor SMILES`, `HTL`, `ETL`, `n_replicates`
   - the 11 features under their short names, in original units
   - the same 11 features with a `_unit` suffix (scaled)
   - the Step 11 flags
3. **`pool_labels.csv`**: 526 rows. Columns: `candidate_id`, `PCE`, `PCE_std`. This file is read only by `query()` and by the evaluation code.
4. **`pool_spec.json`**, using this structure:

```json
{
  "target": {"name": "PCE", "source_column": "calculated PCE (%)", "unit": "%", "direction": "maximize"},
  "n_candidates": 526,
  "features": ["HOMO_D", "LUMO_D", "Eg_D", "HOMO_A", "LUMO_A", "Eg_A",
               "DA_ratio", "additive_conc", "anneal_temp", "HTL_energy", "ETL_energy"],
  "scaling": {
    "HOMO_D": {"min": -5.68, "max": -5.01, "source": "pool_minmax"}
  },
  "distance": {"metric": "euclidean", "space": "unit_scaled", "tie_break": "lowest_candidate_id"},
  "files": {"features": "pool_features.csv", "labels": "pool_labels.csv", "initial_sets": "initial_sets.json"},
  "reference": {"median_pce": 8.94, "top5_threshold": 12.74, "top5_count": 27, "top5_distinct_pairs": 23,
                "best_pce": 15.71, "random_expected_to_first_top5": 18.8, "random_expected_to_best": 263.5},
  "noise": {"replicate_pce_std_median": 0.38, "replicate_pce_std_max": 2.09}
}
```

Fill in scaling for all 11 features.

5. **`initial_sets.json`**: only if Step 10 was run.
6. **`preprocessing_report.md`**, containing:
   - a table of row counts after each step, next to the expected counts
   - the replicate summary (groups, median and maximum `PCE_std`, the mixed-acceptor group)
   - the pair summary (distinct pairs, pairs with more than one candidate)
   - the pool ranges, and the Step 9 reference values
   - the Step 11 flag counts, plus the flagged PBDB-TF:Y6, PBDTTz-N, and PDINO cases
   - the downstream notes from Steps 7 and 8
   - the known limitations listed below

## Sanity checks (expected values)

| Check | Expected |
|---|---|
| Input shape | 558 × 45 |
| max \|Voc × Jsc × FF − calculated PCE\| | < 1e-6 |
| max \|Ehl − (LUMO − HOMO)\| | < 1e-6 |
| Rows after dropping missing | 541 |
| Replicate groups / rows involved | 13 / 28 |
| Candidates / unique feature vectors | 526 / 526 |
| Distinct pairs / pairs with more than one candidate | 474 / 33 |
| Distinct HTL / ETL values | 3 / 14 |
| PCE range | 0.01–15.71% |
| Top 5%: threshold / count / distinct pairs | 12.74% / 27 / 23 |
| Candidates passing all Step 11 checks | 502 |
| PCE column present in `pool_features.csv` | no |

## Known limitations (report these; do not try to fix them)

- **The pool is published, mostly best-case devices.** Low-PCE and failed devices are under-represented. Success in this benchmark shows efficient search of a known space, not discovery of new materials.
- **Optimizers can't go beyond the pool.** New donor–acceptor pairs or new processing conditions cannot be proposed.
- **The top of the pool is sparse.** Only 10 candidates exceed 13.4% PCE, so the last improvements are the hardest to find.
- **Same-pair candidates are correlated.** The 33 pairs with more than one candidate share all six energy features, which can inflate apparent performance. Report distinct pairs found.
- **The best candidate's features are likely wrong.** Y6's recorded optical gap is likely incorrect, which may mislead surrogate models near the top of the pool.
- **Mapping continuous proposals is crude.** Nearest-candidate mapping weights all features equally and can send many proposals to the same few candidates.
- **No structural information.** Materials are distinguished only by their energy levels.
- **Solvent additive identity and annealing time are not features.**
- **Energy levels are noisy.** They come from mixed measurement methods and were averaged per material.
