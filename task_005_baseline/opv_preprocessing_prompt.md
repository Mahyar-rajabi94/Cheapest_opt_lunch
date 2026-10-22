# Task: Pre-process an OPV device dataset for an emulator-based optimization benchmark (all-continuous search space)

## Context

The dataset comes from Seifrid et al., "Beyond molecular structure: critically assessing machine learning for designing organic photovoltaic materials and devices" (*J. Mater. Chem. A*, 2024). It contains 558 literature-reported organic photovoltaic (OPV) devices.

It will be used in an **emulator-based** benchmark:

- An **emulator** (a regression model trained on the processed data) acts as the "experiment". It returns a power conversion efficiency (PCE) for any input.
- Several **optimizers** propose inputs within a bounded, constrained search space and receive the emulator's prediction.

**All 11 search variables are continuous.** Nine are ordinary continuous variables. The other two, `HTL_energy` and `ETL_energy`, can be proposed anywhere in their range but are **snapped to the nearest observed level** before the emulator is queried. This keeps the search space fully continuous for every optimizer, while the emulator is only ever queried at interlayer energies that exist in the data.

Structural descriptors (BigSMILES-derived donor descriptors, acceptor descriptors, fingerprints) and polymer molecular weights (Mn, Mw, PDI) are deliberately **excluded**:

- In tests, donor structure and molecular weights did not improve the emulator.
- Acceptor structure helped only slightly, and only works if acceptors are chosen from a library of real molecules, which isn't compatible with an all-continuous search space.
- Mn, Mw, and PDI are 63–80% missing.

**Your job is pre-processing only.** Do not train the emulator or run any optimizer.

## Input

- File: `pce_cleaned_dataset.csv` (identical to `datasets/Min_2020_n558/cleaned_dataset.csv` in the paper's GitHub repo).
- Expected shape: 558 rows × 45 columns.
- Requires Python with pandas and NumPy.
- Use the exact column names given below. Several contain spaces, units, and special characters.

## Ground rules

1. Do not impute missing values.
2. Do not alter any recorded value, even one that looks wrong. Flag it in the report instead.
3. Do not drop rows except those with missing base features (Step 4).
4. Do not use device outcomes or post-fabrication measurements as features. This excludes `Voc (V)`, `Jsc (mA cm^-2)`, `FF (%)`, `PCE (%)`, all mobility columns, and all thickness columns.
5. Use only the 11 inputs in Step 3 as emulator inputs. Material names, SMILES strings, and polymer molecular weights are not emulator inputs.
6. Do not recompute the fixed search-space bounds given in Step 9.
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

Exclude `Ehl_D (eV)` and `Ehl_A (eV)`. They equal LUMO − HOMO exactly, so they are redundant. Confirm this numerically and report the maximum absolute difference.

### Step 4: Drop rows with missing features

- Drop any row with a missing value in any of the 11 features.
- Expected result: **541 rows**. No other rows are dropped at any later step.

### Step 5: Interlayer levels and snapping

#### 5a. Observed levels

Round `HTL_energy` and `ETL_energy` to 2 decimals and list their distinct values among the 541 rows.

**HTL levels** (3):

| HTL_energy | Material(s) | Devices |
|---|---|---|
| −5.30 | MoOx (228), CuSCN (1) | 229 |
| −5.20 | PEDOT:PSS | 310 |
| −4.70 | V2O5 | 2 |

**ETL levels** (14):

| ETL_energy | Material(s) | Devices |
|---|---|---|
| −4.65 | ZrAcAc | 10 |
| −4.46 | ZnO | 210 |
| −4.23 | TiO2:TOPD | 2 |
| −4.18 | PNDIT-F3N-Br (6), PFNDI-Br (5) | 11 |
| −4.10 | PEIE | 1 |
| −4.08 | ZnO/PFN-Br | 9 |
| −4.05 | EDTA-ZnO | 1 |
| −3.93 | FPI | 1 |
| −3.90 | PDINO (101), C60-bissalt (15), N719:PrC60MAI (4), PNDIT-F3N (3) | 123 |
| −3.60 | PDIN | 17 |
| −3.00 | LiF | 10 |
| −2.90 | Ca | 63 |
| −2.18 | PFN-Br (73), PDINO (1) | 74 |
| −2.14 | PFN (7), ZnO/PFN (2) | 9 |

#### 5b. Snapping rule

Write a function `snap(x, levels)` that returns the observed level nearest to `x`.

- If `x` is exactly halfway between two levels, return the **more negative** one.
- Values outside the range should not occur, because proposals are bounded; if one does, snap it to the nearest end level.
- Apply it to `HTL_energy` and `ETL_energy` in **original units**, after converting a proposal back from [0, 1] scaling.
- All training rows already sit exactly on observed levels, so snapping leaves them unchanged. Verify this.

#### 5c. Capture intervals

Each level "captures" the part of the continuous range that snaps to it, bounded by the midpoints to its neighbors. Compute each level's interval and its share of the full range, and save them in `interlayer_levels.csv`.

Expected:

- **HTL** (range −5.30 to −4.70):
  - −5.30 → [−5.30, −5.25], 8.3% of the range
  - −5.20 → [−5.25, −4.95], 50.0%
  - −4.70 → [−4.95, −4.70], 41.7%
- **ETL** (range −4.65 to −2.14): the largest shares go to −3.60 (17.9%), −2.90 (16.3%), −2.18 (15.1%), and −3.00 (13.9%). ZnO at −4.46 gets 8.4%, and −2.14 gets 0.8%.

Report these, because they mean a level's share of the search space depends on the gaps between levels, not on how many devices support it. For example, V2O5 (2 devices) captures 41.7% of the HTL range, while MoOx (229 devices) captures 8.3%.

#### 5d. Combination support

Build the full 3 × 14 table of (HTL_energy, ETL_energy) combinations with device counts, and save it as `interlayer_support.csv`.

- Expected: 18 of the 42 combinations appear in the data.
- Most unobserved combinations mix the two common device designs: conventional (PEDOT:PSS with Ca, PDINO, PFN-Br, or similar) and inverted (MoOx with ZnO).
- Add each row's combination count as the metadata column `interlayer_support`.

**Flag in the report:** PDINO is recorded at −3.90 eV in 101 rows but at −2.18 eV in 1 row. This is an encoding inconsistency. Do not change it.

### Step 6: Average exact replicates

- Group rows by the 11 features, rounded to 6 decimals.
- For each group, output:
  - `PCE`: mean of the target
  - `PCE_std`: standard deviation of the target (NaN for single rows)
  - `n_replicates`: group size
  - Metadata: the unique values in the group, joined with `"; "`
- Expected result: 13 replicate groups covering 28 rows, leaving **526 rows**.
- Expected `PCE_std` across replicate groups: median 0.38, maximum 2.09. This serves as a noise estimate for later modeling.
- One group contains two different acceptors with identical energy values: PBDB-T with ITBTR-C4 and with ITBTR-C6. This is expected. Note it in the report.

### Step 7: Variable types

| Variables | Type |
|---|---|
| `HOMO_D`, `LUMO_D`, `Eg_D`, `HOMO_A`, `LUMO_A`, `Eg_A`, `DA_ratio`, `additive_conc`, `anneal_temp` | continuous |
| `HTL_energy`, `ETL_energy` | continuous, snapped to the nearest observed level before the emulator is queried |

The emulator always receives 11 numeric values. Do not one-hot encode anything.

### Step 8: Metadata

- Keep `Donor`, `Acceptor`, `DOI`, `Donor SMILES`, `Acceptor SMILES`, `HTL`, `ETL`, `interlayer_support`, `n_replicates`, and `PCE_std` as metadata columns.
- They must never be used as emulator inputs. They exist for reporting results.

### Step 9: Search space

**Bounds** (fixed values; do not recompute)

| Variable | Lower | Upper | Source |
|---|---|---|---|
| `HOMO_D` | −5.60 | −5.06 | 1st–99th percentile of the 541 rows |
| `LUMO_D` | −3.70 | −2.90 | same |
| `Eg_D` | 1.55 | 2.05 | same |
| `HOMO_A` | −6.07 | −5.24 | same |
| `LUMO_A` | −4.18 | −3.32 | same |
| `Eg_A` | 1.23 | 2.19 | same |
| `DA_ratio` | 0.50 | 1.67 | same |
| `additive_conc` | 0.0 | 3.6 | same |
| `anneal_temp` | 25 | 160 | same |
| `HTL_energy` | −5.30 | −4.70 | lowest–highest observed level (snapped) |
| `ETL_energy` | −4.65 | −2.14 | lowest–highest observed level (snapped) |

**Linear constraints** (original units; 0.15 eV tolerance for measurement error)

| Name | Inequality | Physical reason |
|---|---|---|
| `LUMO_offset` | LUMO_D − LUMO_A ≥ −0.15 | electron must move downhill from donor to acceptor |
| `HOMO_offset` | HOMO_D − HOMO_A ≥ −0.15 | hole must stay on the donor |
| `eff_gap_min` | LUMO_A − HOMO_D ≥ 1.05 | effective gap sets the maximum V_OC |
| `eff_gap_max` | LUMO_A − HOMO_D ≤ 1.96 | same |
| `gap_consistency_D` | Eg_D − (LUMO_D − HOMO_D) ≤ 0.15 | optical gap ≤ HOMO–LUMO gap |
| `gap_consistency_A` | Eg_A − (LUMO_A − HOMO_A) ≤ 0.15 | same |

- No constraint involves `HTL_energy` or `ETL_energy`, so snapping never changes whether a proposal is feasible.
- Optional constraints (include in the JSON, marked `"optional": true`):
  - (LUMO_D − HOMO_D) − Eg_D ≤ 0.5
  - (LUMO_A − HOMO_A) − Eg_A ≤ 0.5

### Step 10: Scaling and the query pipeline

- Map every input to [0, 1] with `x_unit = (x − lower) / (upper − lower)`, using the Step 9 bounds (including the HTL and ETL ranges).
- Write `to_unit(df)` and `from_unit(df)` functions driven by `search_space.json`.
- **Do not clip.** Training rows outside the bounds map outside [0, 1]. Keep them, because they are real devices.
- **Do not scale the target** in the saved file.
  - The emulator may standardize PCE internally using all data, because it plays the role of reality.
  - Each optimizer's own surrogate must standardize PCE using only the points it has queried.
- Write a function `prepare_query(x_unit)` that downstream code will use for every optimizer proposal:
  1. `from_unit` → original units
  2. `snap` `HTL_energy` and `ETL_energy` to the nearest level
  3. evaluate the constraints in original units and return a feasibility flag
  4. return the snapped 11-value input, ready for the emulator
- Add two notes to the report for downstream use:
  - The benchmark log should record both the **proposed** and the **snapped** HTL/ETL values.
  - Optimizers see a staircase along these two dimensions.

### Step 11: Feasibility flags

For each of the 526 rows, add boolean columns:

- `in_bounds`: the 9 ordinary continuous variables lie within the Step 9 bounds. HTL and ETL are always within range.
- one column per required constraint in Step 9
- `satisfies_constraints`: all required constraints hold
- `feasible`: `in_bounds` and `satisfies_constraints`

Expected counts: 461 in bounds, 502 satisfying all constraints, 450 feasible. The maximum PCE among feasible rows is 14.29%.

**Known flags to report.** Do not change these values; list them for the user to verify against the source papers (DOIs are in the rows):

- **PBDB-TF:Y6 (15.71%, the highest-PCE device)** fails `gap_consistency_A`. Its acceptor optical gap is recorded as 1.81 eV, but Y6 is usually reported at about 1.3–1.4 eV.
- **PBDTTz-N** fails `gap_consistency_D`. Its HOMO (−5.23 eV) and LUMO (−3.52 eV) give a HOMO–LUMO gap of 1.71 eV, but its optical gap is recorded as 1.95 eV, which is 0.24 eV larger. It is the only donor beyond the tolerance.

## Outputs

1. **`preprocess.py`**: one reproducible script that performs Steps 1–11. It must include `snap`, `to_unit`, `from_unit`, and `prepare_query`.
2. **`opv_emulator_data.csv`**: 526 rows. Columns:
   - metadata (Step 8)
   - the 11 inputs under their short names, in original units
   - the same 11 inputs with a `_unit` suffix (scaled)
   - `PCE`
   - the feasibility flags
3. **`interlayer_levels.csv`**: one row per HTL and ETL level. Columns: variable, level, material(s), `n_devices`, interval lower, interval upper, share of range.
4. **`interlayer_support.csv`**: 42 rows, one per (HTL_energy, ETL_energy) combination. Columns: `HTL_energy`, `ETL_energy`, HTL material(s), ETL material(s), `n_devices` (0 if unobserved).
5. **`search_space.json`**, using this structure:

```json
{
  "target": {"name": "PCE", "source_column": "calculated PCE (%)", "unit": "%", "direction": "maximize"},
  "inputs": ["HOMO_D", "LUMO_D", "Eg_D", "HOMO_A", "LUMO_A", "Eg_A",
             "DA_ratio", "additive_conc", "anneal_temp", "HTL_energy", "ETL_energy"],
  "continuous": {
    "HOMO_D": {"column": "HOMO_D (eV)", "lower": -5.60, "upper": -5.06, "unit": "eV"},
    "HTL_energy": {"column": "HTL energy level (eV)", "lower": -5.30, "upper": -4.70, "unit": "eV",
                   "snap_to_levels": [-5.30, -5.20, -4.70], "tie_break": "more_negative"}
  },
  "linear_constraints": [
    {"name": "LUMO_offset", "coefficients": {"LUMO_D": 1, "LUMO_A": -1}, "op": ">=", "rhs": -0.15, "optional": false}
  ],
  "files": {"interlayer_levels": "interlayer_levels.csv", "interlayer_support": "interlayer_support.csv"},
  "notes": "Bounds for the first 9 inputs are 1st-99th percentiles of the 541 complete rows. HTL_energy and ETL_energy are continuous but snapped to the nearest observed level before querying the emulator. Constraints are in original units."
}
```

Fill in all 11 inputs (with `snap_to_levels` for HTL and ETL) and every constraint. Express two-sided constraints as two entries.

6. **`preprocessing_report.md`**, containing:
   - a table of row counts after each step, next to the expected counts
   - the HTL and ETL level tables with capture intervals and shares, the 3 × 14 support table, and the PDINO inconsistency
   - the replicate summary (groups, median and maximum `PCE_std`, the mixed-acceptor group)
   - the feasibility counts, plus the 10 highest-PCE rows with the constraints each one fails
   - the flagged PBDB-TF:Y6 and PBDTTz-N rows
   - the downstream notes from Step 10
   - the known limitations listed below

## Sanity checks (expected values)

| Check | Expected |
|---|---|
| Input shape | 558 × 45 |
| max \|Voc × Jsc × FF − calculated PCE\| | < 1e-6 |
| max \|Ehl − (LUMO − HOMO)\| | < 1e-6 |
| Rows after dropping missing | 541 |
| Distinct HTL / ETL levels | 3 / 14 |
| Training rows changed by snapping | 0 |
| HTL capture shares (−5.30 / −5.20 / −4.70) | 8.3% / 50.0% / 41.7% |
| Observed HTL × ETL combinations | 18 of 42 |
| Replicate groups / rows involved | 13 / 28 |
| Final rows | 526 |
| Emulator inputs | 11 |
| PCE range (final) | 0.01–15.71% |
| In bounds / satisfy constraints / feasible | 461 / 502 / 450 |
| Max PCE among feasible rows | 14.29% |

## Known limitations (report these; do not try to fix them)

- **Snapping gives levels unequal shares of the search space.** A level's share depends on the gaps to its neighbors, not on how many devices support it. V2O5 (2 devices) captures 41.7% of the HTL range, so a uniform random search would query it about 42% of the time. Several ETL levels with 1–2 devices also capture sizeable intervals.
- **Snapping creates a staircase.** Along HTL and ETL, PCE changes only at interval boundaries. Gradient-based optimizers may struggle along these dimensions.
- **HTL and ETL are independent.** 24 of the 42 combinations never occur in the data, and most of these mix conventional and inverted device designs. Emulator predictions there are not backed by data.
- **Energy levels vary independently.** Donor and acceptor energies are continuous, so they can form combinations that no real molecule has. The constraints limit this but do not prevent it.
- **No structural information.** Molecular structure is not an emulator input, so the emulator distinguishes materials only by their energy levels.
- **Interlayer energies mix different quantities.** HTL/ETL values combine work functions, band edges, and LUMOs, and different materials share the same value.
- **Solvent additive identity is not a feature.** 0.5% DIO and 0.5% CN look identical.
- **Annealing time is not a feature.** Temperature comes with no duration.
- **Energy levels are noisy.** They come from mixed measurement methods and were averaged per material.
- **The emulator can only interpolate.** Results near the bounds, or in sparsely sampled regions, are less reliable.
