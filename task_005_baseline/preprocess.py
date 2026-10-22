"""Pre-process the Seifrid et al. (2024) OPV dataset for the emulator benchmark.

Implements Steps 1-11 of opv_preprocessing_prompt.md (pre-processing only: no
emulator training, no optimization). Run with

    python preprocess.py

It writes, next to this file: opv_emulator_data.csv, interlayer_levels.csv,
interlayer_support.csv, search_space.json and preprocessing_report.md. Every
sanity check raises if its value differs from the expected one.

Downstream code imports snap, to_unit, from_unit and prepare_query from here;
the last three read search_space.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_CSV = HERE / "pce_cleaned_dataset.csv"
SPACE_JSON = HERE / "search_space.json"

TARGET_COL = "calculated PCE (%)"

# Short name -> exact column name (Step 3).
FEATURES = {
    "HOMO_D": "HOMO_D (eV)",
    "LUMO_D": "LUMO_D (eV)",
    "Eg_D": "Eg_D (eV)",
    "HOMO_A": "HOMO_A (eV)",
    "LUMO_A": "LUMO_A (eV)",
    "Eg_A": "Eg_A (eV)",
    "DA_ratio": "D:A ratio (m/m)",
    "additive_conc": "solvent additive conc. (% v/v)",
    "anneal_temp": "temperature of thermal annealing",
    "HTL_energy": "HTL energy level (eV)",
    "ETL_energy": "ETL energy level (eV)",
}
INPUTS = list(FEATURES)
ORDINARY = INPUTS[:9]
SNAPPED = ["HTL_energy", "ETL_energy"]
UNITS = {name: "eV" for name in INPUTS}
UNITS.update({"DA_ratio": "mass ratio (donor/acceptor)", "additive_conc": "% v/v",
              "anneal_temp": "degC (25 = not annealed)"})

TEXT_METADATA = ["Donor", "Acceptor", "DOI", "Donor SMILES", "Acceptor SMILES", "HTL", "ETL"]
METADATA = TEXT_METADATA + ["interlayer_support", "n_replicates", "PCE_std"]

# Step 9 bounds: fixed values, never recomputed from the data.
BOUNDS = {
    "HOMO_D": (-5.60, -5.06),
    "LUMO_D": (-3.70, -2.90),
    "Eg_D": (1.55, 2.05),
    "HOMO_A": (-6.07, -5.24),
    "LUMO_A": (-4.18, -3.32),
    "Eg_A": (1.23, 2.19),
    "DA_ratio": (0.50, 1.67),
    "additive_conc": (0.0, 3.6),
    "anneal_temp": (25, 160),
    "HTL_energy": (-5.30, -4.70),
    "ETL_energy": (-4.65, -2.14),
}

# Step 9 linear constraints in original units: (name, coefficients, op, rhs, optional).
CONSTRAINTS = [
    ("LUMO_offset", {"LUMO_D": 1, "LUMO_A": -1}, ">=", -0.15, False),
    ("HOMO_offset", {"HOMO_D": 1, "HOMO_A": -1}, ">=", -0.15, False),
    ("eff_gap_min", {"LUMO_A": 1, "HOMO_D": -1}, ">=", 1.05, False),
    ("eff_gap_max", {"LUMO_A": 1, "HOMO_D": -1}, "<=", 1.96, False),
    ("gap_consistency_D", {"Eg_D": 1, "LUMO_D": -1, "HOMO_D": 1}, "<=", 0.15, False),
    ("gap_consistency_A", {"Eg_A": 1, "LUMO_A": -1, "HOMO_A": 1}, "<=", 0.15, False),
    ("gap_excess_D", {"LUMO_D": 1, "HOMO_D": -1, "Eg_D": -1}, "<=", 0.5, True),
    ("gap_excess_A", {"LUMO_A": 1, "HOMO_A": -1, "Eg_A": -1}, "<=", 0.5, True),
]
REQUIRED = [name for name, _, _, _, optional in CONSTRAINTS if not optional]

# Absorbs float error in constraint sums and at snapping midpoints (e.g. -3.45 - -3.30).
TOL = 1e-9


# ---------------------------------------------------------------------------
# Query pipeline (Steps 5b and 10)
# ---------------------------------------------------------------------------

def snap(x, levels):
    """Return the observed level nearest to x. Exact ties go to the more negative level;
    values outside the range snap to the nearest end level."""
    levels = np.sort(np.asarray(levels, dtype=float))
    dist = np.abs(levels - x)
    return float(levels[np.flatnonzero(dist <= dist.min() + TOL)[0]])


def load_space(path=SPACE_JSON):
    with open(path) as f:
        return json.load(f)


def to_unit(df, space=None):
    """Map the input columns of df to [0, 1] with the search-space bounds. No clipping."""
    space = space or load_space()
    out = df.copy()
    for name in space["inputs"]:
        b = space["continuous"][name]
        out[name] = (df[name] - b["lower"]) / (b["upper"] - b["lower"])
    return out


def from_unit(df, space=None):
    """Inverse of to_unit: map [0, 1] input columns back to original units."""
    space = space or load_space()
    out = df.copy()
    for name in space["inputs"]:
        b = space["continuous"][name]
        out[name] = b["lower"] + df[name] * (b["upper"] - b["lower"])
    return out


def check_constraints(df, space=None, include_optional=False):
    """Evaluate the linear constraints on df (original units). True = satisfied."""
    space = space or load_space()
    result = {}
    for c in space["linear_constraints"]:
        if c["optional"] and not include_optional:
            continue
        lhs = sum(coef * df[var] for var, coef in c["coefficients"].items())
        if c["op"] == ">=":
            result[c["name"]] = lhs >= c["rhs"] - TOL
        else:
            result[c["name"]] = lhs <= c["rhs"] + TOL
    return pd.DataFrame(result, index=df.index)


def prepare_query(x_unit, space=None):
    """Turn one optimizer proposal in [0, 1] scaling into an emulator query.

    x_unit: 11 values in space["inputs"] order, or a dict/Series keyed by input name.
    Returns a dict with
      x           the snapped 11-value input in original units, in space["inputs"] order
      x_proposed  the same before snapping (log both HTL/ETL values, Step 10)
      feasible    True if every required constraint holds
      constraints per-constraint results (True = satisfied)
    """
    space = space or load_space()
    names = space["inputs"]
    if isinstance(x_unit, (dict, pd.Series)):
        values = [float(x_unit[n]) for n in names]
    else:
        values = np.asarray(x_unit, dtype=float).ravel()
        if values.size != len(names):
            raise ValueError(f"expected {len(names)} values, got {values.size}")
    proposed = from_unit(pd.DataFrame([values], columns=names), space)
    snapped = proposed.copy()
    for name in names:
        levels = space["continuous"][name].get("snap_to_levels")
        if levels is not None:
            snapped[name] = snap(proposed[name].iloc[0], levels)
    ok = check_constraints(snapped, space).iloc[0]
    return {
        "x": snapped.iloc[0].to_numpy(dtype=float),
        "x_proposed": proposed.iloc[0].to_numpy(dtype=float),
        "feasible": bool(ok.all()),
        "constraints": {k: bool(v) for k, v in ok.items()},
    }


# ---------------------------------------------------------------------------
# Helpers for the pre-processing run
# ---------------------------------------------------------------------------

def capture_intervals(levels, lower, upper):
    """Part of [lower, upper] that snaps to each level, bounded by the midpoints to its neighbours."""
    levels = sorted(levels)
    edges = [lower] + [(a + b) / 2 for a, b in zip(levels[:-1], levels[1:])] + [upper]
    return [(lv, edges[i], edges[i + 1]) for i, lv in enumerate(levels)]


def join_unique(s):
    vals = list(dict.fromkeys(s.dropna().astype(str)))
    return "; ".join(vals) if vals else np.nan


def material_counts(names):
    return "; ".join(f"{n} ({c})" for n, c in names.value_counts().items())


def md_table(df, fmt=None):
    """Render df as a Markdown table. fmt maps column -> format string for floats."""
    fmt = fmt or {}
    cells = {}
    for c in df.columns:
        f = fmt.get(c, "{:.2f}")
        if pd.api.types.is_float_dtype(df[c]):
            cells[c] = df[c].map(lambda v: "" if pd.isna(v) else f.format(v))
        else:
            cells[c] = df[c].astype(str)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in zip(*(cells[c] for c in df.columns))]
    return "\n".join(lines)


def build_space(htl_levels, etl_levels):
    continuous = {}
    for name in INPUTS:
        lo, hi = BOUNDS[name]
        continuous[name] = {"column": FEATURES[name], "lower": lo, "upper": hi, "unit": UNITS[name]}
    continuous["HTL_energy"].update(snap_to_levels=htl_levels, tie_break="more_negative")
    continuous["ETL_energy"].update(snap_to_levels=etl_levels, tie_break="more_negative")
    return {
        "target": {"name": "PCE", "source_column": TARGET_COL, "unit": "%", "direction": "maximize"},
        "inputs": INPUTS,
        "continuous": continuous,
        "linear_constraints": [
            {"name": name, "coefficients": coefs, "op": op, "rhs": rhs, "optional": optional}
            for name, coefs, op, rhs, optional in CONSTRAINTS
        ],
        "files": {"interlayer_levels": "interlayer_levels.csv",
                  "interlayer_support": "interlayer_support.csv"},
        "notes": ("Bounds for the first 9 inputs are 1st-99th percentiles of the 541 complete rows. "
                  "HTL_energy and ETL_energy are continuous but snapped to the nearest observed level "
                  "before querying the emulator. Constraints are in original units."),
    }


# ---------------------------------------------------------------------------
# Steps 1-11
# ---------------------------------------------------------------------------

def main():
    checks = []      # (check, observed, expected)
    row_counts = []  # (step, description, rows, expected)

    def check(name, observed, expected, ok):
        checks.append((name, str(observed), str(expected)))
        if not ok:
            raise RuntimeError(f"Sanity check failed: {name}: got {observed}, expected {expected}")

    def count(step, description, df, expected):
        row_counts.append((step, description, len(df), expected))
        check(f"Rows after step {step}", len(df), expected, len(df) == expected)

    # Step 1: load and validate. FF (%) is stored as a fraction, so no /100.
    raw = pd.read_csv(RAW_CSV)
    check("Input shape", f"{raw.shape[0]} x {raw.shape[1]}", "558 x 45", raw.shape == (558, 45))
    ff = raw["FF (%)"]
    check("FF stored as a fraction", f"{ff.min():.2f}-{ff.max():.2f}", "0.24-0.84", ff.max() <= 1)
    product = raw["Voc (V)"] * raw["Jsc (mA cm^-2)"] * raw["FF (%)"]
    check("Target and Voc x Jsc x FF present", int(product.isna().sum() + raw[TARGET_COL].isna().sum()),
          0, product.notna().all() and raw[TARGET_COL].notna().all())
    pce_err = (product - raw[TARGET_COL]).abs().max()
    check("max abs(Voc x Jsc x FF - calculated PCE)", f"{pce_err:.1e}", "< 1e-6", pce_err < 1e-6)
    count(1, "Load and validate", raw, 558)

    # Step 2: target is the calculated PCE (renamed PCE below), not the reported PCE (%).
    count(2, "Target = calculated PCE (%)", raw, 558)

    # Step 3: features. Ehl_D / Ehl_A are redundant with LUMO - HOMO and are excluded.
    ehl_err = max((raw["Ehl_D (eV)"] - (raw["LUMO_D (eV)"] - raw["HOMO_D (eV)"])).abs().max(),
                  (raw["Ehl_A (eV)"] - (raw["LUMO_A (eV)"] - raw["HOMO_A (eV)"])).abs().max())
    check("max abs(Ehl - (LUMO - HOMO))", f"{ehl_err:.1e}", "< 1e-6", ehl_err < 1e-6)
    count(3, "Select 11 features", raw, 558)

    # Step 4: drop rows with a missing feature (the only rows ever dropped).
    missing = raw[list(FEATURES.values())].isna().sum()
    complete = raw.dropna(subset=list(FEATURES.values()))
    df = pd.DataFrame({c: complete[c] for c in TEXT_METADATA})
    for name, col in FEATURES.items():
        df[name] = complete[col]
    df["PCE"] = complete[TARGET_COL]
    df = df.reset_index(drop=True)
    count(4, "Drop rows with missing features", df, 541)

    # Step 5a: observed interlayer levels.
    levels = {v: sorted(df[v].round(2).unique().tolist()) for v in SNAPPED}
    check("Distinct HTL / ETL levels", f"{len(levels['HTL_energy'])} / {len(levels['ETL_energy'])}",
          "3 / 14", (len(levels["HTL_energy"]), len(levels["ETL_energy"])) == (3, 14))

    # Step 5b: training rows already sit on observed levels.
    changed = sum(int((df[v].map(lambda x, lv=levels[v]: snap(x, lv)) != df[v]).sum()) for v in SNAPPED)
    check("Training rows changed by snapping", changed, 0, changed == 0)
    count(5, "Interlayer levels and snapping", df, 541)

    # Step 5c: capture intervals and shares of the range.
    material_col = {"HTL_energy": "HTL", "ETL_energy": "ETL"}
    level_rows = []
    for v in SNAPPED:
        lo, hi = BOUNDS[v]
        for lv, a, b in capture_intervals(levels[v], lo, hi):
            names = df.loc[df[v].round(2) == lv, material_col[v]]
            level_rows.append({"variable": v, "level": lv, "materials": material_counts(names),
                               "n_devices": len(names), "interval_lower": round(a, 6),
                               "interval_upper": round(b, 6), "share_of_range": (b - a) / (hi - lo)})
    level_table = pd.DataFrame(level_rows)
    level_table.to_csv(HERE / "interlayer_levels.csv", index=False)

    share = {(r.variable, r.level): round(100 * r.share_of_range, 1) for r in level_table.itertuples()}
    htl_shares = [share[("HTL_energy", lv)] for lv in levels["HTL_energy"]]
    check("HTL capture shares (-5.30 / -5.20 / -4.70)", " / ".join(f"{s}%" for s in htl_shares),
          "8.3% / 50.0% / 41.7%", htl_shares == [8.3, 50.0, 41.7])
    etl_expected = {-3.60: 17.9, -2.90: 16.3, -2.18: 15.1, -3.00: 13.9, -4.46: 8.4, -2.14: 0.8}
    etl_observed = {lv: share[("ETL_energy", lv)] for lv in etl_expected}
    etl_top4 = level_table[level_table.variable == "ETL_energy"].nlargest(4, "share_of_range")["level"]
    check("ETL capture shares (-3.60 / -2.90 / -2.18 / -3.00 / -4.46 / -2.14)",
          " / ".join(f"{s}%" for s in etl_observed.values()), " / ".join(f"{s}%" for s in etl_expected.values()),
          etl_observed == etl_expected and sorted(etl_top4) == sorted([-3.60, -2.90, -2.18, -3.00]))

    # Step 5d: support of every (HTL, ETL) combination.
    combo = df.groupby([df["HTL_energy"].round(2), df["ETL_energy"].round(2)]).size()
    level_names = {(v, lv): "; ".join(df.loc[df[v].round(2) == lv, material_col[v]].value_counts().index)
                   for v in SNAPPED for lv in levels[v]}
    support = pd.DataFrame(
        [{"HTL_energy": h, "ETL_energy": e,
          "HTL_materials": level_names[("HTL_energy", h)], "ETL_materials": level_names[("ETL_energy", e)],
          "n_devices": int(combo.get((h, e), 0))}
         for h in levels["HTL_energy"] for e in levels["ETL_energy"]])
    support.to_csv(HERE / "interlayer_support.csv", index=False)
    n_observed = int((support.n_devices > 0).sum())
    check("Observed HTL x ETL combinations", f"{n_observed} of {len(support)}", "18 of 42",
          (n_observed, len(support)) == (18, 42))
    df["interlayer_support"] = [int(combo[(h, e)]) for h, e in zip(df["HTL_energy"].round(2),
                                                                   df["ETL_energy"].round(2))]

    # Materials recorded at more than one energy level (e.g. PDINO). Reported, not changed.
    multi_level = []
    for v in SNAPPED:
        per_material = df.groupby(material_col[v])[v].agg(lambda s: s.round(2).value_counts().to_dict())
        for material, counts in per_material.items():
            if len(counts) > 1:
                multi_level.append((material_col[v], material, counts))

    # Step 6: average exact replicates (same 11 features to 6 decimals).
    keys = [f"_key_{name}" for name in INPUTS]
    grouped = df.assign(**{k: df[name].round(6) for k, name in zip(keys, INPUTS)}).groupby(keys, sort=False)
    agg = {name: (name, "first") for name in INPUTS}
    agg.update({c: (c, join_unique) for c in TEXT_METADATA})
    agg.update(interlayer_support=("interlayer_support", "first"), PCE=("PCE", "mean"),
               PCE_std=("PCE", "std"), n_replicates=("PCE", "size"))
    check("interlayer_support constant within replicate groups", "yes", "yes",
          (grouped["interlayer_support"].nunique() == 1).all())
    final = grouped.agg(**agg).reset_index(drop=True)
    reps = final[final.n_replicates > 1]
    check("Replicate groups / rows involved", f"{len(reps)} / {reps.n_replicates.sum()}", "13 / 28",
          (len(reps), reps.n_replicates.sum()) == (13, 28))
    std_med, std_max = reps.PCE_std.median(), reps.PCE_std.max()
    check("Replicate PCE_std median / max", f"{std_med:.2f} / {std_max:.2f}", "0.38 / 2.09",
          (round(std_med, 2), round(std_max, 2)) == (0.38, 2.09))
    mixed = reps[reps.Acceptor.str.contains(";") | reps.Donor.str.contains(";")]
    check("Mixed-acceptor replicate group", "; ".join(mixed.Donor + " with " + mixed.Acceptor),
          "PBDB-T with ITBTR-C4; ITBTR-C6",
          len(mixed) == 1 and mixed.Donor.iloc[0] == "PBDB-T"
          and set(mixed.Acceptor.iloc[0].split("; ")) == {"ITBTR-C4", "ITBTR-C6"})
    count(6, "Average exact replicates", final, 526)

    # Step 7 (types) and Step 8 (metadata) are structural: 11 numeric inputs, no one-hot encoding,
    # metadata kept only for reporting.
    check("Emulator inputs", len(INPUTS), 11, len(INPUTS) == 11)
    count("7-8", "Variable types and metadata", final, 526)

    # Step 9: search space with fixed bounds and linear constraints.
    space = build_space(levels["HTL_energy"], levels["ETL_energy"])
    with open(SPACE_JSON, "w") as f:
        json.dump(space, f, indent=2)
    space = load_space()

    # Step 10: scale to [0, 1] (no clipping, target unscaled) and verify the query pipeline.
    unit = to_unit(final[INPUTS], space)
    roundtrip = (from_unit(unit, space) - final[INPUTS]).abs().max().max()
    check("max abs(from_unit(to_unit(x)) - x)", f"{roundtrip:.1e}", "< 1e-9", roundtrip < 1e-9)
    count("9-10", "Search space and scaling", final, 526)

    # Step 11: feasibility flags.
    in_bounds = pd.Series(True, index=final.index)
    out_of_bounds = pd.Series("", index=final.index)
    for name in ORDINARY:
        lo, hi = BOUNDS[name]
        bad = (final[name] < lo) | (final[name] > hi)
        in_bounds &= ~bad
        out_of_bounds[bad] += name + " "
    flags = check_constraints(final, space)
    flags.insert(0, "in_bounds", in_bounds)
    flags["satisfies_constraints"] = flags[REQUIRED].all(axis=1)
    flags["feasible"] = flags.in_bounds & flags.satisfies_constraints
    n_in, n_sat, n_feas = int(flags.in_bounds.sum()), int(flags.satisfies_constraints.sum()), int(flags.feasible.sum())
    # The prompt expects 461 / 502 / 450. Four rows sit exactly on a constraint boundary; counting them as
    # satisfied (as >= and <= intend) gives 505 / 452. Decided with the user; see the report.
    check("In bounds / satisfy constraints / feasible", f"{n_in} / {n_sat} / {n_feas}",
          "461 / 505 / 452 (prompt: 461 / 502 / 450)", (n_in, n_sat, n_feas) == (461, 505, 452))
    boundary = []
    for c in space["linear_constraints"]:
        if c["optional"]:
            continue
        lhs = sum(coef * final[var] for var, coef in c["coefficients"].items())
        plain = lhs >= c["rhs"] if c["op"] == ">=" else lhs <= c["rhs"]
        for i in final.index[(lhs - c["rhs"]).abs() <= TOL]:
            boundary.append((final.Donor[i], final.Acceptor[i], final.PCE[i], c["name"],
                             f"{c['op']} {c['rhs']}", f"{lhs[i] - c['rhs']:+.1e}",
                             "pass" if plain[i] else "fail", bool(flags.feasible[i])))
    boundary = pd.DataFrame(boundary, columns=["Donor", "Acceptor", "PCE", "Constraint", "Bound",
                                               "Float error", "Plain float comparison", "Feasible"])
    max_feasible = final.loc[flags.feasible, "PCE"].max()
    check("Max PCE among feasible rows", f"{max_feasible:.2f}%", "14.29%", round(max_feasible, 2) == 14.29)
    pce_lo, pce_hi = final.PCE.min(), final.PCE.max()
    check("PCE range (final)", f"{pce_lo:.2f}-{pce_hi:.2f}%", "0.01-15.71%",
          (round(pce_lo, 2), round(pce_hi, 2)) == (0.01, 15.71))

    # prepare_query on every row's scaled inputs must give back the row itself.
    mismatches = 0
    for i, row in unit.iterrows():
        q = prepare_query(row, space)
        mismatches += int(np.abs(q["x"] - final.loc[i, INPUTS].to_numpy(dtype=float)).max() > 1e-9
                          or q["feasible"] != flags.loc[i, "satisfies_constraints"])
    check("prepare_query reproduces training rows and their constraint flags", mismatches, 0, mismatches == 0)
    count(11, "Feasibility flags", final, 526)

    # Known flags (Step 11).
    failures = flags[REQUIRED].apply(lambda r: ", ".join(c for c in REQUIRED if not r[c]), axis=1)
    top = final.PCE.idxmax()
    check("Highest-PCE device is PBDB-TF:Y6 and fails gap_consistency_A",
          f"{final.Donor[top]}:{final.Acceptor[top]}, fails [{failures[top]}]",
          "PBDB-TF:Y6, fails [gap_consistency_A]",
          (final.Donor[top], final.Acceptor[top]) == ("PBDB-TF", "Y6") and failures[top] == "gap_consistency_A")
    bad_donors = sorted(final.loc[~flags.gap_consistency_D, "Donor"].unique())
    check("Donors failing gap_consistency_D", ", ".join(bad_donors), "PBDTTz-N", bad_donors == ["PBDTTz-N"])

    # Outputs.
    unit_cols = unit.add_suffix("_unit")
    out = pd.concat([final[METADATA], final[INPUTS], unit_cols, final[["PCE"]], flags], axis=1)
    out.to_csv(HERE / "opv_emulator_data.csv", index=False)

    write_report(row_counts, checks, missing, level_table, support, levels, multi_level, reps, mixed,
                 final, flags, failures, out_of_bounds, boundary)

    print(md_table(pd.DataFrame(row_counts, columns=["Step", "Description", "Rows", "Expected"])))
    print(f"\nAll {len(checks)} sanity checks passed. Outputs written to {HERE}")


def write_report(row_counts, checks, missing, level_table, support, levels, multi_level, reps, mixed,
                 final, flags, failures, out_of_bounds, boundary):
    L = []
    add = L.append

    add("# OPV dataset pre-processing report\n")
    add("Generated by `preprocess.py` from `pce_cleaned_dataset.csv` (Seifrid et al., *J. Mater. Chem. A*, 2024; "
        "558 devices). Pre-processing only: no emulator was trained and no optimizer was run.\n")

    add("## Row counts\n")
    add(md_table(pd.DataFrame(row_counts, columns=["Step", "Description", "Rows", "Expected"])) + "\n")
    dropped = ", ".join(f"`{c}` ({n})" for c, n in missing.items() if n)
    add(f"Step 4 dropped {row_counts[2][2] - row_counts[3][2]} rows. Missing values per feature "
        f"(a row can miss several): {dropped}.\n")

    add("## Sanity checks\n")
    add("Every check below is enforced by the script, which stops if a value differs.\n")
    add(md_table(pd.DataFrame(checks, columns=["Check", "Observed", "Expected"])) + "\n")

    add("## Target and features\n")
    add("- Target: `calculated PCE (%)` = Voc x Jsc x FF, renamed `PCE`. The reported `PCE (%)` is not used. "
        "`FF (%)` is stored as a fraction (0.24-0.84) and was not divided by 100.")
    add("- 11 inputs (all continuous, no one-hot encoding): " + ", ".join(f"`{n}`" for n in INPUTS) + ".")
    add("- `Ehl_D (eV)` and `Ehl_A (eV)` are excluded: they equal LUMO - HOMO exactly (see sanity checks).")
    add("- Metadata, never emulator inputs: " + ", ".join(f"`{c}`" for c in METADATA) + ".\n")

    add("## Interlayer levels and snapping\n")
    add("`snap(x, levels)` returns the nearest observed level; an exact tie goes to the more negative level, "
        "and values outside the range go to the nearest end level. All 541 training rows already sit on "
        "observed levels, so snapping changes none of them.\n")
    for v in SNAPPED:
        t = level_table[level_table.variable == v].copy()
        t["capture interval"] = [f"[{a:.3f}, {b:.3f}]" for a, b in zip(t.interval_lower, t.interval_upper)]
        t["share of range"] = [f"{100 * s:.1f}%" for s in t.share_of_range]
        t = t.rename(columns={"level": v, "materials": "material(s) (devices)", "n_devices": "devices"})
        lo, hi = BOUNDS[v]
        add(f"### {v.split('_')[0]} levels ({len(t)}; range {lo:.2f} to {hi:.2f})\n")
        add(md_table(t[[v, "material(s) (devices)", "devices", "capture interval", "share of range"]]) + "\n")
    add("A level's share of the search space depends on the gaps to its neighbours, not on how many devices "
        "support it. V2O5 (2 devices) captures 41.7% of the HTL range while MoOx (229 devices) captures 8.3%; "
        "on the ETL axis, -3.60 (PDIN, 17 devices) captures the largest share (17.9%) and ZnO "
        "(-4.46, 210 devices) only 8.4%.\n")

    add("### HTL x ETL support (device counts)\n")
    grid = support.pivot(index="ETL_energy", columns="HTL_energy", values="n_devices")
    grid.columns = [f"HTL {h:.2f} ({support.loc[support.HTL_energy == h, 'HTL_materials'].iloc[0]})"
                    for h in grid.columns]
    etl_names = support.drop_duplicates("ETL_energy").set_index("ETL_energy").ETL_materials
    grid.insert(0, "ETL material(s)", etl_names.reindex(grid.index))
    grid = grid.reset_index().rename(columns={"ETL_energy": "ETL"})
    add(md_table(grid, fmt={"ETL": "{:.2f}"}) + "\n")
    n_obs = int((support.n_devices > 0).sum())
    add(f"{n_obs} of {len(support)} combinations appear in the data; {len(support) - n_obs} never occur. Most "
        "unobserved combinations mix the conventional design (PEDOT:PSS with Ca, PDINO, PFN-Br or similar) and "
        "the inverted design (MoOx with ZnO). Each row's combination count is stored as `interlayer_support`.\n")

    add("### Flag: interlayer materials recorded at more than one energy\n")
    add("Not changed; listed for verification against the source papers.\n")
    for layer, material, counts in multi_level:
        desc = ", ".join(f"{lv:.2f} eV in {n} row{'s' if n > 1 else ''}" for lv, n in counts.items())
        add(f"- {layer} **{material}**: {desc}.")
    add("\nThe single PDINO row at -2.18 eV is an encoding inconsistency (PDINO is otherwise -3.90 eV).\n")

    add("## Replicate averaging\n")
    add(f"Rows with identical values for all 11 features (rounded to 6 decimals) were averaged: "
        f"{len(reps)} groups covering {reps.n_replicates.sum()} rows, leaving {len(final)} rows. `PCE` is the "
        f"group mean, `PCE_std` the sample standard deviation (NaN for single rows) and `n_replicates` the group "
        f"size; metadata columns hold the group's unique values joined with \"; \".\n")
    add(f"Replicate `PCE_std`: median {reps.PCE_std.median():.2f}, maximum {reps.PCE_std.max():.2f} "
        "(percentage points). Use this as a noise estimate for later modelling.\n")
    add(md_table(reps[["Donor", "Acceptor", "n_replicates", "PCE", "PCE_std"]]
                 .sort_values("PCE", ascending=False)) + "\n")
    for r in mixed.itertuples():
        add(f"One group mixes two acceptors with identical energy values: {r.Donor} with "
            f"{r.Acceptor.replace('; ', ' and ')}. This is expected and was kept as one group.\n")

    add("## Search space\n")
    add("Fixed bounds from the prompt (not recomputed) and linear constraints in original units are stored in "
        "`search_space.json`. HTL_energy and ETL_energy span their lowest to highest observed level and are "
        "snapped before the emulator is queried. No constraint involves them, so snapping never changes "
        "feasibility.\n")
    bounds = pd.DataFrame([(n, *BOUNDS[n], UNITS[n]) for n in INPUTS],
                          columns=["Variable", "Lower", "Upper", "Unit"])
    add(md_table(bounds) + "\n")
    cons = []
    for name, coefs, op, rhs, optional in CONSTRAINTS:
        lhs = " ".join(f"{'+' if c > 0 else '-'} {v}" for v, c in coefs.items()).lstrip("+ ")
        cons.append((name, f"{lhs} {op} {rhs}", "optional" if optional else "required"))
    add(md_table(pd.DataFrame(cons, columns=["Name", "Inequality", "Status"])) + "\n")

    add("## Feasibility\n")
    n_in, n_sat, n_feas = int(flags.in_bounds.sum()), int(flags.satisfies_constraints.sum()), int(flags.feasible.sum())
    add(f"- In bounds (9 ordinary variables): {n_in} of {len(final)}")
    add(f"- Satisfy all required constraints: {n_sat}")
    add(f"- Feasible (both): {n_feas}")
    add(f"- Maximum PCE among feasible rows: {final.loc[flags.feasible, 'PCE'].max():.2f}%\n")
    add("Constraint flag columns in `opv_emulator_data.csv` are named after the constraint; True means satisfied.\n")
    fails = pd.DataFrame({"Constraint": REQUIRED, "Rows failing": [int((~flags[c]).sum()) for c in REQUIRED]})
    add(md_table(fails) + "\n")

    add("### Deviation from the expected counts\n")
    add(f"The prompt expects 502 rows satisfying the constraints and 450 feasible; this run gives {n_sat} and "
        f"{n_feas}. The rows below sit exactly on a constraint boundary (their 2-decimal energies give "
        "LUMO_A - HOMO_D = 1.05 or Eg_A - (LUMO_A - HOMO_A) = 0.15 exactly). Floating-point rounding moves them "
        "about 1e-16 either side, so a plain comparison fails some of them on noise alone. Per the user's "
        "decision, boundary rows count as satisfied (as >= and <= intend); `check_constraints` and "
        "`prepare_query` use a 1e-9 tolerance for this. A plain float comparison gives 503 / 450. No rule "
        "tried (comparison order, Ehl columns, rounding) reproduces 502, so one row in the prompt's count "
        "is unexplained. The maximum feasible PCE (14.29%) is the same either way.\n")
    add(md_table(boundary) + "\n")

    add("### Ten highest-PCE rows\n")
    top = final.PCE.sort_values(ascending=False).index[:10]
    t = pd.DataFrame({
        "Donor": final.Donor[top], "Acceptor": final.Acceptor[top], "PCE": final.PCE[top],
        "Out of bounds": out_of_bounds[top].str.strip().replace("", "-"),
        "Fails": failures[top].replace("", "-"), "Feasible": flags.feasible[top],
    })
    add(md_table(t) + "\n")

    add("### Flagged rows (verify against the source papers)\n")
    add("Values were not changed.\n")
    y6 = final[(final.Donor == "PBDB-TF") & (final.Acceptor == "Y6") & ~flags.gap_consistency_A]
    for r in y6.itertuples():
        add(f"- **PBDB-TF:Y6 ({r.PCE:.2f}%, the highest-PCE device)** fails `gap_consistency_A`. Its acceptor "
            f"optical gap is recorded as {r.Eg_A:.2f} eV against a HOMO-LUMO gap of {r.LUMO_A - r.HOMO_A:.2f} eV "
            f"(HOMO {r.HOMO_A:.2f}, LUMO {r.LUMO_A:.2f}); Y6 is usually reported at about 1.3-1.4 eV. DOI: {r.DOI}")
    for donor, g in final[~flags.gap_consistency_D].groupby("Donor"):
        r = g.iloc[0]
        gap = r.LUMO_D - r.HOMO_D
        add(f"- **{donor}** fails `gap_consistency_D` in {len(g)} row(s). HOMO {r.HOMO_D:.2f} eV and LUMO "
            f"{r.LUMO_D:.2f} eV give a HOMO-LUMO gap of {gap:.2f} eV, but its optical gap is recorded as "
            f"{r.Eg_D:.2f} eV ({r.Eg_D - gap:.2f} eV larger). It is the only donor beyond the tolerance. "
            f"DOI(s): {'; '.join(g.DOI.unique())}")
    add("")

    add("## Downstream notes\n")
    add("- Every optimizer proposal goes through `prepare_query(x_unit)`: `from_unit` to original units, snap "
        "HTL_energy and ETL_energy, evaluate the constraints, and return the snapped 11-value input with a "
        "feasibility flag.")
    add("- The benchmark log should record both the **proposed** and the **snapped** HTL/ETL values "
        "(`x_proposed` and `x` in the `prepare_query` result).")
    add("- Optimizers see a staircase along HTL_energy and ETL_energy: PCE changes only at capture-interval "
        "boundaries.")
    add("- Scaled inputs are not clipped; training rows outside the bounds map outside [0, 1] and are kept.")
    add("- The target is not scaled in the saved file. The emulator may standardize PCE with all data; each "
        "optimizer's surrogate must standardize using only the points it has queried.\n")

    add("## Known limitations\n")
    for item in [
        "**Snapping gives levels unequal shares of the search space.** A level's share depends on the gaps to "
        "its neighbours, not on how many devices support it. V2O5 (2 devices) captures 41.7% of the HTL range, "
        "so a uniform random search would query it about 42% of the time. Several ETL levels with 1-2 devices "
        "also capture sizeable intervals.",
        "**Snapping creates a staircase.** Along HTL and ETL, PCE changes only at interval boundaries. "
        "Gradient-based optimizers may struggle along these dimensions.",
        "**HTL and ETL are independent.** 24 of the 42 combinations never occur in the data, and most of these "
        "mix conventional and inverted device designs. Emulator predictions there are not backed by data.",
        "**Energy levels vary independently.** Donor and acceptor energies are continuous, so they can form "
        "combinations that no real molecule has. The constraints limit this but do not prevent it.",
        "**No structural information.** Molecular structure is not an emulator input, so the emulator "
        "distinguishes materials only by their energy levels.",
        "**Interlayer energies mix different quantities.** HTL/ETL values combine work functions, band edges, "
        "and LUMOs, and different materials share the same value.",
        "**Solvent additive identity is not a feature.** 0.5% DIO and 0.5% CN look identical.",
        "**Annealing time is not a feature.** Temperature comes with no duration.",
        "**Energy levels are noisy.** They come from mixed measurement methods and were averaged per material.",
        "**The emulator can only interpolate.** Results near the bounds, or in sparsely sampled regions, are "
        "less reliable.",
    ]:
        add(f"- {item}")
    add("")

    (HERE / "preprocessing_report.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
