"""Pre-process the Seifrid et al. (2024) OPV dataset for the pool-based benchmark.

Implements Steps 1-11 of opv_preprocessing_prompt_pool.md (pre-processing only:
no optimizer is run). Run with

    python preprocess.py

It writes, next to this file: pool_features.csv, pool_labels.csv,
pool_spec.json, initial_sets.json and preprocessing_report.md. Every sanity
check raises if its value differs from the expected one.

Benchmark code imports query, nearest_unqueried and top_k_ids from here
(Step 8). They read only the output files.
"""
import csv
import json
import math
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_CSV = HERE / "pce_cleaned_dataset.csv"
FEATURES_CSV = HERE / "pool_features.csv"
LABELS_CSV = HERE / "pool_labels.csv"
SPEC_JSON = HERE / "pool_spec.json"
INITIAL_SETS_JSON = HERE / "initial_sets.json"
QUERY_LOG = HERE / "query_log.csv"

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
UNIT_COLS = [f"{name}_unit" for name in INPUTS]
UNITS = {name: "eV" for name in INPUTS}
UNITS.update({"DA_ratio": "mass ratio (donor/acceptor)", "additive_conc": "% v/v",
              "anneal_temp": "degC (25 = not annealed)"})

TEXT_METADATA = ["Donor", "Acceptor", "DOI", "Donor SMILES", "Acceptor SMILES", "HTL", "ETL"]

# Step 7 expected pool ranges (checked, not used: scaling uses the computed min and max).
EXPECTED_RANGES = {
    "HOMO_D": (-5.68, -5.01),
    "LUMO_D": (-3.75, -2.78),
    "Eg_D": (1.48, 2.12),
    "HOMO_A": (-6.38, -5.02),
    "LUMO_A": (-4.23, -2.83),
    "Eg_A": (1.10, 2.43),
    "DA_ratio": (0.333, 3.333),
    "additive_conc": (0.0, 6.0),
    "anneal_temp": (25, 175),
    "HTL_energy": (-5.30, -4.70),
    "ETL_energy": (-4.65, -2.14),
}

# Step 11 informational checks in original units: (name, coefficients, op, rhs). Never used as filters.
CHECKS = [
    ("LUMO_offset", [({"LUMO_D": 1, "LUMO_A": -1}, ">=", -0.15)]),
    ("HOMO_offset", [({"HOMO_D": 1, "HOMO_A": -1}, ">=", -0.15)]),
    ("eff_gap_in_range", [({"LUMO_A": 1, "HOMO_D": -1}, ">=", 1.05),
                          ({"LUMO_A": 1, "HOMO_D": -1}, "<=", 1.96)]),
    ("gap_consistency_D", [({"Eg_D": 1, "LUMO_D": -1, "HOMO_D": 1}, "<=", 0.15)]),
    ("gap_consistency_A", [({"Eg_A": 1, "LUMO_A": -1, "HOMO_A": 1}, "<=", 0.15)]),
]
FLAGS = [name for name, _ in CHECKS]

# Absorbs float error in check sums (e.g. -3.90 - -5.05 = 1.15 +- 1e-16) and in distance ties.
TOL = 1e-9

N_INITIAL_SEEDS, INITIAL_SET_SIZE = 30, 15


# ---------------------------------------------------------------------------
# Benchmark functions (Step 8). They read only the output files.
# ---------------------------------------------------------------------------

@cache
def _labels():
    return pd.read_csv(LABELS_CSV, index_col="candidate_id")["PCE"]


@cache
def _unit_features():
    features = pd.read_csv(FEATURES_CSV, index_col="candidate_id").sort_index()
    return features.index.to_numpy(), features[UNIT_COLS].to_numpy(dtype=float)


def query(candidate_id, x_proposed=None, log_path=QUERY_LOG):
    """Return the candidate's PCE from pool_labels.csv and append the query to the log.

    A candidate queried before returns its stored value and is logged with
    repeat=True; it does not count as a new experiment. Pass x_proposed (the
    continuous optimizer's point, in unit scaling) when the candidate came from
    nearest_unqueried, so the log keeps both. Use a fresh log_path per campaign:
    repeats are detected from the log itself.
    """
    candidate_id = int(candidate_id)
    labels = _labels()
    if candidate_id not in labels.index:
        raise KeyError(f"no candidate {candidate_id}")
    log_path = Path(log_path)
    has_log = log_path.exists() and log_path.stat().st_size > 0
    seen = set(pd.read_csv(log_path, usecols=["candidate_id"])["candidate_id"]) if has_log else set()
    repeat = candidate_id in seen
    pce = float(labels[candidate_id])
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not has_log:
            writer.writerow(["candidate_id", "PCE", "timestamp", "repeat", "n_experiments", "x_proposed"])
        writer.writerow([candidate_id, repr(pce), datetime.now(timezone.utc).isoformat(), repeat,
                         len(seen) + (not repeat),
                         "" if x_proposed is None else json.dumps([float(v) for v in np.ravel(x_proposed)])])
    return pce


def nearest_unqueried(x_unit, queried_ids=()):
    """Return the candidate_id closest to x_unit (Euclidean, unit-scaled features), excluding queried_ids.

    x_unit is 11 values in pool_spec.json feature order, or a mapping keyed by
    feature name. Points outside [0, 1] are allowed. Ties go to the lowest
    candidate_id.
    """
    ids, unit = _unit_features()
    if isinstance(x_unit, (dict, pd.Series)):
        x = np.array([float(x_unit[n]) for n in INPUTS])
    else:
        x = np.asarray(x_unit, dtype=float).ravel()
        if x.size != len(INPUTS):
            raise ValueError(f"expected {len(INPUTS)} values, got {x.size}")
    dist = np.sqrt(((unit - x) ** 2).sum(axis=1))
    dist[np.isin(ids, list(queried_ids))] = np.inf
    if np.isinf(dist).all():
        raise ValueError("every candidate has been queried")
    return int(ids[np.flatnonzero(dist <= dist.min() + TOL)[0]])


def top_k_ids(fraction=0.05):
    """IDs of candidates with PCE at or above the (1 - fraction) quantile, best first.

    For evaluation only: an optimizer must never call this.
    """
    labels = _labels()
    threshold = labels.quantile(1 - fraction)
    return labels[labels >= threshold].sort_values(ascending=False, kind="mergesort").index.tolist()


# ---------------------------------------------------------------------------
# Helpers for the pre-processing run
# ---------------------------------------------------------------------------

def evaluate_checks(df):
    """Evaluate the Step 11 checks on df (original units). True = passes."""
    result = {}
    for name, parts in CHECKS:
        ok = pd.Series(True, index=df.index)
        for coefs, op, rhs in parts:
            lhs = sum(coef * df[var] for var, coef in coefs.items())
            ok &= lhs >= rhs - TOL if op == ">=" else lhs <= rhs + TOL
        result[name] = ok
    return pd.DataFrame(result, index=df.index)


def replicate_groups(pool):
    """Merged replicate groups, and those that mix two donor or acceptor names."""
    reps = pool[pool.n_replicates > 1]
    return reps, reps[reps.Acceptor.str.contains(";") | reps.Donor.str.contains(";")]


def join_unique(s):
    vals = list(dict.fromkeys(s.dropna().astype(str)))
    return "; ".join(vals) if vals else np.nan


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

    n_levels = (df.HTL_energy.round(2).nunique(), df.ETL_energy.round(2).nunique())
    check("Distinct HTL / ETL values", "%d / %d" % n_levels, "3 / 14", n_levels == (3, 14))

    # Interlayer materials recorded at more than one energy (e.g. PDINO). Reported, not changed.
    multi_level = []
    for layer, v in (("HTL", "HTL_energy"), ("ETL", "ETL_energy")):
        per_material = df.groupby(layer)[v].agg(lambda s: s.round(2).value_counts().to_dict())
        multi_level += [(layer, material, counts) for material, counts in per_material.items() if len(counts) > 1]
    pdino = next((c for _, material, c in multi_level if material == "PDINO"), {})
    check("PDINO ETL energies (rows)", pdino, "{-3.9: 101, -2.18: 1}", pdino == {-3.9: 101, -2.18: 1})

    # Step 5: merge exact replicates (same 11 features to 6 decimals) into single candidates.
    keys = [f"_key_{name}" for name in INPUTS]
    grouped = df.assign(**{k: df[name].round(6) for k, name in zip(keys, INPUTS)}).groupby(keys, sort=False)
    agg = {name: (name, "first") for name in INPUTS}
    agg.update({c: (c, join_unique) for c in TEXT_METADATA})
    agg.update(PCE=("PCE", "mean"), PCE_std=("PCE", "std"), n_replicates=("PCE", "size"))
    pool = grouped.agg(**agg).reset_index(drop=True)
    reps, mixed = replicate_groups(pool)
    check("Replicate groups / rows involved", f"{len(reps)} / {reps.n_replicates.sum()}", "13 / 28",
          (len(reps), reps.n_replicates.sum()) == (13, 28))
    n_unique = len(pool[INPUTS].round(6).drop_duplicates())
    check("Candidates / unique feature vectors", f"{len(pool)} / {n_unique}", "526 / 526",
          (len(pool), n_unique) == (526, 526))
    std_med, std_max = reps.PCE_std.median(), reps.PCE_std.max()
    check("Replicate PCE_std median / max", f"{std_med:.2f} / {std_max:.2f}", "0.38 / 2.09",
          (round(std_med, 2), round(std_max, 2)) == (0.38, 2.09))
    check("Mixed-acceptor replicate group", "; ".join(mixed.Donor + " with " + mixed.Acceptor),
          "PBDB-T with ITBTR-C4; ITBTR-C6",
          len(mixed) == 1 and mixed.Donor.iloc[0] == "PBDB-T"
          and set(mixed.Acceptor.iloc[0].split("; ")) == {"ITBTR-C4", "ITBTR-C6"})
    count(5, "Merge exact replicates", pool, 526)

    # Step 6: candidate_id from a stable sort; pair_id per (Donor SMILES, Acceptor SMILES), numbered by
    # first candidate_id. The mixed-acceptor group's joined SMILES count as one pair.
    pool = pool.sort_values(["DOI", "Donor", "Acceptor", *INPUTS], kind="mergesort").reset_index(drop=True)
    pool.insert(0, "candidate_id", np.arange(len(pool)))
    pair_key = list(zip(pool["Donor SMILES"], pool["Acceptor SMILES"]))
    pair_ids = {key: i for i, key in enumerate(dict.fromkeys(pair_key))}
    pool.insert(1, "pair_id", [pair_ids[key] for key in pair_key])
    reps, mixed = replicate_groups(pool)
    pair_sizes = pool.pair_id.value_counts()
    shared = pair_sizes[pair_sizes > 1]
    check("Distinct pairs / pairs with more than one candidate / candidates in them",
          f"{len(pair_sizes)} / {len(shared)} / {shared.sum()}", "474 / 33 / 85",
          (len(pair_sizes), len(shared), shared.sum()) == (474, 33, 85))
    # The prompt says same-pair candidates share all six energies. Not quite: a few acceptor molecules are
    # recorded under two names with different energies (e.g. IFTIC / IFT-IC). Reported, not changed.
    split_pairs = [g for _, g in pool[pool.pair_id.isin(shared.index)].groupby("pair_id")
                   if len(g[INPUTS[:6]].drop_duplicates()) > 1]
    check("Pairs with more than one candidate whose candidates share all six energies",
          f"{len(shared) - len(split_pairs)} of {len(shared)}", "29 of 33 (prompt: all)",
          (len(shared) - len(split_pairs), len(shared)) == (29, 33))
    count(6, "Identifiers", pool, 526)

    # Step 7: scale to [0, 1] with the pool min and max (no PCE involved). PCE is never scaled.
    lo, hi = pool[INPUTS].min(), pool[INPUTS].max()
    # The table gives 2 decimals (HOMO_A max -5.0183 is listed as -5.02).
    off = [n for n in INPUTS if abs(lo[n] - EXPECTED_RANGES[n][0]) > 5e-3 or abs(hi[n] - EXPECTED_RANGES[n][1]) > 5e-3]
    check("Pool ranges match the expected table (to 2 decimals)", "all 11" if not off else ", ".join(off), "all 11",
          not off)
    unit = ((pool[INPUTS] - lo) / (hi - lo)).add_suffix("_unit")
    check("Scaled range", f"{unit.min().min():.1f}-{unit.max().max():.1f}", "0.0-1.0",
          unit.min().min() == 0 and unit.max().max() == 1)
    count(7, "Scaling", pool, 526)

    # Step 9: reference values (Step 8 functions are checked after the files are written).
    pce = pool.set_index("candidate_id").PCE
    threshold = pce.quantile(0.95)
    top5 = pce[pce >= threshold].sort_values(ascending=False, kind="mergesort")
    top5_pairs = pool.loc[pool.candidate_id.isin(top5.index), "pair_id"].nunique()
    best = pool.loc[pce.idxmax()]
    top10 = pce.sort_values(ascending=False).head(10).round(2).tolist()
    reference = {
        "median_pce": float(pce.median()), "top5_threshold": float(threshold), "top5_count": len(top5),
        "top5_distinct_pairs": int(top5_pairs), "best_pce": float(pce.max()),
        "random_expected_to_first_top5": (len(pce) + 1) / (len(top5) + 1),
        "random_expected_to_best": (len(pce) + 1) / 2,
    }
    check("Median PCE", f"{reference['median_pce']:.2f}%", "8.94%", round(reference["median_pce"], 2) == 8.94)
    check("Top 5%: threshold / count / distinct pairs", f"{threshold:.2f}% / {len(top5)} / {top5_pairs}",
          "12.74% / 27 / 23", (round(threshold, 2), len(top5), top5_pairs) == (12.74, 27, 23))
    check("Best candidate", f"{best.PCE:.2f}% ({best.Donor}:{best.Acceptor})", "15.71% (PBDB-TF:Y6)",
          (round(best.PCE, 2), best.Donor, best.Acceptor) == (15.71, "PBDB-TF", "Y6"))
    expected_top10 = [15.71, 14.43, 14.29, 14.19, 14.13, 13.65, 13.59, 13.52, 13.49, 13.41]
    check("Top 10 PCE values", ", ".join(f"{v:.2f}" for v in top10), ", ".join(f"{v:.2f}" for v in expected_top10),
          top10 == expected_top10)
    check("Random search: expected queries to first top-5% / to best",
          f"{reference['random_expected_to_first_top5']:.1f} / {reference['random_expected_to_best']:.1f}",
          "18.8 / 263.5", (round(reference["random_expected_to_first_top5"], 1),
                           reference["random_expected_to_best"]) == (18.8, 263.5))
    pce_lo, pce_hi = pce.min(), pce.max()
    check("PCE range", f"{pce_lo:.2f}-{pce_hi:.2f}%", "0.01-15.71%",
          (round(pce_lo, 2), round(pce_hi, 2)) == (0.01, 15.71))
    count("8-9", "Benchmark functions and reference values", pool, 526)

    # Step 10: default starting sets, drawn uniformly from the whole pool (no PCE used).
    initial_sets = {str(seed): sorted(np.random.default_rng(seed).choice(
        pool.candidate_id.to_numpy(), INITIAL_SET_SIZE, replace=False).tolist())
        for seed in range(N_INITIAL_SEEDS)}
    count(10, "Starting sets", pool, 526)

    # Step 11: informational flags, never filters.
    flags = evaluate_checks(pool)
    passes = flags.all(axis=1)
    # The prompt expects 502. Four candidates sit exactly on a check boundary; counting them as passing
    # (as >= and <= intend, the decision made with the user for task_005) gives 505. See the report.
    check("Candidates passing all Step 11 checks / failing one or more", f"{passes.sum()} / {(~passes).sum()}",
          "505 / 21 (prompt: 502 / 24)", (passes.sum(), (~passes).sum()) == (505, 21))
    boundary = []
    for name, parts in CHECKS:
        for coefs, op, rhs in parts:
            lhs = sum(coef * pool[var] for var, coef in coefs.items())
            plain = lhs >= rhs if op == ">=" else lhs <= rhs
            for i in pool.index[(lhs - rhs).abs() <= TOL]:
                boundary.append((pool.candidate_id[i], pool.Donor[i], pool.Acceptor[i], name, f"{op} {rhs}",
                                 f"{lhs[i] - rhs:+.1e}", "pass" if plain[i] else "fail"))
    boundary = pd.DataFrame(boundary, columns=["candidate_id", "Donor", "Acceptor", "Check", "Bound",
                                               "Float error", "Plain float comparison"])
    failures = flags.apply(lambda r: ", ".join(c for c in FLAGS if not r[c]), axis=1)
    top = pce.idxmax()
    check("Best candidate fails gap_consistency_A only", failures[top], "gap_consistency_A",
          failures[top] == "gap_consistency_A")
    bad_donors = sorted(pool.loc[~flags.gap_consistency_D, "Donor"].unique())
    check("Donors failing gap_consistency_D", ", ".join(bad_donors), "PBDTTz-N", bad_donors == ["PBDTTz-N"])
    count(11, "Informational flags", pool, 526)

    # Outputs.
    meta = ["Donor", "Acceptor", "DOI", "Donor SMILES", "Acceptor SMILES", "HTL", "ETL", "n_replicates"]
    features = pd.concat([pool[["candidate_id", "pair_id", *meta, *INPUTS]], unit, flags], axis=1)
    features.to_csv(FEATURES_CSV, index=False)
    pool[["candidate_id", "PCE", "PCE_std"]].to_csv(LABELS_CSV, index=False)
    spec = {
        "target": {"name": "PCE", "source_column": TARGET_COL, "unit": "%", "direction": "maximize"},
        "n_candidates": len(pool),
        "features": INPUTS,
        "scaling": {n: {"min": float(lo[n]), "max": float(hi[n]), "source": "pool_minmax"} for n in INPUTS},
        "distance": {"metric": "euclidean", "space": "unit_scaled", "tie_break": "lowest_candidate_id"},
        "files": {"features": FEATURES_CSV.name, "labels": LABELS_CSV.name, "initial_sets": INITIAL_SETS_JSON.name},
        "reference": reference,
        "noise": {"replicate_pce_std_median": float(std_med), "replicate_pce_std_max": float(std_max)},
    }
    SPEC_JSON.write_text(json.dumps(spec, indent=2) + "\n")
    INITIAL_SETS_JSON.write_text(json.dumps(initial_sets, indent=1) + "\n")

    # Verify the written files and the Step 8 functions, which read only those files.
    written = pd.read_csv(FEATURES_CSV)
    check("PCE column present in pool_features.csv", "PCE" in written.columns or "PCE_std" in written.columns,
          False, "PCE" not in written.columns and "PCE_std" not in written.columns)
    self_hits = sum(nearest_unqueried(written.loc[i, UNIT_COLS].to_numpy(dtype=float)) == written.candidate_id[i]
                    for i in written.index)
    check("nearest_unqueried(candidate's own point) returns the candidate", f"{self_hits} of {len(written)}",
          f"{len(written)} of {len(written)}", self_hits == len(written))
    first = int(written.candidate_id[0])
    runner_up = nearest_unqueried(written.loc[0, UNIT_COLS].to_numpy(dtype=float), [first])
    check("nearest_unqueried skips queried candidates", runner_up != first, True, runner_up != first)
    far = nearest_unqueried(np.full(len(INPUTS), 5.0))
    check("nearest_unqueried accepts points outside [0, 1]", isinstance(far, int), True, isinstance(far, int))
    check("top_k_ids(0.05)", f"{len(top_k_ids())} candidates, best {top_k_ids()[0]}",
          f"27 candidates, best {top}", top_k_ids() == top5.index.tolist())

    write_report(row_counts, checks, missing, multi_level, reps, mixed, pool, shared, split_pairs, lo, hi,
                 reference, top5, top10, initial_sets, flags, passes, failures, boundary, std_med, std_max)

    print(md_table(pd.DataFrame(row_counts, columns=["Step", "Description", "Rows", "Expected"])))
    print(f"\nAll {len(checks)} sanity checks passed. Outputs written to {HERE}")


def write_report(row_counts, checks, missing, multi_level, reps, mixed, pool, shared, split_pairs, lo, hi,
                 reference, top5, top10, initial_sets, flags, passes, failures, boundary, std_med, std_max):
    L = []
    add = L.append

    add("# OPV dataset pre-processing report (pool benchmark)\n")
    add("Generated by `preprocess.py` from `pce_cleaned_dataset.csv` (Seifrid et al., *J. Mater. Chem. A*, 2024; "
        "558 devices), following `opv_preprocessing_prompt_pool.md`. Pre-processing only: no optimizer was run.\n")

    add("## Row counts\n")
    add(md_table(pd.DataFrame(row_counts, columns=["Step", "Description", "Rows", "Expected"])) + "\n")
    dropped = ", ".join(f"`{c}` ({n})" for c, n in missing.items() if n)
    add(f"Step 4 dropped {row_counts[2][2] - row_counts[3][2]} rows, the only rows dropped. Missing values per "
        f"feature (a row can miss several): {dropped}.\n")

    add("## Sanity checks\n")
    add("Every check below is enforced by the script, which stops if a value differs.\n")
    add(md_table(pd.DataFrame(checks, columns=["Check", "Observed", "Expected"])) + "\n")

    add("## Target and features\n")
    add("- Target: `calculated PCE (%)` = Voc x Jsc x FF, renamed `PCE`. The reported `PCE (%)` is not used. "
        "`FF (%)` is stored as a fraction (0.24-0.84) and was not divided by 100.")
    add("- 11 numeric features, no one-hot encoding and no snapping: " + ", ".join(f"`{n}`" for n in INPUTS) + ". "
        "`HTL_energy` (3 values) and `ETL_energy` (14 values) are plain numbers.")
    add("- `Ehl_D (eV)` and `Ehl_A (eV)` are excluded: they equal LUMO - HOMO exactly (see sanity checks).")
    add("- Device outcomes and post-fabrication measurements (Voc, Jsc, FF, PCE, mobilities, thicknesses) are "
        "not features. `PCE` and `PCE_std` are only in `pool_labels.csv`.\n")

    add("## Replicates\n")
    add(f"Rows with identical values for all 11 features (rounded to 6 decimals) were merged into one candidate: "
        f"{len(reps)} groups covering {reps.n_replicates.sum()} rows, leaving {len(pool)} candidates with "
        f"{len(pool)} unique feature vectors. `PCE` is the group mean, `PCE_std` the sample standard deviation "
        f"(NaN for single rows) and `n_replicates` the group size; metadata columns hold the group's unique values "
        f"joined with \"; \".\n")
    add(f"Replicate `PCE_std`: median {std_med:.2f}, maximum {std_max:.2f} (percentage points). Optimizers that "
        "model noise can use this as a noise estimate.\n")
    add(md_table(reps[["candidate_id", "Donor", "Acceptor", "n_replicates", "PCE", "PCE_std"]]
                 .sort_values("PCE", ascending=False)) + "\n")
    for r in mixed.itertuples():
        add(f"One group mixes two acceptors with identical energy values: {r.Donor} with "
            f"{r.Acceptor.replace('; ', ' and ')} (candidate {r.candidate_id}). This is expected and was kept as one "
            "candidate; its joined SMILES count as one pair.\n")

    add("## Identifiers and pairs\n")
    add("- `candidate_id` 0-525: candidates sorted by `DOI`, `Donor`, `Acceptor`, then the 11 features.")
    add(f"- `pair_id`: one per (`Donor SMILES`, `Acceptor SMILES`), numbered in order of first `candidate_id`. "
        f"{pool.pair_id.nunique()} distinct pairs; {len(shared)} pairs have more than one candidate, covering "
        f"{shared.sum()} candidates (largest: {shared.max()} candidates).")
    add("- Candidates of the same pair share the donor's and, almost always, the acceptor's energy features, and "
        "differ only in processing and interlayers. Querying one says a lot about the others, so report results "
        "also as **distinct pairs found**.\n")
    add(f"**Deviation from the prompt.** The prompt says same-pair candidates share all six energy features. "
        f"That holds for {len(shared) - len(split_pairs)} of the {len(shared)} pairs with more than one "
        "candidate. In the other "
        f"{len(split_pairs)}, the same acceptor molecule (same SMILES) is recorded under two names with different "
        "energies, because energy levels were averaged per material name. Donor energies always match. Values "
        "were not changed; the pairs are listed for verification.\n")
    rows = []
    for g in split_pairs:
        for acceptor, a in g.groupby("Acceptor", sort=False):
            r = a.iloc[0]
            rows.append((r.pair_id, r.Donor, acceptor, ", ".join(map(str, a.candidate_id)), r.HOMO_A, r.LUMO_A,
                         r.Eg_A, "; ".join(a.DOI.unique())))
    add(md_table(pd.DataFrame(rows, columns=["pair_id", "Donor", "Acceptor", "candidate_id", "HOMO_A", "LUMO_A",
                                             "Eg_A", "DOI"]), fmt={"LUMO_A": "{:.3f}"}) + "\n")

    add("## Pool ranges and scaling\n")
    add("Each feature is scaled to [0, 1] with its min and max over the 526 candidates (`_unit` columns). Every "
        "candidate's features are known before any query and no PCE is used, so nothing leaks.\n")
    ranges = pd.DataFrame([(n, lo[n], hi[n], UNITS[n]) for n in INPUTS], columns=["Feature", "Min", "Max", "Unit"])
    add(md_table(ranges, fmt={"Min": "{:.3f}", "Max": "{:.3f}"}) + "\n")

    add("## Benchmark reference values\n")
    n = len(pool)
    best = pool.loc[pool.PCE.idxmax()]
    refs = [
        ("Pool size", f"{n} candidates"),
        ("Median PCE", f"{reference['median_pce']:.2f}%"),
        ("Top 5%", f"{reference['top5_count']} candidates with PCE >= {reference['top5_threshold']:.2f}% "
                   f"(95th percentile, {reference['top5_threshold']:.4f}), from {reference['top5_distinct_pairs']} "
                   "distinct pairs"),
        ("Best candidate", f"{best.PCE:.2f}% ({best.Donor}:{best.Acceptor}, candidate {best.candidate_id})"),
        ("Top 10 PCE values", ", ".join(f"{v:.2f}" for v in top10)),
        ("Random search, expected queries to the first top-5% candidate",
         f"(N + 1)/(K + 1) = {reference['random_expected_to_first_top5']:.1f}"),
        ("Random search, expected queries to the best candidate",
         f"(N + 1)/2 = {reference['random_expected_to_best']:.1f}"),
    ]
    add(md_table(pd.DataFrame(refs, columns=["Reference", "Value"])) + "\n")
    t = pool.set_index("candidate_id").loc[top5.index, ["pair_id", "Donor", "Acceptor", "PCE"]].reset_index()
    t["Fails"] = failures.set_axis(pool.candidate_id).loc[top5.index].replace("", "-").to_numpy()
    add("### Top-5% candidates\n")
    add(md_table(t) + "\n")

    add("## Starting sets\n")
    add(f"`initial_sets.json` maps seeds 0-{N_INITIAL_SEEDS - 1} to {INITIAL_SET_SIZE} `candidate_id`s each, drawn "
        "uniformly from the whole pool without replacement (`numpy.random.default_rng(seed)`). No PCE is used. "
        "The below-median option (PCE < 8.94%, 263 candidates) was not generated; it would use PCE to choose the "
        "starting points.\n")
    hits = sum(bool(set(ids) & set(top5.index)) for ids in initial_sets.values())
    add(f"{hits} of {N_INITIAL_SEEDS} starting sets already contain a top-5% candidate.\n")

    add("## Informational flags\n")
    add("Physical-consistency checks stored as boolean columns in `pool_features.csv` (True = passes). For "
        "information only: no candidate was removed.\n")
    rules = ["LUMO_D - LUMO_A >= -0.15", "HOMO_D - HOMO_A >= -0.15", "1.05 <= LUMO_A - HOMO_D <= 1.96",
             "Eg_D - (LUMO_D - HOMO_D) <= 0.15", "Eg_A - (LUMO_A - HOMO_A) <= 0.15"]
    t = pd.DataFrame({"Flag": FLAGS, "Rule": rules, "Candidates failing": [int((~flags[c]).sum()) for c in FLAGS]})
    add(md_table(t) + "\n")
    add(f"{passes.sum()} candidates pass all checks and {(~passes).sum()} fail at least one.\n")

    add("### Deviation from the expected counts\n")
    add(f"The prompt expects 502 passing and 24 failing; this run gives {passes.sum()} and {(~passes).sum()}. The "
        "candidates below sit exactly on a check boundary (their 2-decimal energies give the bound exactly). "
        "Floating-point rounding moves them about 1e-16 either side, so a plain comparison fails some of them on "
        "noise alone. As decided with the user for task_005 (the same 526 devices and the same checks), boundary "
        f"candidates count as passing, as >= and <= intend, with a {TOL:g} tolerance. A plain float comparison "
        "gives 503. No rule tried for task_005 reproduces 502. The flags are informational only, so this changes "
        "no candidate, label or reference value.\n")
    add(md_table(boundary) + "\n")

    add("### Known flags (verify against the source papers)\n")
    add("Values were not changed.\n")
    y6 = pool[(pool.Donor == "PBDB-TF") & (pool.Acceptor == "Y6") & ~flags.gap_consistency_A]
    for r in y6.itertuples():
        add(f"- **PBDB-TF:Y6 ({r.PCE:.2f}%, the best candidate, candidate {r.candidate_id})** fails "
            f"`gap_consistency_A`. Its acceptor optical gap is recorded as {r.Eg_A:.2f} eV against a HOMO-LUMO gap "
            f"of {r.LUMO_A - r.HOMO_A:.2f} eV (HOMO {r.HOMO_A:.2f}, LUMO {r.LUMO_A:.2f}); Y6 is usually reported at "
            "about 1.3-1.4 eV. Its PCE is a real measurement, so it stays in the pool, but its features are likely "
            f"wrong and may mislead surrogate models near the top of the pool. DOI: {r.DOI}")
    for donor, g in pool[~flags.gap_consistency_D].groupby("Donor"):
        r = g.iloc[0]
        gap = r.LUMO_D - r.HOMO_D
        add(f"- **{donor}** fails `gap_consistency_D` in {len(g)} candidate(s). HOMO {r.HOMO_D:.2f} eV and LUMO "
            f"{r.LUMO_D:.2f} eV give a HOMO-LUMO gap of {gap:.2f} eV, but its optical gap is recorded as "
            f"{r.Eg_D:.2f} eV. DOI(s): {'; '.join(g.DOI.unique())}")
    for layer, material, counts in multi_level:
        desc = ", ".join(f"{lv:.2f} eV in {k} row{'s' if k > 1 else ''}" for lv, k in counts.items())
        add(f"- {layer} **{material}** is recorded at {desc} (of the 541 complete rows).")
    add("  The single PDINO row at -2.18 eV is an encoding inconsistency (PDINO is otherwise -3.90 eV).\n")

    add("## Downstream notes\n")
    add("- **PCE is never scaled** in any file. Each optimizer's surrogate must standardize PCE using only the "
        "candidates it has queried, refitting at every iteration.")
    add("- `query(candidate_id, x_proposed=None, log_path=...)` returns PCE from `pool_labels.csv` and appends "
        "`candidate_id`, `PCE`, `timestamp`, `repeat`, `n_experiments` and `x_proposed` to the query log. A repeat "
        "returns the stored value and does not count as a new experiment. Use a fresh log per campaign.")
    add("- `nearest_unqueried(x_unit, queried_ids)` maps a continuous proposal to the closest unqueried candidate "
        "(Euclidean distance on the `_unit` features; ties to the lowest `candidate_id`; points outside [0, 1] "
        "allowed).")
    add("- `top_k_ids(fraction=0.05)` returns the candidates at or above the (1 - fraction) PCE quantile. **For "
        "evaluation only**; an optimizer must never call it.")
    add("- All 11 features count equally in the distance, so differences in processing variables matter as much "
        "as differences in energy levels.")
    add("- Many continuous proposals will map to the same nearby candidates. The query log records both the "
        "proposed point (`x_proposed`) and the candidate it was mapped to.\n")

    add("## Known limitations\n")
    for item in [
        "**The pool is published, mostly best-case devices.** Low-PCE and failed devices are under-represented. "
        "Success in this benchmark shows efficient search of a known space, not discovery of new materials.",
        "**Optimizers can't go beyond the pool.** New donor-acceptor pairs or new processing conditions cannot be "
        "proposed.",
        "**The top of the pool is sparse.** Only 10 candidates exceed 13.4% PCE, so the last improvements are the "
        "hardest to find.",
        f"**Same-pair candidates are correlated.** The {len(shared)} pairs with more than one candidate share the "
        f"donor's energy features ({len(shared) - len(split_pairs)} of them all six), which can inflate apparent "
        "performance. Report distinct pairs found.",
        "**The best candidate's features are likely wrong.** Y6's recorded optical gap is likely incorrect, which "
        "may mislead surrogate models near the top of the pool.",
        "**Mapping continuous proposals is crude.** Nearest-candidate mapping weights all features equally and "
        "can send many proposals to the same few candidates.",
        "**No structural information.** Materials are distinguished only by their energy levels.",
        "**Solvent additive identity and annealing time are not features.**",
        "**Energy levels are noisy.** They come from mixed measurement methods and were averaged per material.",
    ]:
        add(f"- {item}")
    add("")

    (HERE / "preprocessing_report.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
