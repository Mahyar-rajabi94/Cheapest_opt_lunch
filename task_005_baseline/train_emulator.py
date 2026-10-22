"""Train the random-forest PCE emulator on opv_emulator_data.csv.

Run with

    python train_emulator.py

Needs scikit-learn; the saved model does not (see emulator.py). Writes
opv_emulator_rf.npz and emulator_report.md next to this file. The export check
raises if the numpy predictor disagrees with scikit-learn.
"""
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

from emulator import MODEL_NPZ, Emulator, save_forest
from preprocess import INPUTS, ORDINARY, check_constraints, from_unit, load_space, md_table, snap

HERE = Path(__file__).resolve().parent
DATA_CSV = HERE / "opv_emulator_data.csv"
REPORT_MD = HERE / "emulator_report.md"

# Best donor-grouped CV R2 in the TUNING_GRID below.
PARAMS = dict(n_estimators=500, min_samples_leaf=3, max_features=0.33, random_state=0)
TUNING_GRID = {"min_samples_leaf": [1, 2, 3, 5], "max_features": [1.0, 0.6, 0.33]}
N_FOLDS = 10
N_RANDOM = 20000      # random search-space points for the export check and behaviour stats
N_FLAT = 1000         # feasible points used to measure flatness
EXPORT_TOL = 1e-9
SEED = 0

# Model selection, measured once on the 11 *_unit columns with the same two CV schemes.
# Not rerun here (the Gaussian processes take about 10 minutes).
MODEL_SELECTION = [
    ("Random forest (500 trees, min_samples_leaf=2)", 0.456, 2.290, 0.391, 2.422, "flat steps"),
    ("Gradient boosting (HistGradientBoosting, defaults)", 0.480, 2.240, 0.358, 2.488, "flat steps"),
    ("Neural-net ensemble, 10 x ReLU MLP (64, 64)", 0.427, 2.350, 0.377, 2.450, "continuous, kinks"),
    ("Neural-net ensemble, 10 x tanh MLP (32, 32)", 0.359, 2.486, 0.321, 2.558, "smooth"),
    ("Gaussian process, Matern 3/2 ARD + noise", 0.404, 2.397, 0.260, 2.671, "smooth"),
    ("Gaussian process, RBF ARD + noise", 0.392, 2.421, 0.245, 2.698, "smooth"),
    ("Gaussian process, Matern 5/2 ARD + noise", 0.398, 2.410, 0.163, 2.840, "smooth"),
]


def scores(y, p):
    return r2_score(y, p), mean_squared_error(y, p) ** 0.5, mean_absolute_error(y, p)


def cv_schemes(groups):
    return {
        "Random 10-fold": (KFold(N_FOLDS, shuffle=True, random_state=SEED), None),
        "10-fold grouped by donor": (GroupKFold(N_FOLDS), groups),
    }


def random_queries(space, n, rng):
    """n uniform proposals in [0, 1]^11, mapped to original units and snapped."""
    x = from_unit(pd.DataFrame(rng.random((n, len(INPUTS))), columns=INPUTS), space)
    for name in INPUTS:
        levels = space["continuous"][name].get("snap_to_levels")
        if levels is not None:
            x[name] = x[name].map(lambda v: snap(v, levels))
    return x


def main():
    space = load_space()
    data = pd.read_csv(DATA_CSV)
    X = data[INPUTS].to_numpy(dtype=float)
    y = data["PCE"].to_numpy(dtype=float)
    groups = data["Donor"].to_numpy()
    if len(data) != 526 or not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise RuntimeError("opv_emulator_data.csv must have 526 complete rows; rerun preprocess.py")
    rng = np.random.default_rng(SEED)

    # 1. Tune min_samples_leaf and max_features.
    tuning = []
    combos = list(itertools.product(*TUNING_GRID.values()))
    for i, (leaf, mf) in enumerate(combos, 1):
        print(f"Tuning {i}/{len(combos)}: min_samples_leaf={leaf}, max_features={mf}", flush=True)
        m = RandomForestRegressor(**{**PARAMS, "min_samples_leaf": leaf, "max_features": mf}, n_jobs=-1)
        row = {"min_samples_leaf": leaf, "max_features": mf}
        for name, (cv, g) in cv_schemes(groups).items():
            row[f"R2, {name}"] = r2_score(y, cross_val_predict(m, X, y, cv=cv, groups=g))
        tuning.append(row)
    tuning = pd.DataFrame(tuning).sort_values("R2, 10-fold grouped by donor", ascending=False)

    # 2. Cross-validate the final settings.
    print("Cross-validating the final settings", flush=True)
    forest = RandomForestRegressor(**PARAMS, n_jobs=-1)
    cv_rows = []
    for name, (cv, g) in cv_schemes(groups).items():
        r2, rmse, mae = scores(y, cross_val_predict(forest, X, y, cv=cv, groups=g))
        cv_rows.append({"CV scheme": name, "R2": r2, "RMSE (% PCE)": rmse, "MAE (% PCE)": mae})
    cv_table = pd.DataFrame(cv_rows)

    # 3. Fit on all 526 rows and export.
    print("Fitting on all rows and exporting", flush=True)
    forest.fit(X, y)
    in_sample = scores(y, forest.predict(X))
    save_forest(forest, INPUTS, info={
        "trained_on": DATA_CSV.name, "n_train": len(y), "params": PARAMS,
        "sklearn_version": sklearn.__version__, "target": "PCE (%)", "direction": "maximize",
    })
    em = Emulator()

    # 4. Export check: training rows, random queries, and queries sitting exactly on split thresholds.
    queries = random_queries(space, N_RANDOM, rng)
    edge = X[rng.integers(len(X), size=N_RANDOM)].copy()
    inner = np.flatnonzero(em.left >= 0)
    picks = rng.choice(inner, size=N_RANDOM)
    edge[np.arange(N_RANDOM), em.feature[picks]] = em.threshold[picks]
    export = []
    for name, pts in [("Training rows", X), ("Random snapped queries", queries[INPUTS].to_numpy()),
                      ("Queries on split thresholds", edge)]:
        diff = np.abs(em.predict(pts) - forest.predict(pts)).max()
        export.append({"Points": name, "n": len(pts), "max abs difference": diff})
        if diff > EXPORT_TOL:
            raise RuntimeError(f"Export check failed on {name}: max difference {diff:.3g}")
    export = pd.DataFrame(export)

    # 5. Behaviour over the search space.
    feasible = check_constraints(queries, space).all(axis=1).to_numpy()
    pred = em.predict(queries[INPUTS].to_numpy())
    pred_feasible = pred[feasible]
    best_random = queries[feasible].iloc[int(np.argmax(pred_feasible))]
    rows_feasible = data["feasible"].to_numpy(dtype=bool)
    fit_feasible = em.predict(X[rows_feasible])

    # Flatness: share of single-coordinate moves (in [0, 1] scaling) that leave the prediction unchanged.
    base = queries[feasible].iloc[:N_FLAT].reset_index(drop=True)
    base_pred = em.predict(base[INPUTS].to_numpy())
    flat = []
    for step in [1e-3, 1e-2, 1e-1]:
        row = {"Step (fraction of range)": step}
        for name in ORDINARY:
            moved = base.copy()
            lo, hi = space["continuous"][name]["lower"], space["continuous"][name]["upper"]
            up = moved[name] + step * (hi - lo)
            moved[name] = up.where(up <= hi, moved[name] - step * (hi - lo))  # stay inside the bounds
            row[name] = np.mean(em.predict(moved[INPUTS].to_numpy()) == base_pred)
        flat.append(row)
    flat = pd.DataFrame(flat)

    importances = pd.DataFrame({"Input": INPUTS, "Importance": forest.feature_importances_})
    importances = importances.sort_values("Importance", ascending=False)

    one = queries[INPUTS].to_numpy()[0]
    t = time.perf_counter()
    for _ in range(1000):
        em.predict(one)
    t_numpy = (time.perf_counter() - t) / 1000
    t = time.perf_counter()
    for _ in range(100):
        forest.predict(one[None, :])
    t_sklearn = (time.perf_counter() - t) / 100

    behaviour = {
        "n_random": N_RANDOM,
        "feasible_share": feasible.mean(),
        "pred_min": pred_feasible.min(), "pred_median": np.median(pred_feasible), "pred_max": pred_feasible.max(),
        "best_random": best_random,
        "fit_feasible_max": fit_feasible.max(),
        "obs_feasible_max": data.loc[rows_feasible, "PCE"].max(),
        "t_numpy": t_numpy, "t_sklearn": t_sklearn,
        "n_nodes": len(em.left), "n_leaves": int((em.left < 0).sum()),
        "npz_mb": MODEL_NPZ.stat().st_size / 1e6,
    }
    write_report(tuning, cv_table, in_sample, export, behaviour, flat, importances, data)
    print(f"Wrote {MODEL_NPZ.name} and {REPORT_MD.name}")


def write_report(tuning, cv_table, in_sample, export, b, flat, importances, data):
    (cv_r2_k, cv_rmse_k), (cv_r2_g, cv_rmse_g) = cv_table[["R2", "RMSE (% PCE)"]].to_numpy()
    selection = pd.DataFrame(MODEL_SELECTION, columns=[
        "Model", "R2, random", "RMSE, random", "R2, by donor", "RMSE, by donor", "Objective shape"])
    reps = data.loc[data["n_replicates"] > 1, "PCE_std"]
    best = ", ".join(f"{n} = {b['best_random'][n]:.3g}" for n in INPUTS)
    tuning_lo, tuning_hi = tuning["R2, 10-fold grouped by donor"].agg(["min", "max"])
    params = ", ".join(f"{k}={v}" for k, v in PARAMS.items())

    lines = [
        "# OPV emulator report",
        "",
        f"Generated by `train_emulator.py` from `{DATA_CSV.name}` (526 rows). The emulator is a random forest "
        "that plays the role of the experiment: it returns PCE (%) for any snapped 11-value input. "
        "No optimizer was run.",
        "",
        "## Summary",
        "",
        f"- **Model:** scikit-learn `RandomForestRegressor({params})`, trained on all 526 rows "
        f"(scikit-learn {sklearn.__version__}).",
        "- **Inputs:** the 11 search variables in original units, after HTL/ETL snapping. **Target:** PCE (%), "
        "to be maximized. Metadata columns are not used.",
        f"- **Accuracy:** cross-validated R2 {cv_r2_k:.2f} (RMSE {cv_rmse_k:.2f}% PCE) on random held-out devices "
        f"and {cv_r2_g:.2f} (RMSE {cv_rmse_g:.2f}%) on held-out donors.",
        f"- **Saved model:** `{MODEL_NPZ.name}` ({b['npz_mb']:.1f} MB, {b['n_nodes']:,} nodes), read by "
        "`emulator.py` with numpy only. Its predictions match scikit-learn's (see Export check).",
        f"- **Speed:** {b['t_numpy'] * 1e3:.2f} ms per single query (scikit-learn: {b['t_sklearn'] * 1e3:.1f} ms).",
        "",
        "## Usage",
        "",
        "```python",
        "from preprocess import prepare_query",
        "from emulator import Emulator",
        "",
        "em = Emulator()",
        "q = prepare_query(x_unit)   # 11 values in [0, 1]: unscale, snap HTL/ETL, check constraints",
        "pce = em.predict(q['x'])    # PCE (%); q['feasible'] says whether the constraints hold",
        "```",
        "",
        "`Emulator.predict` also takes an (n, 11) array and returns n values. Retrain with "
        "`python train_emulator.py`; the run is deterministic (`random_state=0`).",
        "",
        "## Model selection",
        "",
        "Seven models were compared with two 10-fold cross-validation schemes: random folds, and folds grouped "
        "by donor polymer (every device of a donor is held out together). The grouped scheme is closer to the "
        "benchmark, where optimizers reach donor energy levels that are not in the data. These numbers were "
        "measured once on the `*_unit` columns and are not recomputed by the script. Two further tanh "
        "ensemble settings scored lower and are omitted.",
        "",
        md_table(selection, {c: "{:.3f}" for c in selection.columns[1:5]}),
        "",
        "The random forest was chosen (decided with the user) because it is the most accurate on held-out "
        "donors. The cost is the objective's shape: a forest's output is piecewise constant, so it is flat "
        "almost everywhere (see Flatness below).",
        "",
        "### Random-forest settings",
        "",
        f"Grid over `min_samples_leaf` and `max_features` (500 trees each). R2 on held-out donors ranged "
        f"only {tuning_lo:.3f}-{tuning_hi:.3f}; the top row is used.",
        "",
        md_table(tuning, {c: "{:.3f}" for c in tuning.columns[2:]}),
        "",
        "## Accuracy of the final model",
        "",
        md_table(cv_table, {"R2": "{:.3f}", "RMSE (% PCE)": "{:.2f}", "MAE (% PCE)": "{:.2f}"}),
        "",
        f"- For scale: PCE has standard deviation {data['PCE'].std():.2f}% across the 526 rows, and repeated "
        f"devices differ by a median of {reps.median():.2f}% (max {reps.max():.2f}%). Most of the error is "
        "therefore not measurement noise. It comes from what the 11 inputs leave out, mainly molecular structure.",
        f"- In-sample fit on all 526 rows: R2 {in_sample[0]:.3f}, RMSE {in_sample[1]:.2f}%. This is "
        "optimistic and only shows how closely the emulator reproduces the training devices.",
        "",
        "## Export check",
        "",
        "`emulator.py` evaluates the exported trees with numpy and is compared with scikit-learn's `predict`. "
        "The script stops if any difference exceeds "
        f"{EXPORT_TOL:g}. Threshold points place one input exactly on a split value, which tests the `<=` rule "
        "and the float32 rounding scikit-learn applies to inputs.",
        "",
        md_table(export, {"max abs difference": "{:.1e}"}),
        "",
        "## Behaviour over the search space",
        "",
        f"{b['n_random']:,} uniform random proposals in [0, 1]^11, unscaled and snapped:",
        "",
        f"- {b['feasible_share']:.1%} satisfy all six required constraints.",
        f"- Emulator PCE over the feasible ones: min {b['pred_min']:.2f}%, median {b['pred_median']:.2f}%, "
        f"max {b['pred_max']:.2f}%.",
        f"- Best feasible random proposal (original units): {best}.",
        f"- Over the 450 feasible training devices, the emulator's maximum is {b['fit_feasible_max']:.2f}% "
        f"(observed maximum {b['obs_feasible_max']:.2f}%). A forest averages training targets, so it never "
        f"predicts above {data['PCE'].max():.2f}%, the highest PCE in the data.",
        "",
        "### Flatness",
        "",
        f"Share of single-input moves that leave the prediction unchanged, from {N_FLAT:,} feasible random "
        "points. Moves are a fraction of each input's range; the two snapped inputs are left out.",
        "",
        md_table(flat, {"Step (fraction of range)": "{:g}", **{c: "{:.2f}" for c in ORDINARY}}),
        "",
        "Finite-difference gradients (L-BFGS-B) and small simplex or trust-region steps (Nelder-Mead, BOBYQA) "
        "see zero change for most small moves, so they may stall. Population and model-based methods are less "
        "affected.",
        "",
        "### Input importance",
        "",
        "Impurity-based importances from the fitted forest (they sum to 1; biased towards inputs with many "
        "distinct values).",
        "",
        md_table(importances, {"Importance": "{:.3f}"}),
        "",
        "## Limitations",
        "",
        "- **Flat steps.** The objective is piecewise constant along every input, not only HTL and ETL.",
        f"- **Capped predictions.** The emulator cannot exceed {data['PCE'].max():.2f}%. Beyond the training "
        "data it carries the values at the edge of the data outward unchanged.",
        f"- **Moderate accuracy.** R2 is about {cv_r2_g:.2f} on held-out donors, so the emulator is a plausible "
        "stand-in for reality, not a faithful one. Benchmark conclusions are about optimizing this surface.",
        "- **Pre-processing limitations still apply** (see `preprocessing_report.md`): unsupported HTL x ETL "
        "combinations, independent energy levels, and no structural information.",
        "",
        "## Next steps",
        "",
        "- Estimate `f_opt`, the best feasible emulator PCE, by searching the emulator offline. It is a "
        "best-known value, not a proven optimum.",
        "- Wrap `prepare_query` and `Emulator` in a `Task` subclass (`src/tasks/`) and register it, following "
        "`ADDING_TASKS.md`.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
