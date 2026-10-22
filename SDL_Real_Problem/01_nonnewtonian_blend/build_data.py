"""Build every data file for the blend problem, in dependency order.

1. data/ingredient_calibration.json: concentration scaling (E1) and flow index
   (E2) per thickener, fitted to real AutoVisco measurements
   (data/autovisco_rheology_summary.csv).
2. data/mixing_rule_validation.json: Grunberg-Nissan interaction fitted to the
   real PEPV silicone blends (data/pepv_silicone_blends.csv), the evidence
   that ideal log mixing under-predicts blend viscosity.
3. data/target_flow_curve.csv: the Herschel-Bulkley target on the shear grid.
4. data/simulated_recipes.csv: 2048 Sobol recipes run through the model
   (variables, objective terms, viscosity at every shear rate).
5. data/reference_optimum.json: best-known optimum (f_opt) from 40 CMA-ES
   runs with restarts, each polished by L-BFGS-B.
6. figures/: calibration fits and the target vs. the best recipe.

Usage (from this folder): python build_data.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"

# Real reference measurements used for each fit, selected from the AutoVisco summary.
VALID_N_R2 = 0.99  # power-law fits below this R^2 (or n outside (0, 1)) are not used for E2


def _conc(label):
    m = re.search(r"([\d.]+)%", label)
    return float(m[1]) if m else None


def fit_calibration():
    with open(DATA / "autovisco_rheology_summary.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["c"] = _conc(r["label"])
    select = {
        "carbopol": lambda r: r["sample"] == "carbopol",
        "sepineo": lambda r: r["sample"] == "sepineo",
        "solagum": lambda r: r["sample"] == "solagum",
        "peg600k": lambda r: r["sample"] == "PEG" and "600K" in r["label"],
    }
    cal = {}
    for name, keep in select.items():
        sub = [r for r in rows if keep(r)]
        ref = sorted({(r["c"], float(r["true_mu_cP"])) for r in sub if r["true_mu_cP"]})
        if len(ref) >= 2:  # E1 from certified reference viscosities
            c, mu = np.array(ref).T
            a, log_eta0 = np.polyfit(np.log(c / c[0]), np.log(mu), 1)
            eta_source = "log-log fit of reference viscosities (true_mu_cP)"
        else:  # solagum: one reference value; exponent from apparent-viscosity ratios
            c0, mu0 = ref[0]
            app = np.array([(r["c"], float(r["mu_app_cP"])) for r in sub])
            app0 = app[app[:, 0] == c0, 1].mean()
            others = app[app[:, 0] != c0]
            a = float(np.mean(np.log(others[:, 1] / app0) / np.log(others[:, 0] / c0)))
            c, mu, log_eta0 = np.array([c0]), np.array([mu0]), np.log(mu0)
            eta_source = "anchored at the 1% reference viscosity; exponent from AutoVisco apparent-viscosity ratios"
        n_pts = [
            (r["c"], float(r["n_idx"]))
            for r in sub
            if r["n_idx"] and r["R2_powerlaw"] and float(r["R2_powerlaw"]) >= VALID_N_R2 and 0 < float(r["n_idx"]) < 1
        ]
        entry = {
            "c0_pct": float(c[0]),
            "eta0_cP": float(np.exp(log_eta0)),
            "a": float(a),
            "eta_source": eta_source,
            "reference_points": [[float(ci), float(mi)] for ci, mi in zip(c, mu)],
            "flow_index_points": n_pts,
        }
        if name != "carbopol" and len({p[0] for p in n_pts}) >= 2:
            nc, nv = np.array(n_pts).T
            b, n0 = np.polyfit(np.log(nc / c[0]), nv, 1)
            entry.update(n0=float(n0), b=float(b), n_source="linear fit of n vs ln c, valid power-law fits only")
        elif name != "carbopol":
            entry.update(n0=float(np.mean([p[1] for p in n_pts])), b=0.0, n_source="mean of valid fits")
        else:
            entry.update(n0=None, b=None, n_source="assumed 0.45 (Herschel-Bulkley); only one valid AutoVisco fit, n = 0.48")
        cal[name] = entry
    (DATA / "ingredient_calibration.json").write_text(json.dumps(cal, indent=2))
    return cal


def validate_mixing_rule():
    """Fit ln eta = x1 ln eta1 + x2 ln eta2 + G x1 x2 to the real 1k/10k cP silicone blends."""
    with open(DATA / "pepv_silicone_blends.csv") as f:
        rows = [r for r in csv.DictReader(f) if float(r["30kcp_pct"]) == 0]
    x1 = np.array([float(r["1kcp_pct"]) for r in rows]) / 100
    eta = np.array([float(r["viscosity_kcp"]) for r in rows])
    e1, e2 = eta[x1 == 1][0], eta[x1 == 0][0]
    ideal = x1 * np.log(e1) + (1 - x1) * np.log(e2)
    mix = (x1 > 0) & (x1 < 1)
    G = float(np.sum((np.log(eta) - ideal)[mix] * (x1 * (1 - x1))[mix]) / np.sum((x1 * (1 - x1))[mix] ** 2))
    pred_ideal, pred_gn = np.exp(ideal), np.exp(ideal + G * x1 * (1 - x1))
    out = {
        "data": "PEPV pump-viscometer blends of 1 kcP and 10 kcP silicone oils (real)",
        "G_fitted": G,
        "mean_abs_error_pct_ideal": float(np.mean(np.abs(pred_ideal / eta - 1)[mix]) * 100),
        "mean_abs_error_pct_grunberg_nissan": float(np.mean(np.abs(pred_gn / eta - 1)[mix]) * 100),
        "points": [
            {"frac_1kcP": float(a), "measured_kcP": float(m), "ideal_kcP": float(i), "GN_kcP": float(g)}
            for a, m, i, g in zip(x1, eta, pred_ideal, pred_gn)
        ],
    }
    (DATA / "mixing_rule_validation.json").write_text(json.dumps(out, indent=2))
    return out


def write_target(model_mod):
    eta = model_mod.target_viscosity_Pa_s()
    with open(DATA / "target_flow_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shear_rate_1_per_s", "viscosity_Pa_s", "shear_stress_Pa"])
        for g, e in zip(model_mod.SHEAR_RATES, eta):
            w.writerow([g, e, e * g])


def simulate_recipes(model_mod, model, n=2048):
    from scipy.stats import qmc

    lo = np.array([v[2] for v in model_mod.VARIABLES])
    hi = np.array([v[3] for v in model_mod.VARIABLES])
    X = qmc.scale(qmc.Sobol(len(lo), scramble=True, seed=0).random(n), lo, hi)
    names = [v[0] for v in model_mod.VARIABLES]
    shear_cols = [f"eta_Pa_s@{g:.3g}" for g in model_mod.SHEAR_RATES]
    with open(DATA / "simulated_recipes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names + ["objective", "fit_log_rmse", "cost_usd_per_kg", "cost_term", "time_term", "pH_term"] + shear_cols)
        for x in X:
            t = model.objective_terms(x)
            eta, _ = model.flow_curve(x)
            w.writerow(list(x) + [t["objective"], t["fit_log_rmse"], t["cost_usd_per_kg"], t["cost_term"], t["time_term"], t["pH_term"]] + list(eta))


def reference_optimum(model_mod, model, runs=40):
    import cma
    from scipy.optimize import minimize

    lo = np.array([v[2] for v in model_mod.VARIABLES])
    hi = np.array([v[3] for v in model_mod.VARIABLES])
    f_unit = lambda u: model.objective(lo + np.clip(u, 0, 1) * (hi - lo))  # noqa: E731
    best_u, best_f, found = None, np.inf, []
    for r in range(runs):
        es = cma.CMAEvolutionStrategy(
            np.random.default_rng(r).uniform(0, 1, len(lo)), 0.3,
            {"bounds": [0, 1], "seed": r + 1, "verbose": -9, "maxfevals": 20000, "tolfun": 1e-12},
        )
        es.optimize(f_unit)
        pol = minimize(f_unit, es.result.xbest, method="L-BFGS-B", bounds=[(0, 1)] * len(lo))
        u, f = (pol.x, pol.fun) if pol.fun < es.result.fbest else (es.result.xbest, es.result.fbest)
        found.append(float(f))
        if f < best_f:
            best_u, best_f = u, f
    x = lo + best_u * (hi - lo)
    terms = model.objective_terms(x)
    _, info = model.flow_curve(x)
    found = np.sort(found)
    ref = {
        "f_opt": float(best_f),
        "x_opt": dict(zip([v[0] for v in model_mod.VARIABLES], map(float, x))),
        "terms_at_opt": terms,
        "solvent_cP_at_opt": float(info["eta_s_cP"]),
        "method": f"{runs} CMA-ES runs (20000 evals each, random starts, sigma0 0.3 of the box) + L-BFGS-B polish",
        "run_finals_sorted": found.tolist(),
        "distinct_basins_note": "run_finals_sorted spread shows how many runs end in worse local optima",
    }
    (DATA / "reference_optimum.json").write_text(json.dumps(ref, indent=2))
    return ref, x


def plots(model_mod, model, cal, x_opt):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for name, color in zip(model_mod.INGREDIENTS, ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]):
        pts = np.array(cal[name]["reference_points"])
        lo, hi = model_mod.VARIABLES[model_mod.INGREDIENTS.index(name)][2:]
        cc = np.linspace(max(hi * 0.05, 1e-3), hi, 100)
        ax.plot(cc, model.reference_viscosity_cP(name, cc), color=color, label=f"{name} (a = {cal[name]['a']:.2f})")
        ax.plot(pts[:, 0], pts[:, 1], "o", color=color)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Concentration (wt%)")
    ax.set_ylabel("Reference viscosity at 10 1/s (cP)")
    ax.set_title("E1 calibration: real AutoVisco points, fitted power law")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    g = model_mod.SHEAR_RATES
    ax.plot(g, model_mod.target_viscosity_Pa_s(), "k--", lw=2, label="target spec (E0)")
    eta, _ = model.flow_curve(x_opt)
    ax.plot(g, eta, color="#2a78d6", lw=2, label="best-known recipe")
    for name, color in zip(model_mod.INGREDIENTS, ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]):
        x1 = np.array(x_opt)
        x1[:4] = 0
        x1[model_mod.INGREDIENTS.index(name)] = x_opt[model_mod.INGREDIENTS.index(name)]
        if x1[model_mod.INGREDIENTS.index(name)] > 1e-3:
            ax.plot(g, model.flow_curve(x1)[0], ":", color=color, label=f"{name} alone at its dose")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Shear rate (1/s)")
    ax.set_ylabel("Viscosity (Pa s)")
    ax.set_title("Target flow curve vs. best-known blend")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "calibration_and_best_blend.png", dpi=150)


def main():
    cal = fit_calibration()
    print(json.dumps({k: {kk: v[kk] for kk in ("c0_pct", "eta0_cP", "a", "n0", "b")} for k, v in cal.items()}, indent=1))
    val = validate_mixing_rule()
    print("mixing rule:", {k: v for k, v in val.items() if k != "points"})

    import model as model_mod  # after the calibration file exists

    model = model_mod.BlendModel()
    write_target(model_mod)
    simulate_recipes(model_mod, model)
    ref, x_opt = reference_optimum(model_mod, model)
    print(json.dumps({k: ref[k] for k in ("f_opt", "x_opt", "terms_at_opt")}, indent=1))
    print("run finals:", np.round(ref["run_finals_sorted"], 4))
    plots(model_mod, model, cal, x_opt)


if __name__ == "__main__":
    main()
