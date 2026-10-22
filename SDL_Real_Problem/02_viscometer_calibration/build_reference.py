"""Reference optimum and physical validation for the viscometer calibration task.

1. f_opt by variable projection. For fixed globals (log10 k, p, h_c) the
   objective is a weighted linear least-squares problem in each offset B_j,
   solved in closed form (weights 1/D_ij^2, clipped to the beta bounds). The
   remaining 3-D problem is solved by 200 Nelder-Mead multistarts, then the
   best point is polished in the full 26-D space with L-BFGS-B.
2. Validation against the certified viscosities: with the fitted globals,
   each standard's own curve gives A_j by weighted least squares and the
   predicted viscosity mu^_j = (A_j / k)^(1/p); leave-one-out repeats the
   global fit without standard j before predicting it.

Writes data/reference_optimum.json, data/standards_long.csv,
data/viscosity_validation.csv and figures/calibration_fit.png.

Usage (from this folder): python build_reference.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from task import load_standards

HERE = Path(__file__).resolve().parent
LO = np.array([-11.0, 1.0, 0.01])
HI = np.array([-6.0, 3.0, 1.0])
REPO_CONSTANTS = {"log10_k": float(np.log10(5.893e-9)), "p": 2.007, "h_c": 0.2774}  # rheology_constants.py


def projected(theta, standards):
    """Objective (RMS % relative error) and optimal betas for fixed globals."""
    log_k, p, h_c = theta
    sq, n, betas = 0.0, 0, []
    for _, mu, _, h, drag in standards:
        a = 10.0**log_k * mu**p / (h + h_c)
        w = 1.0 / drag**2
        b = np.clip(np.sum(w * (drag - a)) / np.sum(w), -drag.mean(), drag.mean())
        sq += np.sum(((a + b - drag) / drag) ** 2)
        n += len(drag)
        betas.append(b / drag.mean())
    return 100.0 * np.sqrt(sq / n), np.array(betas)


def fit_globals(standards, starts=200, seed=0):
    rng = np.random.default_rng(seed)

    def f(t):
        if np.any(t < LO) or np.any(t > HI):
            return 1e6
        return projected(t, standards)[0]

    best = None
    for _ in range(starts):
        r = minimize(f, rng.uniform(LO, HI), method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-12, fatol=1e-14))
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def full_objective(x, standards):
    log_k, p, h_c = x[:3]
    sq, n = 0.0, 0
    for (_, mu, _, h, drag), beta in zip(standards, x[3:]):
        sq += np.sum(((10.0**log_k * mu**p / (h + h_c) + beta * drag.mean() - drag) / drag) ** 2)
        n += len(drag)
    return 100.0 * np.sqrt(sq / n)


def predict_viscosity(theta, h, drag):
    """Invert one standard's curve: weighted LS for (A, B), then mu = (A/k)^(1/p)."""
    log_k, p, h_c = theta
    X = np.column_stack([1.0 / (h + h_c), np.ones_like(h)]) / drag[:, None]
    A, _ = np.linalg.lstsq(X, np.ones_like(drag), rcond=None)[0]
    return (A / 10.0**log_k) ** (1.0 / p)


def main():
    standards = load_standards()
    theta = fit_globals(standards)
    _, betas = projected(theta, standards)
    x0 = np.concatenate([theta, betas])
    bounds = [(l, h) for l, h in zip(LO, HI)] + [(-1.0, 1.0)] * len(standards)
    polish = minimize(full_objective, x0, args=(standards,), method="L-BFGS-B", bounds=bounds,
                      options=dict(maxiter=20000, ftol=1e-16, gtol=1e-12))
    x_opt = polish.x if polish.fun < full_objective(x0, standards) else x0
    f_opt = float(full_objective(x_opt, standards))

    repo_theta = [REPO_CONSTANTS["log10_k"], REPO_CONSTANTS["p"], REPO_CONSTANTS["h_c"]]
    f_repo = float(projected(repo_theta, standards)[0])

    rows = []
    for j, (label, mu, rpm, h, drag) in enumerate(standards):
        loo = [s for i, s in enumerate(standards) if i != j]
        theta_loo = fit_globals(loo, starts=20, seed=j)
        rows.append({
            "standard": label, "certified_cP": mu, "rpm": rpm, "n_points": len(h),
            "pred_cP_insample": predict_viscosity(x_opt[:3], h, drag),
            "pred_cP_leave_one_out": predict_viscosity(theta_loo, h, drag),
            "pred_cP_repo_constants": predict_viscosity(repo_theta, h, drag),
        })
    for r in rows:
        for key in ("insample", "leave_one_out", "repo_constants"):
            r[f"err_pct_{key}"] = 100.0 * (r[f"pred_cP_{key}"] / r["certified_cP"] - 1.0)
    with open(HERE / "data" / "viscosity_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    with open(HERE / "data" / "standards_long.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["standard", "certified_cP", "rpm", "h_mm", "torque_pct", "drag_pct_per_rpm"])
        for label, mu, rpm, h, drag in standards:
            for hi, di in zip(h, drag):
                w.writerow([label, mu, rpm, hi, di * rpm, di])

    mape = {k: float(np.mean([abs(r[f"err_pct_{k}"]) for r in rows])) for k in ("insample", "leave_one_out", "repo_constants")}
    ref = {
        "f_opt": f_opt,
        "units": "RMS relative drag error, percent",
        "x_opt": x_opt.tolist(),
        "globals_opt": {"log10_k": x_opt[0], "k": 10 ** x_opt[0], "p": x_opt[1], "h_c_mm": x_opt[2]},
        "method": "variable projection (closed-form offsets) + 200 Nelder-Mead multistarts on (log10 k, p, h_c), "
                  "then L-BFGS-B polish in the full space",
        "repo_constants": REPO_CONSTANTS,
        "f_at_repo_constants_with_optimal_offsets": f_repo,
        "certified_viscosity_MAPE_pct": mape,
        "n_standards": len(standards),
        "n_points": int(sum(len(s[3]) for s in standards)),
    }
    (HERE / "data" / "reference_optimum.json").write_text(json.dumps(ref, indent=2))
    print(json.dumps({k: v for k, v in ref.items() if k != "x_opt"}, indent=2))
    plot_fit(standards, x_opt)


def plot_fit(standards, x):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (HERE / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("viridis")
    mus = np.log10([s[1] for s in standards])
    for (label, mu, _, h, drag), beta in zip(standards, x[3:]):
        color = cmap((np.log10(mu) - mus.min()) / np.ptp(mus))
        hh = np.linspace(h.min(), h.max(), 100)
        ax.plot(h, drag, "o", ms=3, color=color)
        ax.plot(hh, 10 ** x[0] * mu ** x[1] / (hh + x[2]) + beta * drag.mean(), "-", lw=1, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("Spindle height above contact, h (mm)")
    ax.set_ylabel("Drag D = torque % / RPM")
    ax.set_title("23 certified silicone standards (points) and the joint calibration at f_opt (lines)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(mus.min(), mus.max()))
    fig.colorbar(sm, ax=ax, label="log10 certified viscosity (cP)")
    ax.grid(alpha=0.3)
    fig.savefig(HERE / "figures" / "calibration_fit.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
