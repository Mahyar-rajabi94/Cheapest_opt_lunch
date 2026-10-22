"""Flow-curve model of an aqueous blend of four thickeners, glycerol and pH.

Every equation here is written out in PROBLEM.md section 3; the numbered
comments (E1..E9) match the equation numbers there. Ingredient constants
fitted from real SDL5 measurements are loaded from
data/ingredient_calibration.json (written by build_data.py); the remaining
constants are literature-based assumptions, listed in ASSUMED below and in
PROBLEM.md section 4.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

INGREDIENTS = ("carbopol", "sepineo", "solagum", "peg600k")

# Decision variables, in order: (name, unit, lower, upper).
VARIABLES = (
    ("c_carbopol", "wt%", 0.0, 0.5),
    ("c_sepineo", "wt%", 0.0, 2.5),
    ("c_solagum", "wt%", 0.0, 3.0),
    ("c_peg600k", "wt%", 0.0, 10.0),
    ("w_glycerol", "wt%", 0.0, 50.0),
    ("pH", "-", 3.5, 10.0),
    ("t_mix", "s", 10.0, 600.0),
)

ASSUMED = {
    "gamma_ref": 10.0,  # 1/s, shear rate of the reference viscosities (E1)
    "temperature_C": 25.0,  # for the glycerol-water solvent (E5)
    "carbopol_yield_fraction": 0.5,  # share of the reference stress carried by the yield stress (E3)
    "carbopol_n": 0.45,  # HB flow index; AutoVisco's one valid fit gave 0.48 (E3)
    "carbopol_pH": {"rise_mid": 4.8, "rise_width": 0.3, "fall_mid": 9.8, "fall_width": 0.4},  # (E6)
    "peg_carbopol_complex": {"G_max": -1.8, "pH_mid": 5.5, "pH_width": 0.3},  # (E8)
    "hydration_tau_s": {"carbopol": 60.0, "sepineo": 15.0, "solagum": 90.0, "peg600k": 240.0},  # (E7)
    "price_usd_per_kg": {"carbopol": 30.0, "sepineo": 25.0, "solagum": 20.0, "peg600k": 15.0, "glycerol": 1.2},
    "water_reference_cP": 1.0,  # solvent of the reference measurements
}

# Target spec for a skin-pH gel dispensed by the Formulator Pen: a yield
# stress so it holds its shape at rest, shear thinning in the nozzle, and a
# high-shear viscosity floor so it does not run once spread (E0). No single
# ingredient has all three, and the pH window is where Carbopol is only
# partly neutralized and complexes with PEG.
TARGET = {"tau_y_Pa": 8.0, "K_Pa_s^n": 3.0, "n": 0.45, "eta_inf_Pa_s": 0.3, "pH": 5.0, "pH_tol": 0.5}
SHEAR_RATES = np.logspace(-1, 2, 16)  # 1/s

# Objective weights (E9).
WEIGHTS = {"cost_per_usd_per_kg": 0.05, "time_per_600s": 0.02, "pH_spec": 0.5}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def glycerol_water_viscosity_cP(w_glycerol_pct, T=ASSUMED["temperature_C"]):
    """Cheng (2008), Ind. Eng. Chem. Res. 47, 3285 (E5)."""
    cm = w_glycerol_pct / 100.0
    eta_w = 1.790 * np.exp((-1230.0 - T) * T / (36100.0 + 360.0 * T))
    eta_g = 12100.0 * np.exp((-1233.0 + T) * T / (9900.0 + 70.0 * T))
    a = 0.705 - 0.0017 * T
    b = (4.9 + 0.036 * T) * a**2.5
    alpha = 1.0 - cm + a * b * cm * (1.0 - cm) / (a * cm + b * (1.0 - cm))
    return eta_w**alpha * eta_g ** (1.0 - alpha)


def target_viscosity_Pa_s(gamma=SHEAR_RATES):
    """E0: Herschel-Bulkley with a high-shear viscosity floor."""
    t = TARGET
    return t["tau_y_Pa"] / gamma + t["K_Pa_s^n"] * gamma ** (t["n"] - 1.0) + t["eta_inf_Pa_s"]


class BlendModel:
    def __init__(self, calibration_path: Path = HERE / "data" / "ingredient_calibration.json"):
        self.cal = json.loads(Path(calibration_path).read_text())

    def reference_viscosity_cP(self, name, c):
        """E1: eta_ref,i(c) = eta0_i * (c / c0_i)^a_i at gamma_ref (fitted to real data)."""
        p = self.cal[name]
        return p["eta0_cP"] * (np.maximum(c, 0.0) / p["c0_pct"]) ** p["a"]

    def flow_index(self, name, c):
        """E2: n_i(c) = n0_i + b_i ln(c / c0_i), clipped to [0.1, 1] (fitted to real data)."""
        p = self.cal[name]
        if name == "carbopol":
            return ASSUMED["carbopol_n"]
        c = np.maximum(c, 1e-6)
        return float(np.clip(p["n0"] + p["b"] * np.log(c / p["c0_pct"]), 0.1, 1.0))

    def ingredient_viscosity_cP(self, name, c, gamma):
        """E3/E4: viscosity of ingredient i alone in water at concentration c."""
        g_ref = ASSUMED["gamma_ref"]
        eta_ref = self.reference_viscosity_cP(name, c)
        n = self.flow_index(name, c)
        if name == "carbopol":  # Herschel-Bulkley, matched to eta_ref at gamma_ref
            phi = ASSUMED["carbopol_yield_fraction"]
            return phi * eta_ref * g_ref / gamma + (1.0 - phi) * eta_ref * (gamma / g_ref) ** (n - 1.0)
        return eta_ref * (gamma / g_ref) ** (n - 1.0)  # power law

    def flow_curve(self, x, gamma=SHEAR_RATES):
        """Blend viscosity (Pa s) at each shear rate, plus the intermediate terms."""
        c = dict(zip(INGREDIENTS, x[:4]))
        w_gly, pH, t_mix = x[4], x[5], x[6]
        eta_s = glycerol_water_viscosity_cP(w_gly)  # E5
        eta_w = ASSUMED["water_reference_cP"]
        cp = ASSUMED["carbopol_pH"]
        f_pH = {  # E6
            "carbopol": 0.02 + 0.98 * _sigmoid((pH - cp["rise_mid"]) / cp["rise_width"])
            * _sigmoid((cp["fall_mid"] - pH) / cp["fall_width"]),
            "sepineo": 1.0, "solagum": 1.0, "peg600k": 1.0,
        }
        tau = ASSUMED["hydration_tau_s"]
        hyd = {i: 1.0 - np.exp(-t_mix / (tau[i] * np.sqrt(eta_s / eta_w))) for i in INGREDIENTS}  # E7
        S = {}
        for i in INGREDIENTS:
            c_eff = hyd[i] * c[i]
            if c_eff <= 0.0:
                S[i] = np.zeros_like(gamma)
                continue
            eta_sp = np.maximum(self.ingredient_viscosity_cP(i, c_eff, gamma) / eta_w - 1.0, 0.0)
            S[i] = f_pH[i] * eta_sp
        pc = ASSUMED["peg_carbopol_complex"]
        G = pc["G_max"] * _sigmoid((pc["pH_mid"] - pH) / pc["pH_width"])  # E8
        interaction = G * np.sqrt(S["peg600k"] * S["carbopol"])
        eta_cP = eta_s * (1.0 + sum(S.values()) + interaction)  # E8
        return eta_cP / 1000.0, {"eta_s_cP": eta_s, "f_pH": f_pH, "hydration": hyd, "G": G}

    def cost_usd_per_kg(self, x):
        price = ASSUMED["price_usd_per_kg"]
        return sum(x[k] / 100.0 * price[i] for k, i in enumerate(INGREDIENTS)) + x[4] / 100.0 * price["glycerol"]

    def objective_terms(self, x):
        """E9: log-RMSE to the target flow curve + cost, mixing-time and pH-spec penalties."""
        eta, _ = self.flow_curve(x)
        fit = float(np.sqrt(np.mean((np.log10(eta) - np.log10(target_viscosity_Pa_s())) ** 2)))
        cost = self.cost_usd_per_kg(x)
        ph_excess = max(0.0, abs(x[5] - TARGET["pH"]) - TARGET["pH_tol"])
        terms = {
            "fit_log_rmse": fit,
            "cost_term": WEIGHTS["cost_per_usd_per_kg"] * cost,
            "time_term": WEIGHTS["time_per_600s"] * x[6] / 600.0,
            "pH_term": WEIGHTS["pH_spec"] * ph_excess**2,
            "cost_usd_per_kg": cost,
        }
        terms["objective"] = terms["fit_log_rmse"] + terms["cost_term"] + terms["time_term"] + terms["pH_term"]
        return terms

    def objective(self, x):
        return self.objective_terms(np.asarray(x, float))["objective"]
