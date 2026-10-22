"""Joint calibration of the SDL5 gap-stepping cone-plate viscometer (26-D).

Real data: data/height_normalized.csv, 23 certified silicone standards
(1.07 to 124 kcP) measured on the Brookfield DV2T-RV + CNC platform
(Automated_Viscometry). Each column is torque % against spindle height h
(mm above the contact point) at one fixed RPM.

Model (Automated_Viscometry/src/python_64/rheology_prediction.py):
    D_ij = torque_ij / RPM_j                     observed drag
    D^_ij = k * mu_j**p / (h_i + h_c) + B_j      calibrated drag

x = [log10 k, p, h_c, beta_1 .. beta_23], with B_j = beta_j * mean_i(D_ij).
Objective: RMS relative drag error in percent,
    f(x) = 100 * sqrt(mean_ij ((D^_ij - D_ij) / D_ij)**2).
See PROBLEM.md for the derivation and data/reference_optimum.json for f_opt.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "main"))

from src.tasks.base import Task  # noqa: E402

COLUMN = re.compile(r"([\d.]+)kcp_([\d.]+)_torque_%_rpm_([\d.]+)")


def load_standards(path: Path = HERE / "data" / "height_normalized.csv"):
    """Return a list of (label, mu_cP, rpm, h_mm, drag) for each standard."""
    import csv

    with open(path) as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    h = np.array([float(r[0]) for r in body])
    standards = []
    for j, name in enumerate(header[1:], start=1):
        nominal, certified_kcp, rpm = (float(g) for g in COLUMN.fullmatch(name).groups())
        torque = np.array([float(r[j]) if r[j] else np.nan for r in body])
        ok = np.isfinite(torque)
        standards.append((f"{nominal:g}kcP", certified_kcp * 1000.0, rpm, h[ok], torque[ok] / rpm))
    return standards


class ViscometerCalibrationTask(Task):
    def __init__(self):
        self.standards = load_standards()
        m = len(self.standards)
        self.name = f"viscometer_calibration_d{3 + m}"
        self.dim = 3 + m
        self.bounds = np.array(
            [[-11.0, -6.0],  # log10 k
             [1.0, 3.0],  # p, viscosity exponent
             [0.01, 1.0]]  # h_c, mm
            + [[-1.0, 1.0]] * m  # beta_j, offset as a fraction of the mean drag
        )
        self.f_opt = json.loads((HERE / "data" / "reference_optimum.json").read_text())["f_opt"]
        # Flattened arrays for a vectorized objective.
        self._fluid = np.concatenate([np.full(len(s[3]), j) for j, s in enumerate(self.standards)])
        self._h = np.concatenate([s[3] for s in self.standards])
        self._drag = np.concatenate([s[4] for s in self.standards])
        self._mu = np.array([s[1] for s in self.standards])[self._fluid]
        self._mean_drag = np.array([s[4].mean() for s in self.standards])

    def predict(self, x: np.ndarray) -> np.ndarray:
        log_k, p, h_c = x[:3]
        offsets = x[3:] * self._mean_drag
        return 10.0**log_k * self._mu**p / (self._h + h_c) + offsets[self._fluid]

    def _f(self, x: np.ndarray) -> float:
        rel = (self.predict(x) - self._drag) / self._drag
        return 100.0 * float(np.sqrt(np.mean(rel**2)))


def make_task() -> Task:
    return ViscometerCalibrationTask()
