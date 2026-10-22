"""CO2 uptake task using the saved XGBoost surrogate."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import xgboost as xgb


def _reference_bounds(project_root):
    """Match ws_ml.py's descriptor order without importing its training code."""
    data_root = project_root / "data" / "chemunity"
    omitted = {
        "ref code", "filename", "0", "V_cell_2 (A^3)",
        "density_2 (g/cm^3)", "V_cell (A^3)", "ASA (A^2)",
        "NASA (A^2)", "AV (A^3)", "NAV (A^3)",
    }
    names = []
    for filename in ("ChemUnity_Full_geo.csv", "core_desc_df_full.csv"):
        with (data_root / "descriptors" / filename).open(newline="") as f:
            names.extend(name for name in next(csv.reader(f)) if name not in omitted)

    # The objective notebook renames these columns before selecting descriptors.
    source_columns = {
        "Di": "Largest_included_sphere",
        "Df": "Largest_free_sphere",
        "Dif": "Largest_included_sphere_along_free_path",
        "density (g/cm^3)": "Density",
        "ASA (m^2/cm^3)": "ASA_m^2/cm^3",
        "ASA (m^2/g)": "ASA_m^2/g",
        "NASA (m^2/cm^3)": "NASA_m^2/cm^3",
        "NASA (m^2/g)": "NASA_m^2/g",
        "AVF": "POAV_Volume_fraction",
        "AV (cm^3/g)": "POAV_cm^3/g",
        "NAVF": "PONAV_Volume_fraction",
        "NAV (cm^3/g)": "PONAV_cm^3/g",
    }
    columns = [source_columns.get(name, name) for name in names]
    reference_path = data_root / "csv" / "ChemUnity_all_descriptors.csv"
    lower = np.full(len(names), np.inf)
    upper = np.full(len(names), -np.inf)
    with reference_path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing descriptor columns in {reference_path}: {sorted(missing)}")
        for row in reader:
            try:
                values = np.asarray([float(row[column]) for column in columns])
            except (TypeError, ValueError):
                continue  # Use only complete, numeric descriptor rows.
            if np.isfinite(values).all():
                lower = np.minimum(lower, values)
                upper = np.maximum(upper, values)
    if not np.isfinite(lower).all():
        raise ValueError(f"No complete finite descriptor rows in {reference_path}")
    return names, np.column_stack((lower, upper))


class CO2UptakeTask:
    """Minimize negative predicted CO2 uptake.

    Parameters
    ----------
    bounds : array-like of shape (n_descriptors, 2), optional
        Descriptor bounds for optimization. Defaults to per-column minima
        and maxima over all complete finite rows of the local
        src/agent_harness/data/chemunity/csv/ChemUnity_all_descriptors.csv
        reference set.
        Equal bounds keep constant descriptors fixed.
    model_path : str or Path, optional
        Override the default saved-model path.
    instance : int, default 1
        Select co2_uptake_<instance>.model, whose suffix is the model's
        training seed in the notebook. This does not retrain the model or
        set the optimizer seed. Ignored for model selection if model_path
        is supplied.

    Notes
    -----
    Inputs must use exactly the notebook's descriptor_names order,
    with no additional scaling. The default box describes observed feature
    ranges; arbitrary combinations within it need not describe a real MOF.
    """

    def __init__(self, bounds=None, model_path=None, instance=1):
        # Keep model and reference data local to this sandbox's harness.
        harness_root = Path(__file__).resolve().parent
        if model_path is None:
            model_path = (
                harness_root
                / "models"
                / f"co2_uptake_{instance}.model"
            )

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load once; reuse for every evaluation.
        self._model = xgb.Booster(params={"nthread": 1})
        self._model.load_model(str(self.model_path))

        self.dim = int(self._model.num_features())
        self.name = f"co2_uptake_d{self.dim}_i{instance}"
        self.direction = "min"

        self.descriptor_names = None
        if bounds is None:
            self.descriptor_names, bounds = _reference_bounds(harness_root)
        bounds = np.asarray(bounds, dtype=float)

        if bounds.shape != (self.dim, 2):
            raise ValueError(
                f"bounds must have shape ({self.dim}, 2), "
                f"got {bounds.shape}."
            )
        if not np.isfinite(bounds).all():
            raise ValueError("bounds must contain only finite values.")
        if np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("Each lower bound must be at or below its upper bound.")

        self.bounds = bounds.copy()

    def predict_uptake(self, x):
        """Return raw predicted uptake for one descriptor vector."""
        x = np.asarray(x, dtype=float)

        if x.shape != (self.dim,):
            raise ValueError(
                f"x must have shape ({self.dim},), got {x.shape}."
            )
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite descriptor values.")
        if np.any(x < self.bounds[:, 0]) or np.any(x > self.bounds[:, 1]):
            raise ValueError("x must lie within the descriptor bounds.")

        # XGBoost expects (n_samples, n_descriptors).
        return float(self._model.inplace_predict(x.reshape(1, -1))[0])

    def evaluate(self, x):
        """Return (negative_predicted_uptake, elapsed_seconds)."""
        start = time.perf_counter()
        y = -self.predict_uptake(x)
        elapsed = time.perf_counter() - start
        return y, elapsed
