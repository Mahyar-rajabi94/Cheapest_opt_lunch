"""Regression checks against the supplied high-pressure methane model.

Run from the sandbox with:
    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import xgboost as xgb

from src.agent_harness.ch4_uptake import CH4UptakeTask


SANDBOX = Path(__file__).resolve().parents[1]
HARNESS = SANDBOX / "src" / "agent_harness"
MODEL = HARNESS / "xgb" / "ch4_uptake_1.model"

# The notebook places geometric descriptors before RACs, whose source table
# order differs from the combined reference CSV's order.
GEOMETRY = [
    ("Di", "Largest_included_sphere"),
    ("Df", "Largest_free_sphere"),
    ("Dif", "Largest_included_sphere_along_free_path"),
    ("density (g/cm^3)", "Density"),
    ("ASA (m^2/cm^3)", "ASA_m^2/cm^3"),
    ("ASA (m^2/g)", "ASA_m^2/g"),
    ("NASA (m^2/cm^3)", "NASA_m^2/cm^3"),
    ("NASA (m^2/g)", "NASA_m^2/g"),
    ("AVF", "POAV_Volume_fraction"),
    ("AV (cm^3/g)", "POAV_cm^3/g"),
    ("NAVF", "PONAV_Volume_fraction"),
    ("NAV (cm^3/g)", "PONAV_cm^3/g"),
]


class CH4UptakeTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = CH4UptakeTask()
        cls.midpoint = cls.task.bounds.mean(axis=1)

    def test_default_uses_supplied_methane_model_and_real_predictions(self):
        self.assertEqual(self.task.model_path.resolve(), MODEL.resolve())
        self.assertEqual(self.task.name, "ch4_uptake_d188_i1")
        self.assertEqual(self.task.direction, "min")
        self.assertEqual(self.task.dim, 188)

        reference_model = xgb.Booster(params={"nthread": 1})
        reference_model.load_model(str(MODEL))
        self.assertEqual(reference_model.num_features(), self.task.dim)
        lower, upper = self.task.bounds.T
        rng = np.random.default_rng(17)
        points = np.vstack([
            lower, upper, self.midpoint,
            rng.uniform(lower, upper, size=(5, self.task.dim)),
        ])
        expected = reference_model.predict(xgb.DMatrix(points))
        for point, prediction in zip(points, expected):
            with self.subTest(point=point[:3]):
                uptake = self.task.predict_uptake(point)
                value, seconds = self.task.evaluate(point)
                self.assertEqual(uptake, float(prediction))
                self.assertEqual(value, -float(prediction))
                self.assertGreaterEqual(seconds, 0)

    def test_descriptor_order_and_bounds_match_complete_reference_rows(self):
        data = HARNESS / "data" / "chemunity"
        with (data / "descriptors" / "core_desc_df_full.csv").open(newline="") as stream:
            rac_names = next(csv.reader(stream))[:-2]
        self.assertEqual(len(rac_names), 176)
        self.assertEqual(rac_names[:4], [f"f-chi-{i}-all" for i in range(4)])
        self.assertEqual(rac_names[-4:], [f"D_func-alpha-{i}-all" for i in range(4)])
        expected_names = [name for name, _ in GEOMETRY] + rac_names
        self.assertEqual(self.task.descriptor_names, expected_names)
        self.assertEqual(len(set(expected_names)), 188)

        columns = [column for _, column in GEOMETRY] + rac_names
        with (data / "csv" / "ChemUnity_all_descriptors.csv").open(newline="") as stream:
            records = csv.DictReader(stream)
            numeric_rows = []
            for record in records:
                try:
                    numeric_rows.append([float(record[column]) for column in columns])
                except (TypeError, ValueError):
                    continue
        matrix = np.asarray(numeric_rows)
        matrix = matrix[np.isfinite(matrix).all(axis=1)]
        self.assertGreater(len(matrix), 0)
        expected_bounds = np.stack([matrix.min(axis=0), matrix.max(axis=0)], axis=1)
        np.testing.assert_array_equal(self.task.bounds, expected_bounds)
        self.assertEqual(np.count_nonzero(expected_bounds[:, 0] < expected_bounds[:, 1]), 160)

    def test_invalid_descriptors_fail_before_prediction(self):
        invalid = [
            self.midpoint[:-1],
            np.append(self.midpoint, 0),
            self.midpoint.reshape(1, -1),
            np.full(self.task.dim, np.nan),
            np.full(self.task.dim, np.inf),
            np.full(self.task.dim, -np.inf),
        ]
        for column in (0, 12):
            below = self.midpoint.copy()
            below[column] = self.task.bounds[column, 0] - 1
            above = self.midpoint.copy()
            above[column] = self.task.bounds[column, 1] + 1
            invalid.extend([below, above])
        for point in invalid:
            with self.subTest(shape=point.shape, values=point[:3]):
                with self.assertRaises(ValueError):
                    self.task.predict_uptake(point)
                with self.assertRaises(ValueError):
                    self.task.evaluate(point)

    def test_invalid_bounds_and_missing_models_are_rejected(self):
        inverted = self.task.bounds.copy()
        inverted[0] = [2, 1]
        for bounds in (
            self.task.bounds[:-1],
            self.task.bounds[:, 0],
            np.full((self.task.dim, 2), np.nan),
            np.full((self.task.dim, 2), np.inf),
            inverted,
        ):
            with self.subTest(shape=bounds.shape):
                with self.assertRaises(ValueError):
                    CH4UptakeTask(bounds=bounds)
        with tempfile.TemporaryDirectory(prefix="ch4-model-test-") as directory:
            with self.assertRaises(FileNotFoundError):
                CH4UptakeTask(model_path=Path(directory) / "missing.model")

    def test_custom_fixed_bounds_are_copied_and_model_override_is_respected(self):
        bounds = np.column_stack([self.midpoint, self.midpoint])
        task = CH4UptakeTask(bounds=bounds, model_path=MODEL)
        bounds[:] = 0
        np.testing.assert_array_equal(task.bounds[:, 0], self.midpoint)
        np.testing.assert_array_equal(task.bounds[:, 1], self.midpoint)
        self.assertEqual(task.model_path.resolve(), MODEL.resolve())
        self.assertEqual(task.predict_uptake(self.midpoint), self.task.predict_uptake(self.midpoint))
        invalid = self.midpoint.copy()
        invalid[0] += 1
        with self.assertRaises(ValueError):
            task.evaluate(invalid)

if __name__ == "__main__":
    unittest.main()
