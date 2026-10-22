"""Integration checks for the MOF baseline runner using its saved real model.

Run from the sandbox with:
    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SANDBOX = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mof_run_campaign", SANDBOX / "scripts" / "run_campaign.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

from src.optimizers import OptimizerExhausted
from src.optimizers.random_search import RandomSearch


class RunCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = runner.TASKS["co2_uptake"]()
        cls.free = cls.task.bounds[:, 0] < cls.task.bounds[:, 1]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mof-runner-test-")
        self.addCleanup(self.temporary.cleanup)
        self.outdir = Path(self.temporary.name)

    def run_campaign(self, optimizer="random_search", budget=20, seed=0, outdir=None):
        return runner.run_campaign(
            "co2_uptake", optimizer, seed, budget,
            outdir=str(self.outdir if outdir is None else outdir),
        )

    def read_outputs(self, campaign):
        with (campaign / "trace.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            columns = reader.fieldnames
        meta = json.loads((campaign / "meta.json").read_text())
        return columns, rows, meta

    def assert_trace(self, rows, optimizer, budget):
        self.assertEqual(len(rows), budget)
        if not rows:
            return
        points = np.asarray([
            [float(row[f"x{d}"]) for d in range(self.task.dim)] for row in rows
        ])
        self.assertTrue(np.isfinite(points).all())
        self.assertTrue((points >= self.task.bounds[:, 0]).all())
        self.assertTrue((points <= self.task.bounds[:, 1]).all())
        np.testing.assert_array_equal(
            points[:, ~self.free],
            np.broadcast_to(self.task.bounds[~self.free, 0], points[:, ~self.free].shape),
        )
        values = np.asarray([float(row["y"]) for row in rows])
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(
            [float(row["y_best_so_far"]) for row in rows],
            np.minimum.accumulate(values),
        )
        elapsed = np.asarray([float(row["eval_seconds"]) for row in rows])
        self.assertTrue((elapsed >= 0).all())
        np.testing.assert_allclose(
            [float(row["cumulative_seconds"]) for row in rows], elapsed.cumsum()
        )
        self.assertEqual([int(row["eval_idx"]) for row in rows], list(range(1, budget + 1)))
        self.assertEqual([int(row["batch_idx"]) for row in rows], list(range(budget)))
        self.assertTrue(all(row["optimizer"] == optimizer for row in rows))

    def test_random_search_smoke_trace_metadata_and_reproducibility(self):
        campaign = self.run_campaign()
        self.assertEqual(
            campaign,
            self.outdir / "co2_uptake_d188_i1" / "random_search" / "seed_0",
        )
        self.assertEqual({path.name for path in campaign.iterdir()}, {"trace.csv", "meta.json"})
        columns, rows, meta = self.read_outputs(campaign)
        self.assertEqual(columns, [
            "eval_idx", "batch_idx", "optimizer",
            *[f"x{d}" for d in range(188)],
            "y", "y_best_so_far", "eval_seconds", "cumulative_seconds",
        ])
        self.assert_trace(rows, "random_search", 20)
        self.assertEqual(set(meta), {
            "task_name", "task_dim", "task_free_dim", "task_direction",
            "optimizer_name", "optimizer_class", "seed", "requested_budget",
            "evaluations_used", "stopped_early", "stop_reason", "start_time_utc",
            "end_time_utc", "total_wall_seconds", "numpy_version", "scipy_version",
            "xgboost_version", "status",
        })
        self.assertEqual(meta["task_name"], "co2_uptake_d188_i1")
        self.assertEqual(meta["task_dim"], 188)
        self.assertEqual(meta["task_free_dim"], 160)
        self.assertEqual(meta["task_direction"], "min")
        self.assertEqual(meta["optimizer_name"], "random_search")
        self.assertEqual(meta["optimizer_class"], "src.optimizers.random_search.RandomSearch")
        self.assertEqual(meta["seed"], 0)
        self.assertEqual(meta["requested_budget"], 20)
        self.assertEqual(meta["evaluations_used"], 20)
        self.assertIs(meta["stopped_early"], False)
        self.assertIsNone(meta["stop_reason"])
        self.assertEqual(meta["status"], "completed")
        self.assertGreaterEqual(meta["total_wall_seconds"], 0)
        started = datetime.fromisoformat(meta["start_time_utc"])
        ended = datetime.fromisoformat(meta["end_time_utc"])
        self.assertIsNotNone(started.tzinfo)
        self.assertGreaterEqual(ended, started)
        for key in ("numpy_version", "scipy_version", "xgboost_version"):
            self.assertTrue(meta[key])

        # Sampling must use the original units of only the free descriptors.
        reference = RandomSearch(bounds=self.task.bounds[self.free], seed=0)
        for row in rows:
            expected = reference.ask(n=1)
            actual = np.asarray([float(row[f"x{d}"]) for d in range(188)])
            np.testing.assert_array_equal(actual[self.free], expected[0])
            reference.tell(expected, [float(row["y"])])
        first_point = np.asarray([float(rows[0][f"x{d}"]) for d in range(188)])
        self.assertEqual(float(rows[0]["y"]), -self.task.predict_uptake(first_point))

        repeated = self.run_campaign(outdir=self.outdir / "repeated")
        _, repeated_rows, _ = self.read_outputs(repeated)
        deterministic_columns = [
            name for name in columns if name not in {"eval_seconds", "cumulative_seconds"}
        ]
        self.assertEqual(
            [[row[name] for name in deterministic_columns] for row in rows],
            [[row[name] for name in deterministic_columns] for row in repeated_rows],
        )

    def test_sobol_and_differential_evolution_support_fixed_descriptors(self):
        for optimizer in ("sobol", "de"):
            with self.subTest(optimizer=optimizer):
                campaign = self.run_campaign(optimizer=optimizer, budget=4)
                _, rows, meta = self.read_outputs(campaign)
                self.assert_trace(rows, optimizer, 4)
                self.assertEqual(meta["task_free_dim"], 160)
                self.assertEqual(meta["evaluations_used"], 4)
                self.assertIs(meta["stopped_early"], False)

    def test_optimizer_exhaustion_preserves_real_observations_and_tells_free_points(self):
        for limit in (0, 3):
            with self.subTest(limit=limit):
                instances = []

                class StopAfter(RandomSearch):
                    def __init__(self, bounds, seed):
                        super().__init__(bounds, seed)
                        instances.append(self)

                    def ask(self, n=1):
                        if len(self.y_) >= limit:
                            raise OptimizerExhausted("test optimizer completed")
                        return super().ask(n)

                with patch.object(runner, "_load_optimizer", return_value=StopAfter):
                    campaign = self.run_campaign(budget=5, outdir=self.outdir / str(limit))
                _, rows, meta = self.read_outputs(campaign)
                self.assert_trace(rows, "random_search", limit)
                self.assertIs(meta["stopped_early"], True)
                self.assertEqual(meta["stop_reason"], "test optimizer completed")
                self.assertEqual(meta["evaluations_used"], limit)
                self.assertEqual(meta["requested_budget"], 5)
                optimizer = instances[0]
                np.testing.assert_array_equal(optimizer.bounds, self.task.bounds[self.free])
                self.assertEqual(optimizer.X_.shape, (limit, 160))
                np.testing.assert_array_equal(optimizer.y_, [float(row["y"]) for row in rows])
                if rows:
                    full_points = np.asarray([
                        [float(row[f"x{d}"]) for d in range(188)] for row in rows
                    ])
                    np.testing.assert_array_equal(optimizer.X_, full_points[:, self.free])

    def test_nonempty_campaign_is_preserved_before_evaluation(self):
        campaign = self.outdir / "co2_uptake_d188_i1" / "random_search" / "seed_0"
        campaign.mkdir(parents=True)
        sentinel = campaign / "existing.txt"
        sentinel.write_text("preserve this campaign")
        with patch.object(type(self.task), "evaluate", side_effect=AssertionError("evaluated")):
            with self.assertRaises(FileExistsError):
                self.run_campaign()
        self.assertEqual({path.name for path in campaign.iterdir()}, {"existing.txt"})
        self.assertEqual(sentinel.read_text(), "preserve this campaign")

    def test_invalid_budget_and_seed_fail_before_evaluation(self):
        with patch.object(type(self.task), "evaluate", side_effect=AssertionError("evaluated")):
            for budget in (0, -1, 1.5, True, "20"):
                with self.subTest(budget=budget):
                    with self.assertRaises((TypeError, ValueError)):
                        self.run_campaign(budget=budget)
            for seed in (-1, 0.5, True, "0"):
                with self.subTest(seed=seed):
                    with self.assertRaises((TypeError, ValueError)):
                        self.run_campaign(seed=seed)
        self.assertEqual(list(self.outdir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
