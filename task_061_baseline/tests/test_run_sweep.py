"""Regression tests for fixed-pool sweeps; every objective is toy CSV data."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "run_sweep.py"
SPEC = importlib.util.spec_from_file_location("task_061_run_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sweep
SPEC.loader.exec_module(sweep)


class FakeOptimizer:
    """Return off-pool proposals, recording the actual training observations."""

    instances = []
    proposals = np.array([[0.1, 100], [0.9, 100], [0.1, 900], [0.9, 900]])

    def __init__(self, bounds, seed=None):
        self.bounds = np.asarray(bounds)
        self.seed = seed
        self.n_init = 999
        self.ask_count = 0
        self.feedback = []
        self.instances.append(self)

    def ask(self, n=1):
        if n != 1:
            raise AssertionError("Expected sequential asks")
        proposal = self.proposals[self.ask_count].reshape(1, -1).copy()
        self.ask_count += 1
        return proposal

    def tell(self, points, values):
        self.feedback.append((np.asarray(points).copy(), np.asarray(values).copy()))


class SweepTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "toy.csv"
        self.fields = ["", "name", "descriptor_x", "descriptor_y", "constant",
                       "Unnamed: 0", sweep.TARGET]
        self.rows = [
            [100, "alpha", 0, 0, 7, 900, 5],
            [101, "beta", 1, 0, 7, 901, 7],
            [102, "gamma", 0, 1000, 7, 902, 9],
            [103, "delta", 1, 1000, 7, 903, 9],
        ]
        self.write_csv(self.data, self.fields, self.rows)
        FakeOptimizer.instances.clear()

    @staticmethod
    def write_csv(path, fields, rows):
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)

    @staticmethod
    def read_csv(path):
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))

    def arguments(self, **updates):
        values = dict(data=self.data, budget=4, seeds=2, workers=1, n_init=2,
                      optimizers=["gp_bo", "turbo"], outdir=self.root / "sweep")
        values.update(updates)
        return SimpleNamespace(**values)

    @staticmethod
    def fake_classes():
        return {"gp_bo": FakeOptimizer, "turbo": FakeOptimizer}

    def run_toy_sweep(self, args):
        with mock.patch.object(sweep, "_optimizer_classes", self.fake_classes), \
                mock.patch.object(sweep, "_init_worker"), \
                contextlib.redirect_stdout(io.StringIO()):
            return sweep.run_sweep(args)


class PoolTests(SweepTestCase):
    def test_bounds_exclude_objective_and_metadata_and_drop_constant_inputs(self):
        pool = sweep.load_pool(self.data)
        self.assertEqual(pool.descriptor_names,
                         ["descriptor_x", "descriptor_y", "constant"])
        self.assertEqual(pool.names, ["alpha", "beta", "gamma", "delta"])
        np.testing.assert_array_equal(pool.bounds, [[0, 1], [0, 1000], [7, 7]])
        np.testing.assert_array_equal(pool.active, [True, True, False])
        np.testing.assert_array_equal(pool.optimizer_bounds, [[0, 1], [0, 1000]])
        np.testing.assert_array_equal(pool.optimizer_points,
                                      [[0, 0], [1, 0], [0, 1000], [1, 1000]])
        np.testing.assert_array_equal(pool.uptake, [5, 7, 9, 9])

    def test_projection_normalizes_units_and_never_reuses_a_row(self):
        # Raw Euclidean distance would choose row 0 because of the y units;
        # min/max normalization makes row 1 the nearer MOF.
        projection = sweep.PoolProjection(np.array([[0, 1000], [1, 0]]),
                                         np.array([[0, 1], [0, 1000]]))
        self.assertEqual(projection.select(np.array([0.9, 600])), 1)
        self.assertEqual(projection.select(np.array([0.9, 600])), 0)
        with self.assertRaisesRegex(ValueError, "already been evaluated"):
            projection.select(np.array([0.9, 600]))

    def test_projection_rejects_invalid_proposals(self):
        pool = sweep.load_pool(self.data)
        projection = sweep.PoolProjection(pool.optimizer_points, pool.optimizer_bounds)
        for proposal in ([0], [np.nan, 0], [0, np.inf], [[0, 0]]):
            with self.subTest(proposal=proposal):
                with self.assertRaisesRegex(ValueError, "invalid descriptor proposal"):
                    projection.select(np.asarray(proposal))
        self.assertTrue(projection.available.all())

    def test_optimizer_adapter_returns_actual_coordinates_and_configures_initialization(self):
        pool = sweep.load_pool(self.data)
        with mock.patch.object(sweep, "_optimizer_classes", self.fake_classes):
            optimizer = sweep._make_optimizer("gp_bo", pool, seed=12, n_init=3)
        np.testing.assert_array_equal(optimizer.bounds, [[0, 1], [0, 1000]])
        self.assertEqual(optimizer.seed, 12)
        self.assertEqual(optimizer.n_init, 3)
        actual = optimizer.ask()
        np.testing.assert_array_equal(actual, [[0, 0]])
        self.assertEqual(optimizer.row_index, 0)
        with self.assertRaisesRegex(ValueError, "one MOF at a time"):
            optimizer.ask(n=2)

    def test_invalid_and_constant_only_tables_are_rejected(self):
        invalid_rows = [row.copy() for row in self.rows]
        invalid_rows[0][-1] = "nan"
        self.write_csv(self.data, self.fields, invalid_rows)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            sweep.load_pool(self.data)
        self.write_csv(self.data, ["name", "constant", sweep.TARGET],
                       [["one", 7, 1], ["two", 7, 2]])
        with self.assertRaisesRegex(ValueError, "at least one descriptor must vary"):
            sweep.load_pool(self.data)


class CampaignTests(SweepTestCase):
    def test_feedback_uses_actual_coordinates_negative_uptake_and_competition_ranks(self):
        directory = self.root / "campaign"
        with mock.patch.object(sweep, "_optimizer_classes", self.fake_classes):
            summary = sweep.run_campaign(str(self.data), "turbo", 4, 4, 3, str(directory))
        optimizer, = FakeOptimizer.instances
        self.assertEqual(optimizer.n_init, 3)
        np.testing.assert_array_equal(
            np.vstack([points for points, _ in optimizer.feedback]),
            [[0, 0], [1, 0], [0, 1000], [1, 1000]],
        )
        np.testing.assert_array_equal(
            np.concatenate([values for _, values in optimizer.feedback]), [-5, -7, -9, -9],
        )
        trace = self.read_csv(directory / "trace.csv")
        self.assertEqual([int(row["rank"]) for row in trace], [4, 3, 1, 1])
        self.assertEqual([int(row["best_rank_so_far"]) for row in trace], [4, 3, 1, 1])
        self.assertEqual([float(row[sweep.TARGET]) for row in trace], [5, 7, 9, 9])
        self.assertEqual([row["best_mof_so_far"] for row in trace],
                         ["alpha", "beta", "gamma", "gamma"])
        self.assertEqual(len({row["row_index"] for row in trace}), 4)
        self.assertEqual(summary["best_rank"], 1)
        self.assertEqual(summary["best_mof"], "gamma")
        metadata = json.loads((directory / "meta.json").read_text())
        self.assertEqual(metadata["evaluations_used"], 4)
        self.assertEqual(metadata["requested_budget"], 4)
        self.assertEqual(metadata["direction"], "max")
        self.assertEqual(metadata["target"], sweep.TARGET)


class SweepTests(SweepTestCase):
    def test_summary_bounds_and_resume_without_rerunning_completed_jobs(self):
        args = self.arguments()
        self.assertEqual(self.run_toy_sweep(args), 0)
        self.assertEqual(len(FakeOptimizer.instances), 4)
        summary_path = args.outdir / "best_ranks.csv"
        summaries = self.read_csv(summary_path)
        self.assertEqual([(row["optimizer"], int(row["seed"])) for row in summaries],
                         [("gp_bo", 0), ("gp_bo", 1), ("turbo", 0), ("turbo", 1)])
        self.assertTrue(all(row["best_rank"] == "1" for row in summaries))
        self.assertEqual(json.loads((args.outdir / "bounds.json").read_text()),
                         [[0, 1], [0, 1000], [7, 7]])
        self.assertEqual(json.loads((args.outdir / "descriptor_names.json").read_text()),
                         ["descriptor_x", "descriptor_y", "constant"])
        manifest = json.loads((args.outdir / "sweep.json").read_text())
        self.assertEqual(manifest["active_descriptors"], ["descriptor_x", "descriptor_y"])
        traces = {path: path.read_bytes() for path in args.outdir.glob("*/seed_*/trace.csv")}
        with mock.patch.object(sweep, "_optimizer_classes", side_effect=AssertionError("rerun")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sweep.run_sweep(args), 0)
        self.assertEqual(self.read_csv(summary_path), summaries)
        for path, content in traces.items():
            self.assertEqual(path.read_bytes(), content)

    def test_configuration_conflict_and_incomplete_run_preserve_existing_results(self):
        args = self.arguments(seeds=1)
        self.assertEqual(self.run_toy_sweep(args), 0)
        summary_path = args.outdir / "best_ranks.csv"
        before = summary_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "different configuration"):
            sweep.run_sweep(self.arguments(seeds=1, budget=3))
        self.assertEqual(summary_path.read_bytes(), before)
        (args.outdir / "gp_bo" / "seed_0" / "meta.json").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete existing run"):
            sweep.run_sweep(args)
        self.assertEqual(summary_path.read_bytes(), before)

    def test_budget_larger_than_pool_is_rejected_before_writing_outputs(self):
        args = self.arguments(budget=5)
        with mock.patch.object(sweep, "_optimizer_classes", side_effect=AssertionError("import")):
            with self.assertRaisesRegex(ValueError, "budget exceeds the pool of 4"):
                sweep.run_sweep(args)
        self.assertFalse(args.outdir.exists())

    def test_cli_rejects_invalid_budgets_initial_design_and_duplicate_optimizers(self):
        invalid_options = [
            ["--budget", "0"], ["--budget", "-1"], ["--seeds", "0"],
            ["--workers", "0"], ["--n-init", "1"],
            ["--optimizers", "gp_bo", "gp_bo"],
        ]
        for options in invalid_options:
            with self.subTest(options=options), \
                    mock.patch.object(sys, "argv", [str(SCRIPT), *options]), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    sweep._parse_args()
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
