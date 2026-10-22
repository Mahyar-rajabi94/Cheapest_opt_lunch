"""Regression checks for the sandbox sweep; all outputs are temporary.

Run from the sandbox with:
    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SANDBOX = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mof_run_sweep", SANDBOX / "scripts" / "run_sweep.py"
)
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)

from src.optimizers import OptimizerExhausted
from src.optimizers.random_search import RandomSearch


class RunSweepTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="mof-sweep-test-")
        self.addCleanup(temporary.cleanup)
        self.outdir = Path(temporary.name)
        self.output = io.StringIO()

    def sweep(self, budget=2, seeds=2, workers=1, optimizers=None):
        with redirect_stdout(self.output):
            return sweep._sweep(
                "ch4_uptake", budget, seeds, str(self.outdir), workers,
                optimizers=optimizers,
            )

    def campaign_dir(self, name="random_search", seed=0):
        return self.outdir / "ch4_uptake_d188_i1" / name / f"seed_{seed}"

    def read_meta(self, campaign):
        return json.loads((campaign / "meta.json").read_text())

    def write_meta(self, campaign, metadata):
        (campaign / "meta.json").write_text(json.dumps(metadata))

    def run_small_campaign(self):
        return sweep.run_campaign("ch4_uptake", "random_search", 0, 2, str(self.outdir))

    def assert_campaign(self, campaign, seed, budget=2):
        self.assertEqual({p.name for p in campaign.iterdir()}, {"trace.csv", "meta.json"})
        metadata = self.read_meta(campaign)
        self.assertEqual(metadata["task_name"], "ch4_uptake_d188_i1")
        self.assertEqual(metadata["task_direction"], "min")
        self.assertEqual(metadata["task_dim"], 188)
        self.assertEqual(metadata["task_free_dim"], 160)
        self.assertEqual(metadata["optimizer_name"], "random_search")
        self.assertEqual(metadata["seed"], seed)
        self.assertEqual(metadata["requested_budget"], budget)
        self.assertEqual(metadata["evaluations_used"], budget)
        self.assertFalse(metadata["stopped_early"])
        self.assertIsNone(metadata["stop_reason"])
        with (campaign / "trace.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), budget)
        self.assertEqual([int(row["eval_idx"]) for row in rows], list(range(1, budget + 1)))
        ys = [float(row["y"]) for row in rows]
        self.assertEqual(
            [float(row["y_best_so_far"]) for row in rows],
            [min(ys[:index + 1]) for index in range(budget)],
        )

    def test_all_optimizer_plan_caps_and_deterministic_seeds(self):
        with patch.object(sweep, "_timed_campaign", return_value=0.01) as timed:
            directories, failures = self.sweep(budget=150, seeds=3)
        self.assertEqual(failures, [])
        expected_deterministic = {"bobyqa", "nelder_mead", "lbfgsb"}
        expected_caps = {"gp_bo": 100, "turbo": 100, "bobyqa": 100}
        expected = {
            (name, seed, min(150, expected_caps.get(name, 150)))
            for name in sweep.OPTIMIZERS
            for seed in range(1 if name in expected_deterministic else 3)
        }
        actual = {(call.args[1], call.args[2], call.args[3]) for call in timed.call_args_list}
        self.assertEqual(actual, expected)
        self.assertEqual(len(directories), len(expected))
        self.assertEqual(len(timed.call_args_list), len(expected))

    def test_explicit_selection_avoids_unrequested_optimizers(self):
        with patch.object(sweep, "_timed_campaign", return_value=0.01) as timed:
            directories, failures = self.sweep(optimizers=["random_search", "nelder_mead"])
        self.assertEqual(failures, [])
        self.assertEqual(len(directories), 3)
        self.assertEqual(
            {(call.args[1], call.args[2]) for call in timed.call_args_list},
            {("random_search", 0), ("random_search", 1), ("nelder_mead", 0)},
        )

    def test_invalid_inputs_fail_before_campaign_calls(self):
        cases = [
            {key: value}
            for key in ("budget", "seeds", "workers")
            for value in (0, -1, 0.5, True, "2")
        ]
        cases.extend([{"optimizers": ["missing"]}, {"optimizers": []}])
        with patch.object(sweep, "_timed_campaign", side_effect=AssertionError("ran a campaign")):
            for kwargs in cases:
                with self.subTest(**kwargs):
                    with self.assertRaises((TypeError, ValueError)):
                        self.sweep(**kwargs)
            with self.assertRaises(ValueError):
                sweep._sweep("missing", 2, 2, str(self.outdir), 1)
        self.assertEqual(list(self.outdir.iterdir()), [])

    def test_serial_random_search_and_matching_resume(self):
        directories, failures = self.sweep(optimizers=["random_search"])
        self.assertEqual(failures, [])
        self.assertEqual(set(directories), {self.campaign_dir(seed=0), self.campaign_dir(seed=1)})
        for seed in (0, 1):
            self.assert_campaign(self.campaign_dir(seed=seed), seed)
        before = {file: file.read_bytes() for campaign in directories for file in campaign.iterdir()}
        with patch.object(sweep, "run_campaign", side_effect=AssertionError("resumed completed run")):
            resumed, failures = self.sweep(optimizers=["random_search"])
        self.assertEqual(set(resumed), set(directories))
        self.assertEqual(failures, [])
        self.assertEqual({file: file.read_bytes() for file in before}, before)
        self.assertIn("skipped 2", self.output.getvalue())

    def test_different_budget_aborts_before_any_new_campaign(self):
        campaign = self.run_small_campaign()
        before = {file: file.read_bytes() for file in campaign.iterdir()}
        with patch.object(sweep, "_timed_campaign", side_effect=AssertionError("started before preflight")):
            with self.assertRaises((ValueError, FileExistsError)):
                self.sweep(budget=3, optimizers=["random_search"])
        self.assertFalse(self.campaign_dir(seed=1).exists())
        self.assertEqual({file: file.read_bytes() for file in before}, before)

    def test_partial_and_corrupt_outputs_abort_before_new_campaigns(self):
        campaign = self.run_small_campaign()
        original_meta = (campaign / "meta.json").read_bytes()
        original_trace = (campaign / "trace.csv").read_bytes()
        for kind in ("no_meta", "bad_meta", "no_trace", "truncated_trace"):
            with self.subTest(kind=kind):
                (campaign / "meta.json").write_bytes(original_meta)
                (campaign / "trace.csv").write_bytes(original_trace)
                if kind == "no_meta":
                    (campaign / "meta.json").unlink()
                elif kind == "bad_meta":
                    (campaign / "meta.json").write_text("{invalid-json")
                elif kind == "no_trace":
                    (campaign / "trace.csv").unlink()
                else:
                    (campaign / "trace.csv").write_bytes(original_trace.splitlines(keepends=True)[0])
                before = {file: file.read_bytes() for file in campaign.iterdir()}
                with patch.object(sweep, "_timed_campaign", side_effect=AssertionError("started before preflight")):
                    with self.assertRaises((ValueError, FileExistsError)):
                        self.sweep(optimizers=["random_search"])
                self.assertEqual({file: file.read_bytes() for file in before}, before)
                self.assertFalse(self.campaign_dir(seed=1).exists())

    def test_clean_optimizer_exhaustion_can_resume(self):
        campaign_module = sys.modules[sweep.run_campaign.__module__]

        class StopAfterOne(RandomSearch):
            def ask(self, n=1):
                if len(self.y_) == 1:
                    raise OptimizerExhausted("test optimizer completed")
                return super().ask(n)

        with patch.object(campaign_module, "_load_optimizer", return_value=StopAfterOne):
            campaign = self.run_small_campaign()
        metadata = self.read_meta(campaign)
        self.assertEqual(metadata["evaluations_used"], 1)
        self.assertTrue(metadata["stopped_early"])
        with patch.object(sweep, "run_campaign", side_effect=AssertionError("reran clean early stop")):
            directories, failures = self.sweep(seeds=1, optimizers=["random_search"])
        self.assertEqual(directories, [campaign])
        self.assertEqual(failures, [])

    def test_error_metadata_is_not_treated_as_completed(self):
        campaign_module = sys.modules[sweep.run_campaign.__module__]

        class FailingTell(RandomSearch):
            def tell(self, X, y):
                super().tell(X, y)
                if len(self.y_) == 2:
                    raise RuntimeError("test failure after recording the final observation")

        with patch.object(campaign_module, "_load_optimizer", return_value=FailingTell):
            with self.assertRaises(RuntimeError):
                self.run_small_campaign()
        campaign = self.campaign_dir()
        metadata = self.read_meta(campaign)
        self.assertEqual(metadata["evaluations_used"], 2)
        self.assertEqual(metadata["status"], "failed")
        self.assertTrue(metadata["stop_reason"].startswith("RuntimeError:"))
        with patch.object(sweep, "_timed_campaign", side_effect=AssertionError("started before preflight")):
            with self.assertRaises((ValueError, FileExistsError)):
                self.sweep(optimizers=["random_search"])
        self.assertFalse(self.campaign_dir(seed=1).exists())

    def test_failed_jobs_are_reported_and_other_jobs_continue(self):
        def timed(task_name, name, seed, budget, outdir):
            if seed == 0:
                raise RuntimeError("test failure")
            return 0.01

        with patch.object(sweep, "_timed_campaign", side_effect=timed):
            directories, failures = self.sweep(optimizers=["random_search"])
        self.assertEqual(directories, [self.campaign_dir(seed=1)])
        self.assertEqual(failures, [("random_search", 0)])
        self.assertIn("FAILED", self.output.getvalue())
        self.assertIn("ran 1, skipped 0, failed 1", self.output.getvalue())

    def test_public_wrapper_returns_completed_directories(self):
        expected = [self.campaign_dir()]
        with patch.object(sweep, "_sweep", return_value=(expected, [])) as internal:
            actual = sweep.run_sweep("ch4_uptake", optimizers=["random_search"])
        self.assertEqual(actual, expected)
        internal.assert_called_once()

    def test_parallel_cli_random_search(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_sweep.py", "--task", "ch4_uptake",
             "--optimizers", "random_search", "--budget", "2", "--seeds", "2",
             "--workers", "2", "--outdir", str(self.outdir)],
            cwd=SANDBOX, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ran 2, skipped 0, failed 0", result.stdout)
        for seed in (0, 1):
            self.assert_campaign(self.campaign_dir(seed=seed), seed)


if __name__ == "__main__":
    unittest.main()
