"""Finite-pool Bayesian optimization checks use only synthetic observations."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))

from src.optimizers import OptimizerExhausted
from src.optimizers.pool_gp_bo import GPBO
from src.optimizers.pool_turbo import TuRBO


class PoolOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def pool(self, rows=23, dimensions=3):
        pool = np.random.default_rng(10).random((rows, dimensions))
        pool[:, -1] = 7.0  # Constant columns must remain finite after normalization.
        return pool

    @staticmethod
    def fake_model(dimensions):
        return SimpleNamespace(covar_module=SimpleNamespace(
            lengthscale=torch.ones(1, dimensions, dtype=torch.double)))

    @staticmethod
    def last_row_sampler(**kwargs):
        return lambda points, num_samples: points[-num_samples:]

    def test_gpbo_scores_every_available_row_and_maximizes_raw_y(self):
        pool = self.pool()
        pool[:, 0] = np.arange(len(pool))
        optimizer = GPBO(candidates=pool, direction="max", seed=4)
        y = np.arange(10, dtype=float) - 20
        optimizer.tell(pool[:10], y)
        optimizer.pool_chunk_size = 3
        seen = []
        acquisition_options = {}

        def acquisition(**kwargs):
            acquisition_options.update(kwargs)

            def score(points):
                seen.extend(points[:, 0, 0].tolist())
                return points[:, 0, 0]

            return score

        available = np.array([14, 10, 22, 12, 20, 11, 18, 15])
        with patch.object(optimizer, "_fit_model", return_value=object()), patch(
                "src.optimizers.pool_gp_bo.LogExpectedImprovement", side_effect=acquisition):
            index = optimizer.ask_pool(available)
        self.assertEqual(index, 22)
        self.assertEqual(seen, pool[available, 0].tolist())
        self.assertTrue(acquisition_options["maximize"])
        self.assertEqual(acquisition_options["best_f"], -11)
        np.testing.assert_array_equal(optimizer.best()[0], pool[9])
        self.assertEqual(optimizer.best()[1], -11)

    def test_193_dimensions_use_bayesian_path_after_ten_seeds(self):
        pool = self.pool(dimensions=193)
        gpbo = GPBO(candidates=pool, direction="max", seed=1)
        turbo = TuRBO(candidates=pool, direction="max", seed=1)
        for optimizer in (gpbo, turbo):
            self.assertEqual(optimizer.n_init, 10)
            optimizer.tell(pool[:10], np.arange(10))
            with self.assertRaisesRegex(RuntimeError, "ask_pool"):
                optimizer.ask()
        with patch.object(gpbo, "_fit_model", return_value=object()) as fit, patch(
                "src.optimizers.pool_gp_bo.LogExpectedImprovement",
                return_value=lambda X: X[:, 0, 0]):
            self.assertIn(gpbo.ask_pool(np.arange(10, len(pool))), range(10, len(pool)))
            fit.assert_called_once()
        with patch.object(turbo, "_fit_pool_model", return_value=self.fake_model(193)) as fit, patch(
                "src.optimizers.pool_turbo.MaxPosteriorSampling", side_effect=self.last_row_sampler):
            self.assertIn(turbo.ask_pool(np.arange(10, len(pool))), range(10, len(pool)))
            fit.assert_called_once()
        self.assertEqual(turbo.batch_size, 1)
        self.assertEqual(turbo._phase, "evolve")
        self.assertFalse(turbo._to_ask)

    def test_turbo_empty_region_expands_to_actual_rows_and_bounds_sampling(self):
        pool = np.zeros((30, 3))
        pool[:10, 0] = np.linspace(0, 0.01, 10)
        pool[10:, 0] = np.linspace(0.8, 1, 20)
        pool[:, -1] = 7
        optimizer = TuRBO(candidates=pool, direction="max", seed=1)
        optimizer.tell(pool[:10], np.arange(10))
        optimizer._length = 0.01
        optimizer.n_candidates = 4
        sampled = []

        def sampler(**kwargs):
            def draw(points, num_samples):
                sampled.append(points.numpy())
                return points[-1:]
            return draw

        with patch.object(optimizer, "_fit_pool_model", return_value=self.fake_model(3)), patch(
                "src.optimizers.pool_turbo.MaxPosteriorSampling", side_effect=sampler):
            index = optimizer.ask_pool(np.arange(10, 30))
        self.assertIn(index, range(10, 30))
        self.assertGreater(optimizer._last_pool_effective_length, optimizer._length)
        self.assertEqual(len(sampled[0]), 4)
        self.assertTrue(np.isfinite(sampled[0]).all())
        for point in sampled[0]:
            self.assertTrue(np.any(np.all(optimizer._pool_unit[10:] == point, axis=1)))
        np.testing.assert_array_equal(sampled[0][:, -1], 0)

    def test_turbo_feedback_updates_each_point_and_never_discards_history(self):
        pool = self.pool(rows=30)
        optimizer = TuRBO(candidates=pool, direction="max", seed=1)
        optimizer.tell(pool[:10], np.arange(10))
        optimizer._success_tolerance = 2
        optimizer.tell(pool[10], [10])
        self.assertEqual(optimizer._success_counter, 1)
        optimizer.tell(pool[11], [11])
        self.assertEqual(optimizer._length, 1.6)
        optimizer._failure_tolerance = 1
        optimizer._length = optimizer._length_min
        optimizer.tell(pool[12], [0])
        self.assertEqual(optimizer._length, optimizer._length_init)
        self.assertEqual(optimizer._phase, "evolve")
        self.assertEqual(len(optimizer.X_), 13)
        self.assertEqual(len(optimizer._local_y), 13)
        self.assertEqual(optimizer.best()[1], 11)

    def test_real_backend_fits_observed_labels_and_switches_warm(self):
        pool = self.pool(rows=15)
        gpbo = GPBO(candidates=pool, direction="max", seed=3)
        turbo = TuRBO(candidates=pool, direction="max", seed=3)
        y = np.linspace(-3, 1, 10)
        for optimizer in (gpbo, turbo):
            optimizer.tell(pool[:10], y)
        available = list(range(10, 15))
        fits = []
        original_gp_fit = gpbo._fit_model
        original_turbo_fit = turbo._fit_pool_model

        def record_fit(optimizer, original):
            fits.append((type(optimizer).__name__, len(optimizer.y_)))
            model = original()
            untransformed = model.outcome_transform.untransform(
                model.train_targets.unsqueeze(-1))[0].squeeze(-1)
            np.testing.assert_allclose(untransformed.detach().numpy(), optimizer.y_)
            self.assertTrue(torch.isfinite(model.train_inputs[0]).all())
            return model

        with patch.object(gpbo, "_fit_model", side_effect=lambda: record_fit(gpbo, original_gp_fit)), patch.object(
                turbo, "_fit_pool_model", side_effect=lambda: record_fit(turbo, original_turbo_fit)):
            selected = []
            for step, optimizer in enumerate((gpbo, turbo, gpbo)):
                index = optimizer.ask_pool(np.array(available))
                self.assertIn(index, available)
                self.assertNotIn(index, selected)
                selected.append(index)
                available.remove(index)
                for recipient in (gpbo, turbo):
                    recipient.tell(pool[index], [2 + step])
                    self.assertEqual(len(recipient.y_), 11 + step)
                    np.testing.assert_array_equal(recipient.y_[:10], y)
                    self.assertEqual(recipient.best()[1], 2 + step)
        self.assertEqual(fits, [("GPBO", 10), ("TuRBO", 11), ("GPBO", 12)])

    def test_pool_preconditions_and_pending_proposal(self):
        pool = self.pool()
        for cls in (GPBO, TuRBO):
            with self.subTest(optimizer=cls.__name__):
                optimizer = cls(candidates=pool, direction="max")
                with self.assertRaises(OptimizerExhausted):
                    optimizer.ask_pool([])
                with self.assertRaisesRegex(RuntimeError, "n_init"):
                    optimizer.ask_pool([10])
                optimizer.tell(pool[:10], np.arange(10))
                for invalid in ([10.5], [10, 10], [-1], [len(pool)], [[10]]):
                    with self.assertRaises(ValueError):
                        optimizer.ask_pool(invalid)
                with self.assertRaises(ValueError):
                    optimizer.tell(pool[10], [1, 2])
                fit_name = "_fit_model" if cls is GPBO else "_fit_pool_model"
                score_name = ("src.optimizers.pool_gp_bo.LogExpectedImprovement" if cls is GPBO
                              else "src.optimizers.pool_turbo.MaxPosteriorSampling")
                scorer = (lambda **kwargs: lambda X: X[:, 0, 0]) if cls is GPBO else self.last_row_sampler
                with patch.object(optimizer, fit_name, return_value=self.fake_model(3)), patch(
                        score_name, side_effect=scorer):
                    optimizer.ask_pool([10])
                    with self.assertRaisesRegex(RuntimeError, "tell"):
                        optimizer.ask_pool([11])
                with self.assertRaisesRegex(ValueError, "last proposed"):
                    optimizer.tell(pool[11], [2])
                optimizer.tell(pool[10], [3])
                self.assertIsNone(optimizer._pending_pool_index)

    def test_continuous_default_still_minimizes(self):
        for cls in (GPBO, TuRBO):
            optimizer = cls(bounds=[(-1, 1)] * 2, seed=1)
            self.assertEqual(optimizer.direction, "min")
            X = optimizer.ask()
            self.assertEqual(X.shape, (1, 2))
            optimizer.tell(X, [3])
            optimizer.tell([[0, 0]], [-2])
            self.assertEqual(optimizer.best()[1], -2)
            with self.assertRaisesRegex(RuntimeError, "requires candidates"):
                optimizer.ask_pool([0])

    def test_pool_initial_observation_requirement_is_configurable(self):
        pool = self.pool()
        for cls in (GPBO, TuRBO):
            optimizer = cls(candidates=pool, n_init=2, seed=0)
            optimizer.tell(pool[:2], [3., 1.])
            np.testing.assert_array_equal(optimizer._pool_indices([2, 3]), [2, 3])
            self.assertEqual(optimizer.best()[1], 1.)


if __name__ == "__main__":
    unittest.main()
