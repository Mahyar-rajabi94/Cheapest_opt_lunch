"""Pool selection and runner integration; each test constructs a fresh pool."""
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

import numpy as np

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN / 'scripts'))

from src.blind_runner.evaluator import Evaluator
from src.optimizers.pool_random import PoolRandomSearch
from src.tasks.registry import TASKS
from run_campaign import run_campaign
from run_sweep import _sweep


class OPVPoolIntegrity(unittest.TestCase):
    def test_repeated_query_exhausts_pool_without_replacement(self):
        task = TASKS['opv_pool']()
        ids, values = [], []
        for _ in range(task.n_candidates):
            values.append(task.evaluate(np.full(task.dim, .5)).y)
            ids.append(task.last_candidate)
        self.assertEqual(len(set(ids)), task.n_candidates)
        self.assertEqual(min(values), task.f_opt)
        with self.assertRaisesRegex(ValueError, 'all .* candidates'):
            task.evaluate(np.full(task.dim, .5))
        # A different campaign owns a fresh tested set.
        fresh = TASKS['opv_pool']()
        fresh.evaluate(np.full(fresh.dim, .5))
        self.assertEqual(fresh.last_candidate, ids[0])

    def test_pool_random_selects_each_intended_device(self):
        task = TASKS['opv_pool']()
        optimizer = PoolRandomSearch(task.bounds, seed=0, candidates=task.candidates)
        for point in optimizer.ask(task.n_candidates):
            row = np.flatnonzero(np.all(task.candidates == point, axis=1))
            self.assertEqual(len(row), 1)
            task.evaluate(point)
            self.assertEqual(task.last_candidate, int(task._ids[row[0]]))

    def test_transport_retry_does_not_consume_another_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = TASKS['opv_pool']()
            evaluator = Evaluator({'budget': 2, 'campaigns': [{'id': 'run'}]},
                                  Path(tmp) / 'journal.sqlite', lambda _: task)
            self.addCleanup(evaluator.db.close)
            evaluator.handle('run', {'request_id': uuid.uuid4().hex, 'op': 'label',
                'name': 'probe', 'documents': {'roadmap_initial.md': 'Two probes', 'changelog.md': ''}})
            request = {'request_id': uuid.uuid4().hex, 'op': 'evaluate', 'x': [.5] * task.dim}
            first = evaluator.handle('run', request)
            self.assertTrue(first['ok'])
            first_id = task.last_candidate
            self.assertEqual(evaluator.handle('run', request), first)
            self.assertEqual(task.last_candidate, first_id)
            self.assertEqual(evaluator.count('run'), 1)
            second = evaluator.handle('run', {**request, 'request_id': uuid.uuid4().hex})
            self.assertTrue(second['ok'])
            self.assertNotEqual(task.last_candidate, first_id)
            rows = list(csv.DictReader(io.StringIO(evaluator.artifacts('run')['trace.csv'])))
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[-1]['best_y']), min(first['result'], second['result']))
            public = json.loads(evaluator.artifacts('run')['meta.json'])
            for private_key in ('f_opt', 'candidates', 'last_candidate', 'task', 'task_name'):
                self.assertNotIn(private_key, public)

    def test_oversized_budget_rejected_before_sweep_jobs(self):
        limit = TASKS['opv_pool']().n_candidates
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'budget .* exceeds'):
                run_campaign('opv_pool', 'random_search', 0, limit + 1, tmp)
            with patch('run_sweep._timed_campaign') as launch:
                with self.assertRaisesRegex(ValueError, 'budget .* exceeds'):
                    _sweep('opv_pool', limit + 1, 10, tmp, workers=1)
                launch.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
