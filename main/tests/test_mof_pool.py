"""Exact-pool integrity uses synthetic labels, duplicate features and tied optima."""
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

from src.agent_harness import harness as client
from src.blind_runner.evaluator import Evaluator
from src.blind_runner.runner import Bridge
from src.tasks.mof_pool import MOFPoolTask
from scripts.run_campaign import optimizers_for_task, run_campaign


class MOFPoolIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.csv = self.root / 'private.csv'
        self.csv.write_text('id,x0,x1,y\nSECRET_A,0,7,1\nSECRET_B,0,7,9\n'
                            'SECRET_C,1,7,9\nSECRET_D,2,7,4\n')

    def evaluator(self, budget=4):
        e = Evaluator({'budget': budget, 'seed': 2, 'campaigns': [{'id': 'run'}]},
                      self.root / 'journal.sqlite', lambda _: MOFPoolTask(self.csv))
        return e

    def call(self, e, op, **kwargs):
        return e.handle('run', {'request_id': uuid.uuid4().hex, 'op': op, **kwargs})

    def initialize(self, e):
        self.assertTrue(self.call(e, 'label', name='test', documents={
            'roadmap_initial.md': 'Select two rows', 'changelog.md': ''})['ok'])

    def test_exact_identity_and_tied_rank_conventions(self):
        task = MOFPoolTask(self.csv)
        self.assertEqual(task.evaluate_index(1).y, -9.)
        self.assertEqual(task.evaluate_index(0).y, -1.)
        with self.assertRaisesRegex(ValueError, 'already'):
            task.evaluate_index(1)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            task.evaluate([.5, 7.])
        self.assertEqual(task.offline_observation(1)['descending_rank'], 1)
        self.assertEqual(task.offline_observation(2)['descending_rank'], 1)
        self.assertEqual(task.offline_observation(1)['rank'], 3)
        self.assertEqual(task.f_opt, -9.)

    def test_retries_duplicates_invalid_rows_and_restart(self):
        e = self.evaluator()
        self.initialize(e)
        request = {'request_id': uuid.uuid4().hex, 'op': 'evaluate', 'candidate_index': 1}
        first = e.handle('run', request)
        self.assertEqual(first['result'], -9.)
        self.assertEqual(e.handle('run', request), first)
        for index in (1, True, -1, 4, 1.5, '2'):
            self.assertFalse(self.call(e, 'evaluate', candidate_index=index)['ok'])
        self.assertFalse(self.call(e, 'evaluate', x=[0., 7.])['ok'])
        self.assertEqual(e.count('run'), 1)
        e.db.close()
        e = self.evaluator()
        self.addCleanup(e.db.close)
        self.assertEqual(e.status('run')['available_indices'], [0, 2, 3])
        self.assertEqual(e.handle('run', request), first)
        self.assertFalse(self.call(e, 'evaluate', candidate_index=1)['ok'])
        self.assertEqual(self.call(e, 'evaluate', candidate_index=0)['result'], -1.)
        self.assertEqual(self.call(e, 'evaluate', candidate_index=2)['result'], -9.)
        meta = json.loads(e.artifacts('run')['meta.json'])
        self.assertEqual(meta['best_candidate_index'], 1)  # Keep first observed tie.
        self.assertEqual(meta['seed'], 2)
        public = json.dumps([e.status('run'), e.artifacts('run')])
        for hidden in ('SECRET_', 'f_opt', 'raw_y', 'rank', 'private.csv'):
            self.assertNotIn(hidden, public)

    def test_generic_transport_client_and_prospective_logs(self):
        workspace = self.root / 'workspace'
        for path in ('.bridge/requests', '.bridge/responses', 'results/agent_run'):
            (workspace / path).mkdir(parents=True)
        results = workspace / 'results/agent_run'
        (results / 'roadmap_initial.md').write_text('First block: two candidates')
        (results / 'changelog.md').write_text('')
        task = MOFPoolTask(self.csv)
        np.save(workspace / 'candidates.npy', task.candidates, allow_pickle=False)
        e = self.evaluator(budget=2)
        bridge = Bridge(e, workspace)
        bridge.start()
        try:
            with patch.object(client, 'ROOT', workspace):
                h = client.AgentTaskHarness(budget=2)
                self.assertEqual(h.seed, 2)
                self.assertFalse(h.candidates.flags.writeable)
                np.testing.assert_array_equal(h.candidates, task.candidates)
                h.set_active_optimizer('selected_rows')
                self.assertEqual(h.evaluate_index(np.int64(1)), -9.)
                reconnect = client.AgentTaskHarness(budget=2)
                self.assertEqual(reconnect.evaluations_used, 1)
                self.assertNotIn(1, reconnect.available_indices)
                with self.assertRaisesRegex(RuntimeError, 'already'):
                    reconnect.evaluate_index(1)
                (results / 'changelog.md').write_text('Decision at 1: one more row')
                self.assertEqual(reconnect.evaluate_index(0), -1.)
                with self.assertRaises(client.BudgetExhausted):
                    reconnect.evaluate_index(2)
                reconnect.save()
                rows = list(csv.DictReader(io.StringIO((results / 'trace.csv').read_text())))
                self.assertEqual([row['candidate_index'] for row in rows], ['1', '0'])
            snapshots = e.db.execute("SELECT eval_idx FROM snapshots WHERE name='changelog.md'").fetchall()
            self.assertEqual(snapshots, [(1,)])
        finally:
            bridge.close()
            e.db.close()

    def test_baseline_dispatch_preserves_exact_rows_and_minimization(self):
        factory = lambda: MOFPoolTask(self.csv)
        self.assertEqual(set(optimizers_for_task(factory())), {'gp_bo', 'turbo', 'pool_random'})
        with patch.dict('scripts.run_campaign.TASKS', {'synthetic': factory}):
            out = run_campaign('synthetic', 'pool_random', 0, 4, self.root / 'results')
            with (out / 'trace.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len({r['candidate_index'] for r in rows}), 4)
            self.assertEqual(float(rows[-1]['y_best_so_far']), -9.)


if __name__ == '__main__':
    unittest.main()
