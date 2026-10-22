"""Integrity tests use a fake objective; real campaign budgets remain untouched."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import uuid
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.blind_runner.evaluator import Evaluator

class FakeTask:
    dim = 10
    bounds = [[-5., 5.]] * 10
    def __init__(self):
        self.calls = 0
    def evaluate(self, x):
        self.calls += 1
        return SimpleNamespace(y=sum(v*v for v in x), elapsed=0.)

class Integrity(unittest.TestCase):
    def test_crash_keeps_reserved_charge(self):
        class Crash(BaseException):
            pass
        class CrashingTask:
            dim = 10
            bounds = [[-5., 5.]] * 10
            def evaluate(self, x):
                raise Crash()
        with tempfile.TemporaryDirectory() as tmp:
            suite = {'budget': 2, 'campaigns': [{'id': 'a'}]}
            path = Path(tmp) / 'state.sqlite'
            e = Evaluator(suite, path, lambda c: CrashingTask())
            e.handle('a', {'request_id': uuid.uuid4().hex, 'op': 'label', 'name': 'test',
                'documents': {'roadmap_initial.md': 'First block', 'changelog.md': ''}})
            request = {'request_id': uuid.uuid4().hex, 'op': 'evaluate', 'x': [0.] * 10}
            with self.assertRaises(Crash):
                e.handle('a', request)
            e.db.close()
            e = Evaluator(suite, path, lambda c: FakeTask())
            self.assertEqual(e.count('a'), 1)
            with self.assertRaisesRegex(ValueError, 'unknown outcome'):
                e.handle('a', request)
            reply = e.handle('a', {**request, 'request_id': uuid.uuid4().hex})
            self.assertFalse(reply['ok'])
            self.assertEqual(e.tasks['a'].calls, 0)
            e.db.close()

    def test_budget_idempotence_documents_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = {'budget': 2, 'campaigns': [{'id': 'a'}, {'id': 'b'}]}
            tasks = {}
            def factory(c):
                tasks[c['id']] = FakeTask()
                return tasks[c['id']]
            path = Path(tmp) / 'state.sqlite'
            e = Evaluator(suite, path, factory)
            def call(op, **kwargs):
                return e.handle('a', {'request_id': uuid.uuid4().hex, 'op': op, **kwargs})
            self.assertFalse(call('evaluate', x=[0.]*10)['ok'])
            self.assertEqual(e.count('a'), 0)
            docs = {'roadmap_initial.md': 'First block only', 'changelog.md': ''}
            self.assertTrue(call('label', name='test', documents=docs)['ok'])
            request = {'request_id': uuid.uuid4().hex, 'op': 'evaluate', 'x': [1.]*10}
            first = e.handle('a', request)
            self.assertEqual(first['result'], 10.)
            self.assertEqual(e.handle('a', request), first)
            self.assertEqual(tasks['a'].calls, 1)
            self.assertEqual(e.count('b'), 0)
            self.assertFalse(call('save', documents={**docs, 'roadmap_initial.md': 'rewritten'})['ok'])
            self.assertFalse(call('evaluate', x=[6.]*10)['ok'])
            self.assertEqual(e.count('a'), 1)
            self.assertTrue(call('evaluate', x=[0.]*10)['ok'])
            self.assertEqual(call('evaluate', x=[0.]*10)['error_type'], 'BudgetExhausted')
            e.db.close()
            e = Evaluator(suite, path, factory)
            self.assertEqual(e.count('a'), 2)
            self.assertEqual(e.handle('a', request), first)
            self.assertEqual(call('evaluate', x=[0.]*10)['error_type'], 'BudgetExhausted')
            self.assertEqual(tasks['a'].calls, 0)
            self.assertNotIn('f_opt', json.dumps(e.artifacts('a')))
            e.db.close()

    def test_budget_cannot_change_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = {'budget': 2, 'campaigns': [{'id': 'a'}]}
            path = Path(tmp) / 'state.sqlite'
            evaluator = Evaluator(suite, path, lambda _: FakeTask())
            evaluator.db.close()
            with self.assertRaisesRegex(ValueError, 'Cannot change'):
                Evaluator({**suite, 'budget': 3}, path, lambda _: FakeTask())

    def test_backend_exception_does_not_reveal_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            class BrokenTask(FakeTask):
                def evaluate(self, x):
                    raise ValueError('SECRET_OBJECTIVE_NAME and private optimum')
            e = Evaluator({'budget': 2, 'campaigns': [{'id': 'a'}]},
                          Path(tmp) / 'state.sqlite', lambda _: BrokenTask())
            e.handle('a', {'request_id': uuid.uuid4().hex, 'op': 'label', 'name': 'test',
                'documents': {'roadmap_initial.md': 'First block', 'changelog.md': ''}})
            reply = e.handle('a', {'request_id': uuid.uuid4().hex, 'op': 'evaluate', 'x': [0.]*10})
            self.assertEqual(reply['error_type'], 'EvaluatorError')
            self.assertNotIn('SECRET_OBJECTIVE_NAME', json.dumps(reply))
            self.assertIn('SECRET_OBJECTIVE_NAME', e.error_log.read_text())
            self.assertEqual(e.count('a'), 1)
            e.db.close()

    def test_dimension_bounds_and_append_only_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            class SmallTask(FakeTask):
                dim = 2
                bounds = [[0., 1.], [-2., 2.]]
            e = Evaluator({'budget': 4, 'campaigns': [{'id': 'a'}]},
                          Path(tmp) / 'state.sqlite', lambda _: SmallTask())
            def call(op, **kwargs):
                return e.handle('a', {'request_id': uuid.uuid4().hex, 'op': op, **kwargs})
            docs = {'roadmap_initial.md': 'First block only', 'changelog.md': ''}
            self.assertEqual(call('status')['result']['bounds'], SmallTask.bounds)
            self.assertTrue(call('label', name='random', documents=docs)['ok'])
            self.assertFalse(call('evaluate', x=[-1., 0.])['ok'])
            self.assertFalse(call('evaluate', x=[True, 0.])['ok'])
            self.assertFalse(call('evaluate', x=[0., 0., 0.])['ok'])
            self.assertTrue(call('evaluate', x=[.5, 0.])['ok'])
            changed = {**docs, 'changelog.md': 'Decision at evaluation 1'}
            self.assertTrue(call('save', documents=changed)['ok'])
            self.assertFalse(call('save', documents=docs)['ok'])
            snapshot = e.db.execute("SELECT eval_idx FROM snapshots WHERE name='changelog.md'").fetchone()
            self.assertEqual(snapshot[0], 1)
            self.assertEqual(e.count('a'), 1)
            e.db.close()

if __name__ == '__main__':
    unittest.main()
