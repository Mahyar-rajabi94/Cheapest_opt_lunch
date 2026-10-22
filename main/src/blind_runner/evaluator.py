"""Private evaluator: durable budgets, idempotent requests and prospective decisions.

Only the trusted runner imports this module. Agents receive the transport client.
"""
import csv
import hashlib
import hmac
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
import traceback


class BudgetExhausted(Exception):
    pass

class Evaluator:
    def __init__(self, suite, database, task_factory):
        self.suite = suite
        self.lock = threading.Lock()
        self.tasks = {c['id']: task_factory(c) for c in suite['campaigns']}
        self.error_log = Path(database).parent / 'evaluator_errors.log'
        if isinstance(suite['budget'], bool) or not isinstance(suite['budget'], int) or suite['budget'] < 1:
            raise ValueError('Budget must be a positive integer.')
        self.db = sqlite3.connect(database, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS requests(campaign TEXT, id TEXT, digest TEXT, reply TEXT,
              PRIMARY KEY(campaign,id));
            CREATE TABLE IF NOT EXISTS evaluations(campaign TEXT, idx INTEGER, request_id TEXT,
              x TEXT, y REAL, label TEXT, elapsed REAL, wall_ns INTEGER,
              PRIMARY KEY(campaign,idx));
            CREATE TABLE IF NOT EXISTS labels(campaign TEXT PRIMARY KEY, label TEXT);
            CREATE TABLE IF NOT EXISTS documents(campaign TEXT, name TEXT, content TEXT,
              PRIMARY KEY(campaign,name));
            CREATE TABLE IF NOT EXISTS snapshots(campaign TEXT, eval_idx INTEGER, name TEXT,
              content TEXT, wall_ns INTEGER);
        ''')
        self.db.execute('CREATE TABLE IF NOT EXISTS configuration(value TEXT NOT NULL)')
        config = json.dumps(suite, sort_keys=True)
        previous = self.db.execute('SELECT value FROM configuration').fetchone()
        if previous and previous[0] != config:
            self.db.close()
            raise ValueError('Cannot change a saved campaign configuration or budget.')
        if not previous:
            self.db.execute('INSERT INTO configuration VALUES(?)', (config,))
        self.db.commit()

    def count(self, campaign):
        return self.db.execute('SELECT COUNT(*) FROM evaluations WHERE campaign=?', (campaign,)).fetchone()[0]

    def status(self, campaign):
        used = self.count(campaign)
        task = self.tasks[campaign]
        return {'dim': int(task.dim), 'bounds': [[float(v) for v in pair] for pair in task.bounds],
                'direction': 'min', 'noisy': False,
                'budget': self.suite['budget'], 'evaluations_used': used,
                'evaluations_remaining': self.suite['budget'] - used}

    def documents(self, campaign, docs):
        if not isinstance(docs, dict) or set(docs) != {'roadmap_initial.md', 'changelog.md'}:
            raise ValueError('Invalid decision documents.')
        for name, content in docs.items():
            if not isinstance(content, str) or len(content.encode()) > 200000:
                raise ValueError('Invalid decision document size.')
            old = self.db.execute('SELECT content FROM documents WHERE campaign=? AND name=?',
                                  (campaign, name)).fetchone()
            old = old[0] if old else ''
            if name == 'roadmap_initial.md' and old and content != old:
                raise ValueError('The initial decision is immutable.')
            if name == 'changelog.md' and not content.startswith(old):
                raise ValueError('The changelog must be append-only.')
            if content != old:
                self.db.execute('INSERT OR REPLACE INTO documents VALUES(?,?,?)', (campaign, name, content))
                self.db.execute('INSERT INTO snapshots VALUES(?,?,?,?,?)',
                                (campaign, self.count(campaign), name, content, time.time_ns()))

    def artifacts(self, campaign):
        rows = self.db.execute('SELECT idx,x,y,label,elapsed,wall_ns FROM evaluations '
                               'WHERE campaign=? AND y IS NOT NULL ORDER BY idx', (campaign,)).fetchall()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['eval_idx', 'x', 'y', 'best_y', 'optimizer', 'eval_seconds', 'wall_ns'])
        best, best_x = math.inf, None
        for idx, x, y, label, elapsed, wall_ns in rows:
            if y < best:
                best, best_x = y, json.loads(x)
            writer.writerow([idx, x, y, best, label, elapsed, wall_ns])
        meta = {**self.status(campaign), 'best_y': best if rows else None, 'best_x': best_x}
        artifacts = {'trace.csv': output.getvalue(), 'meta.json': json.dumps(meta, indent=2)}
        return artifacts

    def handle(self, campaign, request):
        with self.lock:
            return self._handle(campaign, request)

    def _handle(self, campaign, request):
        if campaign not in self.tasks or not isinstance(request, dict):
            raise ValueError('Invalid campaign or request.')
        key = request.get('request_id')
        if not isinstance(key, str) or not re.fullmatch('[0-9a-f]{32}', key):
            raise ValueError('Invalid request identifier.')
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        old = self.db.execute('SELECT digest,reply FROM requests WHERE campaign=? AND id=?',
                              (campaign, key)).fetchone()
        if old:
            if not hmac.compare_digest(old[0], digest):
                raise ValueError('Request identifier was reused with different content.')
            if old[1] is None:
                raise ValueError('A previous request has an unknown outcome; stop and report.')
            return json.loads(old[1])
        self.db.execute('INSERT INTO requests VALUES(?,?,?,NULL)', (campaign, key, digest))
        self.db.commit()
        try:
            if 'documents' in request:
                self.documents(campaign, request['documents'])
            op = request.get('op')
            artifacts = None
            if op == 'status':
                result = self.status(campaign)
            elif op == 'label':
                label = request.get('name')
                if not isinstance(label, str) or not re.fullmatch(r'[A-Za-z0-9_.+ -]{1,100}', label):
                    raise ValueError('Invalid optimizer label.')
                self.db.execute('INSERT OR REPLACE INTO labels VALUES(?,?)', (campaign, label))
                result = None
            elif op == 'evaluate':
                n = self.count(campaign)
                if n >= self.suite['budget']:
                    raise BudgetExhausted('Evaluation budget exhausted.')
                if self.db.execute('SELECT 1 FROM evaluations WHERE campaign=? AND y IS NULL', (campaign,)).fetchone():
                    raise ValueError('An evaluation has an unknown outcome; stop and report.')
                x = request.get('x')
                bounds = self.status(campaign)['bounds']
                if not isinstance(x, list) or len(x) != len(bounds) or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    or not lo <= v <= hi for v, (lo, hi) in zip(x, bounds)):
                    raise ValueError('Expected finite coordinates within the public bounds.')
                label = self.db.execute('SELECT label FROM labels WHERE campaign=?', (campaign,)).fetchone()
                initial = self.db.execute('SELECT content FROM documents WHERE campaign=? AND name=?',
                                          (campaign, 'roadmap_initial.md')).fetchone()
                if not label or not initial or not initial[0].strip():
                    raise ValueError('Record the first decision and optimizer label before evaluating.')
                # Reserve a charged slot durably before calling the objective. A crash cannot reset it.
                self.db.execute('INSERT INTO evaluations VALUES(?,?,?,?,NULL,?,NULL,?)',
                                (campaign, n + 1, key, json.dumps(x), label[0], time.time_ns()))
                self.db.commit()
                try:
                    ev = self.tasks[campaign].evaluate(x)
                except Exception as exc:
                    # Backend exceptions may identify the objective. Log the
                    # chained exception privately and expose only a generic error.
                    raise RuntimeError('Objective evaluation failed.') from exc
                result = float(ev.y)
                if not math.isfinite(result):
                    raise RuntimeError('Nonfinite backend result')
                self.db.execute('UPDATE evaluations SET y=?,elapsed=? WHERE campaign=? AND idx=?',
                                (result, ev.elapsed, campaign, n + 1))
                if n + 1 == self.suite['budget']:
                    artifacts = self.artifacts(campaign)
            elif op == 'save':
                artifacts = self.artifacts(campaign)
                result = None
            else:
                raise ValueError('Unsupported evaluator operation.')
            reply = {'ok': True, 'result': result}
            if artifacts is not None:
                reply['artifacts'] = artifacts
        except (ValueError, BudgetExhausted) as exc:
            reply = {'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)}
        except Exception:
            with self.error_log.open('a') as log:
                traceback.print_exc(file=log)
            reply = {'ok': False, 'error_type': 'EvaluatorError', 'error': 'Evaluator failed; stop and report.'}
        self.db.execute('UPDATE requests SET reply=? WHERE campaign=? AND id=?',
                        (json.dumps(reply, allow_nan=False), campaign, key))
        self.db.commit()
        return reply
