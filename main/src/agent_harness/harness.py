"""Public transport client. No objective implementation or task identity is here."""
import json
import math
import os
from pathlib import Path
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]

class BudgetExhausted(Exception):
    pass

class AgentTaskHarness:
    def __init__(self, budget=None, outdir='results/agent_run'):
        if str(outdir) != 'results/agent_run':
            raise ValueError('Use outdir="results/agent_run".')
        self._previous_docs = {}
        state = self._request('status')
        if budget is not None and (type(budget) is not int or budget != state['budget']):
            raise ValueError('Budget must match the public campaign budget.')
        for k in ['dim', 'bounds', 'direction', 'noisy', 'budget']:
            setattr(self, k, state[k])

    def _request(self, op, **kwargs):
        request_id = uuid.uuid4().hex
        data = {'request_id': request_id, 'op': op, **kwargs}
        if op in ('evaluate', 'label', 'save'):
            docs = {name: (ROOT / 'results/agent_run' / name).read_text()
                    for name in ['roadmap_initial.md', 'changelog.md']}
            if docs != self._previous_docs:
                data['documents'] = docs
                self._previous_docs = docs
        request = ROOT / '.bridge/requests' / (request_id + '.json')
        tmp = request.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, allow_nan=False))
        os.replace(tmp, request)
        response = ROOT / '.bridge/responses' / request.name
        deadline = time.monotonic() + 180
        while not response.exists():
            if time.monotonic() > deadline:
                raise RuntimeError('Evaluator timed out; outcome unknown. Stop and report.')
            time.sleep(.005)
        data = json.loads(response.read_text())
        if not data['ok']:
            if data['error_type'] == 'BudgetExhausted':
                raise BudgetExhausted(data['error'])
            raise RuntimeError(data['error'])
        return data['result']

    def set_active_optimizer(self, name):
        return self._request('label', name=str(name))

    def evaluate(self, x):
        x = [float(v) for v in x]
        if len(x) != self.dim or any(not math.isfinite(v) or not lo <= v <= hi for v, (lo, hi) in zip(x, self.bounds)):
            raise ValueError('Expected finite coordinates within the public bounds.')
        return self._request('evaluate', x=x)

    @property
    def evaluations_used(self):
        return self._request('status')['evaluations_used']

    @property
    def evaluations_remaining(self):
        return self._request('status')['evaluations_remaining']

    def save(self):
        return self._request('save')
