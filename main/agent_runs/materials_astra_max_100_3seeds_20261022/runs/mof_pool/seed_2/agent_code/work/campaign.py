import sys
import json
import traceback
import time
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import numpy as np
sys.path.insert(0, str(Path.cwd()))
from src.agent_harness.harness import AgentTaskHarness

# One live harness and observation store for all manually chosen blocks.
h = AgentTaskHarness(budget=100, outdir='results/agent_run')
X = np.asarray(h.candidates, dtype=float)
bounds = np.asarray(h.bounds, dtype=float)
active = np.ptp(X, axis=0) > 0
span = bounds[active, 1] - bounds[active, 0]
Z = (X[:, active] - bounds[active, 0]) / span
observed_indices, observed_y = [], []
step_end = 0
fatal_evaluator_error = False
rng = np.random.default_rng(h.seed)

def start_block(label, k):
    global step_end
    if fatal_evaluator_error:
        raise RuntimeError('Evaluator previously failed; campaign is halted')
    if not 0 < k <= h.evaluations_remaining:
        raise ValueError('Invalid block allowance')
    step_end = h.evaluations_used + k
    h.set_active_optimizer(label)

def buy(index, details=None):
    global fatal_evaluator_error
    if fatal_evaluator_error or h.evaluations_used >= step_end:
        raise RuntimeError('Block limit reached or evaluator halted')
    index = int(index)
    if index not in set(map(int, h.available_indices)):
        raise ValueError('Selection is not an available row')
    try:
        y = float(h.evaluate_index(index))
    except BaseException:
        fatal_evaluator_error = True
        raise
    observed_indices.append(index)
    observed_y.append(y)
    print(json.dumps({'evaluation': h.evaluations_used, 'index': index, 'y': y,
                      'best': min(observed_y), 'details': details}), flush=True)
    if not np.isfinite(y):
        fatal_evaluator_error = True
        raise RuntimeError('Nonfinite objective; campaign is halted')
    return y

def checkpoint():
    h.save()
    oi, oy = np.array(observed_indices), np.array(observed_y)
    top = np.argsort(oy)[:10]
    print(json.dumps({'checkpoint': h.evaluations_used,
                      'remaining': h.evaluations_remaining,
                      'top_observed': [{'index': int(oi[i]), 'y': float(oy[i])} for i in top],
                      'quantiles': np.quantile(oy, [0, .25, .5, .75, 1]).tolist() if len(oy) else []}), flush=True)

s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
cumvar = np.cumsum(s*s) / np.sum(s*s)
versions = {}
for pkg in ['numpy', 'scipy', 'torch', 'botorch', 'gpytorch', 'scikit-learn']:
    try:
        versions[pkg] = version(pkg)
    except PackageNotFoundError:
        versions[pkg] = None
print(json.dumps({'ready': True, 'shape': X.shape, 'active_features': int(active.sum()),
                  'unique_feature_rows': int(len(np.unique(X, axis=0))),
                  'pca_dims_90_95_99': [int(np.searchsorted(cumvar, q)+1) for q in [.9,.95,.99]],
                  'pca_first10_cumvar': cumvar[:10].tolist(), 'seed': h.seed,
                  'used': h.evaluations_used, 'versions': versions,
                  'existing_results': [p.name for p in Path('results/agent_run').iterdir()]}), flush=True)
for line in sys.stdin:
    try:
        command = json.loads(line)
        if command.get('quit'):
            break
        path = Path(command['file'])
        if path.parent != Path('work'):
            raise ValueError('Scripts belong in work/')
        exec(compile(path.read_text(), str(path), 'exec'), globals())
        print('READY_FOR_NEXT_DECISION', flush=True)
    except BaseException as exc:
        traceback.print_exc()
        print(json.dumps({'error': str(exc), 'fatal_evaluator_error': fatal_evaluator_error}), flush=True)
        if fatal_evaluator_error:
            break
