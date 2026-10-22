import json
import sys
import traceback
from pathlib import Path
import numpy as np
from src.agent_harness.harness import AgentTaskHarness

h = AgentTaskHarness(budget=100, outdir='results/agent_run')
C = np.asarray(h.candidates, dtype=float)
bounds = np.asarray(h.bounds, dtype=float)
active = bounds[:, 1] > bounds[:, 0]
U = (C[:, active] - bounds[active, 0]) / (bounds[active, 1] - bounds[active, 0])
observed_indices, observed_y = [], []
rng = np.random.default_rng(h.seed)
step_end = 0

def begin_block(k, label):
    global step_end
    if not 0 < k <= h.evaluations_remaining:
        raise ValueError('Invalid block allowance')
    step_end = h.evaluations_used + k
    h.set_active_optimizer(label)

def purchase(index):
    if h.evaluations_used >= step_end:
        raise RuntimeError('Block limit reached')
    index = int(index)
    if index not in set(h.available_indices):
        raise ValueError('Selection must be an unused row')
    try:
        value = float(h.evaluate_index(index))
    except Exception:
        traceback.print_exc()
        print('EVALUATOR_ERROR_STOP', flush=True)
        raise SystemExit(2)
    observed_indices.append(index)
    observed_y.append(value)
    print(json.dumps({'evaluation': h.evaluations_used, 'index': index, 'y': value,
                      'best': min(observed_y)}), flush=True)
    return value

def report():
    h.save()
    y = np.asarray(observed_y)
    data = {'used': h.evaluations_used, 'remaining': h.evaluations_remaining}
    if len(y):
        order = np.argsort(y)[:10]
        data.update(best_index=observed_indices[int(order[0])], best_y=float(y.min()),
                    y_quantiles=np.quantile(y, [0, .25, .5, .75, 1]).tolist(),
                    best_observations=[{'index':observed_indices[i], 'y':float(y[i])} for i in order])
    print('CHECKPOINT ' + json.dumps(data), flush=True)

singular = np.linalg.svd(U - U.mean(0), compute_uv=False)
variance = np.cumsum(singular**2) / np.sum(singular**2)
print('PUBLIC_GEOMETRY ' + json.dumps({'shape':C.shape, 'nonconstant_features':int(active.sum()),
      'distinct_feature_rows':int(len(np.unique(C, axis=0))),
      'pca_components_90_95_99':[int(np.searchsorted(variance, p)+1) for p in [.9,.95,.99]],
      'seed':int(h.seed), 'used':h.evaluations_used, 'remaining':h.evaluations_remaining}), flush=True)
print('READY', flush=True)
for line in sys.stdin:
    try:
        command = json.loads(line)
        if command['action'] == 'execfile':
            path = Path(command['path'])
            if path.parent != Path('work'):
                raise ValueError('Scripts must be in work/')
            exec(compile(path.read_text(), str(path), 'exec'), globals())
        elif command['action'] == 'finish':
            report()
            print('FINISHED', flush=True)
            break
        else:
            raise ValueError('Unknown command')
        print('READY', flush=True)
    except Exception:
        traceback.print_exc()
        print('SCRIPT_ERROR_NO_AUTOMATIC_RETRY', flush=True)
