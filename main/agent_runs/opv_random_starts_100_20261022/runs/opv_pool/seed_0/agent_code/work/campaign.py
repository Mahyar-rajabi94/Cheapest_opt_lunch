"""Persistent, explicitly commanded optimization blocks; no automatic schedule."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from src.agent_harness.harness import AgentTaskHarness

h = AgentTaskHarness(budget=100, outdir='results/agent_run')
X, Y = [], []
step_stop = 0

def evaluate(x):
    """The only objective-call site, guarded by both step and campaign caps."""
    if h.evaluations_used >= step_stop or h.evaluations_remaining <= 0:
        raise RuntimeError('Objective call would exceed the authorized block')
    x = np.asarray(x, dtype=float)
    bounds = np.asarray(h.bounds, dtype=float)
    if x.shape != (h.dim,) or not np.isfinite(x).all():
        raise ValueError('Invalid point shape or nonfinite coordinate')
    if np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1]):
        raise ValueError('Point outside public bounds')
    y = float(h.evaluate(x.tolist()))
    X.append(x.copy())
    Y.append(y)
    print(json.dumps({'eval': h.evaluations_used, 'y': y, 'best': min(Y), 'x': x.tolist()}), flush=True)
    if not np.isfinite(y):
        raise RuntimeError('Nonfinite objective; stop without retry')
    return y

def summary():
    i = int(np.argmin(Y)) if Y else None
    print(json.dumps({'checkpoint': h.evaluations_used, 'remaining': h.evaluations_remaining,
                      'best_y': Y[i] if i is not None else None,
                      'best_x': X[i].tolist() if i is not None else None}), flush=True)

print(json.dumps({'ready': True, 'dim': h.dim, 'budget': h.budget, 'seed': h.seed,
                  'used': h.evaluations_used}), flush=True)
for line in sys.stdin:
    try:
        command = json.loads(line)
        if command['action'] == 'block':
            limit = int(command['limit'])
            if not 0 < limit <= h.evaluations_remaining:
                raise ValueError('Block exceeds available budget')
            step_stop = h.evaluations_used + limit
            h.set_active_optimizer(command['label'])
            script = Path(command['file'])
            exec(compile(script.read_text(), str(script), 'exec'), globals())
            h.save()
            summary()
        elif command['action'] == 'inspect':
            exec(command['code'], globals())
        elif command['action'] == 'finish':
            h.save()
            summary()
            print(json.dumps({'files': sorted(p.name for p in Path('results/agent_run').iterdir())}), flush=True)
            break
        else:
            raise ValueError('Unknown command')
    except Exception:
        traceback.print_exc()
        print('STOPPED: error; no evaluation retried.', flush=True)
        break
