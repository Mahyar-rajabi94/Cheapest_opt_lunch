"""Persistent campaign workspace: explicit, bounded blocks supplied over stdin.

The single harness owns evaluation and budget accounting. This adapter only
checks each block's limit, keeps the observed data in memory, and prints results.
It does not schedule any subsequent optimization block.
"""
import json
import sys
import traceback
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.cwd()))
from src.agent_harness.harness import AgentTaskHarness
from src.optimizers.sobol_search import SobolSearch

h = AgentTaskHarness(budget=100, outdir='results/agent_run')
X, Y, records = [], [], []
block_start = block_end = int(h.evaluations_used)
rng = np.random.default_rng(1)

class EvaluationFailure(BaseException):
    pass


def begin_block(name, calls):
    global block_start, block_end
    calls = int(calls)
    if calls <= 0 or calls > h.evaluations_remaining:
        raise ValueError('Invalid block budget')
    block_start = int(h.evaluations_used)
    block_end = block_start + calls
    h.set_active_optimizer(name)
    print(json.dumps({'block': name, 'start': block_start, 'limit': block_end}), flush=True)


def evaluate(x, note=''):
    if h.evaluations_used >= block_end or h.evaluations_remaining <= 0:
        raise RuntimeError('Block or campaign evaluation limit reached')
    x = np.asarray(x, dtype=float)
    bounds = np.asarray(h.bounds)
    if x.shape != (h.dim,) or not np.isfinite(x).all() or np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1]):
        raise ValueError('Invalid candidate')
    try:
        y = float(h.evaluate(x.tolist()))
    except Exception as exc:
        print(json.dumps({'fatal_evaluator_error': repr(exc)}), flush=True)
        raise EvaluationFailure from exc
    X.append(x.copy())
    Y.append(y)
    row = {'eval': int(h.evaluations_used), 'x': x.tolist(), 'y': y, 'best': float(min(Y)), 'note': note}
    records.append(row)
    print(json.dumps(row), flush=True)
    if not np.isfinite(y):
        raise EvaluationFailure('Non-finite objective; stop and review')
    return y


def checkpoint():
    h.save()
    i = int(np.argmin(Y)) if Y else None
    print(json.dumps({'checkpoint': int(h.evaluations_used), 'remaining': int(h.evaluations_remaining), 'best_y': Y[i] if i is not None else None, 'best_x': X[i].tolist() if i is not None else None}), flush=True)


print(json.dumps({'ready': True, 'dim': h.dim, 'bounds': h.bounds, 'direction': h.direction, 'noisy': h.noisy, 'used': int(h.evaluations_used)}), flush=True)
for line in sys.stdin:
    try:
        command = json.loads(line)
        if command.get('quit'):
            h.save()
            break
        exec(compile(command['code'], '<campaign-block>', 'exec'), globals())
        print(json.dumps({'ready': True, 'used': int(h.evaluations_used)}), flush=True)
    except EvaluationFailure:
        traceback.print_exc()
        h.save()
        break
    except Exception:
        traceback.print_exc()
        print(json.dumps({'paused_after_script_error': True, 'used': int(h.evaluations_used)}), flush=True)
