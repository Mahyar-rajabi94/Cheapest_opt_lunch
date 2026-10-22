"""Persistent campaign client: execute one explicitly supplied, bounded block."""
import json
import sys
import traceback
from pathlib import Path
import numpy as np
from src.agent_harness.harness import AgentTaskHarness

h = AgentTaskHarness(budget=100, outdir="results/agent_run")
assert h.dim == 11 and h.direction == "min" and h.evaluations_used == 0
assert np.array_equal(np.asarray(h.bounds), np.tile([0., 1.], (11, 1)))
observations = []
active_method = None
block_end = 0
initial_record = Path("results/agent_run/roadmap_initial.md").read_bytes()
assert initial_record


def method(name):
    global active_method
    h.set_active_optimizer(name)
    active_method = name


def evaluate(x):
    """Only route to the public evaluator; enforce both call limits locally."""
    assert active_method is not None
    assert h.evaluations_used < block_end <= 100
    assert h.evaluations_remaining > 0
    assert Path("results/agent_run/roadmap_initial.md").read_bytes() == initial_record
    x = np.asarray(x, dtype=float)
    assert x.shape == (11,) and np.isfinite(x).all()
    assert (x >= 0).all() and (x <= 1).all()
    y = float(h.evaluate(x.tolist()))
    record = dict(n=int(h.evaluations_used), method=active_method, x=x.tolist(), y=y)
    observations.append(record)
    print(json.dumps(record), flush=True)
    if not np.isfinite(y):
        raise RuntimeError("Non-finite objective result: stop without retry")
    return y


def checkpoint(start):
    h.save()
    all_y = np.array([r['y'] for r in observations])
    recent = all_y[start:]
    best = observations[int(np.argmin(all_y))] if len(all_y) else None
    print(json.dumps(dict(checkpoint=True, used=int(h.evaluations_used),
          remaining=int(h.evaluations_remaining), best=best,
          block_count=len(recent), block_min=float(recent.min()) if len(recent) else None,
          block_median=float(np.median(recent)) if len(recent) else None,
          block_max=float(recent.max()) if len(recent) else None)), flush=True)


print(json.dumps(dict(ready=True, dim=h.dim, budget=h.budget, seed=h.seed,
                     noisy=h.noisy, used=h.evaluations_used)), flush=True)
for line in sys.stdin:
    try:
        command = json.loads(line)
        if command['action'] == 'run':
            assert int(command['expected']) == h.evaluations_used
            limit = int(command['limit'])
            assert 0 < limit <= h.evaluations_remaining
            block_end = h.evaluations_used + limit
            if h.evaluations_used:
                log = Path('results/agent_run/changelog.md').read_text()
                assert f'## Decision after evaluation {h.evaluations_used}\n' in log
            start = len(observations)
            source = Path(command['script'])
            assert source.parent == Path('work')
            exec(compile(source.read_text(), str(source), 'exec'), globals())
            checkpoint(start)
        elif command['action'] == 'inspect':
            block_end = h.evaluations_used
            source = Path(command['script'])
            assert source.parent == Path('work')
            exec(compile(source.read_text(), str(source), 'exec'), globals())
        elif command['action'] == 'finish':
            h.save()
            print(json.dumps(dict(finished=True, used=int(h.evaluations_used),
                  files=sorted(p.name for p in Path('results/agent_run').iterdir()))), flush=True)
            break
        else:
            raise ValueError('Unknown command')
    except Exception:
        traceback.print_exc()
        print('STOPPED: no evaluation will be retried', flush=True)
        break
