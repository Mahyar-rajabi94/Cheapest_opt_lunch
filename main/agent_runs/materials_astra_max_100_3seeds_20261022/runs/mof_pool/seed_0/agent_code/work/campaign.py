import sys, json, traceback
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
from src.agent_harness.harness import AgentTaskHarness

h = AgentTaskHarness(budget=100, outdir='results/agent_run')
assert h.evaluations_used == 0, 'Refusing to restart an existing campaign'
X = np.asarray(h.candidates, dtype=float)
bounds = np.asarray(h.bounds, dtype=float)
active = bounds[:, 1] > bounds[:, 0]
Z = (X[:, active] - bounds[active, 0]) / (bounds[active, 1] - bounds[active, 0])
rng = np.random.default_rng(0)
indices, values, diagnostics = [], [], []
fatal = False

def emit(data):
    print(json.dumps(data, allow_nan=False), flush=True)

def summary():
    yy = np.asarray(values)
    result = {'used':h.evaluations_used, 'remaining':h.evaluations_remaining}
    if len(yy):
        order = np.argsort(yy)
        result.update(best_index=indices[int(order[0])], best_y=float(yy[order[0]]),
                      quantiles=np.quantile(yy,[0,.25,.5,.75,1]).tolist(),
                      top=[{'index':indices[int(i)], 'y':values[int(i)]} for i in order[:10]])
    return result

def run_block(tag, limit, choose, observe=None, early_stop=None):
    global fatal
    if fatal:
        raise RuntimeError('Campaign stopped after evaluator error')
    assert isinstance(limit,int) and 0 < limit <= h.evaluations_remaining
    start = h.evaluations_used
    ceiling = start + limit
    h.set_active_optimizer(tag)
    for step in range(limit):
        assert h.evaluations_used < ceiling and h.evaluations_remaining > 0
        proposal = choose()
        idx, detail = proposal if isinstance(proposal,tuple) else (proposal,{})
        idx = int(idx)
        assert idx in set(map(int,h.available_indices))
        previous_best = min(values) if values else None
        try:
            y = float(h.evaluate_index(idx))
        except Exception:
            fatal = True
            raise
        indices.append(idx)
        values.append(y)
        diagnostics.append(detail)
        emit({'evaluation':h.evaluations_used,'index':idx,'y':y,'best':min(values),'prediction':detail})
        if not np.isfinite(y):
            fatal = True
            raise RuntimeError('Nonfinite objective; stop campaign')
        if observe is not None:
            observe(idx,y)
        if early_stop is not None and early_stop(idx,y,previous_best,detail):
            emit({'early_checkpoint':True})
            break
    h.save()
    Path('work/observations.json').write_text(json.dumps({'indices':indices,'values':values,'diagnostics':diagnostics},allow_nan=False))
    emit({'checkpoint':summary()})

emit({'ready':True,'dim':h.dim,'pool_shape':list(X.shape),'budget':h.budget,'seed':h.seed,'used':h.evaluations_used})
for line in sys.stdin:
    try:
        command=json.loads(line)
        if command.get('action')=='finish':
            h.save()
            emit({'finished':summary()})
            break
        if command.get('action')=='summary':
            emit(summary())
            continue
        path=Path(command['script']).resolve()
        assert path.is_relative_to(Path('work').resolve())
        exec(compile(path.read_text(),str(path),'exec'),globals())
    except Exception as exc:
        emit({'error':str(exc),'fatal_evaluator_error':fatal})
        traceback.print_exc()
        if fatal:
            break
