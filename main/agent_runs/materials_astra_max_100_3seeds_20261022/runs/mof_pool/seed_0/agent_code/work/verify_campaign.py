from pathlib import Path
import csv,json,numpy as np
out=Path('results/agent_run')
expected={'roadmap_initial.md','changelog.md','trace.csv','meta.json'}
actual={p.name for p in out.iterdir()}
assert actual==expected,(actual,expected)
obs=json.loads(Path('work/observations.json').read_text())
assert len(obs['indices'])==len(set(obs['indices']))==len(obs['values'])==100
assert np.isfinite(obs['values']).all()
with (out/'trace.csv').open(newline='') as f:
    reader=csv.DictReader(f);rows=list(reader);columns=reader.fieldnames
assert len(rows)==100,len(rows)
ykey=next((k for k in ('y','value','objective','objective_value') if k in columns),None)
assert ykey is not None,columns
trace_y=np.array([float(row[ykey]) for row in rows])
assert np.allclose(trace_y,obs['values'],rtol=0,atol=1e-12)
indexkey=next((k for k in ('candidate_index','pool_index','row_index') if k in columns),None)
if indexkey is not None:
    assert [int(row[indexkey]) for row in rows]==obs['indices']
best=int(np.argmin(trace_y))
assert obs['indices'][best]==561 and abs(trace_y[best]-(-9.2928301557))<1e-12
meta=json.loads((out/'meta.json').read_text())
assert isinstance(meta,dict)
log=(out/'changelog.md').read_text()
checkpoints=[12,20,27,35,45,57,67,75,81,88,94,97,100]
assert [int(s.split('\n')[0]) for s in log.split('## Decision after evaluation ')[1:]]==checkpoints
print(json.dumps({'verified':True,'files':sorted(actual),'trace_rows':len(rows),'unique_purchases':len(set(obs['indices'])),
                  'trace_values_match_real_observations':True,'trace_index_column':indexkey,'best_index':obs['indices'][best],
                  'best_y':float(trace_y[best]),'best_evaluation':best+1,'decision_checkpoints':checkpoints,
                  'file_sizes':{p.name:p.stat().st_size for p in out.iterdir()}},indent=2))
