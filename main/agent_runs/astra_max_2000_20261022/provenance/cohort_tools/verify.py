"""Independent offline checks of published scores; never evaluates an objective."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]

def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def same(actual, expected):
    if expected is None:
        assert actual in ('', None)
    else:
        assert math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)

def verify(task, cohort):
    run = Path(task['run_dir'])
    public = run/'workspace/results/agent_run'
    private = run/'private'
    manifest = json.loads((private/'manifest.json').read_text())
    checks = json.loads((private/'isolation_checks.json').read_text())
    audit = json.loads((private/'integrity_audit.json').read_text())
    result_dir = ROOT/'figures'/task['task_name']
    result = json.loads((result_dir/'result.json').read_text())
    assert manifest['source_commit'] == cohort['source_commit']
    assert manifest['model'] == 'gpt-6-astra' and manifest['reasoning_effort'] == 'max'
    assert manifest['budget'] == 2000 and manifest['seed'] == 0
    assert not manifest['agent_runtime']['backends']
    assert checks['evaluations'] == 0 and checks['budget'] == 2000 and len(checks['checks']) == 24
    assert audit['unknown_evaluations'] == 0 and audit['artifacts_match_journal'] and audit['only_workspace_execution_tools']
    assert json.loads((private/'cli_exit.json').read_text())['returncode'] == 0
    for name in ('AGENTS.md', 'PUBLIC_EVALUATOR.md', 'src/agent_harness/harness.py'):
        assert hashlib.sha256((run/'workspace'/name).read_bytes()).hexdigest() == manifest['public_sha256'][name]
    assert manifest['public_sha256']['AGENTS.md'] == hashlib.sha256((REPO/'AGENTS.md').read_bytes()).hexdigest()
    trace = read_rows(public/'trace.csv')
    scored = read_rows(private/'scored_trace.csv')
    analysis = read_rows(ROOT/'analysis'/task['task_name']/'trace.csv')
    with sqlite3.connect('file:'+str(private/'evaluations.sqlite')+'?mode=ro',uri=True) as db:
        journal = db.execute('SELECT idx,x,y,g,y_true,label FROM evaluations ORDER BY idx').fetchall()
        snapshots = db.execute('SELECT eval_idx,name,content FROM snapshots ORDER BY rowid').fetchall()
    assert 0 < len(trace) <= 2000
    assert len(trace) == len(scored) == len(journal) == len(analysis) == result['evaluations'] == audit['evaluations_used']
    best = math.inf
    best_score = None
    for index, (row, score, exported, raw) in enumerate(zip(trace,scored,analysis,journal),1):
        idx,x,y,g,true,label = raw
        assert index == idx == int(row['eval_idx']) == int(score['eval_idx'])
        assert json.loads(row['x']) == json.loads(x) and row['optimizer'] == label
        same(row['y'],y); same(score['y'],y)
        violations = json.loads(g or '[]')
        feasible = sum(max(v,0.) for v in violations) <= 1e-10
        if task['n_constraints']:
            assert json.loads(row['g']) == violations
            assert (row['feasible']=='True') == feasible
        assert (score['feasible']=='True') == feasible
        if feasible and y < best:
            best = y
            best_score = true if true is not None else y
        same(row['best_y'],None if not math.isfinite(best) else best)
        same(score['y_best_so_far'],best_score)
        same(exported['y_best_so_far'],best_score)
        same(score['y_true'],true)
        assert 'y_true' not in row
    regret = None if best_score is None else best_score-task['f_opt']
    same(result['agent_regret'],regret)
    previous = ''
    initial = []
    for idx,name,content in snapshots:
        assert 0 <= idx <= len(trace)
        if name == 'roadmap_initial.md':
            initial.append((idx,content))
        else:
            assert content.startswith(previous)
            previous = content
    assert len(initial)==1 and initial[0][0]==0
    assert initial[0][1] == (public/'roadmap_initial.md').read_text()
    final_log = (public/'changelog.md').read_text()
    # A final note appended after the last save is not a prospective checkpoint.
    # It may extend the journaled history, but must never rewrite it.
    assert final_log.startswith(previous)
    blocks=json.loads((ROOT/'analysis'/task['task_name']/'decisions.json').read_text())
    next_eval=1
    for block in blocks:
        assert block['first_eval']==next_eval
        assert block['calls']==block['last_eval']-block['first_eval']+1
        next_eval=block['last_eval']+1
    assert next_eval==len(trace)+1
    for method,mean in result['baseline_mean_regrets'].items():
        values=[]
        for path in (REPO/'main/results_full'/task['task_name']/method).glob('seed_*/trace.csv'):
            with path.open() as stream:
                reader=csv.DictReader(stream)
                last=None
                for last in reader:
                    pass
            assert last is not None
            if last['y_best_so_far']:
                values.append(float(last['y_best_so_far'])-task['f_opt'])
        same(mean,sum(values)/len(values) if values else None)
    for stem in ('01_final_regret','02_convergence','03_decision_timeline'):
        for ext in ('png','pdf','svg'):
            assert (result_dir/f'{stem}.{ext}').stat().st_size>1000
    return {'task':task['task'],'passed':True,'evaluations':len(trace),'decision_blocks':len(blocks),
            'status':result['status'],'agent_regret':regret,'isolation_checks':24,
            'trailing_unsnapshotted_characters':len(final_log)-len(previous),
            'checks':'journal / public artifacts / private scoring / exported scores / baseline means / decision coverage / figure files'}

if __name__=='__main__':
    cohort=json.loads((ROOT/'cohort.json').read_text())
    state=json.loads((ROOT/'state.json').read_text())
    results=[]
    for task in state['runs']:
        if not (ROOT/'figures'/task['task_name']/'result.json').exists():
            continue
        try:
            result=verify(task,cohort)
        except Exception as exc:
            result={'task':task['task'],'passed':False,'error':repr(exc)}
        results.append(result)
        print(result,flush=True)
    report={'source_commit':cohort['source_commit'],'checked':len(results),'expected':30,
            'all_finished':state['done'],'failures':sum(not r['passed'] for r in results),'tasks':results}
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(bool(report['failures']))
