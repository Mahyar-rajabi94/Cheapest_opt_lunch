"""Bundle completed, verified figures and research provenance for handoff."""
import argparse
import csv
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile

ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--partial',action='store_true',help='Explicitly label an incomplete-cohort export.')
    args=parser.parse_args()
    state=json.loads((ROOT/'state.json').read_text())
    verification=json.loads((ROOT/'verification.json').read_text())
    assert verification['failures']==0, 'Resolve verification failures before packaging.'
    results=[json.loads(p.read_text()) for p in sorted((ROOT/'figures').glob('*/result.json'))]
    assert len(results)==verification['checked']
    if not args.partial:
        assert state['done'] and len(results)==30, 'Cohort is not complete; --partial explicitly exports available results.'
    by_task={r['task']:r for r in results}
    metrics=[]
    usage_totals=Counter()
    concurrency_events=[]
    for task in state['runs']:
        if task.get('started_at'):
            concurrency_events.append((task['started_at'],1))
        if task.get('finished_at'):
            concurrency_events.append((task['finished_at'],-1))
        if task['task'] not in by_task:
            continue
        result=by_task[task['task']]
        usage=Counter()
        with (Path(task['run_dir'])/'private/cli_events.jsonl').open() as stream:
            for line in stream:
                event=json.loads(line)
                if event.get('type')=='turn.completed':
                    usage.update(event.get('usage',{}))
        usage_totals.update(usage)
        elapsed=(datetime.fromisoformat(task['finished_at'])-datetime.fromisoformat(task['started_at'])).total_seconds()
        metrics.append({'task':task['task'],'status':result['status'],'evaluations':result['evaluations'],
                        'wall_seconds_including_setup':round(elapsed,3),**usage})
    current=peak=0
    for _,delta in sorted(concurrency_events):
        current+=delta
        peak=max(peak,current)
    statistics={'campaigns_plotted':len(results),'allowance_per_task':2000,
                'actual_evaluations':sum(r['evaluations'] for r in results),
                'outcomes':dict(Counter(r['status'] for r in results)),
                'peak_simultaneous_campaigns':peak,'started_at':state['started_at'],
                'finished_at':state.get('finished_at'),'llm_usage_totals_as_reported_by_cli':dict(usage_totals)}
    (ROOT/'statistics.json').write_text(json.dumps(statistics,indent=2)+'\n')
    fields=list(dict.fromkeys(key for row in metrics for key in row))
    with (ROOT/'execution_metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(metrics)
    paths={}
    for name in ('README.md','index.html','summary.csv','cohort.json','tasks.json','verification.json','annotations.json',
                 'statistics.json','execution_metrics.csv',
                 'baseline_integrity.json','baseline-sha256.jsonl','source_snapshot.tar.gz',
                 'agent-requirements.txt','evaluator-requirements.txt','codex-version.txt',
                 'supervision_events.jsonl','state.json','campaign_status.json','resources.json','render_errors.json',
                 'supervise.py','watch.py','report.py','verify.py','package.py'):
        path=ROOT/name
        if path.exists(): paths[name]=path
    for folder in ('figures','analysis'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and not path.is_symlink():
                paths[str(path.relative_to(ROOT))]=path
    for result in results:
        run=Path(result['run_dir'])
        target='provenance/'+result['task']
        for name in ('manifest.json','isolation_checks.json','integrity_audit.json','scoring.json',
                     'scored_trace.csv','agent_final.txt','cli_exit.json','evaluations.sqlite','cli_events.jsonl'):
            path=run/'private'/name
            if name == 'evaluations.sqlite':
                # SQLite backup includes committed WAL pages; copying only the
                # main file could otherwise omit recent transactions.
                snapshot=ROOT/'export_journals'/(result['task']+'.sqlite')
                snapshot.parent.mkdir(exist_ok=True)
                source=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
                destination_db=sqlite3.connect(snapshot)
                try:
                    source.backup(destination_db)
                    assert destination_db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
                finally:
                    destination_db.close();source.close()
                path=snapshot
            if path.exists(): paths[target+'/private/'+name]=path
        for name in ('trace.csv','meta.json','roadmap_initial.md','changelog.md'):
            paths[target+'/public/'+name]=run/'workspace/results/agent_run'/name
        for name in ('AGENTS.md','PUBLIC_EVALUATOR.md'):
            paths[target+'/'+name]=run/'workspace'/name
    suffix='-partial' if args.partial else ''
    destination=ROOT/f'paper-results-2000{suffix}.zip'
    temporary=destination.with_suffix('.zip.tmp')
    hashes=[]
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for relative,path in sorted(paths.items()):
            data=path.read_bytes()
            archive.writestr(relative,data)
            hashes.append({'path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        archive.writestr('ARTIFACT_HASHES.json',json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
    temporary.replace(destination)
    print(destination)
    print(len(results),'campaigns;',len(hashes),'files;',round(destination.stat().st_size/1024**2,1),'MiB')
