"""Trusted local queue; agents never receive this file, task list, or results."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PYTHON = REPO / '.venv/bin/python'
CODEX = '/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex'

def now():
    return datetime.now(timezone.utc).isoformat()

def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def count(run_dir):
    if not run_dir:
        return 0
    db = Path(run_dir) / 'private/evaluations.sqlite'
    if not db.exists():
        return 0
    try:
        with sqlite3.connect('file:' + str(db) + '?mode=ro', uri=True, timeout=2) as conn:
            return conn.execute('SELECT COUNT(*) FROM evaluations').fetchone()[0]
    except sqlite3.Error:
        return 0

def main():
    if (ROOT / 'state.json').exists():
        raise RuntimeError('Cohort already started; preserve it and inspect state rather than relaunching.')
    cohort = json.loads((ROOT / 'cohort.json').read_text())
    tasks = cohort['tasks']
    first = ['rosenbrock10', 'pymoo_g04', 'noisy_rosenbrock_gauss5', 'mixint_rosenbrock10']
    tasks = sorted(tasks, key=lambda t: first.index(t['task']) if t['task'] in first else 4)
    rows = [{**t, 'status': 'queued', 'evaluations': 0} for t in tasks]
    state = {'started_at': now(), 'supervisor_pid': os.getpid(), 'done': False, 'paused': False, 'runs': rows}
    save(ROOT / 'control.json', {'max_workers': 4, 'paused': False})
    running = {}
    save(ROOT / 'state.json', state)
    env = os.environ.copy()
    env.update({'MPLCONFIGDIR': str(ROOT / 'cache/matplotlib'), 'XDG_CACHE_HOME': str(ROOT / 'cache'),
                'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    while any(r['status'] in ('queued', 'running') for r in rows):
        control = json.loads((ROOT / 'control.json').read_text())
        state['paused'] = control['paused']
        for key, (proc, stream, row) in list(running.items()):
            log = ROOT / 'logs' / (key + '.log')
            if not row.get('run_dir'):
                for line in log.read_text(errors='replace').splitlines():
                    if line.startswith('Run directory: '):
                        row['run_dir'] = line.partition(': ')[2]
                        break
            row['evaluations'] = count(row.get('run_dir'))
            rc = proc.poll()
            if rc is None:
                continue
            stream.close()
            del running[key]
            row.update(returncode=rc, finished_at=now())
            run_dir = Path(row['run_dir']) if row.get('run_dir') else None
            audit_path = run_dir / 'private/integrity_audit.json' if run_dir else None
            if audit_path and audit_path.exists():
                audit = json.loads(audit_path.read_text())
                row['integrity'] = audit
            else:
                audit = {}
            valid = (rc == 0 and row['evaluations'] == 2000 and audit.get('unknown_evaluations') == 0
                     and audit.get('artifacts_match_journal') and audit.get('only_workspace_execution_tools'))
            row['status'] = 'complete' if valid else 'failed_or_incomplete'
            print(now(), key, row['status'], row['evaluations'], flush=True)
            if rc != 0 and row['evaluations'] == 0:
                control['paused'] = True
                save(ROOT / 'control.json', control)
                state['paused'] = True
            save(ROOT / 'state.json', state)
        while not state['paused'] and len(running) < control['max_workers']:
            row = next((r for r in rows if r['status'] == 'queued'), None)
            if row is None:
                break
            current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
            dirty = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=REPO, text=True).strip()
            if current != cohort['source_commit'] or dirty:
                state['paused'] = True
                control['paused'] = True
                save(ROOT / 'control.json', control)
                print(now(), 'Source changed; pending launches paused.', flush=True)
                break
            key = row['task']
            command = [str(PYTHON), '-B', str(REPO / 'main/scripts/run_agent.py'), '--task', key,
                       '--budget', '2000', '--seed', '0', '--model', 'gpt-6-astra', '--effort', 'max',
                       '--runs-dir', str(ROOT / 'campaigns'), '--codex', CODEX]
            stream = (ROOT / 'logs' / (key + '.log')).open('w')
            proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            row.update(status='running', started_at=now(), pid=proc.pid, command=command)
            running[key] = (proc, stream, row)
            print(now(), key, 'launched', proc.pid, flush=True)
            save(ROOT / 'state.json', state)
        state['updated_at'] = now()
        save(ROOT / 'state.json', state)
        time.sleep(5)
    state.update(done=True, finished_at=now())
    save(ROOT / 'state.json', state)
    print(now(), 'All campaigns ended; see state.json for completion and integrity.', flush=True)

if __name__ == '__main__':
    main()
