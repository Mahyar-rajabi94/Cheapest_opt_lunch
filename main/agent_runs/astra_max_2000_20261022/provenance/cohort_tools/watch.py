"""Trusted resource monitor and incremental figure generation; no agent steering."""
import json
import re
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]

def event(**values):
    with (ROOT/'supervision_events.jsonl').open('a') as f:
        f.write(json.dumps({'at':datetime.now(timezone.utc).isoformat(),**values})+'\n')

def main():
    while True:
        state=json.loads((ROOT/'state.json').read_text())
        result=subprocess.run(['/usr/bin/memory_pressure','-Q'],capture_output=True,text=True,timeout=15)
        match=re.search(r'free percentage:\s*(\d+)%',result.stdout)
        if match:
            free=int(match.group(1))
            target=16 if free>=40 else 12 if free>=36 else 8 if free>=30 else 4 if free>=22 else 0
            path=ROOT/'control.json'
            stamp=path.stat().st_mtime_ns
            control=json.loads(path.read_text())
            if control['max_workers']!=target:
                control['max_workers']=target
                # Keep any concurrent infrastructure pause set by the launcher.
                if path.stat().st_mtime_ns==stamp:
                    temp=path.with_suffix('.watch.tmp');temp.write_text(json.dumps(control,indent=2)+'\n');temp.replace(path)
                    event(action='memory_guard',free_percent=free,max_workers=target)
            (ROOT/'resources.json').write_text(json.dumps({'free_percent':free,'launch_limit':target,'updated_at':datetime.now(timezone.utc).isoformat()})+'\n')
        with (ROOT/'plots.log').open('a') as out:
            proc=subprocess.run([str(REPO/'.agent-venv/bin/python'),'-B',str(ROOT/'report.py')],stdout=out,stderr=subprocess.STDOUT,cwd=REPO)
        if proc.returncode:
            event(action='plot_error',returncode=proc.returncode)
        if state['done']:
            break
        time.sleep(20)

if __name__=='__main__':
    main()
