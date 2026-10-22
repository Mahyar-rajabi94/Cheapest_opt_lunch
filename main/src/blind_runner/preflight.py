"""Fail closed before inference if the native execution boundary leaks access."""
import json
import os
from pathlib import Path
import shlex

from .execution import run


PROBE = r'''
import errno, importlib.util, json, os, pathlib, socket, subprocess, sys
root = pathlib.Path.cwd()
checks = []
def denied(label, fn):
    try:
        fn()
    except OSError as e:
        assert e.errno in (errno.EACCES, errno.EPERM, errno.ENOENT, errno.EROFS), (label, str(e))
        checks.append(label)
    else:
        raise AssertionError('Isolation failure: ' + label)
for value in json.loads(sys.argv[1]):
    denied('private read denied', lambda p=pathlib.Path(value): p.read_bytes())
for name in ['AGENTS.md', 'PUBLIC_EVALUATOR.md', 'src/agent_harness/harness.py',
             'results/agent_run/trace.csv', 'results/agent_run/meta.json']:
    denied(name + ' protected', lambda p=root/name: p.open('a'))
denied('trace removal denied', lambda: (root/'results/agent_run/trace.csv').unlink())
denied('trace replacement denied', lambda: (root/'results/agent_run/trace.csv').rename(root/'work/stolen_trace.csv'))
denied('response forgery denied', lambda: (root/'.bridge/responses/forged.json').write_text('{}'))
denied('home configuration protected', lambda: (root/'.home/agent_config').write_text('override'))
denied('project configuration protected', lambda: (root/'.codex').mkdir())
denied('bridge replacement denied', lambda: (root/'.bridge').rename(root/'work/moved_bridge'))
link = root/'work/escape'
link.symlink_to(json.loads(sys.argv[1])[0])
denied('symlink escape denied', lambda: link.read_bytes())
link.unlink()
denied('hardlink escape denied', lambda: os.link(json.loads(sys.argv[1])[0], root/'work/hardlink'))
def connect(address):
    with socket.socket() as s:
        s.settimeout(1)
        s.connect(address)
denied('external network denied', lambda: connect(('1.1.1.1', 443)))
denied('local service denied', lambda: connect(('127.0.0.1', int(sys.argv[3]))))
if sys.platform == 'darwin':
    try:
        p = subprocess.run(['/bin/ps', '-p', sys.argv[2], '-o', 'command='], capture_output=True, text=True)
        assert p.returncode != 0 or not p.stdout.strip(), 'Supervisor process arguments exposed'
    except PermissionError:
        pass
    checks.append('supervisor process inaccessible')
assert not (root/'objective.py').exists() and not (root/'src/tasks').exists()
assert not (root/'.git').exists()
assert importlib.util.find_spec('ioh') is None, 'Use an optimizer-only runtime without ioh'
assert importlib.util.find_spec('cocoex') is None, 'Use an optimizer-only runtime without cocoex'
checks.append('objective sources and backends absent')
import numpy, scipy, cma, torch, botorch, gpytorch, pybobyqa
from src.agent_harness.harness import AgentTaskHarness
h = AgentTaskHarness()
assert h.evaluations_used == 0
assert h.evaluations_remaining == h.budget
try:
    AgentTaskHarness(budget=h.budget+1)
except ValueError:
    pass
else:
    raise AssertionError('Client budget override accepted')
h.save()
print(json.dumps({'checks': checks, 'evaluations': 0, 'budget': h.budget}))
'''


def check(policy, forbidden, private):
    import socket
    # A listening socket distinguishes actual denial from "nothing listening".
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        script = '.venv/bin/python -B -c ' + shlex.quote(PROBE)
        script += ' ' + shlex.quote(json.dumps([str(p) for p in forbidden]))
        script += ' ' + str(os.getpid()) + ' ' + str(listener.getsockname()[1])
        result = run(policy, script, timeout=120)
    (private / 'preflight.stdout').write_text(result.stdout)
    (private / 'preflight.stderr').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError('Isolation preflight failed. See ' + str(private / 'preflight.stderr')
                           + '\n' + result.stdout[-3000:] + result.stderr[-3000:])
    report = json.loads(result.stdout.splitlines()[-1])
    (private / 'isolation_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
