"""Prepare one fresh campaign, serve its evaluator, and launch a clean CLI.

Trusted files live in private/; only workspace/ and an optimizer-only runtime
are exposed to agent commands. No VM, socket API, or shared campaign state.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import threading
import uuid

from .evaluator import Evaluator
from .execution import overrides, permissions
from .preflight import check

MAIN = Path(__file__).resolve().parents[2]
REPO = MAIN.parent


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def atomic(path, text):
    # Parent directories are read-only in the agent's native sandbox.
    temp = path.with_suffix('.tmp')
    temp.write_text(text)
    os.replace(temp, path)


class Bridge:
    def __init__(self, evaluator, workspace):
        self.evaluator, self.workspace = evaluator, workspace
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.error = None

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('Evaluator still running; preserve the journal before retrying.')
        if self.error:
            raise RuntimeError('Evaluator transport failed') from self.error

    def serve(self):
        try:
            while not self.stop.is_set():
                for path in (self.workspace / '.bridge/requests').glob('*.json'):
                    if not re.fullmatch('[0-9a-f]{32}\.json', path.name):
                        continue
                    dest = self.workspace / '.bridge/responses' / path.name
                    if dest.exists():
                        path.unlink(missing_ok=True)
                        continue
                    try:
                        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                        with os.fdopen(fd, 'rb') as stream:
                            info = os.fstat(stream.fileno())
                            if not stat.S_ISREG(info.st_mode) or info.st_size > 500000:
                                raise ValueError('Invalid request file')
                            data = stream.read(500001)
                        if len(data) > 500000:
                            raise ValueError('Request too large')
                        request = json.loads(data)
                        if not isinstance(request, dict) or request.get('request_id') != path.stem:
                            raise ValueError('Invalid request identifier')
                        reply = self.evaluator.handle('run', request)
                        for name, content in reply.pop('artifacts', {}).items():
                            if name not in ('trace.csv', 'meta.json'):
                                raise ValueError('Invalid artifact')
                            atomic(self.workspace / 'results/agent_run' / name, content)
                    except Exception:
                        reply = {'ok': False, 'error_type': 'TransportError',
                                 'error': 'Evaluator request failed; stop and report. Do not reset.'}
                    atomic(dest, json.dumps(reply, allow_nan=False))
                    path.unlink(missing_ok=True)
                self.stop.wait(.005)
        except BaseException as exc:
            self.error = exc


def runtime_info(python):
    query = """import sys,json,importlib.util,importlib.metadata as m
names=['numpy','scipy','cma','botorch','gpytorch','torch','Py-BOBYQA','pandas','matplotlib']
print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,
 'executable':sys.executable,'version':sys.version,
 'backends':[n for n in ['ioh','cocoex','opfunu','pymoo','yahpo_gym']
   if importlib.util.find_spec(n) is not None],
 'packages':{n:m.version(n) for n in names}}))
"""
    info = json.loads(subprocess.check_output([str(python), '-I', '-c', query], text=True))
    if info['backends']:
        raise ValueError('Agent runtime contains objective backends: ' + ', '.join(info['backends'])
                         + '. Install main/requirements-agent.txt in a separate environment.')
    return info


def prepare(args, task, runtime):
    # An opaque basename keeps the task identity out of paths visible to the model.
    root = args.runs_dir.resolve() / ('run-' + uuid.uuid4().hex[:12])
    root.mkdir(parents=True, mode=0o700)
    workspace, private = root / 'workspace', root / 'private'
    private.mkdir(mode=0o700)
    for relative in ['work', '.tmp', '.home', '.bridge/requests', '.bridge/responses',
                     'results/agent_run', 'src/agent_harness']:
        (workspace / relative).mkdir(parents=True, exist_ok=True)
    for name in ['roadmap_initial.md', 'changelog.md', 'trace.csv', 'meta.json']:
        (workspace / 'results/agent_run' / name).touch()
    shutil.copyfile(REPO / 'AGENTS.md', workspace / 'AGENTS.md')
    (workspace / 'CLAUDE.md').write_text('@AGENTS.md\n')
    for name in ['src/__init__.py', 'src/agent_harness/__init__.py']:
        (workspace / name).write_text('')
    shutil.copyfile(MAIN / 'src/agent_harness/harness.py', workspace / 'src/agent_harness/harness.py')
    shutil.copytree(MAIN / 'src/optimizers', workspace / 'src/optimizers',
                    ignore=shutil.ignore_patterns('__pycache__'))
    # A read-only venv view removes installation aliases (e.g. uv's versionless
    # symlink) without granting access to their parent directories on macOS.
    venv = workspace / '.venv'
    (venv / 'bin').mkdir(parents=True)
    executable = Path(runtime['executable']).resolve()
    (venv / 'bin/python').symlink_to(executable)
    (venv / 'pyvenv.cfg').write_text(
        f'home = {executable.parent}\ninclude-system-site-packages = false\n')
    (venv / 'lib').symlink_to(Path(runtime['prefix']).resolve() / 'lib', target_is_directory=True)
    if (Path(runtime['prefix']) / 'lib64').exists():
        (venv / 'lib64').symlink_to('lib', target_is_directory=True)
    # Scripts in work/ can import the public src package without modifying it.
    (workspace / 'work/src').symlink_to('../src', target_is_directory=True)
    bounds = [[float(v) for v in pair] for pair in task.bounds]
    public = f'''# Public evaluation interface

Minimize a deterministic scalar black-box objective over {task.dim} continuous
coordinates with bounds {json.dumps(bounds)}. There are no additional constraints
and no prior observations. The hard budget is {args.budget} objective evaluations,
including repeated points, initialization, line searches and finite differences.
The objective identity, implementation and optimum are private.

Use `.venv/bin/python` from the workspace root. Put scripts in `work/`.

```python
from src.agent_harness.harness import AgentTaskHarness, BudgetExhausted
h = AgentTaskHarness(budget={args.budget}, outdir="results/agent_run")
h.set_active_optimizer("your_method")
# y = h.evaluate(x)
h.save()
```

Public metadata: dim, bounds, direction (min), noisy (False), budget.
Live counters: evaluations_used and evaluations_remaining.
The client is inspectable transport code; evaluation runs in a separate process.
Reconnecting or creating another client cannot reset the campaign budget.

Follow AGENTS.md. Record only the first block in results/agent_run/roadmap_initial.md
before any evaluation. Keep it unchanged; append each subsequent decision to
changelog.md before its block. Choose the method, relevant hyperparameters and
block length yourself. Return control to the LLM at every checkpoint.
The evaluator snapshots these documents and rejects retrospective edits.
It writes trace.csv and meta.json; call save() at checkpoints to refresh them.

Installed packages include numpy, scipy, cma, torch, botorch, gpytorch, pybobyqa,
pandas and matplotlib. Wrappers are in src/optimizers. You may use their libraries
or write adapters in work/. Network access and dependency installation are disabled.
On an evaluator error, stop and report; do not retry an unknown evaluation.
'''
    (workspace / 'PUBLIC_EVALUATOR.md').write_text(public)
    (private / 'canary.txt').write_text('Private evaluator data: access must fail.')
    (root / 'sibling_canary.txt').write_text('Another campaign: access must fail.')
    roots = sorted({runtime['prefix'], runtime['base_prefix'],
                    str(Path(runtime['prefix']).resolve()), str(Path(runtime['base_prefix']).resolve())})
    policy = {'codex': str(args.codex), 'python': str(args.agent_python),
              'workspace': str(workspace), 'permissions': permissions(workspace, roots)}
    write_json(private / 'execution.json', policy)
    manifest = {
        'task': args.task, 'budget': args.budget, 'model': args.model, 'reasoning_effort': args.effort,
        'source_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_status': subprocess.check_output(['git', '-C', str(REPO), 'status', '--short'], text=True),
        'codex_version': subprocess.check_output([str(args.codex), '--version'], text=True).strip(),
        'agent_runtime': runtime,
        'source_sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in ['src/blind_runner', 'src/tasks']
                          for p in (MAIN / folder).glob('*.py')},
        'public_sha256': {str(p.relative_to(workspace)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in workspace.rglob('*') if p.is_file() and not p.is_symlink()},
    }
    write_json(private / 'manifest.json', manifest)
    return root, workspace, private, policy


def launch(args, workspace, private, policy):
    config = dict(policy['permissions'])
    config.update({
        'developer_instructions': (workspace / 'AGENTS.md').read_text() +
            '\nAll campaign execution and file access use the workspace MCP tools. '
            'Read PUBLIC_EVALUATOR.md. Put scripts in work/; use .venv/bin/python. '
            'Use write_stdin to keep optimizer processes alive between blocks.\n',
        'model_reasoning_effort': args.effort,
        'mcp_servers': {'workspace': {'command': sys.executable,
            'args': ['-B', str(Path(__file__).with_name('execution.py')), str(private / 'execution.json')],
            'enabled': True, 'required': True, 'startup_timeout_sec': 30, 'tool_timeout_sec': 45,
            'enabled_tools': ['run_command', 'write_stdin'], 'default_tools_approval_mode': 'approve'}},
    })
    for feature in ['shell_tool', 'unified_exec', 'view_image', 'apps', 'plugins', 'remote_plugin',
                    'memories', 'hooks', 'shell_snapshot', 'browser_use', 'browser_use_external',
                    'computer_use', 'in_app_browser', 'multi_agent', 'multi_agent_v2',
                    'image_generation', 'skill_search', 'workspace_dependencies', 'code_mode']:
        config['features.' + feature] = False
    config['features.code_mode_host'] = True
    write_json(private / 'effective_overrides.json', config)
    prompt = ('Carry out the fully authorized optimization campaign in PUBLIC_EVALUATOR.md, following '
              'AGENTS.md. Choose and execute only one bounded block at a time, then review your '
              'observations and decide again. You choose relevant hyperparameters and block lengths. '
              'Work autonomously without asking for confirmation between blocks. Use the available '
              'budget thoughtfully and finish with the four required campaign files and a brief summary.')
    (private / 'initial_prompt.txt').write_text(prompt)
    command = [str(args.codex), '--no-daemon', 'exec', '--ignore-user-config', '--ignore-rules',
               '--ephemeral', '--skip-git-repo-check', '--json', '--color', 'never',
               '-m', args.model, '-C', str(workspace), *overrides(config),
               '-o', str(private / 'agent_final.txt'), '-']
    with (private / 'cli_events.jsonl').open('w') as events, (private / 'cli_stderr.log').open('w') as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=events, stderr=errors, text=True)
        try:
            process.communicate(prompt)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    write_json(private / 'cli_exit.json', {'returncode': process.returncode})
    return process.returncode


def audit(evaluator, workspace, private):
    expected = evaluator.artifacts('run')
    matches = all((workspace / 'results/agent_run' / name).read_bytes() == content.encode()
                  for name, content in expected.items())
    unknown = evaluator.db.execute('SELECT COUNT(*) FROM evaluations WHERE y IS NULL').fetchone()[0]
    events = private / 'cli_events.jsonl'
    tools = []
    if events.exists():
        for line in events.read_text().splitlines():
            event = json.loads(line)
            if event.get('type') != 'item.completed':
                continue
            item = event.get('item', {})
            if item.get('type') == 'mcp_tool_call':
                tools.append((item.get('server'), item.get('tool')))
            elif item.get('type') in ('command_execution', 'file_change', 'web_search'):
                tools.append(('unexpected', item['type']))
    report = {
        **evaluator.status('run'), 'unknown_evaluations': unknown,
        'artifacts_match_journal': matches,
        'only_workspace_execution_tools': all(s == 'workspace' and t in ('run_command', 'write_stdin') for s, t in tools),
        'tool_calls': len(tools),
    }
    write_json(private / 'integrity_audit.json', report)
    if not matches or unknown or not report['only_workspace_execution_tools']:
        raise RuntimeError('Integrity audit failed; see ' + str(private / 'integrity_audit.json'))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run the canonical agent with native filesystem isolation.')
    parser.add_argument('--task', required=True, help='A registered deterministic continuous BBOB task, e.g. step_ellipsoid10')
    parser.add_argument('--budget', required=True, type=int)
    parser.add_argument('--model', default='gpt-6-astra')
    parser.add_argument('--effort', default='xhigh', choices=['low', 'medium', 'high', 'xhigh'])
    parser.add_argument('--agent-python', type=Path, default=REPO / '.agent-venv/bin/python')
    parser.add_argument('--codex', type=Path, default=shutil.which('codex'))
    parser.add_argument('--runs-dir', type=Path, default=REPO / 'runs')
    parser.add_argument('--check-only', action='store_true', help='Validate isolation with zero evaluations and no model call')
    args = parser.parse_args(argv)
    if sys.platform not in ('darwin', 'linux'):
        parser.error('The native executor currently supports macOS and Linux only.')
    if args.budget < 1 or args.codex is None:
        parser.error('Use a positive budget and install/sign in to Codex CLI.')
    # Preserve the venv executable path: resolving its symlink would bypass the venv.
    args.agent_python = args.agent_python.expanduser().absolute()
    args.codex = args.codex.expanduser().absolute()
    from src.tasks.registry import TASKS
    from src.tasks.bbob import BBOBTask
    if args.task not in TASKS:
        parser.error('Unknown task. See main/src/tasks/registry.py.')
    task = TASKS[args.task]()
    if not isinstance(task, BBOBTask):
        parser.error('This isolated runner currently supports continuous deterministic BBOB tasks. '
                     'Other families require their public variable/constraint contracts first.')
    runtime = runtime_info(args.agent_python)
    root, workspace, private, policy = prepare(args, task, runtime)
    print('Run directory: ' + str(root), flush=True)
    suite = {'budget': args.budget, 'campaigns': [{'id': 'run', 'task': args.task}]}
    evaluator = Evaluator(suite, private / 'evaluations.sqlite', lambda _: task)
    write_json(private / 'scoring.json', {'task': args.task, 'f_opt': float(task.f_opt)})
    bridge = Bridge(evaluator, workspace)
    bridge.start()
    exit_code = 0
    try:
        forbidden = [private / 'canary.txt', root / 'sibling_canary.txt', REPO / 'README.md',
                     MAIN / 'src/tasks/bbob.py', private / 'evaluations.sqlite',
                     Path.home() / '.codex/auth.json', Path(__file__)]
        report = check(policy, forbidden, private)
        print(f"Isolation passed: {len(report['checks'])} checks, 0/{args.budget} evaluations.", flush=True)
        if not args.check_only:
            exit_code = launch(args, workspace, private, policy)
    finally:
        bridge.close()
        # The private journal remains recoverable on failure. Do not overwrite
        # public artifacts before comparing them with that journal.
        if sys.exc_info()[0] is None:
            report = audit(evaluator, workspace, private)
        evaluator.db.close()
    print(f"Finished: {report['evaluations_used']}/{args.budget} evaluations. Results: {workspace / 'results/agent_run'}")
    return exit_code
