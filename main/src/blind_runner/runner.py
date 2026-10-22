"""Prepare one fresh campaign, serve its evaluator, and launch a clean CLI.

Trusted files live in private/; only workspace/ and an optimizer-only runtime
are exposed to agent commands. No VM, socket API, or shared campaign state.
"""
import argparse
import hashlib
import inspect
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
# MCP server name per agent CLI; Claude Code reserves "workspace".
SERVER = {'codex': 'workspace', 'claude': 'isolated'}
TOOLS = ['run_command', 'write_stdin']
# Expose pool wrappers only when the task explicitly exposes candidate features.
POOL_ONLY_FILES = ['pool_random.py', 'pool_gp_bo.py', 'pool_turbo.py']


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
    indexed = getattr(task, 'indexed_pool', False)
    excluded = [] if indexed else POOL_ONLY_FILES
    shutil.copytree(MAIN / 'src/optimizers', workspace / 'src/optimizers',
                    ignore=shutil.ignore_patterns('__pycache__', *excluded))
    if indexed:
        import numpy as np
        # Deliberately export only features, with anonymous row positions.
        np.save(workspace / 'candidates.npy', task.candidates, allow_pickle=False)
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
    kind = 'noisy' if task.noisy else 'noise-free' if task.pool and not indexed else 'deterministic'
    if task.n_constraints:
        constraints = (f'There are {task.n_constraints} black-box inequality constraints: a point is '
                       'feasible iff every constraint value is <= 0 (tolerance 1e-10). Infeasible points '
                       'are still evaluated and charged. The result is the best feasible observation.')
        returns = 'y, g = h.evaluate(x)  # g: list of constraint values'
    else:
        constraints, returns = 'There are no additional constraints.', 'y = h.evaluate(x)'
    noise = (' Each evaluation returns a noisy observation of the objective; the campaign is '
             'scored by the noise-free value at the point with the lowest observed y.'
             if task.noisy else
             # A pool task maps each point to an untested candidate, so a repeat differs.
             ' The objective is not repeatable: the value returned for x depends on which '
             'points were already evaluated, so evaluating the same x again generally '
             'returns a different value.'
             if task.pool and not indexed else '')
    pool_interface = ''
    if indexed:
        returns = 'y = h.evaluate_index(index)  # index: one currently available row'
        pool_interface = f'''This is an exact finite-candidate problem with {task.n_candidates} rows.
Only these rows are valid choices; the bounding box is descriptive, not a continuous search space.
Read h.candidates (shape {task.n_candidates} by {task.dim}) or candidates.npy for ALL public feature
values, and h.available_indices for the unused row indices. Choose a row with
h.evaluate_index(index). Each row may be purchased once. Generated points and duplicate purchases
are rejected without consuming budget. Identical feature rows, if present, remain distinct indices.
No labels, source identifiers, global ranks, or target-derived statistics accompany the feature table.
Choose your own initial design, methods, hyperparameters, and bounded blocks under AGENTS.md.
Pool-aware GPBO and TuRBO wrappers are in src.optimizers.pool_gp_bo and src.optimizers.pool_turbo:
construct with candidates=h.candidates, bounds=h.bounds, seed=h.seed, direction="min";
tell ONLY observed rows and returned y, then ask_pool(h.available_indices).
Their default n_init=10 is configurable; it is not a task requirement. Other methods or adapters
are allowed, provided their final selections are unused row indices. Constant features may exist.
'''
    public = f'''# Public evaluation interface

Minimize a {kind} scalar black-box objective over {task.dim} real-valued
coordinates with bounds {json.dumps(bounds)}.{noise} {constraints}
There are no prior observations. The hard budget is {args.budget} objective evaluations,
including repeated points, initialization, line searches and finite differences.
The objective identity, implementation and optimum are private.
Campaign seed: {args.seed}. Use this seed for stochastic initial designs and optimizer randomness;
record any derived seeds or justified changes in your decision log.
{pool_interface}

Use `.venv/bin/python` from the workspace root. Put scripts in `work/`.

```python
from src.agent_harness.harness import AgentTaskHarness, BudgetExhausted
h = AgentTaskHarness(budget={args.budget}, outdir="results/agent_run")
h.set_active_optimizer("your_method")
# {returns}
h.save()
```

Public metadata: dim, bounds, direction (min), noisy ({task.noisy}),
n_constraints ({task.n_constraints}), budget, seed.
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
        'task': args.task, 'budget': args.budget, 'seed': args.seed, 'agent': args.agent,
        'model': args.model, 'reasoning_effort': args.effort,
        'source_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_status': subprocess.check_output(['git', '-C', str(REPO), 'status', '--short'], text=True),
        'codex_version': subprocess.check_output([str(args.codex), '--version'], text=True).strip(),
        **({'claude_version': subprocess.check_output([str(args.claude), '--version'], text=True).strip()}
           if args.agent == 'claude' else {}),
        'agent_runtime': runtime,
        'private_data_sha256': {str(Path(p).relative_to(REPO)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                for p in getattr(task, 'private_data_paths', ())},
        'source_sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in ['src/blind_runner', 'src/tasks']
                          for p in (MAIN / folder).glob('*.py')},
        'public_sha256': {str(p.relative_to(workspace)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in workspace.rglob('*') if p.is_file() and not p.is_symlink()},
    }
    write_json(private / 'manifest.json', manifest)
    if getattr(args, 'fast', False):
        manifest['service_tier'] = 'fast'
        write_json(private / 'manifest.json', manifest)
    if getattr(args, 'literature', 'none') == 'valency':
        from . import valency
        directory = private / 'literature'
        directory.mkdir(mode=0o700)
        config = {'max_calls': args.literature_max_calls, 'timeout_seconds': args.literature_timeout,
                  'token_file': str(args.valency_token_file) if args.valency_token_file else None}
        write_json(directory / 'config.json', config)
        manifest['literature'] = {'provider': 'valency', 'endpoint': valency.URL,
                                 'allowed_tools': valency.TOOLS, **config}
        write_json(private / 'manifest.json', manifest)
    return root, workspace, private, policy


def instructions(args, workspace):
    text = ((workspace / 'AGENTS.md').read_text() +
            f'\nAll campaign execution and file access use the {SERVER[args.agent]} MCP tools. '
            'Read PUBLIC_EVALUATOR.md. Put scripts in work/; use .venv/bin/python. '
            'Use write_stdin to keep optimizer processes alive between blocks.\n')
    if getattr(args, 'literature', 'none') == 'valency':
        from .valency import INSTRUCTIONS
        text += INSTRUCTIONS
    return text


def executor_args(private):
    return ['-B', str(Path(__file__).with_name('execution.py')), str(private / 'execution.json')]


def literature_command(private):
    return [sys.executable, '-B', str(Path(__file__).with_name('valency.py')),
            str(private / 'literature/config.json')]


def literature_tools(private):
    """Use the actual discovered subset, never every tool a remote server offers."""
    from .valency import PAGE_TOOL, REMOTE_TOOLS
    snapshot = json.loads((private / 'literature/tools.json').read_text())
    return [t['name'] for t in snapshot['tools'] if t['name'] in REMOTE_TOOLS] + [PAGE_TOOL]


def codex_command(args, workspace, private, policy):
    config = dict(policy['permissions'])
    config.update({
        'developer_instructions': instructions(args, workspace),
        'model_reasoning_effort': args.effort,
        'mcp_servers': {'workspace': {'command': sys.executable, 'args': executor_args(private),
            'enabled': True, 'required': True, 'startup_timeout_sec': 30, 'tool_timeout_sec': 45,
            'enabled_tools': TOOLS, 'default_tools_approval_mode': 'approve'}},
    })
    if getattr(args, 'fast', False):
        config['service_tier'] = 'fast'
        config['features.fast_mode'] = True
    if getattr(args, 'literature', 'none') == 'valency':
        command = literature_command(private)
        config['mcp_servers']['valency'] = {
            'command': command[0], 'args': command[1:], 'enabled': True, 'required': True,
            'startup_timeout_sec': 30, 'tool_timeout_sec': args.literature_timeout + 15,
            'enabled_tools': literature_tools(private), 'default_tools_approval_mode': 'approve',
            'env_vars': ['VALENCY_API_TOKEN'],
        }
    for feature in ['shell_tool', 'unified_exec', 'view_image', 'apps', 'plugins', 'remote_plugin',
                    'memories', 'hooks', 'shell_snapshot', 'browser_use', 'browser_use_external',
                    'computer_use', 'in_app_browser', 'multi_agent', 'multi_agent_v2',
                    'image_generation', 'skill_search', 'workspace_dependencies', 'code_mode']:
        config['features.' + feature] = False
    config['features.code_mode_host'] = True
    write_json(private / 'effective_overrides.json', config)
    return [str(args.codex), '--no-daemon', 'exec', '--ignore-user-config', '--ignore-rules',
            '--ephemeral', '--skip-git-repo-check', '--json', '--color', 'never',
            '-m', args.model, '-C', str(workspace), *overrides(config),
            '-o', str(private / 'agent_final.txt'), '-'], os.environ


def claude_command(args, workspace, private):
    # No built-in tools, settings, memory, CLAUDE.md, skills or connectors: the only
    # tools are the sandboxed MCP executor's, and anything else is denied unprompted.
    config = {'mcpServers': {SERVER['claude']: {'type': 'stdio', 'command': sys.executable,
                                                'args': executor_args(private)}}}
    allowed = [f"mcp__{SERVER['claude']}__{t}" for t in TOOLS]
    if getattr(args, 'literature', 'none') == 'valency':
        literature = literature_command(private)
        config['mcpServers']['valency'] = {'type': 'stdio', 'command': literature[0],
                                           'args': literature[1:]}
        allowed += ['mcp__valency__' + t for t in literature_tools(private)]
    write_json(private / 'claude_mcp.json', config)
    command = [str(args.claude), '-p', '--output-format', 'stream-json', '--verbose',
               '--model', args.model, '--effort', args.effort, '--tools', '',
               '--strict-mcp-config', '--mcp-config', str(private / 'claude_mcp.json'),
               '--allowedTools', ','.join(allowed),
               '--permission-mode', 'dontAsk', '--setting-sources', '', '--disable-slash-commands',
               '--no-session-persistence', '--system-prompt', instructions(args, workspace)]
    # The ceiling keeps the enclosing repository's git status out of the CLI's context.
    env = {**os.environ, 'ENABLE_CLAUDEAI_MCP_SERVERS': 'false', 'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1',
           'GIT_CEILING_DIRECTORIES': str(workspace.parent)}
    write_json(private / 'effective_overrides.json', {'command': command[1:], 'mcp': config})
    return command, env


def launch(args, workspace, private, policy):
    if args.agent == 'codex':
        command, env = codex_command(args, workspace, private, policy)
    else:
        command, env = claude_command(args, workspace, private)
    prompt = ('Carry out the fully authorized optimization campaign in PUBLIC_EVALUATOR.md, following '
              'AGENTS.md. Choose and execute only one bounded block at a time, then review your '
              'observations and decide again. You choose relevant hyperparameters and block lengths. '
              'Work autonomously without asking for confirmation between blocks. Use the available '
              'budget thoughtfully and finish with the four required campaign files and a brief summary.')
    (private / 'initial_prompt.txt').write_text(prompt)
    with (private / 'cli_events.jsonl').open('w') as events, (private / 'cli_stderr.log').open('w') as errors:
        process = subprocess.Popen(command, cwd=workspace, env=env, stdin=subprocess.PIPE,
                                   stdout=events, stderr=errors, text=True)
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
    if args.agent == 'claude':
        results = [e for e in events_of(private) if e.get('type') == 'result']
        (private / 'agent_final.txt').write_text((results[-1].get('result') or '') if results else '')
    return process.returncode


def events_of(private):
    path = private / 'cli_events.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def audit(evaluator, workspace, private, agent, literature='none'):
    expected = evaluator.artifacts('run')
    matches = all((workspace / 'results/agent_run' / name).read_bytes() == content.encode()
                  for name, content in expected.items())
    unknown = evaluator.db.execute('SELECT COUNT(*) FROM evaluations WHERE y IS NULL').fetchone()[0]
    tools = []
    for event in events_of(private):
        # Codex reports completed items; Claude reports tool_use blocks named mcp__<server>__<tool>.
        item = event.get('item', {}) if event.get('type') == 'item.completed' else {}
        if item.get('type') == 'mcp_tool_call':
            tools.append((item.get('server'), item.get('tool')))
        elif item.get('type') in ('command_execution', 'file_change', 'web_search'):
            tools.append(('unexpected', item['type']))
        elif event.get('type') == 'assistant':
            for block in event.get('message', {}).get('content', []):
                if block.get('type') in ('tool_use', 'server_tool_use'):
                    _, server, tool = (block.get('name', '').split('__', 2) + ['', ''])[:3]
                    tools.append((server, tool) if block['type'] == 'tool_use' else ('unexpected', block['name']))
    allowed_literature = set(literature_tools(private)) if literature == 'valency' else set()
    allowed = {(SERVER[agent], t) for t in TOOLS} | {('valency', t) for t in allowed_literature}
    tools_valid = all(pair in allowed for pair in tools)
    report = {
        **evaluator.status('run'), 'unknown_evaluations': unknown,
        'artifacts_match_journal': matches,
        'only_workspace_execution_tools': all(s == SERVER[agent] and t in TOOLS for s, t in tools),
        'tool_calls': len(tools),
    }
    if literature == 'valency':
        report.update({'only_allowed_tools': tools_valid,
                       'literature_tool_calls': sum(s == 'valency' for s, _ in tools)})
    write_json(private / 'integrity_audit.json', report)
    if not matches or unknown or not tools_valid:
        raise RuntimeError('Integrity audit failed; see ' + str(private / 'integrity_audit.json'))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run the canonical agent with native filesystem isolation.')
    parser.add_argument('--task', required=True, help='A key of main/src/tasks/registry.py, e.g. step_ellipsoid10')
    parser.add_argument('--budget', required=True, type=int)
    parser.add_argument('--seed', type=int, default=0, help='Public campaign seed and private observation-noise seed')
    parser.add_argument('--agent', default='codex', choices=['codex', 'claude'], help='CLI that drives the model')
    parser.add_argument('--model', help='Default: gpt-6-astra (codex), claude-opus-5-5 (claude)')
    parser.add_argument('--effort', default='max', choices=['low', 'medium', 'high', 'xhigh', 'max'],
                        help='Reasoning effort (default: max).')
    parser.add_argument('--fast', action='store_true',
                        help='Request Codex Fast service tier; preserves model and reasoning effort, increases usage.')
    parser.add_argument('--agent-python', type=Path, default=REPO / '.agent-venv/bin/python')
    parser.add_argument('--codex', type=Path, default=shutil.which('codex'),
                        help='Codex CLI; also the per-command sandbox for both agents')
    parser.add_argument('--claude', type=Path, default=shutil.which('claude'))
    parser.add_argument('--runs-dir', type=Path, default=REPO / 'runs')
    parser.add_argument('--literature', choices=['none', 'valency'], default='none',
                        help='Optional literature MCP tools; agent command networking remains disabled')
    parser.add_argument('--valency-token-file', type=Path,
                        help='Private bearer-token file; otherwise use VALENCY_API_TOKEN')
    parser.add_argument('--literature-max-calls', type=int, default=50,
                        help='Maximum uncached remote retrieval attempts per run (default: 50)')
    parser.add_argument('--literature-timeout', type=int, default=30,
                        help='Seconds per retrieval attempt, including connection setup (default: 30)')
    parser.add_argument('--check-only', action='store_true', help='Validate isolation with zero evaluations and no model call')
    args = parser.parse_args(argv)
    if args.fast and args.agent != 'codex':
        parser.error('--fast is supported only by the Codex launcher.')
    if args.literature_max_calls < 1 or args.literature_timeout < 1:
        parser.error('Literature call limit and timeout must be positive.')
    if args.valency_token_file and args.literature != 'valency':
        parser.error('--valency-token-file requires --literature valency.')
    if args.literature == 'valency':
        from .valency import token
        if args.valency_token_file:
            args.valency_token_file = args.valency_token_file.expanduser().resolve()
        try:
            import mcp
            token({'token_file': args.valency_token_file})
        except ImportError:
            parser.error('Install main/requirements-literature.txt in the private runner environment.')
        except (OSError, ValueError):
            parser.error('Set VALENCY_API_TOKEN or provide a readable --valency-token-file containing a bearer token.')
    if sys.platform not in ('darwin', 'linux'):
        parser.error('The native executor currently supports macOS and Linux only.')
    if args.budget < 1 or args.seed < 0 or args.codex is None:
        parser.error('Use a positive budget, nonnegative seed, and install Codex CLI (signed in, for --agent codex).')
    if args.agent == 'claude' and args.claude is None:
        parser.error('Install and sign in to Claude Code CLI, or pass --claude.')
    args.model = args.model or {'codex': 'gpt-6-astra', 'claude': 'claude-opus-5-5'}[args.agent]
    # Preserve the venv executable path: resolving its symlink would bypass the venv.
    args.agent_python = args.agent_python.expanduser().absolute()
    args.codex = args.codex.expanduser().absolute()
    if args.claude is not None:
        args.claude = args.claude.expanduser().absolute()
    from src.tasks.registry import TASKS
    if args.task not in TASKS:
        parser.error('Unknown task. See main/src/tasks/registry.py.')
    task = TASKS[args.task]()
    if task.pool and args.budget > task.n_candidates:
        parser.error(f'A pool task allows at most {task.n_candidates} evaluations.')
    task.reseed(args.seed)
    runtime = runtime_info(args.agent_python)
    root, workspace, private, policy = prepare(args, task, runtime)
    print('Run directory: ' + str(root), flush=True)
    suite = {'budget': args.budget, 'seed': args.seed, 'campaigns': [{'id': 'run', 'task': args.task}]}
    evaluator = Evaluator(suite, private / 'evaluations.sqlite', lambda _: task)
    write_json(private / 'scoring.json', {'task': args.task, 'task_name': task.name, 'f_opt': float(task.f_opt),
                                          'noisy': task.noisy, 'n_constraints': task.n_constraints,
                                          'seed': args.seed})
    bridge = Bridge(evaluator, workspace)
    bridge.start()
    exit_code = 0
    try:
        forbidden = [private / 'canary.txt', root / 'sibling_canary.txt', REPO / 'README.md',
                     Path(inspect.getfile(type(task))), private / 'evaluations.sqlite',
                     Path.home() / '.codex/auth.json', Path.home() / '.claude.json', Path(__file__)]
        forbidden.extend(getattr(task, 'private_data_paths', ()))
        if args.literature == 'valency':
            forbidden.append(private / 'literature/config.json')
            if args.valency_token_file:
                forbidden.append(args.valency_token_file)
        report = check(policy, forbidden, private)
        print(f"Isolation passed: {len(report['checks'])} checks, 0/{args.budget} evaluations.", flush=True)
        if args.literature == 'valency':
            # Fail setup clearly before inference; outages after setup are ordinary
            # tool errors, so the agent can continue its optimization campaign.
            subprocess.run([*literature_command(private), '--check'], check=True,
                           timeout=args.literature_timeout + 15)
        if not args.check_only:
            exit_code = launch(args, workspace, private, policy)
    finally:
        bridge.close()
        # The private journal remains recoverable on failure. Do not overwrite
        # public artifacts before comparing them with that journal.
        if sys.exc_info()[0] is None:
            report = audit(evaluator, workspace, private, args.agent, args.literature)
            (private / 'scored_trace.csv').write_text(evaluator.scored_trace('run'))
        evaluator.db.close()
    print(f"Finished: {report['evaluations_used']}/{args.budget} evaluations. Results: {workspace / 'results/agent_run'}")
    return exit_code
