"""Trusted MCP executor. Every agent command passes through the native sandbox.

The model sees only run_command/write_stdin. This process owns the policy;
neither the policy nor this source is readable or writable by agent commands.
"""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


def toml(value):
    if isinstance(value, dict):
        return '{' + ', '.join(json.dumps(k) + '=' + toml(v) for k, v in value.items()) + '}'
    return json.dumps(value)


def overrides(config):
    return [part for key, value in config.items() for part in ('-c', key + '=' + toml(value))]


def environment(workspace, python):
    return {
        'PATH': str(Path(python).parent) + ':/usr/bin:/bin:/usr/sbin:/sbin',
        'TMPDIR': str(workspace / '.tmp'), 'HOME': str(workspace / '.home'),
        'XDG_CACHE_HOME': str(workspace / '.tmp/cache'),
        'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'MPLCONFIGDIR': str(workspace / '.tmp/matplotlib'),
    }


def permissions(workspace, runtime_roots):
    files = {':root': 'deny', ':minimal': 'read', ':tmpdir': 'deny', ':slash_tmp': 'deny'}
    files.update({str(path): 'read' for path in runtime_roots})
    files[str(workspace)] = 'read'
    # Writable subdirectories have immutable parents. An agent cannot replace
    # the bridge/results directories with symlinks and redirect trusted writes.
    for path in ['work', '.tmp', '.bridge/requests', 'results/agent_run']:
        files[str(workspace / path)] = 'write'
    for path in ['trace.csv', 'meta.json']:
        files[str(workspace / 'results/agent_run' / path)] = 'read'
    return {
        'default_permissions': 'blind',
        'permissions.blind.filesystem': files,
        'permissions.blind.network.enabled': False,
        'approval_policy': 'never', 'allow_login_shell': False,
        'shell_environment_policy.inherit': 'none',
        'web_search': 'disabled', 'project_doc_max_bytes': 0,
        'project_root_markers': ['PUBLIC_EVALUATOR.md'],
        'memories.use_memories': False, 'memories.generate_memories': False,
        'mcp_servers': {}, 'features.skip_host_skill_discovery': True,
    }


def command(policy, script):
    return [policy['codex'], '--no-daemon', *overrides(policy['permissions']),
            'sandbox', '-P', 'blind', '-C', policy['workspace'], '--', '/bin/bash', '-c', script]


def run(policy, script, timeout=60):
    """Run preflight probes using exactly the executor's command boundary."""
    return subprocess.run(command(policy, script), cwd=policy['workspace'],
                          env=environment(Path(policy['workspace']), policy['python']),
                          capture_output=True, text=True, timeout=timeout)


TOOLS = [
    {'name': 'run_command', 'description':
     'Run a command in your isolated workspace. Use .venv/bin/python. Put scripts in work/. '
     'Network access is disabled. Long-running commands return a session_id.',
     'inputSchema': {'type': 'object', 'properties': {'command': {'type': 'string'},
        'yield_time_ms': {'type': 'integer', 'minimum': 1, 'maximum': 10000}},
        'required': ['command'], 'additionalProperties': False}},
    {'name': 'write_stdin', 'description':
     'Read output or send input to an existing command. Retain live optimizer state between blocks.',
     'inputSchema': {'type': 'object', 'properties': {'session_id': {'type': 'string'},
        'chars': {'type': 'string'},
        'yield_time_ms': {'type': 'integer', 'minimum': 1, 'maximum': 10000}},
        'required': ['session_id'], 'additionalProperties': False}},
]


class Executor:
    def __init__(self, policy):
        self.policy = policy
        self.sessions = {}

    def call(self, name, args):
        if name == 'run_command':
            script = args['command']
            if not isinstance(script, str) or len(script) > 250000:
                raise ValueError('Invalid command')
            process = subprocess.Popen(command(self.policy, script),
                cwd=self.policy['workspace'],
                env=environment(Path(self.policy['workspace']), self.policy['python']),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True)
            key = str(len(self.sessions) + 1)
            self.sessions[key] = process
        elif name == 'write_stdin':
            key = str(args['session_id'])
            process = self.sessions[key]
            if args.get('chars'):
                process.stdin.write(args['chars'].encode())
                process.stdin.flush()
        else:
            raise ValueError('Unknown tool')
        deadline = time.monotonic() + min(max(args.get('yield_time_ms', 1000), 1), 10000) / 1000
        chunks = []
        while time.monotonic() < deadline:
            ready, _, _ = select.select([process.stdout], [], [], .05)
            if ready:
                data = os.read(process.stdout.fileno(), 65536)
                if not data:
                    break
                chunks.append(data)
            elif process.poll() is not None:
                break
        result = {'output': b''.join(chunks).decode(errors='replace')[-40000:],
                  'exit_code': process.poll()}
        if process.poll() is None:
            result['session_id'] = key
        return {'content': [{'type': 'text', 'text': json.dumps(result)}]}

    def close(self):
        for process in self.sessions.values():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in self.sessions.values():
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main():
    executor = Executor(json.loads(Path(sys.argv[1]).read_text()))
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if 'id' not in request:
                continue
            try:
                method = request['method']
                if method == 'initialize':
                    result = {'protocolVersion': request['params']['protocolVersion'],
                              'capabilities': {'tools': {}},
                              'serverInfo': {'name': 'isolated-workspace', 'version': '1.0'}}
                elif method == 'tools/list':
                    result = {'tools': TOOLS}
                elif method == 'tools/call':
                    result = executor.call(request['params']['name'], request['params'].get('arguments', {}))
                elif method in ('resources/list', 'resources/templates/list'):
                    result = {'resources' if method == 'resources/list' else 'resourceTemplates': []}
                elif method == 'ping':
                    result = {}
                else:
                    raise ValueError('Unsupported RPC')
                response = {'jsonrpc': '2.0', 'id': request['id'], 'result': result}
            except Exception as exc:
                response = {'jsonrpc': '2.0', 'id': request['id'],
                            'error': {'code': -32603, 'message': str(exc)}}
            print(json.dumps(response), flush=True)
    finally:
        executor.close()


if __name__ == '__main__':
    main()
