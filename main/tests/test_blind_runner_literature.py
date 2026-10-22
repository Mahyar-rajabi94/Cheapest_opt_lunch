"""Runner checks without inference, credentials, network, or real objectives."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.blind_runner import runner, valency
from src.blind_runner.evaluator import Evaluator
from src.blind_runner.execution import environment, permissions


class RunnerLiteratureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace, self.private = self.root / 'workspace', self.root / 'private'
        self.workspace.mkdir()
        self.private.mkdir()
        (self.workspace / 'AGENTS.md').write_text('Canonical optimization instructions')
        (self.private / 'literature').mkdir()
        (self.private / 'literature/tools.json').write_text(json.dumps({'tools': [
            {'name': 'semantic_search_papers'}, {'name': 'get_paper_by_id'}]}))
        self.args = SimpleNamespace(agent='codex', codex=Path('/bin/codex'), claude=Path('/bin/claude'),
            model='test', effort='max', literature='none', literature_timeout=30)
        self.policy = {'permissions': permissions(self.workspace, []), 'python': sys.executable}

    def tearDown(self):
        self.temp.cleanup()

    def test_disabled_instructions_and_tools_are_unchanged(self):
        expected = ('Canonical optimization instructions\nAll campaign execution and file access use the workspace MCP tools. '
                    'Read PUBLIC_EVALUATOR.md. Put scripts in work/; use .venv/bin/python. '
                    'Use write_stdin to keep optimizer processes alive between blocks.\n')
        self.assertEqual(runner.instructions(self.args, self.workspace), expected)
        runner.codex_command(self.args, self.workspace, self.private, self.policy)
        config = json.loads((self.private / 'effective_overrides.json').read_text())
        self.assertEqual(set(config['mcp_servers']), {'workspace'})
        self.assertEqual(config['mcp_servers']['workspace']['enabled_tools'], runner.TOOLS)
        self.assertFalse(config['permissions.blind.network.enabled'])
        self.assertNotIn('service_tier', config)
        self.args.agent = 'claude'
        command, _ = runner.claude_command(self.args, self.workspace, self.private)
        self.assertEqual(command[command.index('--allowedTools') + 1],
                         'mcp__isolated__run_command,mcp__isolated__write_stdin')

    def test_enabled_tools_are_explicit_for_both_clients(self):
        self.args.literature = 'valency'
        with patch.dict(os.environ, {valency.TOKEN_ENV: 'TEST_SECRET'}):
            runner.codex_command(self.args, self.workspace, self.private, self.policy)
            config = json.loads((self.private / 'effective_overrides.json').read_text())
            self.assertEqual(set(config['mcp_servers']), {'workspace', 'valency'})
            self.assertFalse(config['permissions.blind.network.enabled'])
            self.assertNotIn('TEST_SECRET', json.dumps(config))
            self.assertEqual(config['mcp_servers']['valency']['env_vars'], ['VALENCY_API_TOKEN'])
            self.args.agent = 'claude'
            command, _ = runner.claude_command(self.args, self.workspace, self.private)
            allowed = command[command.index('--allowedTools') + 1]
            self.assertIn('mcp__valency__get_paper_by_id', allowed)
            self.assertNotIn('submit_feedback', allowed)
            self.assertNotIn('TEST_SECRET', (self.private / 'claude_mcp.json').read_text())
            self.assertNotIn(valency.TOKEN_ENV, environment(self.workspace, sys.executable))
        self.assertIn(valency.INSTRUCTIONS, runner.instructions(self.args, self.workspace))
        self.assertEqual((self.workspace / 'AGENTS.md').read_text(), 'Canonical optimization instructions')

    def test_fast_tier_preserves_reasoning_and_sandbox(self):
        self.args.fast = True
        self.args.literature = 'valency'
        command, _ = runner.codex_command(self.args, self.workspace, self.private, self.policy)
        config = json.loads((self.private / 'effective_overrides.json').read_text())
        self.assertEqual(config['service_tier'], 'fast')
        self.assertTrue(config['features.fast_mode'])
        self.assertEqual(config['model_reasoning_effort'], 'max')
        self.assertEqual(command[command.index('-m') + 1], 'test')
        self.assertFalse(config['permissions.blind.network.enabled'])
        self.assertEqual(set(config['mcp_servers']), {'workspace', 'valency'})

    def test_audit_accepts_only_enabled_retrieval_and_preserves_evaluator_checks(self):
        class Task:
            dim, bounds = 1, [[0., 1.]]
        evaluator = Evaluator({'budget': 2, 'campaigns': [{'id': 'run'}]},
                              self.private / 'state.sqlite', lambda _: Task())
        directory = self.workspace / 'results/agent_run'
        directory.mkdir(parents=True)
        for name, text in evaluator.artifacts('run').items():
            (directory / name).write_text(text)
        events = self.private / 'cli_events.jsonl'
        def codex(tool):
            events.write_text(json.dumps({'type': 'item.completed',
                'item': {'type': 'mcp_tool_call', 'server': 'valency', 'tool': tool}}) + '\n')
        try:
            codex('semantic_search_papers')
            with self.assertRaisesRegex(RuntimeError, 'Integrity audit'):
                runner.audit(evaluator, self.workspace, self.private, 'codex')
            report = runner.audit(evaluator, self.workspace, self.private, 'codex', 'valency')
            self.assertTrue(report['only_allowed_tools'])
            self.assertEqual(report['literature_tool_calls'], 1)
            self.assertEqual(report['evaluations_used'], 0)
            codex('submit_feedback')
            with self.assertRaisesRegex(RuntimeError, 'Integrity audit'):
                runner.audit(evaluator, self.workspace, self.private, 'codex', 'valency')
            events.write_text(json.dumps({'type': 'assistant', 'message': {'content': [
                {'type': 'tool_use', 'name': 'mcp__valency__get_paper_by_id'}]}}) + '\n')
            self.assertTrue(runner.audit(evaluator, self.workspace, self.private,
                                        'claude', 'valency')['only_allowed_tools'])
            (directory / 'trace.csv').write_text('forged trace')
            with self.assertRaisesRegex(RuntimeError, 'Integrity audit'):
                runner.audit(evaluator, self.workspace, self.private, 'claude', 'valency')
        finally:
            evaluator.db.close()

    def test_token_file_and_missing_token(self):
        with patch.dict(os.environ, {valency.TOKEN_ENV: ''}):
            with self.assertRaises(ValueError):
                valency.token({})
            path = self.private / 'token'
            path.write_text('private-token\n')
            self.assertEqual(valency.token({'token_file': path}), 'private-token')


if __name__ == '__main__':
    unittest.main()
