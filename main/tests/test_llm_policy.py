import json

try:
    import httpx2 as httpx  # newer SDKs ship on httpx2
except ImportError:  # pragma: no cover
    import httpx
import pytest

from agent.controller import Controller
from agent.dev_tasks import get_task
from agent.intake import Intake
from agent.policies.llm import AnthropicProvider, LLMPolicy, OpenAIProvider
from agent.policies.scripted import ScriptedPolicy


def ctl_for(policy, task="dev_f8_d5", budget=300):
    t = get_task(task)
    intake = Intake(dim=t.dim, bounds=t.bounds.tolist(), budget_evals=budget,
                    eval_cost="cheap", noise="deterministic")
    return Controller(t, intake, policy, seed=0)


class FakeProvider:
    """Plays the LLM by delegating to the scripted policy; optionally
    returns an invalid action first to exercise the retry path."""

    def __init__(self, invalid_first=False, always_invalid=False):
        self.inner = ScriptedPolicy()
        self.invalid_first = invalid_first
        self.always_invalid = always_invalid
        self.calls = []

    def call(self, system, messages, schema):
        self.calls.append((system, messages))
        if self.always_invalid or (self.invalid_first and len(self.calls) == 1):
            return {"kind": "race", "candidates": ["cma_es"]}, {"input": 10, "output": 5}, {}
        report = json.loads(messages[0]["content"].split("```json\n")[1].split("\n```")[0])
        return self.inner.decide(report).model_dump(), {"input": 100, "output": 20}, {}

    @staticmethod
    def retry_messages(messages, args, error):
        return messages + [{"role": "user", "content": f"error {error}"}]


def test_llm_policy_end_to_end_with_fake_provider(tmp_path):
    fake = FakeProvider(invalid_first=True)
    pol = LLMPolicy(provider="anthropic", model="fake", client=fake)
    ctl = ctl_for(pol)
    ctl.run()
    ctl.save(tmp_path)
    assert pol.usage["retries"] == 1
    assert all(not e["source"].startswith("fallback") for e in ctl.log.entries)
    lines = (tmp_path / "llm_transcript.jsonl").read_text().splitlines()
    assert "system_prompt" in json.loads(lines[0])
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["playbook_version"] and meta["llm_usage"]["calls"] == len(fake.calls)
    # the system prompt is the playbook, and the report never names the task
    system, messages = fake.calls[0]
    assert "# Skills" in system
    assert "dev_f8" not in json.dumps(messages)


def test_llm_policy_falls_back_after_failed_retry():
    pol = LLMPolicy(provider="openai", model="fake", client=FakeProvider(always_invalid=True))
    ctl = ctl_for(pol, budget=150)
    ctl.max_rounds = 2
    ctl.run()
    assert all(e["source"].startswith("fallback") for e in ctl.log.entries)


def test_no_playbook_ablation_sends_role_only():
    fake = FakeProvider()
    pol = LLMPolicy(provider="anthropic", model="fake", client=fake, playbook=False)
    ctl = ctl_for(pol, budget=100)
    ctl.max_rounds = 1
    ctl.run()
    assert "# Skills" not in fake.calls[0][0]


# ---- real SDKs against a mocked HTTP layer (no network, no key needed)
ACTION = {"kind": "probe", "rationale": "no data yet"}


def test_anthropic_provider_request_and_parse(monkeypatch):
    seen = {}

    def handler(req: httpx.Request):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "tool_use", "id": "tu_1", "name": "take_action", "input": ACTION}],
            "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 50,
                      "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0}})

    import anthropic
    p = AnthropicProvider.__new__(AnthropicProvider)
    p.client = anthropic.Anthropic(api_key="x", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    p.model, p.max_tokens = "claude-test", 512
    pol = LLMPolicy(provider="anthropic", model="claude-test", client=p)
    a = pol.decide(ctl_for(pol).report())
    assert a.kind == "probe"
    b = seen["body"]
    assert b["tool_choice"] == {"type": "tool", "name": "take_action"}
    assert b["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert b["tools"][0]["input_schema"]["properties"]["kind"]
    assert pol.usage["cache_read"] == 900


def test_openai_provider_request_and_parse():
    seen = {}

    def handler(req: httpx.Request):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "t1", "type": "function", "function": {
                    "name": "take_action", "arguments": json.dumps(ACTION)}}]}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 40, "total_tokens": 1040}})

    import openai
    p = OpenAIProvider.__new__(OpenAIProvider)
    p.client = openai.OpenAI(api_key="x", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    p.model, p.max_tokens = "gpt-test", 512
    pol = LLMPolicy(provider="openai", model="gpt-test", client=p)
    a = pol.decide(ctl_for(pol).report())
    assert a.kind == "probe"
    b = seen["body"]
    assert b["tool_choice"]["function"]["name"] == "take_action"
    assert b["messages"][0]["role"] == "system"
