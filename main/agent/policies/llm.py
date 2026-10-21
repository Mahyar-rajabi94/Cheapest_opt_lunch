"""LLM strategist (design §3, roadmap Phase 6).

One stateless call per round: system prompt = playbook (ROLE + skills),
user message = the StateReport JSON. The model must answer by calling a
single tool, `take_action`, whose input schema is the Action JSON schema,
so the output is always structured. On an invalid action it gets one retry
with the validation error; after that the call raises and the controller
falls back to a safe action (logged as "fallback").

Providers (switch with --provider):
    anthropic  ANTHROPIC_API_KEY, model from --model / $AGENT_ANTHROPIC_MODEL
    openai     OPENAI_API_KEY,    model from --model / $AGENT_OPENAI_MODEL

If VP_API_KEY is set, both providers go through the Vector AI Gateway
(https://proxy.vectorinstitute.ai) with that key instead of the provider keys.
The gateway reaches every model with either request format.

Every request/response is kept in a transcript (llm_transcript.jsonl in the
run folder) so runs can be audited and replayed.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from agent.playbook_loader import system_prompt, version
from agent.schema import Action, action_json_schema

DEFAULT_MODELS = {
    # Check these against your account's available models; override with
    # --model or the environment variables.
    "anthropic": os.environ.get("AGENT_ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    "openai": os.environ.get("AGENT_OPENAI_MODEL", "gpt-5"),
}

GATEWAY_URL = os.environ.get("VP_BASE_URL", "https://proxy.vectorinstitute.ai")

TOOL_NAME = "take_action"
TOOL_DESC = ("Submit exactly one Action for the next round of the optimization run. "
             "Fill `rationale` with 1-3 sentences citing numbers from the StateReport.")


# ------------------------------------------------------------------ providers
class AnthropicProvider:
    def __init__(self, model: str, max_tokens: int = 1024):
        import anthropic
        key = os.environ.get("VP_API_KEY")
        # The gateway reads only the Bearer header (auth_token, not api_key),
        # and the SDK appends /v1 itself.
        self.client = (anthropic.Anthropic(auth_token=key, base_url=GATEWAY_URL)
                       if key else anthropic.Anthropic())
        self.model = model
        self.max_tokens = max_tokens

    def call(self, system: str, messages: list[dict], schema: dict):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            # The playbook is identical every round: cache it.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=[{"name": TOOL_NAME, "description": TOOL_DESC, "input_schema": schema}],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=messages,
        )
        args = next((b.input for b in resp.content if b.type == "tool_use"), None)
        u = resp.usage
        usage = {"input": u.input_tokens, "output": u.output_tokens,
                 "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                 "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0}
        return args, usage, {"stop_reason": resp.stop_reason}

    @staticmethod
    def retry_messages(messages, args, error):
        # Echo the failed tool call and return the error as its result.
        return messages + [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_0", "name": TOOL_NAME, "input": args or {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_0", "is_error": True,
                 "content": f"Invalid action: {error}. Call {TOOL_NAME} again with a valid action."}]},
        ]


class OpenAIProvider:
    # Reasoning models (gpt-5) count hidden reasoning tokens against
    # max_completion_tokens; 1024 ran out before the tool call was emitted.
    def __init__(self, model: str, max_tokens: int = 16000):
        import openai
        key = os.environ.get("VP_API_KEY")
        self.client = (openai.OpenAI(api_key=key, base_url=f"{GATEWAY_URL}/v1")
                       if key else openai.OpenAI())
        self.model = model
        self.max_tokens = max_tokens

    def call(self, system: str, messages: list[dict], schema: dict):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + messages,
            tools=[{"type": "function", "function": {
                "name": TOOL_NAME, "description": TOOL_DESC, "parameters": schema}}],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
            max_completion_tokens=self.max_tokens,
        )
        msg = resp.choices[0].message
        args = None
        if msg.tool_calls:
            args = json.loads(msg.tool_calls[0].function.arguments)
        u = resp.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        usage = {"input": u.prompt_tokens, "output": u.completion_tokens,
                 "cache_read": cached, "cache_write": 0}
        return args, usage, {"finish_reason": resp.choices[0].finish_reason}

    @staticmethod
    def retry_messages(messages, args, error):
        return messages + [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_0", "type": "function",
                "function": {"name": TOOL_NAME, "arguments": json.dumps(args or {})}}]},
            {"role": "tool", "tool_call_id": "call_0",
             "content": f"Invalid action: {error}. Call {TOOL_NAME} again with a valid action."},
        ]


PROVIDERS = {"anthropic": AnthropicProvider, "openai": OpenAIProvider}


# -------------------------------------------------------------------- policy
class LLMPolicy:
    def __init__(self, provider: str = "anthropic", model: Optional[str] = None,
                 playbook: bool = True, max_calls: int = 200,
                 max_total_tokens: int = 3_000_000, client=None):
        self.provider_name = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.playbook = playbook
        self.name = f"llm_{provider}"
        self.max_calls = max_calls
        self.max_total_tokens = max_total_tokens
        # `client` lets tests inject a fake provider.
        self.provider = client or PROVIDERS[provider](self.model)
        self.schema = action_json_schema()
        self.transcript: list[dict] = []
        self.usage = {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
                      "retries": 0, "seconds": 0.0}
        self._system: Optional[str] = None

    def meta(self) -> dict:
        return {"llm_provider": self.provider_name, "llm_model": self.model,
                "playbook": self.playbook, "playbook_version": version() if self.playbook else None,
                "llm_usage": self.usage}

    def _user_message(self, report: dict) -> dict:
        return {"role": "user", "content":
                f"StateReport (round {report['round']}):\n```json\n"
                f"{json.dumps(report, default=str, separators=(',', ':'))}\n```\n"
                f"Choose the next action by calling {TOOL_NAME}."}

    def decide(self, report: dict) -> Action:
        if self._system is None:
            self._system = system_prompt(report["problem"]["allowed_methods"],
                                         with_playbook=self.playbook)
        if self.usage["calls"] >= self.max_calls:
            raise RuntimeError("LLM call budget exhausted")
        if self.usage["input"] + self.usage["output"] >= self.max_total_tokens:
            raise RuntimeError("LLM token budget exhausted")

        messages = [self._user_message(report)]
        last_err = None
        for attempt in range(2):
            t = time.perf_counter()
            args, usage, info = self.provider.call(self._system, messages, self.schema)
            dt = time.perf_counter() - t
            self.usage["calls"] += 1
            self.usage["seconds"] += dt
            for k in ("input", "output", "cache_read", "cache_write"):
                self.usage[k] += usage.get(k, 0)
            entry = {"round": report["round"], "attempt": attempt, "request": messages[-1],
                     "response": args, "usage": usage, "info": info, "seconds": round(dt, 3)}
            try:
                if args is None:
                    raise ValueError("no tool call in response")
                action = Action(**args)
                entry["valid"] = True
                self.transcript.append(entry)
                return action
            except (ValidationError, ValueError, TypeError) as e:
                last_err = str(e)[:500]
                entry["valid"] = False
                entry["error"] = last_err
                self.transcript.append(entry)
                self.usage["retries"] += 1
                messages = self.provider.retry_messages(messages, args, last_err)
        raise ValueError(f"LLM produced no valid action after retry: {last_err}")

    def save_transcript(self, path) -> None:
        with open(Path(path), "w") as f:
            if self._system is not None:
                f.write(json.dumps({"system_prompt": self._system,
                                    "playbook_version": version()}) + "\n")
            for e in self.transcript:
                f.write(json.dumps(e, default=str) + "\n")
