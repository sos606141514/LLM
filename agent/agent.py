"""The function-calling loop: ask the model, run the tools it asks for, feed results back."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient
from .tools.registry import ConfirmFn, ToolRegistry

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are an internal assistant with tools for querying MongoDB and calling internal APIs.
Rules:
- Use tools to get facts; never invent data.
- If you don't know a collection's fields, call mongo_sample_schema first.
- Keep queries small (use limit, projection, count, aggregate).
- Before calling an API that changes something, make sure the arguments are correct.
- Answer in the user's language."""


class Agent:
    def __init__(self, llm: LLMClient, registry: ToolRegistry, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 max_steps: int = 8):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        # Tool calls made during the last run(): [{"name", "arguments", "result"}]
        self.last_trace: List[Dict[str, Any]] = []
        # SkillLibrary set by build_agent (None when the agent is built by hand)
        self.skills = None

    def reset(self) -> None:
        self.messages = self.messages[:1]

    def run(self, user_input: str, confirm: Optional[ConfirmFn] = None) -> str:
        """Answer one user message. `confirm` overrides the registry's confirmation handler."""
        self.last_trace = []
        self.messages.append({"role": "user", "content": user_input})
        tools = self.registry.schemas()

        for step in range(self.max_steps):
            resp = self.llm.chat(self.messages, tools)
            if not resp.tool_calls:
                answer = resp.content or ""
                self.messages.append({"role": "assistant", "content": answer})
                return answer

            self.messages.append({
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [{
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.name,
                                 "arguments": tc.arguments if isinstance(tc.arguments, str)
                                 else json.dumps(tc.arguments, ensure_ascii=False)},
                } for tc in resp.tool_calls],
            })
            for tc in resp.tool_calls:
                result = self.registry.call(tc.name, tc.arguments, confirm=confirm)
                self.last_trace.append({"name": tc.name, "arguments": tc.arguments, "result": result})
                logger.debug("step %d %s -> %s", step, tc.name, result[:300])
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result})

        msg = f"Stopped after {self.max_steps} tool steps without a final answer."
        self.messages.append({"role": "assistant", "content": msg})
        return msg
