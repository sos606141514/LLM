"""Client for the internal LLM API.

Two modes:
- native: the API speaks the OpenAI-compatible /chat/completions format with `tools`
  (vLLM, TGI, Ollama, Azure, most internal gateways).
- prompt: the API has no function-calling support; the tool list goes into the system prompt
  and the model replies with JSON that we parse.

If your internal API uses a different request/response shape, subclass LLMClient and override
`_post` (build the request) and `_parse_native` (read the response).
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Any  # dict or JSON string


@dataclass
class LLMResponse:
    content: Optional[str]
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Any = None


PROMPT_MODE_INSTRUCTIONS = """You can call tools. Available tools (JSON Schema):
{tools}

Reply with ONLY one JSON object, no other text:
- To call tools: {{"tool_calls": [{{"name": "<tool name>", "arguments": {{...}}}}]}}
- To give the final answer: {{"final": "<answer to the user>"}}
Tool results come back in messages that start with TOOL_RESULT."""


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None,
                 tool_mode: str = "native", extra_headers: Optional[Dict[str, str]] = None,
                 temperature: float = 0.0, timeout: float = 120, verify_ssl: Any = True,
                 session: Optional[requests.Session] = None):
        if tool_mode not in ("native", "prompt"):
            raise ValueError("tool_mode must be 'native' or 'prompt'")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.tool_mode = tool_mode
        self.temperature = temperature
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.headers = {"Content-Type": "application/json", **(extra_headers or {})}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.session = session or requests.Session()

    # ---- public ----------------------------------------------------------------
    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> LLMResponse:
        if self.tool_mode == "native":
            data = self._post({"messages": messages, "tools": tools, "tool_choice": "auto"} if tools
                              else {"messages": messages})
            return self._parse_native(data)
        data = self._post({"messages": self._to_prompt_messages(messages, tools)})
        return self._parse_prompt(data)

    # ---- transport (override for non-OpenAI APIs) -------------------------------
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = {"model": self.model, "temperature": self.temperature, **payload}
        resp = self.session.post(self.url, json=body, headers=self.headers,
                                 timeout=self.timeout, verify=self.verify_ssl)
        if resp.status_code >= 400:
            raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    def _parse_native(self, data: Dict[str, Any]) -> LLMResponse:
        msg = data["choices"][0]["message"]
        calls = [
            ToolCall(id=tc.get("id") or _new_id(), name=tc["function"]["name"],
                     arguments=tc["function"].get("arguments") or "{}")
            for tc in (msg.get("tool_calls") or [])
        ]
        return LLMResponse(content=msg.get("content"), tool_calls=calls, raw=data)

    # ---- prompt mode -------------------------------------------------------------
    def _to_prompt_messages(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
        """Rewrite tool-call history into plain user/assistant text for APIs without tool support."""
        tool_text = json.dumps([t["function"] for t in tools], ensure_ascii=False, indent=1)
        out: List[Dict[str, str]] = []
        sys_added = False
        for m in messages:
            role = m["role"]
            if role == "system" and not sys_added:
                out.append({"role": "system",
                            "content": m["content"] + "\n\n" + PROMPT_MODE_INSTRUCTIONS.format(tools=tool_text)})
                sys_added = True
            elif role == "assistant" and m.get("tool_calls"):
                calls = [{"name": c["function"]["name"], "arguments": _maybe_json(c["function"]["arguments"])}
                         for c in m["tool_calls"]]
                out.append({"role": "assistant", "content": json.dumps({"tool_calls": calls}, ensure_ascii=False)})
            elif role == "tool":
                out.append({"role": "user", "content": f"TOOL_RESULT ({m.get('name', '')}):\n{m['content']}"})
            else:
                out.append({"role": role, "content": m.get("content") or ""})
        if not sys_added:
            out.insert(0, {"role": "system", "content": PROMPT_MODE_INSTRUCTIONS.format(tools=tool_text)})
        return out

    def _parse_prompt(self, data: Dict[str, Any]) -> LLMResponse:
        text = data["choices"][0]["message"].get("content") or ""
        obj = _extract_json(text)
        if isinstance(obj, dict) and obj.get("tool_calls"):
            calls = [ToolCall(id=_new_id(), name=c["name"], arguments=c.get("arguments", {}))
                     for c in obj["tool_calls"]]
            return LLMResponse(content=None, tool_calls=calls, raw=data)
        if isinstance(obj, dict) and "final" in obj:
            return LLMResponse(content=str(obj["final"]), raw=data)
        return LLMResponse(content=text, raw=data)  # model ignored the format: treat as final answer


def _new_id() -> str:
    return "call_" + uuid.uuid4().hex[:12]


def _maybe_json(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def _extract_json(text: str) -> Any:
    """Find the first JSON object in the text (handles ```json fences and surrounding prose)."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fence.group(1)] if fence else []
    start = text.find("{")
    if start != -1:
        candidates.append(text[start:])
    decoder = json.JSONDecoder()
    for c in candidates:
        try:
            obj, _ = decoder.raw_decode(c)
            return obj
        except json.JSONDecodeError:
            continue
    return None
