"""Tool registry: holds tool definitions (JSON Schema) and dispatches calls."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Called before a tool that needs confirmation runs. Return True to allow it.
ConfirmFn = Callable[[str, Dict[str, Any]], bool]


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any]
    require_confirmation: bool = False

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolRegistry:
    tools: Dict[str, Tool] = field(default_factory=dict)
    confirm: Optional[ConfirmFn] = None
    max_result_chars: int = 8000

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self.tools[tool.name] = tool

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.to_openai_schema() for t in self.tools.values()]

    def call(self, name: str, arguments: Any, confirm: Optional[ConfirmFn] = None) -> str:
        """Run a tool and always return a string (errors go back to the model as text).

        `confirm` overrides the registry-wide confirmation handler for this call only.
        """
        tool = self.tools.get(name)
        if tool is None:
            return _err(f"Unknown tool '{name}'. Available: {list(self.tools)}")

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as e:
                return _err(f"Arguments are not valid JSON: {e}")
        if not isinstance(arguments, dict):
            return _err("Arguments must be a JSON object")

        if tool.require_confirmation:
            handler = confirm or self.confirm
            if handler is None or not handler(name, arguments):
                return _err("User rejected or no confirmation handler; the call was not executed.")

        logger.info("tool call %s %s", name, arguments)
        try:
            result = tool.func(**arguments)
        except TypeError as e:
            return _err(f"Bad arguments for {name}: {e}")
        except Exception as e:  # noqa: BLE001 - surface any failure to the model
            logger.exception("tool %s failed", name)
            return _err(f"{type(e).__name__}: {e}")

        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        if len(text) > self.max_result_chars:
            text = text[: self.max_result_chars] + f"... [truncated, {len(text)} chars total]"
        return text


def _err(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False)
