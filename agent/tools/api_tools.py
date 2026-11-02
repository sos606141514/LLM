"""Turn each API endpoint declared in config into its own tool.

Only endpoints listed in the config can be called, so the model cannot reach arbitrary URLs.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import requests

from .registry import Tool, ToolRegistry

_PATH_PARAM = re.compile(r"\{(\w+)\}")
_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_ENV_VAR = re.compile(r"\$\{(\w+)\}")


def _expand_env(value: Any) -> Any:
    """Replace ${VAR} with environment variables (unset -> empty string)."""
    if isinstance(value, str):
        return _ENV_VAR.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


class ApiEndpoint:
    def __init__(self, spec: Dict[str, Any], session: Optional[requests.Session] = None):
        self.name: str = spec["name"]
        if not _TOOL_NAME.match(self.name):
            raise ValueError(f"Invalid endpoint name '{self.name}' (letters, digits, _ and - only)")
        self.description: str = spec.get("description", "")
        self.method: str = spec.get("method", "POST").upper()
        self.url: str = spec["url"]
        self.headers: Dict[str, str] = spec.get("headers", {}) or {}
        self.timeout: float = float(spec.get("timeout", 30))
        self.require_confirmation: bool = bool(spec.get("require_confirmation", self.method != "GET"))
        self.parameters: Dict[str, Any] = spec.get("parameters") or {"type": "object", "properties": {}}
        self.session = session or requests.Session()

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        args = dict(kwargs)
        url = _expand_env(self.url)

        def fill(m: re.Match) -> str:
            key = m.group(1)
            if key not in args:
                raise ValueError(f"Missing path parameter '{key}'")
            return requests.utils.quote(str(args.pop(key)), safe="")

        url = _PATH_PARAM.sub(fill, url)
        req: Dict[str, Any] = {"headers": _expand_env(self.headers), "timeout": self.timeout}
        if self.method in ("GET", "DELETE"):
            req["params"] = args
        else:
            req["json"] = args

        resp = self.session.request(self.method, url, **req)
        try:
            body: Any = resp.json()
        except ValueError:
            body = resp.text
        return {"status_code": resp.status_code, "ok": resp.ok, "body": body}

    def to_tool(self) -> Tool:
        return Tool(
            name=self.name,
            description=f"[{self.method} API] {self.description}",
            parameters=self.parameters,
            func=self,
            require_confirmation=self.require_confirmation,
        )


def register_api_tools(registry: ToolRegistry, endpoint_specs: List[Dict[str, Any]],
                       session: Optional[requests.Session] = None) -> None:
    session = session or requests.Session()
    for spec in endpoint_specs or []:
        registry.register(ApiEndpoint(spec, session).to_tool())
