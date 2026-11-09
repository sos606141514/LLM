"""Turn each API endpoint declared in config into its own tool.

Only endpoints listed in the config can be called, so the model cannot reach arbitrary URLs.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import requests

from .api_hooks import Request, Step, build_steps
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
    """A generic tool for one HTTP API from the config: builds the request and sends it.

    It knows nothing about any particular API. Optional per-API behaviour (picking the endpoint per
    site, combining fields, ...) comes from steps in api_hooks.py, switched on by keys in the config.
    Argument checks are done by validation.py before the call reaches this class.
    """

    def __init__(self, spec: Dict[str, Any], session: Optional[requests.Session] = None):
        self.name: str = spec["name"]
        if not _TOOL_NAME.match(self.name):
            raise ValueError(f"Invalid endpoint name '{self.name}' (letters, digits, _ and - only)")
        self.description: str = spec.get("description", "")
        self.method: str = spec.get("method", "POST").upper()
        self.url: Optional[str] = spec.get("url")
        self.headers: Dict[str, str] = spec.get("headers", {}) or {}
        self.timeout: float = float(spec.get("timeout", 30))
        self.parameters: Dict[str, Any] = spec.get("parameters") or {"type": "object", "properties": {}}
        self.session = session or requests.Session()

        self.steps: List[Step] = build_steps(spec)
        if not self.url and not any(s.provides_url() for s in self.steps):
            raise ValueError(f"API '{self.name}' needs a url (or route_by + routes)")
        methods = {self.method}.union(*(s.methods(self.method) for s in self.steps))
        self.require_confirmation: bool = bool(spec.get("require_confirmation", methods != {"GET"}))

    # ---- hooks handed to the tool (run by validation.py before confirmation) -------
    def schema(self) -> Dict[str, Any]:
        params = self.parameters
        for step in self.steps:
            params = step.schema(params)
        return params

    def prepare(self, args: Dict[str, Any]) -> Dict[str, Any]:
        for step in self.steps:
            args = step.prepare(args)
        return args

    def validate(self, args: Dict[str, Any]) -> Optional[str]:
        for step in self.steps:
            problem = step.validate(args)
            if problem:
                return problem
        return None

    # ---- call ---------------------------------------------------------------------
    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        req = Request(method=self.method, url=self.url, headers=dict(self.headers), timeout=self.timeout,
                      args=dict(kwargs), checked=dict(kwargs))
        for step in self.steps:
            req = step.apply(req)
        return self.send(req)

    def send(self, req: Request) -> Dict[str, Any]:
        args = dict(req.args)
        url = _expand_env(req.url)

        def fill(m: re.Match) -> str:
            key = m.group(1)
            if key not in args:
                raise ValueError(f"Missing path parameter '{key}'")
            return requests.utils.quote(str(args.pop(key)), safe="")

        url = _PATH_PARAM.sub(fill, url)
        kw: Dict[str, Any] = {"headers": _expand_env(req.headers), "timeout": req.timeout}
        if req.method in ("GET", "DELETE"):
            kw["params"] = args
        else:
            kw["json"] = args

        resp = self.session.request(req.method, url, **kw)
        try:
            body: Any = resp.json()
        except ValueError:
            body = resp.text
        return {"status_code": resp.status_code, "ok": resp.ok, "body": body}

    def to_tool(self) -> Tool:
        kind = ", ".join([f"{self.method} API"] + [s.label() for s in self.steps if s.label()])
        return Tool(
            name=self.name,
            description=f"[{kind}] {self.description}",
            parameters=self.schema(),
            func=self,
            require_confirmation=self.require_confirmation,
            prepare=self.prepare if self.steps else None,
            validate=self.validate if self.steps else None,
        )


def register_api_tools(registry: ToolRegistry, endpoint_specs: List[Dict[str, Any]],
                       session: Optional[requests.Session] = None) -> None:
    session = session or requests.Session()
    for spec in endpoint_specs or []:
        registry.register(ApiEndpoint(spec, session).to_tool())
