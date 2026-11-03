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
    """One tool backed by one HTTP endpoint, or by several (`routes`) picked by an argument.

    With routes, e.g. one API per site:

        route_by: site                 # the argument that picks the route (added as a required enum)
        routes:
          TW01: {label: Hsinchu, url: https://a/regen, headers: {Authorization: Bearer ${A_TOKEN}}}
          TW02: {label: Tainan,  url: https://b/regen}
        send_route_param: false        # true = also send `site` to the API (default: only used to pick)

    A route may override method, url, headers (merged over the top-level headers) and timeout.
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

        self.route_by: Optional[str] = spec.get("route_by")
        self.routes: Dict[str, Dict[str, Any]] = {str(k): (v or {}) for k, v in (spec.get("routes") or {}).items()}
        self.send_route_param: bool = bool(spec.get("send_route_param", False))
        if self.route_by or self.routes:
            self._check_routes()
        elif not self.url:
            raise ValueError(f"API '{self.name}' needs a url (or route_by + routes)")

        methods = {self.method} | {str(r.get("method", self.method)).upper() for r in self.routes.values()}
        self.require_confirmation: bool = bool(spec.get("require_confirmation", methods != {"GET"}))

    # ---- routes -------------------------------------------------------------------
    def _check_routes(self) -> None:
        if not self.route_by or not self.routes:
            raise ValueError(f"API '{self.name}': route_by and routes must be set together")
        if not _TOOL_NAME.match(self.route_by):
            raise ValueError(f"API '{self.name}': invalid route_by '{self.route_by}'")
        for key, route in self.routes.items():
            if not isinstance(route, dict) or not (route.get("url") or self.url):
                raise ValueError(f"API '{self.name}': route '{key}' needs a url")

    def _route_key(self, value: Any) -> Optional[str]:
        """Exact match first, then case-insensitive (the model may write tw02 for TW02)."""
        if value is None:
            return None
        value = str(value).strip()
        if value in self.routes:
            return value
        matches = [k for k in self.routes if k.lower() == value.lower()]
        return matches[0] if len(matches) == 1 else None

    def _choices(self) -> str:
        return "; ".join(f"{k}: {r['label']}" if r.get("label") else k for k, r in self.routes.items())

    def validate(self, args: Dict[str, Any]) -> Optional[str]:
        """Runs before the confirmation prompt, so nobody is asked to approve a call that can't be sent."""
        if not self.routes:
            return None
        value = args.get(self.route_by)
        if value is None or str(value).strip() == "":
            return (f"'{self.route_by}' is required. Ask the user which one; do not guess. "
                    f"Choices: {self._choices()}")
        if self._route_key(value) is None:
            return f"Unknown {self.route_by} '{value}'. Choices: {self._choices()}. Ask the user if unsure."
        return None

    def _schema(self) -> Dict[str, Any]:
        if not self.routes:
            return self.parameters
        schema = dict(self.parameters)
        props = dict(schema.get("properties") or {})
        given = dict(props.get(self.route_by) or {})
        extra = given.get("description", "")
        given.update(
            type="string",
            enum=list(self.routes),
            description=(f"Which {self.route_by}. {self._choices()}. Must be stated explicitly by the user; "
                         f"if they haven't, ask them and list these choices. Never guess." +
                         (f" {extra}" if extra else "")),
        )
        schema["properties"] = {self.route_by: given, **{k: v for k, v in props.items() if k != self.route_by}}
        schema["required"] = [self.route_by] + [r for r in (schema.get("required") or []) if r != self.route_by]
        schema.setdefault("type", "object")
        return schema

    # ---- call ---------------------------------------------------------------------
    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        args = dict(kwargs)
        method, url, headers, timeout = self.method, self.url, dict(self.headers), self.timeout
        if self.routes:
            problem = self.validate(args)
            if problem:
                raise ValueError(problem)
            key = self._route_key(args[self.route_by])
            route = self.routes[key]
            if self.send_route_param:
                args[self.route_by] = key
            else:
                args.pop(self.route_by)
            method = str(route.get("method", method)).upper()
            url = route.get("url") or url
            headers.update(route.get("headers") or {})
            timeout = float(route.get("timeout", timeout))
        url = _expand_env(url)

        def fill(m: re.Match) -> str:
            key = m.group(1)
            if key not in args:
                raise ValueError(f"Missing path parameter '{key}'")
            return requests.utils.quote(str(args.pop(key)), safe="")

        url = _PATH_PARAM.sub(fill, url)
        req: Dict[str, Any] = {"headers": _expand_env(headers), "timeout": timeout}
        if method in ("GET", "DELETE"):
            req["params"] = args
        else:
            req["json"] = args

        resp = self.session.request(method, url, **req)
        try:
            body: Any = resp.json()
        except ValueError:
            body = resp.text
        return {"status_code": resp.status_code, "ok": resp.ok, "body": body}

    def to_tool(self) -> Tool:
        kind = f"{self.method} API, one per {self.route_by}" if self.routes else f"{self.method} API"
        return Tool(
            name=self.name,
            description=f"[{kind}] {self.description}",
            parameters=self._schema(),
            func=self,
            require_confirmation=self.require_confirmation,
            validate=self.validate if self.routes else None,
        )


def register_api_tools(registry: ToolRegistry, endpoint_specs: List[Dict[str, Any]],
                       session: Optional[requests.Session] = None) -> None:
    session = session or requests.Session()
    for spec in endpoint_specs or []:
        registry.register(ApiEndpoint(spec, session).to_tool())
