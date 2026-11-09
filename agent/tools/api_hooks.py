"""Optional processing steps for API tools, switched on per API by keys in its config.

The API tool itself (api_tools.ApiEndpoint) only builds and sends the request. Anything specific to
one API lives here as a step, and only APIs whose config uses a step's keys get it:

    route_by / routes / send_route_param   RouteStep: pick the endpoint by an argument (e.g. per site)
    combine                                CombineStep: send one field built from several arguments

To add a new kind of step: subclass Step, implement `from_spec`, and add it to STEP_TYPES.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

_FIELD = re.compile(r"\{(\w+)\}")
_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@dataclass
class Request:
    """What will be sent. Steps change it; `checked` holds the model's (already checked) arguments."""
    method: str
    url: Optional[str]
    headers: Dict[str, Any]
    timeout: float
    args: Dict[str, Any]
    checked: Dict[str, Any] = field(default_factory=dict)


class Step:
    """Base class: every hook is optional, the defaults change nothing."""

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> Optional["Step"]:
        """Return a step when the API's config uses this step's keys, else None."""
        raise NotImplementedError

    def schema(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Change the parameters the model sees."""
        return parameters

    def prepare(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up the model's arguments before they are checked."""
        return args

    def validate(self, args: Dict[str, Any]) -> Optional[str]:
        """Extra check before confirmation; return an error message to reject the call."""
        return None

    def apply(self, req: Request) -> Request:
        """Change the request just before it is sent."""
        return req

    def provides_url(self) -> bool:
        """True when the step supplies the URL, so the API itself may leave `url` out."""
        return False

    def methods(self, default: str) -> Set[str]:
        """HTTP methods this step may send with (used for the require_confirmation default)."""
        return {default}

    def label(self) -> str:
        """Extra text for the tool description, e.g. 'one per site'."""
        return ""


class RouteStep(Step):
    """Same action, a different endpoint per value of one argument (e.g. one API per site).

        route_by: site
        routes:
          TW01: {label: Hsinchu, url: https://a/regen, headers: {Authorization: Bearer ${A_TOKEN}}}
          TW02: {label: Tainan,  url: https://b/regen}
        send_route_param: false   # true = also send `site` (default: it only picks the endpoint)

    A route may override method, url, headers (merged over the API's headers) and timeout.
    """

    def __init__(self, name: str, route_by: str, routes: Dict[str, Dict[str, Any]], send: bool, base_url):
        self.name, self.route_by, self.routes, self.send = name, route_by, routes, send
        if not _NAME.match(route_by):
            raise ValueError(f"API '{name}': invalid route_by '{route_by}'")
        for key, route in routes.items():
            if not isinstance(route, dict) or not (route.get("url") or base_url):
                raise ValueError(f"API '{name}': route '{key}' needs a url")

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> Optional["RouteStep"]:
        route_by, routes = spec.get("route_by"), spec.get("routes")
        if not route_by and not routes:
            return None
        if not route_by or not routes:
            raise ValueError(f"API '{spec.get('name')}': route_by and routes must be set together")
        return cls(spec.get("name"), str(route_by), {str(k): (v or {}) for k, v in routes.items()},
                   bool(spec.get("send_route_param", False)), spec.get("url"))

    def key(self, value: Any) -> Optional[str]:
        """Exact match first, then case-insensitive (the model may write tw02 for TW02)."""
        if value is None:
            return None
        value = str(value).strip()
        if value in self.routes:
            return value
        matches = [k for k in self.routes if k.lower() == value.lower()]
        return matches[0] if len(matches) == 1 else None

    def choices(self) -> str:
        return "; ".join(f"{k}: {r['label']}" if r.get("label") else k for k, r in self.routes.items())

    def schema(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        schema = dict(parameters)
        props = dict(schema.get("properties") or {})
        given = dict(props.get(self.route_by) or {})
        extra = given.get("description", "")
        given.update(
            type="string",
            enum=list(self.routes),
            description=(f"Which {self.route_by}. {self.choices()}. Must be stated explicitly by the user; "
                         f"if they haven't, ask them and list these choices. Never guess." +
                         (f" {extra}" if extra else "")),
        )
        schema["properties"] = {self.route_by: given, **{k: v for k, v in props.items() if k != self.route_by}}
        schema["required"] = [self.route_by] + [r for r in (schema.get("required") or []) if r != self.route_by]
        schema.setdefault("type", "object")
        return schema

    def prepare(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self.route_by in args:
            key = self.key(args[self.route_by])
            if key is not None:
                args[self.route_by] = key
        return args

    def validate(self, args: Dict[str, Any]) -> Optional[str]:
        value = args.get(self.route_by)
        if value is None or str(value).strip() == "":
            return f"'{self.route_by}' is required. Ask the user which one; do not guess. Choices: {self.choices()}"
        if self.key(value) is None:
            return f"Unknown {self.route_by} '{value}'. Choices: {self.choices()}. Ask the user if unsure."
        return None

    def apply(self, req: Request) -> Request:
        problem = self.validate(req.checked)
        if problem:
            raise ValueError(problem)
        key = self.key(req.checked[self.route_by])
        route = self.routes[key]
        req.method = str(route.get("method", req.method)).upper()
        req.url = route.get("url") or req.url
        req.headers = {**req.headers, **(route.get("headers") or {})}
        req.timeout = float(route.get("timeout", req.timeout))
        if self.send:
            req.args[self.route_by] = key
        else:
            req.args.pop(self.route_by, None)
        return req

    def provides_url(self) -> bool:
        return all(r.get("url") for r in self.routes.values())

    def methods(self, default: str) -> Set[str]:
        return {str(r.get("method", default)).upper() for r in self.routes.values()}

    def label(self) -> str:
        return f"one per {self.route_by}"


class CombineStep(Step):
    """Send one field built from several arguments, which the model fills (and are checked) separately.

        combine:
          lot_key: "{lot_num}_{work_order}"   # sends lot_key; lot_num and work_order are not sent
    """

    def __init__(self, name: str, combine: Dict[str, str], known: Set[str]):
        self.name, self.combine = name, combine
        for target, template in combine.items():
            fields = _FIELD.findall(template)
            if not fields:
                raise ValueError(f"API '{name}': combine.{target} uses no {{field}}: '{template}'")
            unknown = [f for f in fields if f not in known]
            if unknown:
                raise ValueError(f"API '{name}': combine.{target} uses {unknown}, "
                                 f"which are not in parameters.properties")

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> Optional["CombineStep"]:
        combine = spec.get("combine")
        if not combine:
            return None
        known = set(((spec.get("parameters") or {}).get("properties") or {}))
        if spec.get("route_by"):
            known.add(str(spec["route_by"]))
        return cls(spec.get("name"), {str(k): str(v) for k, v in combine.items()}, known)

    def fields(self) -> List[str]:
        return sorted({f for t in self.combine.values() for f in _FIELD.findall(t)})

    def validate(self, args: Dict[str, Any]) -> Optional[str]:
        missing = [f for f in self.fields() if args.get(f) in (None, "")]
        if missing:
            return (f"{missing} {'is' if len(missing) == 1 else 'are'} required. "
                    "Ask the user for them; do not make them up.")
        return None

    def apply(self, req: Request) -> Request:
        problem = self.validate(req.checked)
        if problem:
            raise ValueError(problem)
        for f in self.fields():
            req.args.pop(f, None)
        for target, template in self.combine.items():
            req.args[target] = _FIELD.sub(lambda m: str(req.checked[m.group(1)]), template)
        return req


# Applied in this order. Each step reads the model's checked arguments from `Request.checked`,
# so the order only matters when two steps write the same request field.
STEP_TYPES = [RouteStep, CombineStep]


def build_steps(spec: Dict[str, Any]) -> List[Step]:
    return [step for step in (t.from_spec(spec) for t in STEP_TYPES) if step is not None]
