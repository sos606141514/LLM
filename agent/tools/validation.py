"""Argument checks that run before any tool is executed (or offered for confirmation).

The rules come from each tool's JSON Schema `parameters` (required, type, enum, pattern, ...), so the
model sees the same formats it is checked against. Tools only describe their own extra checks
(`Tool.prepare` / `Tool.validate`); the checking itself happens here, in one place.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from .registry import Tool

_TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def check_call(tool: "Tool", arguments: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Clean up and check one call. Returns (arguments to use, error message or None).

    Order: the tool's clean-up (e.g. "tw02" -> "TW02") -> the tool's own check -> schema check.
    """
    if tool.prepare is not None:
        arguments = tool.prepare(dict(arguments))
    if tool.validate is not None:
        problem = tool.validate(arguments)
        if problem:
            return arguments, problem
    if tool.check_schema:
        problems = check_arguments(tool.parameters, arguments)
        if problems:
            return arguments, ("Invalid arguments, the call was not executed: " + "; ".join(problems) +
                               ". Fix them, or ask the user if the value they gave doesn't fit.")
    return arguments, None


# ---- JSON Schema subset ---------------------------------------------------------------
# Covers the keywords used in tool definitions: type, required, enum, pattern, properties, items,
# minLength/maxLength, minimum/maximum, minItems/maxItems. Unknown keywords are ignored and extra
# properties are allowed, so existing schemas keep working.

def check_arguments(schema: Dict[str, Any], arguments: Any) -> List[str]:
    """Return a list of problems (empty when the arguments are valid)."""
    problems: List[str] = []
    _check(schema or {}, arguments, "", problems)
    return problems


@lru_cache(maxsize=256)
def _regex(pattern: str):
    return re.compile(pattern)


def _show(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _check(schema: Dict[str, Any], value: Any, path: str, problems: List[str]) -> None:
    name = path or "arguments"
    hint = schema.get("description")
    hint = f" ({hint})" if isinstance(hint, str) and hint and len(hint) <= 200 else ""

    types = schema.get("type")
    if types:
        allowed = types if isinstance(types, list) else [types]
        if not any(_TYPES.get(t, lambda v: True)(value) for t in allowed):
            problems.append(f"{name} must be {' or '.join(allowed)}, got {_show(value)}{hint}")
            return

    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{name} {_show(value)} is not one of {schema['enum']}{hint}")

    if isinstance(value, str):
        pattern = schema.get("pattern")
        if pattern and not _regex(pattern).search(value):
            problems.append(f"{name} {_show(value)} does not match the required format {pattern}{hint}")
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"{name} must be at least {schema['minLength']} characters{hint}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            problems.append(f"{name} must be at most {schema['maxLength']} characters{hint}")

    if _TYPES["number"](value):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{name} must be >= {schema['minimum']}, got {value}{hint}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{name} must be <= {schema['maximum']}, got {value}{hint}")

    if isinstance(value, dict):
        for key in schema.get("required") or []:
            if key not in value or value[key] is None:
                sub = (schema.get("properties") or {}).get(key) or {}
                desc = sub.get("description")
                problems.append(f"{_join(path, key)} is required" + (f" ({desc})" if desc else ""))
        for key, sub in (schema.get("properties") or {}).items():
            if key in value and value[key] is not None and isinstance(sub, dict):
                _check(sub, value[key], _join(path, key), problems)

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{name} needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{name} allows at most {schema['maxItems']} items")
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                _check(items, item, f"{name}[{i}]", problems)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key
