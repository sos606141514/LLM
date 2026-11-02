"""Skills: written procedures the model loads only when a task needs them.

Each skill is a Markdown file with a small YAML header:

    ---
    name: monthly_revenue
    description: Paid-order revenue and order count for one month
    tools: [mongo_aggregate, mongo_count]      # optional: tools the steps use
    ---
    1. Use source `sales`, collection `orders`, status = "paid" ...

The system prompt lists only each skill's name and description. The model calls
`load_skill(name)` to get the full steps, so adding skills barely grows the prompt.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import yaml

from .tools.registry import Tool, ToolRegistry

_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_HEADER_RE = re.compile(r"\s*---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", re.S)


@dataclass
class Skill:
    name: str
    description: str
    instructions: str
    tools: List[str] = field(default_factory=list)
    path: Optional[str] = None


def parse_skill(text: str, path: Optional[str] = None) -> Skill:
    where = path or "skill"
    m = _HEADER_RE.match(text.lstrip("﻿"))
    if not m:
        raise ValueError(f"{where}: must start with a header between '---' lines containing name and description")
    header, body = m.group(1), text.lstrip("﻿")[m.end():]
    meta = yaml.safe_load(header) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"{where}: header must be key: value pairs")
    name = str(meta.get("name") or (os.path.splitext(os.path.basename(path))[0] if path else "")).strip()
    if not name or not _NAME_RE.match(name):
        raise ValueError(f"{where}: name must be letters, digits, _ or - (got '{name}')")
    description = str(meta.get("description") or "").strip()
    if not description:
        raise ValueError(f"{where}: description is required (the model picks skills by it)")
    tools = meta.get("tools") or []
    if isinstance(tools, str):
        tools = [tools]
    return Skill(name=name, description=description, instructions=body.strip(),
                 tools=[str(t) for t in tools], path=path)


def load_skills(directory: str) -> List[Skill]:
    """Every *.md file in `directory` (subfolders included), sorted by path."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"skills_dir not found: {directory}")
    paths = []
    for root, _, files in os.walk(directory):
        paths += [os.path.join(root, f) for f in files if f.endswith(".md") and f.lower() != "readme.md"]
    skills = []
    for path in sorted(paths):
        with open(path, encoding="utf-8") as f:
            skills.append(parse_skill(f.read(), path))
    return skills


class SkillLibrary:
    def __init__(self, skills: Iterable[Skill]):
        self.skills: Dict[str, Skill] = {}
        for s in skills:
            if s.name in self.skills:
                raise ValueError(f"Duplicate skill name '{s.name}' ({self.skills[s.name].path} and {s.path})")
            self.skills[s.name] = s

    def __len__(self) -> int:
        return len(self.skills)

    def check_tools(self, registry: ToolRegistry) -> None:
        """Fail early when a skill names a tool that doesn't exist (e.g. a typo or a removed API)."""
        for s in self.skills.values():
            missing = [t for t in s.tools if t not in registry.tools]
            if missing:
                raise ValueError(f"Skill '{s.name}' uses unknown tools {missing}. Available: {list(registry.tools)}")

    def load(self, name: str) -> str:
        s = self.skills.get(name)
        if s is None:
            raise ValueError(f"Unknown skill '{name}'. Available: {list(self.skills)}")
        return json.dumps({"name": s.name, "description": s.description, "tools": s.tools,
                           "instructions": s.instructions}, ensure_ascii=False)

    def prompt_section(self) -> str:
        if not self.skills:
            return ""
        lines = ["Skills (step-by-step procedures for specific tasks):"]
        lines += [f"- {s.name}: {s.description}" for s in self.skills.values()]
        lines.append("When a request matches a skill, call load_skill first and follow its instructions.")
        return "\n".join(lines)

    def register(self, registry: ToolRegistry) -> None:
        if not self.skills:
            return
        registry.register(Tool(
            name="load_skill",
            description="Load the full step-by-step instructions of a skill listed in the system prompt.",
            parameters={"type": "object", "properties": {
                "name": {"type": "string", "enum": list(self.skills), "description": "Skill name"},
            }, "required": ["name"]},
            func=self.load,
        ))
