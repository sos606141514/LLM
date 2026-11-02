"""Build an Agent from a YAML config file (values like ${VAR} are read from the environment)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import yaml

from .agent import DEFAULT_SYSTEM_PROMPT, Agent
from .llm_client import LLMClient
from .tools.api_tools import _expand_env as _env, register_api_tools
from .tools.mongo_tools import MongoTools
from .tools.registry import ConfirmFn, ToolRegistry


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_agent(cfg: Dict[str, Any], confirm: Optional[ConfirmFn] = None, mongo_client=None) -> Agent:
    llm_cfg = cfg["llm"]
    llm = LLMClient(
        base_url=_env(llm_cfg["base_url"]),
        model=_env(llm_cfg["model"]),
        api_key=_env(llm_cfg.get("api_key")) or None,
        tool_mode=llm_cfg.get("tool_mode", "native"),
        extra_headers={k: _env(v) for k, v in (llm_cfg.get("headers") or {}).items()},
        temperature=llm_cfg.get("temperature", 0.0),
        timeout=llm_cfg.get("timeout", 120),
        verify_ssl=_env(llm_cfg.get("verify_ssl", True)),
    )

    registry = ToolRegistry(confirm=confirm, max_result_chars=cfg.get("max_result_chars", 8000))

    mongo_cfg = cfg.get("mongodb")
    if mongo_cfg:
        if mongo_client is None:
            from pymongo import MongoClient
            mongo_client = MongoClient(_env(mongo_cfg["uri"]), serverSelectionTimeoutMS=5000)
        MongoTools(
            mongo_client[_env(mongo_cfg["database"])],
            allowed_collections=mongo_cfg.get("allowed_collections"),
            max_limit=mongo_cfg.get("max_limit", 50),
            max_time_ms=mongo_cfg.get("max_time_ms", 10000),
        ).register(registry)

    register_api_tools(registry, cfg.get("apis") or [])

    return Agent(llm, registry, system_prompt=cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
                 max_steps=cfg.get("max_steps", 8))
