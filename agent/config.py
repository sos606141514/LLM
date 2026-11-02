"""Build an Agent from a YAML config file (values like ${VAR} are read from the environment)."""
from __future__ import annotations

import importlib
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


def _load_factory(path: str):
    """'package.module:function' -> the function object."""
    module_name, _, func_name = path.partition(":")
    if not func_name:
        raise ValueError(f"mongodb.factory must look like 'module:function', got '{path}'")
    return getattr(importlib.import_module(module_name), func_name)


def _has_method(obj: Any, name: str) -> bool:
    # Look on the class: pymongo's __getattr__ turns any unknown attribute into a database/collection.
    return callable(getattr(type(obj), name, None))


def _as_database(obj: Any, db_name: str):
    """Accept whatever the connection helper returns and get a pymongo-style Database from it."""
    if _has_method(obj, "list_database_names"):
        return obj[db_name]             # a MongoClient
    if _has_method(obj, "list_collection_names"):
        return obj                      # already a Database
    for attr in ("db", "database", "client", "conn", "connection"):  # a wrapper class holding one
        inner = vars(obj).get(attr) if hasattr(obj, "__dict__") else None
        if inner is not None and (_has_method(inner, "list_database_names")
                                  or _has_method(inner, "list_collection_names")):
            return _as_database(inner, db_name)
    raise TypeError(
        f"Could not get a MongoDB Database from {type(obj).__name__}. "
        "The connection helper should return a pymongo Database or MongoClient "
        "(or an object with a .db / .client attribute holding one)."
    )


def connect_mongo(mongo_cfg: Dict[str, Any], mongo_client=None):
    """Return the Database the Mongo tools will query.

    Two ways to configure it:
    - factory: your own helper, called as factory(user, password, ip, database)
    - uri: a plain MongoDB connection string, opened with pymongo
    """
    db_name = _env(mongo_cfg["database"])
    if mongo_client is not None:
        return _as_database(mongo_client, db_name)
    if mongo_cfg.get("factory"):
        factory = _load_factory(mongo_cfg["factory"])
        conn = factory(_env(mongo_cfg["user"]), _env(mongo_cfg["password"]), _env(mongo_cfg["ip"]), db_name)
        return _as_database(conn, db_name)
    from pymongo import MongoClient
    return MongoClient(_env(mongo_cfg["uri"]), serverSelectionTimeoutMS=5000)[db_name]


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
        MongoTools(
            connect_mongo(mongo_cfg, mongo_client),
            allowed_collections=mongo_cfg.get("allowed_collections"),
            max_limit=mongo_cfg.get("max_limit", 50),
            max_time_ms=mongo_cfg.get("max_time_ms", 10000),
        ).register(registry)

    register_api_tools(registry, cfg.get("apis") or [])

    return Agent(llm, registry, system_prompt=cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
                 max_steps=cfg.get("max_steps", 8))
