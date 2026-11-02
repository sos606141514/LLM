"""Build an Agent from a YAML config file (values like ${VAR} are read from the environment)."""
from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List, Optional

import yaml

from .agent import DEFAULT_SYSTEM_PROMPT, Agent
from .llm_client import LLMClient
from .skills import SkillLibrary, load_skills
from .tools.api_tools import _expand_env as _env, register_api_tools
from .tools.mongo_tools import MongoSource, MongoTools
from .tools.registry import ConfirmFn, ToolRegistry


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Relative paths in the config (skills_dir, notes_file) are resolved against the config file's folder.
    cfg.setdefault("_config_dir", os.path.dirname(os.path.abspath(path)))
    return cfg


def _resolve_path(cfg: Dict[str, Any], path: str) -> str:
    path = _env(path)
    if os.path.isabs(path):
        return path
    return os.path.join(cfg.get("_config_dir") or os.getcwd(), path)


def _load_factory(path: str):
    """'package.module:function' -> the function object."""
    module_name, _, func_name = path.partition(":")
    if not func_name:
        raise ValueError(f"mongodb.factory must look like 'module:function', got '{path}'")
    return getattr(importlib.import_module(module_name), func_name)


def _has_method(obj: Any, name: str) -> bool:
    # Look on the class: pymongo's __getattr__ turns any unknown attribute into a database/collection.
    return callable(getattr(type(obj), name, None))


def _is_client(obj: Any) -> bool:
    return _has_method(obj, "list_database_names")


def _is_database(obj: Any) -> bool:
    return _has_method(obj, "list_collection_names") and not _is_client(obj)


def _as_database(obj: Any, db_name: str):
    """Accept whatever the connection helper gives back and get a pymongo-style Database from it."""
    if _is_client(obj):
        return obj[db_name]             # a MongoClient
    if _is_database(obj):
        return obj                      # already a Database
    # An object of your own class: look through its attributes for a Database first, then a MongoClient.
    attrs = vars(obj) if hasattr(obj, "__dict__") else {}
    for value in attrs.values():
        if _is_database(value):
            return value
    for value in attrs.values():
        if _is_client(value):
            return value[db_name]
    raise TypeError(
        f"Could not find a MongoDB connection in {type(obj).__name__} (attributes: {sorted(attrs)}). "
        "After connecting, the object should hold a pymongo Database or MongoClient in one of its "
        "attributes, or the connect method should return one."
    )


def _args_from(mongo_cfg: Dict[str, Any], keys: list, db_name: str) -> list:
    args = []
    for key in keys:
        if key == "database":
            args.append(db_name)
            continue
        if key not in mongo_cfg:
            raise KeyError(f"mongodb.{key} is used as an argument but not set")
        value = _env(mongo_cfg[key])
        if key == "port" and isinstance(value, str) and value.strip().isdigit():
            value = int(value)  # "${MONGO_PORT}" comes in as text
        args.append(value)
    return args


def _factory_args(mongo_cfg: Dict[str, Any], db_name: str) -> list:
    if "factory_args" in mongo_cfg:   # may be [] for a class whose __init__ takes nothing
        order = mongo_cfg["factory_args"] or []
    else:
        order = (["user", "password", "ip", "port", "database"] if "port" in mongo_cfg
                 else ["user", "password", "ip", "database"])
    return _args_from(mongo_cfg, order, db_name)


def connect_mongo(mongo_cfg: Dict[str, Any], mongo_client=None):
    """Return the Database the Mongo tools will query.

    Three ways to configure it (checked in this order):
    - factory: your own function or class, called with the fields listed in `factory_args`, in that order.
      Default order: (user, password, ip, database), or (user, password, ip, port, database) when port is set.
      connect_method (optional): a method to call on the created object to open the connection,
      with the fields listed in `connect_args` (default: none).
    - uri: a plain MongoDB connection string, opened with pymongo
    - user / password / ip / port: opened with pymongo directly (the default)
    """
    db_name = _env(mongo_cfg["database"])
    if mongo_client is not None:
        return _as_database(mongo_client, db_name)
    if mongo_cfg.get("factory"):
        factory = _load_factory(mongo_cfg["factory"])
        obj = factory(*_factory_args(mongo_cfg, db_name))
        method = mongo_cfg.get("connect_method")
        if method:
            returned = getattr(obj, method)(*_args_from(mongo_cfg, mongo_cfg.get("connect_args") or [], db_name))
            if returned is not None and (_is_client(returned) or _is_database(returned)):
                return _as_database(returned, db_name)
        return _as_database(obj, db_name)
    from pymongo import MongoClient
    return MongoClient(**mongo_client_kwargs(mongo_cfg))[db_name]


def mongo_client_kwargs(mongo_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Keyword arguments for pymongo.MongoClient from the config.

    Either `uri`, or user / password / ip / port (+ optional auth_source, options).
    """
    kwargs: Dict[str, Any] = {"serverSelectionTimeoutMS": 5000}
    if mongo_cfg.get("uri"):
        kwargs["host"] = _env(mongo_cfg["uri"])
    else:
        if not mongo_cfg.get("ip"):
            raise KeyError("mongodb needs either 'uri' or 'ip' (plus user/password/port)")
        port = _env(mongo_cfg.get("port", 27017))
        kwargs.update(host=_env(mongo_cfg["ip"]), port=int(port))
        user = _env(mongo_cfg.get("user"))
        if user:  # leave out for a server without login
            kwargs.update(username=user, password=_env(mongo_cfg.get("password")),
                          authSource=_env(mongo_cfg.get("auth_source", "admin")))
    # Anything else pymongo accepts, e.g. {tls: true, directConnection: true, replicaSet: rs0}
    kwargs.update({k: _env(v) for k, v in (mongo_cfg.get("options") or {}).items()})
    return kwargs


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

    mongo_tools = build_mongo_tools(cfg, mongo_client)
    if mongo_tools:
        mongo_tools.register(registry)

    register_api_tools(registry, cfg.get("apis") or [])

    skills = SkillLibrary(load_skills(_resolve_path(cfg, cfg["skills_dir"])) if cfg.get("skills_dir") else [])
    skills.check_tools(registry)
    skills.register(registry)

    prompt_parts = [cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT]
    if mongo_tools:
        prompt_parts.append(mongo_tools.prompt_section())
    prompt_parts.append(skills.prompt_section())
    system_prompt = "\n\n".join(p.strip() for p in prompt_parts if p and p.strip())

    agent = Agent(llm, registry, system_prompt=system_prompt, max_steps=cfg.get("max_steps", 8))
    agent.skills = skills
    return agent


def mongo_source_configs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """`mongodb:` may be one mapping (a single database) or a list of them (several databases)."""
    mongo_cfg = cfg.get("mongodb")
    if not mongo_cfg:
        return []
    if isinstance(mongo_cfg, dict):
        return [{"name": "default", **mongo_cfg}]
    if not isinstance(mongo_cfg, list):
        raise ValueError("mongodb must be a mapping or a list of mappings")
    for i, item in enumerate(mongo_cfg):
        if not isinstance(item, dict) or not item.get("name"):
            raise ValueError(f"mongodb[{i}] needs a 'name' when several databases are listed")
    return mongo_cfg


def _source_notes(cfg: Dict[str, Any], src_cfg: Dict[str, Any]) -> str:
    notes = _env(src_cfg.get("notes") or "")
    if src_cfg.get("notes_file"):
        with open(_resolve_path(cfg, src_cfg["notes_file"]), encoding="utf-8") as f:
            notes = (notes + "\n\n" + f.read()).strip()
    return notes


def build_mongo_tools(cfg: Dict[str, Any], mongo_client=None) -> Optional[MongoTools]:
    """`mongo_client` (tests): one client for every source, or a dict {source name: client}."""
    sources = []
    for src_cfg in mongo_source_configs(cfg):
        name = src_cfg["name"]
        client = mongo_client.get(name) if isinstance(mongo_client, dict) else mongo_client
        sources.append(MongoSource(
            connect_mongo(src_cfg, client),
            name=name,
            description=_env(src_cfg.get("description") or ""),
            notes=_source_notes(cfg, src_cfg),
            allowed_collections=src_cfg.get("allowed_collections"),
            max_limit=src_cfg.get("max_limit", 50),
            max_time_ms=src_cfg.get("max_time_ms", 10000),
        ))
    return MongoTools(sources=sources) if sources else None


def new_session(template: Agent) -> Agent:
    """A fresh conversation that shares the template's LLM client and tools."""
    agent = Agent(template.llm, template.registry, system_prompt=template.messages[0]["content"],
                  max_steps=template.max_steps)
    agent.skills = template.skills
    return agent
