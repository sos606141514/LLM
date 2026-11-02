"""Read-only MongoDB tools."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from bson import json_util

from .registry import Tool, ToolRegistry

# Stages/operators that write data or run server-side code are rejected.
_FORBIDDEN_KEYS = {"$out", "$merge", "$where", "$function", "$accumulator"}


def _check_forbidden(obj: Any) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _FORBIDDEN_KEYS:
                raise PermissionError(f"Operator {k} is not allowed (read-only)")
            _check_forbidden(v)
    elif isinstance(obj, list):
        for v in obj:
            _check_forbidden(v)


def _to_bson(value: Any) -> Any:
    """Convert Extended JSON from the model ({"$oid": ...}, {"$date": ...}) into BSON types."""
    if value is None:
        return None
    return json_util.loads(json.dumps(value))


def _dump(docs: Any) -> str:
    return json_util.dumps(docs, ensure_ascii=False, json_options=json_util.RELAXED_JSON_OPTIONS)


class MongoTools:
    def __init__(self, db, allowed_collections: Optional[Iterable[str]] = None, max_limit: int = 50,
                 max_time_ms: int = 10000):
        self.db = db
        self.allowed = set(allowed_collections) if allowed_collections else None
        self.max_limit = max_limit
        self.max_time_ms = max_time_ms

    def _coll(self, name: str):
        if self.allowed is not None and name not in self.allowed:
            raise PermissionError(f"Collection '{name}' is not allowed. Allowed: {sorted(self.allowed)}")
        return self.db[name]

    def _limit(self, limit: Optional[int]) -> int:
        if not limit or limit <= 0:
            return self.max_limit
        return min(int(limit), self.max_limit)

    # ---- tool implementations -------------------------------------------------
    def list_collections(self) -> str:
        names = self.db.list_collection_names()
        if self.allowed is not None:
            names = [n for n in names if n in self.allowed]
        return json.dumps(sorted(names))

    def sample_schema(self, collection: str, sample_size: int = 3) -> str:
        docs = list(self._coll(collection).find({}).limit(min(max(sample_size, 1), 5)))
        fields: Dict[str, str] = {}
        for d in docs:
            for k, v in d.items():
                fields.setdefault(k, type(v).__name__)
        return _dump({"fields": fields, "examples": docs})

    def find(self, collection: str, filter: Optional[dict] = None, projection: Optional[dict] = None,
             sort: Optional[dict] = None, limit: Optional[int] = None, skip: int = 0) -> str:
        flt = _to_bson(filter) or {}
        _check_forbidden(flt)
        cur = self._coll(collection).find(flt, _to_bson(projection))
        if sort:
            cur = cur.sort(list(sort.items()))
        if skip:
            cur = cur.skip(int(skip))
        cur = cur.limit(self._limit(limit))
        try:
            cur = cur.max_time_ms(self.max_time_ms)
        except (AttributeError, NotImplementedError):
            pass
        return _dump(list(cur))

    def count(self, collection: str, filter: Optional[dict] = None) -> str:
        flt = _to_bson(filter) or {}
        _check_forbidden(flt)
        return json.dumps({"count": self._coll(collection).count_documents(flt)})

    def aggregate(self, collection: str, pipeline: List[dict], limit: Optional[int] = None) -> str:
        if not isinstance(pipeline, list):
            raise ValueError("pipeline must be a list of stages")
        pipe = _to_bson(pipeline)
        _check_forbidden(pipe)
        pipe = list(pipe) + [{"$limit": self._limit(limit)}]
        kwargs = {"maxTimeMS": self.max_time_ms}
        try:
            docs = list(self._coll(collection).aggregate(pipe, **kwargs))
        except TypeError:  # some mocks don't accept maxTimeMS
            docs = list(self._coll(collection).aggregate(pipe))
        return _dump(docs)

    # ---- registration ---------------------------------------------------------
    def register(self, registry: ToolRegistry) -> None:
        coll_desc = "Collection name"
        if self.allowed:
            coll_desc += f". One of: {sorted(self.allowed)}"
        coll = {"type": "string", "description": coll_desc}

        registry.register(Tool(
            name="mongo_list_collections",
            description="List the MongoDB collections that can be queried.",
            parameters={"type": "object", "properties": {}},
            func=self.list_collections,
        ))
        registry.register(Tool(
            name="mongo_sample_schema",
            description="Show field names/types and a few example documents of a collection. "
                        "Use this before querying a collection you don't know.",
            parameters={"type": "object", "properties": {
                "collection": coll,
                "sample_size": {"type": "integer", "description": "1-5, default 3"},
            }, "required": ["collection"]},
            func=self.sample_schema,
        ))
        registry.register(Tool(
            name="mongo_find",
            description="Query documents with a MongoDB filter. Use Extended JSON for special types, "
                        'e.g. {"_id": {"$oid": "..."}} or {"created": {"$gte": {"$date": "2024-01-01T00:00:00Z"}}}.',
            parameters={"type": "object", "properties": {
                "collection": coll,
                "filter": {"type": "object", "description": "MongoDB query filter"},
                "projection": {"type": "object", "description": "Fields to include/exclude, e.g. {\"name\": 1}"},
                "sort": {"type": "object", "description": "e.g. {\"created\": -1}"},
                "limit": {"type": "integer", "description": f"Max documents (<= {self.max_limit})"},
                "skip": {"type": "integer"},
            }, "required": ["collection"]},
            func=self.find,
        ))
        registry.register(Tool(
            name="mongo_count",
            description="Count documents matching a filter.",
            parameters={"type": "object", "properties": {
                "collection": coll,
                "filter": {"type": "object"},
            }, "required": ["collection"]},
            func=self.count,
        ))
        registry.register(Tool(
            name="mongo_aggregate",
            description="Run a read-only aggregation pipeline (group, statistics, joins). "
                        "$out/$merge/$where/$function are not allowed.",
            parameters={"type": "object", "properties": {
                "collection": coll,
                "pipeline": {"type": "array", "items": {"type": "object"}},
                "limit": {"type": "integer", "description": f"Max result documents (<= {self.max_limit})"},
            }, "required": ["collection", "pipeline"]},
            func=self.aggregate,
        ))
