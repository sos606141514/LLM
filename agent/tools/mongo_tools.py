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


# Always added to the system prompt when Mongo tools are available.
AMBIGUOUS_COLLECTION_RULE = (
    "If the question doesn't make clear which source or collection to use (e.g. \"show the first 10 "
    "records\"), do not guess and do not query: ask the user which one they mean, listing the candidate "
    "collections with their descriptions. If earlier messages already established the source and "
    "collection, keep using them without asking again."
)


class MongoSource:
    """One MongoDB database the agent can query, with a name and a description of what it is for."""

    def __init__(self, db, name: str = "default", description: str = "", notes: str = "",
                 allowed_collections: Optional[Iterable[str]] = None, max_limit: int = 50,
                 max_time_ms: int = 10000, collections: Optional[Dict[str, str]] = None):
        """`collections`: {collection name: what it holds}. Only the listed collections can be queried,
        and the descriptions help the model pick the right one. Use it instead of `allowed_collections`."""
        if collections and allowed_collections:
            raise ValueError(f"Source '{name}': set either collections or allowed_collections, not both")
        self.db = db
        self.name = name
        self.description = description or ""
        self.notes = notes or ""
        self.collections: Dict[str, str] = {str(k): str(v or "").strip() for k, v in (collections or {}).items()}
        if self.collections:
            self.allowed: Optional[set] = set(self.collections)
        else:
            self.allowed = set(allowed_collections) if allowed_collections else None
        self.max_limit = max_limit
        self.max_time_ms = max_time_ms

    def coll(self, name: str):
        if self.allowed is not None and name not in self.allowed:
            raise PermissionError(f"Collection '{name}' is not allowed in source '{self.name}'. "
                                  f"Allowed: {sorted(self.allowed)}")
        return self.db[name]

    @property
    def has_collection_descriptions(self) -> bool:
        return any(self.collections.values())

    def limit(self, limit: Optional[int]) -> int:
        if not limit or limit <= 0:
            return self.max_limit
        return min(int(limit), self.max_limit)


class MongoTools:
    """Read-only query tools over one or more MongoDB sources.

    Single database (as before):   MongoTools(db, allowed_collections=[...])
    Several databases:             MongoTools(sources=[MongoSource(db1, "sales", ...), MongoSource(db2, ...)])

    With several sources every tool takes a required `source` argument; the tool set stays the same size.
    """

    def __init__(self, db=None, allowed_collections: Optional[Iterable[str]] = None, max_limit: int = 50,
                 max_time_ms: int = 10000, sources: Optional[List[MongoSource]] = None,
                 max_prompt_collections: int = 30):
        # Collection descriptions go into the system prompt only while there are at most this many in total;
        # beyond that the model gets them from mongo_source_info / mongo_list_collections.
        self.max_prompt_collections = max_prompt_collections
        if sources is None:
            if db is None:
                raise ValueError("MongoTools needs either db or sources")
            sources = [MongoSource(db, allowed_collections=allowed_collections, max_limit=max_limit,
                                   max_time_ms=max_time_ms)]
        if not sources:
            raise ValueError("MongoTools needs at least one source")
        self.sources: Dict[str, MongoSource] = {}
        for s in sources:
            if s.name in self.sources:
                raise ValueError(f"Duplicate MongoDB source name: {s.name}")
            self.sources[s.name] = s

    @property
    def multi(self) -> bool:
        return len(self.sources) > 1

    def _source(self, source: Optional[str]) -> MongoSource:
        if not source:
            if self.multi:
                raise ValueError(f"'source' is required. One of: {list(self.sources)}")
            return next(iter(self.sources.values()))
        if source not in self.sources:
            raise ValueError(f"Unknown source '{source}'. One of: {list(self.sources)}")
        return self.sources[source]

    # ---- tool implementations -------------------------------------------------
    def source_info(self, source: Optional[str] = None) -> str:
        s = self._source(source)
        info: Dict[str, Any] = {"source": s.name, "description": s.description}
        if s.collections:
            info["collections"] = s.collections
        else:
            info["allowed_collections"] = sorted(s.allowed) if s.allowed is not None else "all"
        info["notes"] = s.notes
        return json.dumps(info, ensure_ascii=False)

    def list_collections(self, source: Optional[str] = None) -> str:
        s = self._source(source)
        names = s.db.list_collection_names()
        if s.allowed is not None:
            names = [n for n in names if n in s.allowed]
        if s.has_collection_descriptions:
            return json.dumps([{"name": n, "description": s.collections.get(n, "")} for n in sorted(names)],
                              ensure_ascii=False)
        return json.dumps(sorted(names))

    def sample_schema(self, collection: str, sample_size: int = 3, source: Optional[str] = None) -> str:
        s = self._source(source)
        docs = list(s.coll(collection).find({}).limit(min(max(sample_size, 1), 5)))
        fields: Dict[str, str] = {}
        for d in docs:
            for k, v in d.items():
                fields.setdefault(k, type(v).__name__)
        return _dump({"fields": fields, "examples": docs})

    def find(self, collection: str, filter: Optional[dict] = None, projection: Optional[dict] = None,
             sort: Optional[dict] = None, limit: Optional[int] = None, skip: int = 0,
             source: Optional[str] = None) -> str:
        s = self._source(source)
        flt = _to_bson(filter) or {}
        _check_forbidden(flt)
        cur = s.coll(collection).find(flt, _to_bson(projection))
        if sort:
            cur = cur.sort(list(sort.items()))
        if skip:
            cur = cur.skip(int(skip))
        cur = cur.limit(s.limit(limit))
        try:
            cur = cur.max_time_ms(s.max_time_ms)
        except (AttributeError, NotImplementedError):
            pass
        return _dump(list(cur))

    def count(self, collection: str, filter: Optional[dict] = None, source: Optional[str] = None) -> str:
        s = self._source(source)
        flt = _to_bson(filter) or {}
        _check_forbidden(flt)
        return json.dumps({"count": s.coll(collection).count_documents(flt)})

    def aggregate(self, collection: str, pipeline: List[dict], limit: Optional[int] = None,
                  source: Optional[str] = None) -> str:
        s = self._source(source)
        if not isinstance(pipeline, list):
            raise ValueError("pipeline must be a list of stages")
        pipe = _to_bson(pipeline)
        _check_forbidden(pipe)
        pipe = list(pipe) + [{"$limit": s.limit(limit)}]
        try:
            docs = list(s.coll(collection).aggregate(pipe, maxTimeMS=s.max_time_ms))
        except TypeError:  # some mocks don't accept maxTimeMS
            docs = list(s.coll(collection).aggregate(pipe))
        return _dump(docs)

    # ---- prompt / registration ------------------------------------------------
    def prompt_section(self) -> str:
        """Sources (and, when there are few, their collections) for the system prompt.

        Notes are never put here; the model reads them with mongo_source_info.
        """
        srcs = list(self.sources.values())
        described = [s for s in srcs if s.has_collection_descriptions]
        total = sum(len(s.collections) for s in described)
        inline = bool(described) and total <= self.max_prompt_collections

        lines: List[str] = []
        if self.multi or any(s.description or s.notes or s.has_collection_descriptions for s in srcs):
            lines.append("MongoDB sources (pass the name as `source`):" if self.multi else "MongoDB source:")
            for s in srcs:
                lines.append(f"- {s.name}: {s.description or '(no description)'}")
                if inline and s.has_collection_descriptions:
                    lines += [f"    - {c}: {d or '(no description)'}" for c, d in s.collections.items()]
            lines.append("Pick the source and collection whose description matches the question.")
        lines.append(AMBIGUOUS_COLLECTION_RULE)
        if described and not inline:
            lines.append("Collection descriptions are not listed here: call mongo_source_info for a source "
                         "to see what each of its collections holds before choosing one.")
        if any(s.notes for s in srcs):
            lines.append("Before querying a source for the first time, call mongo_source_info to read its "
                         "notes (field meanings and business rules).")
        return "\n".join(lines)

    def register(self, registry: ToolRegistry) -> None:
        max_limit = max(s.max_limit for s in self.sources.values())

        def coll_desc() -> str:
            if self.multi:
                parts = [f"{s.name}: {sorted(s.allowed) if s.allowed else 'any'}" for s in self.sources.values()]
                return "Collection name. Allowed per source: " + "; ".join(parts)
            s = next(iter(self.sources.values()))
            return "Collection name" + (f". One of: {sorted(s.allowed)}" if s.allowed else "")

        coll = {"type": "string", "description": coll_desc()}
        source_prop = {"type": "string", "enum": list(self.sources),
                       "description": "Which database. " + "; ".join(
                           f"{s.name}: {s.description or s.name}" for s in self.sources.values())}

        def params(props: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
            props, required = dict(props), list(required)
            if self.multi:
                props = {"source": source_prop, **props}
                required = ["source"] + required
            out: Dict[str, Any] = {"type": "object", "properties": props}
            if required:
                out["required"] = required
            return out

        if self.multi or any(s.description or s.notes or s.has_collection_descriptions
                             for s in self.sources.values()):
            registry.register(Tool(
                name="mongo_source_info",
                description="Show what a MongoDB source is for, its collections (with what each one holds), "
                            "and its notes (field meanings, units, business rules). Read this before querying "
                            "a source.",
                parameters=params({}, []),
                func=self.source_info,
            ))
        registry.register(Tool(
            name="mongo_list_collections",
            description="List the MongoDB collections that can be queried.",
            parameters=params({}, []),
            func=self.list_collections,
        ))
        registry.register(Tool(
            name="mongo_sample_schema",
            description="Show field names/types and a few example documents of a collection. "
                        "Use this before querying a collection you don't know.",
            parameters=params({
                "collection": coll,
                "sample_size": {"type": "integer", "description": "1-5, default 3"},
            }, ["collection"]),
            func=self.sample_schema,
        ))
        registry.register(Tool(
            name="mongo_find",
            description="Query documents with a MongoDB filter. Use Extended JSON for special types, "
                        'e.g. {"_id": {"$oid": "..."}} or {"created": {"$gte": {"$date": "2024-01-01T00:00:00Z"}}}.',
            parameters=params({
                "collection": coll,
                "filter": {"type": "object", "description": "MongoDB query filter"},
                "projection": {"type": "object", "description": "Fields to include/exclude, e.g. {\"name\": 1}"},
                "sort": {"type": "object", "description": "e.g. {\"created\": -1}"},
                "limit": {"type": "integer", "description": f"Max documents (<= {max_limit})"},
                "skip": {"type": "integer"},
            }, ["collection"]),
            func=self.find,
        ))
        registry.register(Tool(
            name="mongo_count",
            description="Count documents matching a filter.",
            parameters=params({
                "collection": coll,
                "filter": {"type": "object"},
            }, ["collection"]),
            func=self.count,
        ))
        registry.register(Tool(
            name="mongo_aggregate",
            description="Run a read-only aggregation pipeline (group, statistics, joins). "
                        "$out/$merge/$where/$function are not allowed.",
            parameters=params({
                "collection": coll,
                "pipeline": {"type": "array", "items": {"type": "object"}},
                "limit": {"type": "integer", "description": f"Max result documents (<= {max_limit})"},
            }, ["collection", "pipeline"]),
            func=self.aggregate,
        ))
