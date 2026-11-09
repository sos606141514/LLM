import json
import os
import sys
from datetime import datetime

import mongomock
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, ToolRegistry  # noqa: E402
from agent.llm_client import LLMClient, LLMResponse, ToolCall, _extract_json  # noqa: E402
from agent.tools import MongoTools, register_api_tools  # noqa: E402


@pytest.fixture
def db():
    d = mongomock.MongoClient()["shop"]
    d.orders.insert_many([
        {"order_no": "A1", "amount": 100, "status": "paid", "created": datetime(2026, 1, 5)},
        {"order_no": "A2", "amount": 250, "status": "paid", "created": datetime(2026, 2, 1)},
        {"order_no": "A3", "amount": 80, "status": "refund", "created": datetime(2026, 2, 3)},
    ])
    d.secrets.insert_one({"k": "v"})
    return d


@pytest.fixture
def registry(db):
    r = ToolRegistry(confirm=lambda name, args: True)
    MongoTools(db, allowed_collections=["orders"], max_limit=2).register(r)
    return r


# ---- mongo tools -------------------------------------------------------------
def test_find_with_limit_cap_and_date_filter(registry):
    out = json.loads(registry.call("mongo_find", {
        "collection": "orders",
        "filter": {"created": {"$gte": {"$date": "2026-02-01T00:00:00Z"}}},
        "sort": {"amount": -1}, "limit": 100,
    }))
    assert [d["order_no"] for d in out] == ["A2", "A3"]


def test_count_and_aggregate(registry):
    assert json.loads(registry.call("mongo_count", {"collection": "orders", "filter": {"status": "paid"}})) == {"count": 2}
    out = json.loads(registry.call("mongo_aggregate", {
        "collection": "orders",
        "pipeline": [{"$group": {"_id": "$status", "total": {"$sum": "$amount"}}}, {"$sort": {"_id": 1}}],
    }))
    assert out == [{"_id": "paid", "total": 350}, {"_id": "refund", "total": 80}]


def test_disallowed_collection_and_write_stage(registry):
    assert "not allowed" in registry.call("mongo_find", {"collection": "secrets"})
    assert "not allowed" in registry.call("mongo_aggregate", {"collection": "orders", "pipeline": [{"$out": "x"}]})
    assert json.loads(registry.call("mongo_list_collections", "{}")) == ["orders"]


def test_bad_arguments_return_error_text(registry):
    assert "error" in json.loads(registry.call("mongo_find", "{not json"))
    assert "error" in json.loads(registry.call("mongo_find", {"nope": 1}))
    assert "error" in json.loads(registry.call("no_such_tool", {}))


# ---- api tools ---------------------------------------------------------------
class FakeResp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}
        self.ok = status < 400
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        return FakeResp(200, {"done": True})


def test_api_tool_path_params_env_and_confirmation(monkeypatch):
    monkeypatch.setenv("TOKEN", "secret")
    sess = FakeSession()
    approved = []
    r = ToolRegistry(confirm=lambda n, a: approved.append(n) or n != "deny_me")
    register_api_tools(r, [
        {"name": "get_status", "method": "GET", "url": "https://x/orders/{order_id}",
         "headers": {"Authorization": "Bearer ${TOKEN}"},
         "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}},
        {"name": "run_job", "method": "POST", "url": "https://x/jobs"},
        {"name": "deny_me", "method": "POST", "url": "https://x/deny"},
    ], session=sess)

    out = json.loads(r.call("get_status", {"order_id": "a/b", "verbose": 1}))
    assert out["status_code"] == 200
    method, url, kw = sess.calls[0]
    assert (method, url) == ("GET", "https://x/orders/a%2Fb")
    assert kw["params"] == {"verbose": 1} and kw["headers"]["Authorization"] == "Bearer secret"

    r.call("run_job", {"report": "daily"})
    assert sess.calls[1][0] == "POST" and sess.calls[1][2]["json"] == {"report": "daily"}
    assert approved == ["run_job"]  # GET needs no confirmation

    assert "rejected" in r.call("deny_me", {})
    assert len(sess.calls) == 2


# ---- agent loop ----------------------------------------------------------------
class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []

    def chat(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        return self.responses.pop(0)


def test_agent_runs_tools_then_answers(registry):
    llm = ScriptedLLM([
        LLMResponse(None, [ToolCall("c1", "mongo_count", '{"collection": "orders", "filter": {"status": "paid"}}')]),
        LLMResponse("There are 2 paid orders."),
    ])
    agent = Agent(llm, registry)
    assert agent.run("How many paid orders?") == "There are 2 paid orders."
    tool_msg = llm.seen[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert json.loads(tool_msg["content"]) == {"count": 2}


def test_agent_stops_after_max_steps(registry):
    loop = LLMResponse(None, [ToolCall("c", "mongo_list_collections", "{}")])
    agent = Agent(ScriptedLLM([loop] * 3), registry, max_steps=3)
    assert "Stopped after 3" in agent.run("hi")


# ---- llm client ----------------------------------------------------------------
class FakeLLMSession:
    def __init__(self, reply):
        self.reply, self.payloads = reply, []

    def post(self, url, json=None, **kw):
        self.payloads.append(json)
        return FakeResp(200, self.reply)


def test_native_mode_parsing():
    sess = FakeLLMSession({"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "mongo_count", "arguments": "{\"collection\":\"orders\"}"}}]}}]})
    c = LLMClient("http://llm/v1", "m", session=sess)
    resp = c.chat([{"role": "user", "content": "hi"}], [{"type": "function", "function": {"name": "t"}}])
    assert resp.tool_calls[0].name == "mongo_count"
    assert sess.payloads[0]["tools"] and sess.payloads[0]["model"] == "m"


def test_prompt_mode_roundtrip():
    sess = FakeLLMSession({"choices": [{"message": {"content":
        'Sure:\n```json\n{"tool_calls": [{"name": "mongo_count", "arguments": {"collection": "orders"}}]}\n```'}}]})
    c = LLMClient("http://llm/v1", "m", tool_mode="prompt", session=sess)
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "a", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "name": "a", "content": "[]"},
    ]
    resp = c.chat(history, [{"type": "function", "function": {"name": "mongo_count", "parameters": {}}}])
    assert resp.tool_calls[0].arguments == {"collection": "orders"}
    sent = sess.payloads[0]["messages"]
    assert "tools" not in sess.payloads[0]
    assert "mongo_count" in sent[0]["content"] and sent[-1]["content"].startswith("TOOL_RESULT")

    assert _extract_json('{"final": "done"} trailing') == {"final": "done"}


# ---- custom mongo connection helper --------------------------------------------
def test_connect_mongo_with_custom_factory(monkeypatch):
    import types

    from agent.config import connect_mongo

    seen = {}
    client = mongomock.MongoClient()
    client["mydb"].orders.insert_one({"x": 1})

    class Wrapper:  # e.g. a class that keeps the MongoClient on .client
        def __init__(self, user, pw, ip, db):
            seen.update(user=user, pw=pw, ip=ip, db=db)
            self.client = client

    returns = {
        "Database": lambda u, p, i, d: client[d],
        "MongoClient": lambda u, p, i, d: client,
        "Wrapper": Wrapper,
    }
    monkeypatch.setenv("MONGO_PW", "secret")
    for name, fn in returns.items():
        mod = types.ModuleType("fake_mongo_" + name)
        mod.Clinet_mongo = fn
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
        db = connect_mongo({"factory": f"{mod.__name__}:Clinet_mongo", "user": "u",
                            "password": "${MONGO_PW}", "ip": "10.0.0.1", "database": "mydb"})
        assert db.orders.count_documents({}) == 1, name
    assert seen == {"user": "u", "pw": "secret", "ip": "10.0.0.1", "db": "mydb"}


def test_factory_args_with_port(monkeypatch):
    from agent.config import _factory_args

    monkeypatch.setenv("MONGO_PORT", "27018")
    base = {"user": "u", "password": "p", "ip": "1.2.3.4"}
    assert _factory_args(base, "db") == ["u", "p", "1.2.3.4", "db"]
    assert _factory_args({**base, "port": "${MONGO_PORT}"}, "db") == ["u", "p", "1.2.3.4", 27018, "db"]
    assert _factory_args({**base, "port": 27017, "factory_args": ["ip", "port", "user", "password", "database"]},
                         "db") == ["1.2.3.4", 27017, "u", "p", "db"]
    with pytest.raises(KeyError):
        _factory_args({**base, "factory_args": ["user", "port"]}, "db")


def test_class_with_connect_method(monkeypatch):
    import types

    from agent.config import connect_mongo

    client = mongomock.MongoClient()
    client["mydb"].orders.insert_one({"x": 1})

    class Mongo:  # connection opened by a method, stored under a custom attribute name
        def __init__(self, user, pw, ip, port, db):
            self.args = (user, pw, ip, port, db)
            self._mongo = None

        def connectmongo(self):
            self._mongo = client

        def save(self, *a):
            raise AssertionError("agent must never call save")

    class MongoNoInit:  # __init__ takes nothing, connect method takes the credentials and returns the db
        def connectmongo(self, user, pw, ip, port, db):
            assert (user, pw, ip, port, db) == ("u", "p", "1.2.3.4", 27017, "mydb")
            return client[db]

    mod = types.ModuleType("fake_mongo_cls")
    mod.Mongo, mod.MongoNoInit = Mongo, MongoNoInit
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    base = {"user": "u", "password": "p", "ip": "1.2.3.4", "port": 27017, "database": "mydb"}

    db = connect_mongo({**base, "factory": "fake_mongo_cls:Mongo", "connect_method": "connectmongo"})
    assert db.orders.count_documents({}) == 1

    db = connect_mongo({**base, "factory": "fake_mongo_cls:MongoNoInit", "factory_args": [],
                        "connect_method": "connectmongo",
                        "connect_args": ["user", "password", "ip", "port", "database"]})
    assert db.orders.count_documents({}) == 1

    with pytest.raises(TypeError, match="Could not find a MongoDB connection"):
        connect_mongo({**base, "factory": "fake_mongo_cls:Mongo"})  # forgot connect_method


def test_builtin_mongo_connection(monkeypatch):
    from agent.config import mongo_client_kwargs

    monkeypatch.setenv("MONGO_PW", "p@ss:word")
    kw = mongo_client_kwargs({"user": "u", "password": "${MONGO_PW}", "ip": "10.0.0.5", "port": "27018",
                              "database": "d", "options": {"directConnection": True}})
    assert kw == {"serverSelectionTimeoutMS": 5000, "host": "10.0.0.5", "port": 27018, "username": "u",
                  "password": "p@ss:word", "authSource": "admin", "directConnection": True}
    assert mongo_client_kwargs({"ip": "h", "database": "d"}) == {"serverSelectionTimeoutMS": 5000,
                                                                 "host": "h", "port": 27017}
    assert mongo_client_kwargs({"uri": "mongodb://x", "database": "d"})["host"] == "mongodb://x"
    with pytest.raises(KeyError):
        mongo_client_kwargs({"database": "d"})

    # the kwargs are accepted by the real pymongo client (connect=False: no network)
    import pymongo
    c = pymongo.MongoClient(connect=False, **kw)  # raises if any option name/value is invalid
    c.close()


# ---- several MongoDB sources ---------------------------------------------------
def _two_source_cfg(tmp_path):
    notes = tmp_path / "wms_notes.md"
    notes.write_text("qty is in boxes, not pieces", encoding="utf-8")
    return {
        "_config_dir": str(tmp_path),
        "llm": {"base_url": "http://llm/v1", "model": "m"},
        "mongodb": [
            {"name": "sales", "description": "orders and customers", "uri": "mongodb://s", "database": "shop",
             "allowed_collections": ["orders"], "notes": "amount is in TWD; only count status=paid"},
            {"name": "logistics", "description": "shipments and stock", "uri": "mongodb://l", "database": "wms",
             "notes_file": "wms_notes.md", "max_limit": 1},
        ],
    }


def _two_clients():
    sales, wms = mongomock.MongoClient(), mongomock.MongoClient()
    sales["shop"].orders.insert_many([{"amount": 100, "status": "paid"}, {"amount": 5, "status": "refund"}])
    sales["shop"].secrets.insert_one({"k": "v"})
    wms["wms"].stock.insert_many([{"sku": "X", "qty": 3}, {"sku": "Y", "qty": 7}])
    return {"sales": sales, "logistics": wms}


def test_multiple_sources_share_one_tool_set(tmp_path):
    from agent.config import build_agent

    agent = build_agent(_two_source_cfg(tmp_path), mongo_client=_two_clients())
    reg = agent.registry
    assert sorted(n for n in reg.tools if n.startswith("mongo_")) == [
        "mongo_aggregate", "mongo_count", "mongo_find", "mongo_list_collections", "mongo_sample_schema",
        "mongo_source_info"]
    schema = reg.tools["mongo_find"].parameters
    assert schema["properties"]["source"]["enum"] == ["sales", "logistics"]
    assert "source" in schema["required"]

    assert json.loads(reg.call("mongo_count", {"source": "sales", "collection": "orders",
                                               "filter": {"status": "paid"}})) == {"count": 1}
    assert len(json.loads(reg.call("mongo_find", {"source": "logistics", "collection": "stock"}))) == 1  # max_limit
    assert json.loads(reg.call("mongo_list_collections", {"source": "logistics"})) == ["stock"]

    # each source keeps its own allow-list; source is required and must exist
    assert "not allowed" in reg.call("mongo_find", {"source": "sales", "collection": "secrets"})
    assert "required" in reg.call("mongo_find", {"collection": "orders"})
    assert "is not one of ['sales', 'logistics']" in reg.call("mongo_find", {"source": "hr", "collection": "orders"})

    info = json.loads(reg.call("mongo_source_info", {"source": "logistics"}))
    assert info["notes"] == "qty is in boxes, not pieces" and info["allowed_collections"] == "all"

    # the system prompt lists sources by name + description only; notes are loaded on demand
    prompt = agent.messages[0]["content"]
    assert "- sales: orders and customers" in prompt and "mongo_source_info" in prompt
    assert "TWD" not in prompt


def test_single_mongodb_mapping_still_works():
    from agent.config import build_agent

    client = mongomock.MongoClient()
    client["shop"].orders.insert_one({"x": 1})
    agent = build_agent({"llm": {"base_url": "http://llm/v1", "model": "m"},
                         "mongodb": {"uri": "mongodb://x", "database": "shop"}}, mongo_client=client)
    reg = agent.registry
    assert "source" not in reg.tools["mongo_find"].parameters["properties"]
    assert "mongo_source_info" not in reg.tools
    assert json.loads(reg.call("mongo_count", {"collection": "orders"})) == {"count": 1}


def test_source_list_needs_unique_names():
    from agent.config import mongo_source_configs
    from agent.tools import MongoSource

    with pytest.raises(ValueError, match="needs a 'name'"):
        mongo_source_configs({"mongodb": [{"database": "d"}]})
    db = mongomock.MongoClient()["d"]
    with pytest.raises(ValueError, match="Duplicate"):
        MongoTools(sources=[MongoSource(db, "a"), MongoSource(db, "a")])


# ---- skills --------------------------------------------------------------------
def _write_skill(folder, filename, text):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_text(text, encoding="utf-8")


def test_skills_are_listed_in_prompt_and_loaded_on_demand(tmp_path):
    from agent.config import build_agent

    skills = tmp_path / "skills"
    _write_skill(skills, "revenue.md", "---\nname: revenue\ndescription: monthly paid revenue\n"
                                       "tools: [mongo_aggregate]\n---\nStep 1: only status=paid\n---\nStep 2")
    _write_skill(skills / "ops", "refund.md", "---\ndescription: refund an order\n---\nCheck the order first")
    _write_skill(skills, "README.md", "not a skill")
    client = mongomock.MongoClient()
    llm = ScriptedLLM([
        LLMResponse(None, [ToolCall("c1", "load_skill", '{"name": "revenue"}')]),
        LLMResponse("done"),
    ])
    agent = build_agent({"_config_dir": str(tmp_path), "llm": {"base_url": "http://llm/v1", "model": "m"},
                         "mongodb": {"uri": "mongodb://x", "database": "shop"}, "skills_dir": "skills"},
                        mongo_client=client)
    assert sorted(agent.skills.skills) == ["refund", "revenue"]  # name defaults to the file name
    prompt = agent.messages[0]["content"]
    assert "- revenue: monthly paid revenue" in prompt and "load_skill" in prompt
    assert "only status=paid" not in prompt

    agent.llm = llm
    assert agent.run("revenue for March?") == "done"
    loaded = json.loads(llm.seen[1][-1]["content"])
    assert loaded["tools"] == ["mongo_aggregate"]
    assert loaded["instructions"] == "Step 1: only status=paid\n---\nStep 2"  # '---' in the body is kept
    assert "is not one of" in agent.registry.call("load_skill", {"name": "nope"})  # rejected by the schema check
    with pytest.raises(ValueError, match="Unknown skill"):
        agent.skills.load("nope")


def test_skill_errors_are_reported_at_startup(tmp_path):
    from agent.config import build_agent
    from agent.skills import SkillLibrary, parse_skill

    with pytest.raises(ValueError, match="description is required"):
        parse_skill("---\nname: x\n---\nbody")
    with pytest.raises(ValueError, match="header"):
        parse_skill("no header here")
    with pytest.raises(ValueError, match="Duplicate skill"):
        SkillLibrary([parse_skill("---\nname: a\ndescription: d\n---\n")] * 2)

    _write_skill(tmp_path / "s", "bad.md", "---\nname: bad\ndescription: d\ntools: [mongo_findd]\n---\n")
    with pytest.raises(ValueError, match="unknown tools"):
        build_agent({"_config_dir": str(tmp_path), "llm": {"base_url": "http://llm/v1", "model": "m"},
                     "mongodb": {"uri": "mongodb://x", "database": "d"}, "skills_dir": "s"},
                    mongo_client=mongomock.MongoClient())


def test_no_skills_dir_means_no_load_skill_tool():
    from agent.config import build_agent

    agent = build_agent({"llm": {"base_url": "http://llm/v1", "model": "m"}})
    assert "load_skill" not in agent.registry.tools and "Skills" not in agent.messages[0]["content"]


def test_example_skills_folder_loads():
    from agent.skills import load_skills

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    names = [s.name for s in load_skills(os.path.join(root, "skills"))]
    assert "monthly_revenue" in names


# ---- collection descriptions ---------------------------------------------------
def _described_cfg(**extra):
    return {
        "llm": {"base_url": "http://llm/v1", "model": "m"},
        "mongodb": [
            {"name": "sales", "description": "orders and customers", "uri": "mongodb://s", "database": "shop",
             "collections": {"orders": "one document per order: amount, status, created",
                             "customers": "customer profiles"}},
            {"name": "logistics", "description": "shipments and stock", "uri": "mongodb://l", "database": "wms",
             "collections": {"shipments": "shipping records, linked to orders by order_no"}},
        ],
        **extra,
    }


def _described_clients():
    sales, wms = mongomock.MongoClient(), mongomock.MongoClient()
    sales["shop"].orders.insert_one({"order_no": "A1"})
    sales["shop"].customers.insert_one({"name": "x"})
    sales["shop"].secrets.insert_one({"k": "v"})
    wms["wms"].shipments.insert_one({"order_no": "A1", "shipped": True})
    return {"sales": sales, "logistics": wms}


def test_collection_descriptions_guide_the_model(registry):
    from agent.config import build_agent

    agent = build_agent(_described_cfg(), mongo_client=_described_clients())
    prompt = agent.messages[0]["content"]
    assert "- logistics: shipments and stock" in prompt
    assert "    - shipments: shipping records, linked to orders by order_no" in prompt
    assert "    - customers: customer profiles" in prompt

    reg = agent.registry
    listed = json.loads(reg.call("mongo_list_collections", {"source": "sales"}))
    assert listed == [{"name": "customers", "description": "customer profiles"},
                      {"name": "orders", "description": "one document per order: amount, status, created"}]
    info = json.loads(reg.call("mongo_source_info", {"source": "logistics"}))
    assert info["collections"] == {"shipments": "shipping records, linked to orders by order_no"}

    # only the described collections can be queried
    assert "not allowed" in reg.call("mongo_find", {"source": "sales", "collection": "secrets"})
    out = json.loads(reg.call("mongo_find", {"source": "logistics", "collection": "shipments",
                                             "filter": {"order_no": "A1"}}))
    assert out[0]["shipped"] is True


def test_many_collections_are_left_out_of_the_prompt():
    from agent.config import build_agent

    agent = build_agent(_described_cfg(max_prompt_collections=2), mongo_client=_described_clients())
    prompt = agent.messages[0]["content"]
    assert "- sales: orders and customers" in prompt
    assert "customer profiles" not in prompt and "call mongo_source_info" in prompt
    info = json.loads(agent.registry.call("mongo_source_info", {"source": "sales"}))
    assert info["collections"]["customers"] == "customer profiles"


def test_collections_config_forms():
    from agent.config import _collections_cfg
    from agent.tools import MongoSource

    assert _collections_cfg({"collections": ["a", "b"]}) == {"a": "", "b": ""}
    assert _collections_cfg({"collections": {"a": "desc", "b": None}}) == {"a": "desc", "b": ""}
    assert _collections_cfg({}) is None
    db = mongomock.MongoClient()["d"]
    with pytest.raises(ValueError, match="not both"):
        MongoSource(db, "x", collections={"a": "d"}, allowed_collections=["a"])

    # a single database with described collections also gets them in the prompt
    tools = MongoTools(sources=[MongoSource(db, collections={"orders": "orders"})])
    assert "    - orders: orders" in tools.prompt_section()
    # plain names (no descriptions): list_collections keeps returning names
    db.a.insert_one({})
    plain = MongoTools(sources=[MongoSource(db, collections={"a": ""})])
    reg = ToolRegistry()
    plain.register(reg)
    assert json.loads(reg.call("mongo_list_collections", {})) == ["a"]


def test_prompt_tells_model_to_ask_when_collection_is_unclear():
    from agent.config import build_agent
    from agent.tools.mongo_tools import AMBIGUOUS_COLLECTION_RULE

    described = build_agent(_described_cfg(), mongo_client=_described_clients()).messages[0]["content"]
    assert AMBIGUOUS_COLLECTION_RULE in described
    assert described.index("customer profiles") < described.index(AMBIGUOUS_COLLECTION_RULE)

    # also for a single database without any descriptions, and with a custom system_prompt
    client = mongomock.MongoClient()
    plain = build_agent({"llm": {"base_url": "http://llm/v1", "model": "m"}, "system_prompt": "Be brief.",
                         "mongodb": {"uri": "mongodb://x", "database": "shop"}}, mongo_client=client)
    prompt = plain.messages[0]["content"]
    assert prompt.startswith("Be brief.") and AMBIGUOUS_COLLECTION_RULE in prompt
    assert "MongoDB source" not in prompt

    # no Mongo configured: no rule
    assert AMBIGUOUS_COLLECTION_RULE not in build_agent(
        {"llm": {"base_url": "http://llm/v1", "model": "m"}}).messages[0]["content"]


# ---- one API per site (route_by / routes) ----------------------------------------
def _site_api(monkeypatch, **extra):
    monkeypatch.setenv("HSINCHU_TOKEN", "hc-secret")
    monkeypatch.setenv("TAINAN_TOKEN", "tn-secret")
    sess = FakeSession()
    asked = []
    r = ToolRegistry(confirm=lambda name, args: asked.append(dict(args)) or True)
    register_api_tools(r, [{
        "name": "regenerate_system_data",
        "description": "Regenerate system data",
        "method": "POST",
        "headers": {"X-Source": "agent"},
        "route_by": "site",
        "routes": {
            "TW01": {"label": "Hsinchu", "url": "https://hsinchu/api/regen",
                     "headers": {"Authorization": "Bearer ${HSINCHU_TOKEN}"}},
            "TW02": {"label": "Tainan", "url": "https://tainan/v2/regenerate",
                     "headers": {"Authorization": "Bearer ${TAINAN_TOKEN}"}},
        },
        "parameters": {"type": "object", "properties": {
            "lot_num": {"type": "string"}, "work_order": {"type": "string"}},
            "required": ["lot_num", "work_order"]},
        **extra,
    }], session=sess)
    return r, sess, asked


def test_routes_pick_the_site_api(monkeypatch):
    r, sess, asked = _site_api(monkeypatch)
    tool = r.tools["regenerate_system_data"]
    assert tool.require_confirmation
    site = tool.parameters["properties"]["site"]
    assert site["enum"] == ["TW01", "TW02"] and "TW02: Tainan" in site["description"]
    assert "ask" in site["description"].lower()
    assert tool.parameters["required"] == ["site", "lot_num", "work_order"]
    assert list(tool.parameters["properties"])[0] == "site"

    args = {"site": "tw02", "lot_num": "AB123456-0C-01", "work_order": "L2L16"}  # case-insensitive
    out = json.loads(r.call("regenerate_system_data", args))
    assert out["ok"]
    method, url, kw = sess.calls[0]
    assert (method, url) == ("POST", "https://tainan/v2/regenerate")
    assert kw["headers"] == {"X-Source": "agent", "Authorization": "Bearer tn-secret"}
    assert kw["json"] == {"lot_num": "AB123456-0C-01", "work_order": "L2L16"}  # site only picks the API
    assert asked == [{**args, "site": "TW02"}]  # the confirmation shows the site, with its configured spelling


def test_missing_or_unknown_site_is_rejected_before_confirmation(monkeypatch):
    r, sess, asked = _site_api(monkeypatch)
    missing = json.loads(r.call("regenerate_system_data", {"lot_num": "x", "work_order": "y"}))
    assert "required" in missing["error"] and "TW01: Hsinchu" in missing["error"]
    unknown = json.loads(r.call("regenerate_system_data", {"site": "CN01", "lot_num": "x", "work_order": "y"}))
    assert "Unknown site" in unknown["error"]
    assert sess.calls == [] and asked == []  # nothing sent, nobody asked to approve


def test_route_options(monkeypatch):
    r, sess, _ = _site_api(monkeypatch, send_route_param=True)
    r.call("regenerate_system_data", {"site": "tw01", "lot_num": "x", "work_order": "y"})
    assert sess.calls[0][1] == "https://hsinchu/api/regen"
    assert sess.calls[0][2]["json"]["site"] == "TW01"  # sent with the configured spelling

    with pytest.raises(ValueError, match="together"):
        register_api_tools(ToolRegistry(), [{"name": "a", "route_by": "site"}])
    with pytest.raises(ValueError, match="needs a url"):
        register_api_tools(ToolRegistry(), [{"name": "a", "route_by": "site", "routes": {"X": {"label": "x"}}}])
    with pytest.raises(ValueError, match="needs a url"):
        register_api_tools(ToolRegistry(), [{"name": "a"}])


# ---- argument format checks + combine ---------------------------------------------
LOT = "^[A-Za-z0-9]{8}-[A-Za-z0-9]{2}-[A-Za-z0-9]{2}$"
LNUM = "^L[0-9]+L[0-9]+$"


def _combined_api(monkeypatch):
    sess = FakeSession()
    asked = []
    r = ToolRegistry(confirm=lambda name, args: asked.append(dict(args)) or True)
    register_api_tools(r, [{
        "name": "regenerate_system_data",
        "method": "POST",
        "route_by": "site",
        "routes": {"TW01": {"label": "Hsinchu", "url": "https://hsinchu/regen"},
                   "TW02": {"label": "Tainan", "url": "https://tainan/regen"}},
        "parameters": {"type": "object", "properties": {
            "lot_num": {"type": "string", "pattern": LOT, "description": "lot number 8-2-2"},
            "work_order": {"type": "string", "pattern": LNUM, "description": "L number"}},
            "required": ["lot_num", "work_order"]},
        "combine": {"lot_key": "{lot_num}_{work_order}"},
    }], session=sess)
    return r, sess, asked


def test_combine_sends_one_field_built_from_checked_parts(monkeypatch):
    r, sess, asked = _combined_api(monkeypatch)
    args = {"site": "TW02", "lot_num": "AB123456-0C-01", "work_order": "L2L16"}
    assert json.loads(r.call("regenerate_system_data", args))["ok"]
    assert sess.calls[0][1] == "https://tainan/regen"
    assert sess.calls[0][2]["json"] == {"lot_key": "AB123456-0C-01_L2L16"}
    assert asked == [args]  # the confirmation shows the two parts separately
    # the model still sees two separate arguments
    assert set(r.tools["regenerate_system_data"].parameters["properties"]) == {"site", "lot_num", "work_order"}


def test_wrong_format_is_rejected_before_confirmation(monkeypatch):
    r, sess, asked = _combined_api(monkeypatch)
    bad = [
        {"site": "TW01", "lot_num": "AB123456-0C-01", "work_order": "L2-L16"},
        {"site": "TW01", "lot_num": "AB12345-0C-01", "work_order": "L2L16"},     # 7 characters
        {"site": "TW01", "lot_num": "AB123456_0C_01", "work_order": "L2L16"},
        {"site": "TW01", "lot_num": "L2L16", "work_order": "AB123456-0C-01"},    # swapped
    ]
    for args in bad:
        err = json.loads(r.call("regenerate_system_data", args))["error"]
        assert "does not match the required format" in err, args
    missing = json.loads(r.call("regenerate_system_data", {"site": "TW01", "lot_num": "AB123456-0C-01"}))
    assert "work_order" in missing["error"] and "required" in missing["error"]
    assert sess.calls == [] and asked == []


def test_combine_config_errors():
    with pytest.raises(ValueError, match="not in parameters"):
        register_api_tools(ToolRegistry(), [{"name": "a", "url": "https://x", "combine": {"k": "{nope}_x"},
                                             "parameters": {"type": "object", "properties": {"a": {}}}}])
    with pytest.raises(ValueError, match="uses no"):
        register_api_tools(ToolRegistry(), [{"name": "a", "url": "https://x", "combine": {"k": "fixed"}}])


def test_schema_check_rules():
    from agent.tools.validation import check_arguments

    schema = {"type": "object", "required": ["n"], "properties": {
        "n": {"type": "integer", "minimum": 1, "maximum": 31},
        "kind": {"type": "string", "enum": ["a", "b"]},
        "tags": {"type": "array", "items": {"type": "string", "pattern": "^[a-z]+$"}, "maxItems": 2},
        "opt": {"type": "object", "required": ["x"], "properties": {"x": {"type": "boolean"}}},
    }}
    assert check_arguments(schema, {"n": 5, "kind": "a", "tags": ["ok"], "opt": {"x": True}, "extra": 1}) == []
    assert check_arguments(schema, {"n": 5, "kind": None}) == []           # null optional = not given
    assert "n must be integer" in check_arguments(schema, {"n": "5"})[0]   # strings aren't coerced
    assert "n must be integer" in check_arguments(schema, {"n": True})[0]
    assert "<= 31" in check_arguments(schema, {"n": 40})[0]
    assert "not one of" in check_arguments(schema, {"n": 1, "kind": "c"})[0]
    assert "tags[1]" in check_arguments(schema, {"n": 1, "tags": ["ok", "NO"]})[0]
    assert "at most 2" in check_arguments(schema, {"n": 1, "tags": ["a", "b", "c"]})[0]
    assert check_arguments(schema, {"n": 1, "opt": {}}) == ["opt.x is required"]
    assert check_arguments(schema, {}) == ["n is required"]


def test_tools_can_opt_out_of_schema_check():
    from agent import Tool

    r = ToolRegistry()
    r.register(Tool(name="t", description="", parameters={"type": "object", "properties": {
        "n": {"type": "integer"}}}, func=lambda n: {"n": n}, check_schema=False))
    assert json.loads(r.call("t", {"n": "5"})) == {"n": "5"}


def test_api_tool_is_generic_and_steps_are_opt_in():
    from agent.tools.api_hooks import CombineStep, RouteStep
    from agent.tools.api_tools import ApiEndpoint

    plain = ApiEndpoint({"name": "plain", "method": "POST", "url": "https://x/a"})
    assert plain.steps == []
    tool = plain.to_tool()
    assert tool.prepare is None and tool.validate is None and tool.description == "[POST API] "

    both = ApiEndpoint({"name": "both", "route_by": "site", "routes": {"A": {"url": "https://a"}},
                        "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
                        "combine": {"key": "{site}-{x}"}})
    assert [type(s) for s in both.steps] == [RouteStep, CombineStep]
    sess = FakeSession()
    both.session = sess
    r = ToolRegistry(confirm=lambda n, a: True)
    r.register(both.to_tool())
    r.call("both", {"site": "a", "x": "1"})
    assert sess.calls[0][1] == "https://a" and sess.calls[0][2]["json"] == {"key": "A-1"}


# ---- fill a field from the conversation (email the agent's answer) -----------------
EMAIL = r"^[A-Za-z0-9._%+-]+@company\.com$"


def _mail_agent(responses):
    sess = FakeSession()
    shown = []
    reg = ToolRegistry(confirm=lambda name, args: shown.append(dict(args)) or True)
    register_api_tools(reg, [{
        "name": "send_email",
        "method": "POST",
        "url": "https://mail/send",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "array", "minItems": 1, "items": {"type": "string", "pattern": EMAIL}},
            "cc": {"type": "array", "items": {"type": "string", "pattern": EMAIL}},
            "subject": {"type": "string", "minLength": 1, "maxLength": 100},
            "content": {"type": "string"}},
            "required": ["to", "subject", "content"]},
        "fill": {"content": {"from": "answer", "format": "html"}},
    }], session=sess)
    llm = ScriptedLLM(responses)
    return Agent(llm, reg), sess, shown


def test_email_sends_the_chosen_answer_as_html():
    answer = "上個月共 **152** 筆\n\n| 月份 | 金額 |\n|---|---|\n| 10 | 1,234,500 |\n\n<script>x</script>"
    agent, sess, shown = _mail_agent([
        LLMResponse(answer),
        LLMResponse("請問要寄給誰？"),
        LLMResponse(None, [ToolCall("c1", "send_email",
                                    '{"to": ["amy@company.com"], "cc": ["bob@company.com"], '
                                    '"subject": "10 月營收", "which_answer": 2}')]),
        LLMResponse("已寄出"),
    ])
    agent.run("上個月營收？")
    agent.run("寄出去")
    assert agent.run("amy@company.com，cc bob") == "已寄出"

    # the model doesn't write the content; it picks the answer
    schema = agent.registry.tools["send_email"].parameters
    assert "content" not in schema["properties"] and "content" not in schema["required"]
    assert "which_answer" in schema["properties"]

    # the confirmation shows the exact text that will be sent
    assert shown == [{"to": ["amy@company.com"], "cc": ["bob@company.com"], "subject": "10 月營收",
                      "which_answer": 2, "content": answer}]
    body = sess.calls[0][2]["json"]
    assert set(body) == {"to", "cc", "subject", "content"}  # which_answer is not sent
    assert body["subject"] == "10 月營收"
    assert schema["required"] == ["to", "subject"]
    html_out = body["content"]
    assert "<strong>152</strong>" in html_out and '<table style="border-collapse:collapse">' in html_out
    assert "<td" in html_out and "1,234,500" in html_out
    assert "<script>" not in html_out and "&lt;script&gt;" in html_out  # HTML in the answer is not rendered


def test_email_rejected_without_an_answer_or_with_bad_recipients():
    agent, sess, shown = _mail_agent([])
    reg = agent.registry
    from agent.tools import context as tool_context

    err = json.loads(reg.call("send_email", {"to": ["amy@company.com"], "subject": "s"}))["error"]
    assert "no earlier answer" in err
    with tool_context.use(tool_context.CallContext(answers=["only one"])):
        assert "between 1 and 1" in json.loads(reg.call("send_email", {"to": ["amy@company.com"], "subject": "s",
                                                                         "which_answer": 3}))["error"]
        bad = json.loads(reg.call("send_email", {"to": ["amy@gmail.com"], "subject": "s"}))["error"]
        assert "to[0]" in bad and "does not match" in bad
        no_subject = json.loads(reg.call("send_email", {"to": ["amy@company.com"]}))["error"]
        assert "subject is required" in no_subject
        # a content the model writes itself is replaced by the answer
        reg.call("send_email", {"to": ["amy@company.com"], "subject": "s", "content": "made up"})
    assert sess.calls[0][2]["json"]["content"] == "<p>only one</p>"
    assert len(shown) == 1


def test_fill_config_errors_and_text_format():
    with pytest.raises(ValueError, match="from must be"):
        register_api_tools(ToolRegistry(), [{"name": "a", "url": "https://x", "fill": {"c": "nope"}}])
    with pytest.raises(ValueError, match="format must be"):
        register_api_tools(ToolRegistry(), [{"name": "a", "url": "https://x",
                                             "fill": {"c": {"from": "answer", "format": "pdf"}}}])
    from agent.tools import context as tool_context

    sess = FakeSession()
    r = ToolRegistry(confirm=lambda n, a: True)
    register_api_tools(r, [{"name": "a", "url": "https://x", "fill": {"c": "answer"}}], session=sess)
    with tool_context.use(tool_context.CallContext(answers=["**raw**"])):
        r.call("a", {})
    assert sess.calls[0][2]["json"] == {"c": "**raw**"}
