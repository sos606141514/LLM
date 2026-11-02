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
