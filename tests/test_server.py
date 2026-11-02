import json
import os
import sys

import mongomock
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, Tool, ToolRegistry  # noqa: E402
from agent.llm_client import LLMResponse, ToolCall  # noqa: E402
from agent.tools import MongoTools  # noqa: E402
from server import create_app  # noqa: E402


class ScriptedLLM:
    """Plays back responses; records the message history it was shown each time."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []

    def chat(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        return self.responses.pop(0)


def make_client(responses, api_key=""):
    db = mongomock.MongoClient()["shop"]
    db.orders.insert_many([{"status": "paid"}, {"status": "paid"}, {"status": "refund"}])
    reg = ToolRegistry()
    MongoTools(db).register(reg)
    fired = []
    reg.register(Tool(name="run_job", description="start job",
                      parameters={"type": "object", "properties": {"kind": {"type": "string"}}},
                      func=lambda kind: fired.append(kind) or {"started": kind}, require_confirmation=True))
    llm = ScriptedLLM(responses)
    client = TestClient(create_app(agent=Agent(llm, reg), api_key=api_key))
    client.__enter__()  # run lifespan
    return client, llm, fired


def test_chat_with_tool_and_session_memory():
    client, llm, _ = make_client([
        LLMResponse(None, [ToolCall("c1", "mongo_count", {"collection": "orders", "filter": {"status": "paid"}})]),
        LLMResponse("2 paid orders"),
        LLMResponse("Yes, 2."),
    ])
    r = client.post("/chat", json={"message": "how many paid?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "2 paid orders"
    assert body["tool_calls"] == [{"name": "mongo_count",
                                   "arguments": {"collection": "orders", "filter": {"status": "paid"}},
                                   "result": {"count": 2}}]

    r2 = client.post("/chat", json={"message": "sure?", "session_id": body["session_id"]})
    assert r2.json()["answer"] == "Yes, 2." and r2.json()["tool_calls"] == []
    # the second request saw the first exchange
    assert any(m.get("content") == "how many paid?" for m in llm.seen[-1])


def test_actions_need_permission():
    call = LLMResponse(None, [ToolCall("c", "run_job", {"kind": "daily"})])
    client, _, fired = make_client([call, LLMResponse("needs approval"), call, LLMResponse("started")])

    body = client.post("/chat", json={"message": "run daily job"}).json()
    assert fired == [] and body["pending_actions"] == [{"name": "run_job", "arguments": {"kind": "daily"}}]
    assert "rejected" in body["tool_calls"][0]["result"]["error"]

    body = client.post("/chat", json={"message": "yes", "session_id": body["session_id"],
                                      "allow_actions": ["run_job"]}).json()
    assert fired == ["daily"] and body["pending_actions"] == []
    assert body["tool_calls"][0]["result"] == {"started": "daily"}


def test_api_key_tools_and_delete():
    client, _, _ = make_client([LLMResponse("hi")], api_key="k")
    assert client.get("/health").status_code == 200
    assert client.post("/chat", json={"message": "x"}).status_code == 401
    assert client.get("/tools", headers={"X-API-Key": "wrong"}).status_code == 401

    h = {"X-API-Key": "k"}
    names = [t["name"] for t in client.get("/tools", headers=h).json()]
    assert "mongo_find" in names and "run_job" in names

    sid = client.post("/chat", json={"message": "x"}, headers=h).json()["session_id"]
    assert client.delete(f"/sessions/{sid}", headers=h).status_code == 200
    assert client.delete(f"/sessions/{sid}", headers=h).status_code == 404


def test_llm_failure_returns_502():
    class Broken:
        def chat(self, *a):
            raise RuntimeError("LLM API error 500: boom")

    client = TestClient(create_app(agent=Agent(Broken(), ToolRegistry()), api_key=""))
    with client:
        r = client.post("/chat", json={"message": "x"})
    assert r.status_code == 502 and "boom" in r.json()["detail"]
