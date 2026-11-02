"""HTTP service for the agent.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000
    # or: python server.py

Environment variables:
    AGENT_CONFIG          path to the config file (default: config.yaml)
    AGENT_API_KEY         if set, every request must send header  X-API-Key: <value>
    SESSION_TTL_SECONDS   drop a conversation after this long without use (default: 3600)
    MAX_SESSIONS          max conversations kept in memory (default: 1000)
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from agent import Agent
from agent.config import build_agent, load_config, new_session

logger = logging.getLogger("agent.server")


# ---- request / response models -------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's question")
    session_id: Optional[str] = Field(None, description="Omit to start a new conversation")
    allow_actions: Union[bool, List[str]] = Field(
        False,
        description="APIs that need confirmation (non-GET by default) only run when allowed here: "
                    "true = allow all in this request, or a list of tool names to allow. "
                    "Blocked calls are returned in pending_actions.",
    )


class ToolCallInfo(BaseModel):
    name: str
    arguments: Any
    result: Any


class PendingAction(BaseModel):
    name: str
    arguments: Any


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tool_calls: List[ToolCallInfo]
    pending_actions: List[PendingAction]


# ---- conversation store ----------------------------------------------------------
class _Session:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.lock = threading.Lock()  # one request at a time per conversation
        self.last_used = time.time()


class SessionStore:
    def __init__(self, template: Agent, ttl: float, max_sessions: int):
        self.template = template
        self.ttl = ttl
        self.max_sessions = max_sessions
        self._sessions: Dict[str, _Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: Optional[str]) -> tuple[str, _Session]:
        with self._lock:
            self._evict()
            if session_id and session_id in self._sessions:
                s = self._sessions[session_id]
            else:
                session_id = session_id or uuid.uuid4().hex
                s = self._sessions[session_id] = _Session(new_session(self.template))
            s.last_used = time.time()
            return session_id, s

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _evict(self) -> None:
        now = time.time()
        for sid in [k for k, s in self._sessions.items() if now - s.last_used > self.ttl]:
            del self._sessions[sid]
        while len(self._sessions) >= self.max_sessions:
            oldest = min(self._sessions, key=lambda k: self._sessions[k].last_used)
            del self._sessions[oldest]


# ---- app -------------------------------------------------------------------------
def _maybe_json(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def create_app(agent: Optional[Agent] = None, api_key: Optional[str] = None) -> FastAPI:
    """`agent` and `api_key` can be passed directly (tests); otherwise they come from the environment."""
    api_key = api_key if api_key is not None else os.environ.get("AGENT_API_KEY", "")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        template = agent or build_agent(load_config(os.environ.get("AGENT_CONFIG", "config.yaml")))
        app.state.sessions = SessionStore(
            template,
            ttl=float(os.environ.get("SESSION_TTL_SECONDS", 3600)),
            max_sessions=int(os.environ.get("MAX_SESSIONS", 1000)),
        )
        logger.info("agent ready with tools: %s", list(template.registry.tools))
        yield

    app = FastAPI(title="LLM Agent", version="1.0", lifespan=lifespan)

    def check_key(x_api_key: Optional[str] = Header(None)) -> None:
        if api_key and not hmac.compare_digest(x_api_key or "", api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/tools", dependencies=[Depends(check_key)])
    def list_tools():
        tools = app.state.sessions.template.registry.tools.values()
        return [{"name": t.name, "description": t.description,
                 "require_confirmation": t.require_confirmation, "parameters": t.parameters} for t in tools]

    @app.get("/skills", dependencies=[Depends(check_key)])
    def list_skills():
        skills = app.state.sessions.template.skills
        if not skills:
            return []
        return [{"name": s.name, "description": s.description, "tools": s.tools} for s in skills.skills.values()]

    # Plain `def`: FastAPI runs it in a thread pool, so the blocking LLM/Mongo/HTTP calls don't block the server.
    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(check_key)])
    def chat(req: ChatRequest):
        session_id, session = app.state.sessions.get_or_create(req.session_id)
        pending: List[PendingAction] = []

        def confirm(name: str, args: Dict[str, Any]) -> bool:
            allowed = req.allow_actions is True or (isinstance(req.allow_actions, list) and name in req.allow_actions)
            if not allowed:
                pending.append(PendingAction(name=name, arguments=args))
            return allowed

        if not session.lock.acquire(timeout=1):
            raise HTTPException(status_code=409, detail="This session is already handling a request")
        try:
            answer = session.agent.run(req.message, confirm=confirm)
            trace = session.agent.last_trace
        except Exception as e:  # noqa: BLE001
            logger.exception("chat failed")
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e
        finally:
            session.lock.release()

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            tool_calls=[ToolCallInfo(name=t["name"], arguments=_maybe_json(t["arguments"]),
                                     result=_maybe_json(t["result"])) for t in trace],
            pending_actions=pending,
        )

    @app.delete("/sessions/{session_id}", dependencies=[Depends(check_key)])
    def delete_session(session_id: str):
        if not app.state.sessions.delete(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": session_id}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8000)))
