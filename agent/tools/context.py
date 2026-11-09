"""What a tool may know about the conversation it is called from.

The agent sets this for the duration of one `Agent.run`; tools read it with `current()`. It is a
ContextVar, so parallel requests (e.g. the HTTP server's threads) each see their own conversation.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator, List, Optional


@dataclass
class CallContext:
    # The agent's earlier final answers in this conversation, most recent first.
    answers: List[str] = field(default_factory=list)


_current: ContextVar[Optional[CallContext]] = ContextVar("tool_call_context", default=None)


def current() -> CallContext:
    return _current.get() or CallContext()


@contextmanager
def use(ctx: CallContext) -> Iterator[None]:
    token = _current.set(ctx)
    try:
        yield
    finally:
        _current.reset(token)
