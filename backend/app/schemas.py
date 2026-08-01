from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field  # type: ignore


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str
    mode: Literal["bot", "live_agent"]
    intent: str
    suggestions: list[str] = []
    session_id: str


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class SessionSnapshot(BaseModel):
    session_id: str
    mode: Literal["bot", "live_agent"]
    messages: list[dict[str, str]]
    suggestions: list[str]
