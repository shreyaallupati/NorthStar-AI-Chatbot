from __future__ import annotations

from threading import Lock

from app.graph.state import ChatState
from app.graph.workflow import initial_state


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ChatState] = {}
        self._lock = Lock()

    def get_or_create(self, session_id: str) -> ChatState:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = initial_state(session_id)
            return self._sessions[session_id]

    def set(self, session_id: str, state: ChatState) -> None:
        with self._lock:
            self._sessions[session_id] = state

    def reset(self, session_id: str) -> ChatState:
        with self._lock:
            state = initial_state(session_id)
            self._sessions[session_id] = state
            return state


store = SessionStore()
