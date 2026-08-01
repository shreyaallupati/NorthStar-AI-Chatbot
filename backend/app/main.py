from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents import classifier
from app.config import settings
from app.db import init_db
from app.graph.workflow import describe_graph, run_turn
from app.rag import store as rag_store
from app.schemas import ChatRequest, ChatResponse, ResetRequest, SessionSnapshot
from app.sessions import store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENGINE_INFO: dict[str, str] = {
    "engine": "local",
    "router": classifier.BACKEND_KEYWORD,
    "retriever": rag_store.BACKEND_KEYWORD,
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    ENGINE_INFO["router"] = classifier.warmup()
    ENGINE_INFO["retriever"] = rag_store.warmup()
    logger.info(
        "North Star engine ready: router=%s retriever=%s",
        ENGINE_INFO["router"],
        ENGINE_INFO["retriever"],
    )
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "engine": ENGINE_INFO["engine"],
        "router": ENGINE_INFO["router"],
        "retriever": ENGINE_INFO["retriever"],
        "graph": describe_graph(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    state = store.get_or_create(payload.session_id)
    updated = run_turn(state, payload.message)
    store.set(payload.session_id, updated)
    return ChatResponse(
        reply=updated["reply"],
        mode=updated.get("mode", "bot"),
        intent=updated.get("intent", "fallback"),
        suggestions=updated.get("suggestions", []),
        session_id=payload.session_id,
    )


@app.post("/chat/reset", response_model=SessionSnapshot)
def reset(payload: ResetRequest) -> SessionSnapshot:
    state = store.reset(payload.session_id)
    return SessionSnapshot(
        session_id=payload.session_id,
        mode=state.get("mode", "bot"),
        messages=state.get("messages", []),
        suggestions=state.get("suggestions", []),
    )


@app.get("/chat/{session_id}", response_model=SessionSnapshot)
def get_session(session_id: str) -> SessionSnapshot:
    state = store.get_or_create(session_id)
    return SessionSnapshot(
        session_id=session_id,
        mode=state.get("mode", "bot"),
        messages=state.get("messages", []),
        suggestions=state.get("suggestions", []),
    )
