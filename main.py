# api/main.py
#
# Endpoints:
#   POST /query              — ask a question (with optional session_id for memory)
#   POST /session/new        — create a new session, returns session_id
#   DELETE /session/{id}     — clear a session's history
#   GET  /health             — check Qdrant + Ollama are alive
#
# HOW SESSIONS WORK:
#   1. Client calls POST /session/new → gets back a session_id (UUID)
#   2. Client includes session_id in every /query request
#   3. Server loads history from in-memory dict, rewrites query if needed,
#      retrieves, generates, saves turn back to dict
#   Note: history is lost on server restart (single-session, no persistence needed)
#
# Run:  uvicorn api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional

from retriever import build_retriever, retrieve
from generate import rewrite_query, generate
from config import QDRANT_URL, OLLAMA_BASE_URL, MAX_HISTORY_TURNS

# In-memory session store: { session_id: [{"role": ..., "content": ...}, ...] }
sessions: dict = {}

# Retriever built once at startup, reused across all requests
retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    print("Loading retriever from Qdrant...")
    retriever = build_retriever()
    print("Retriever ready.")
    yield

app = FastAPI(
    title="RAG Pipeline",
    description="Hybrid search + reranking + local LLM + in-memory conversation history",
    lifespan=lifespan
)


# --- Helpers ---

def format_history(history: list[dict]) -> str:
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def get_history(session_id: str) -> list[dict]:
    return sessions.get(session_id, [])


def save_turn(session_id: str, user_message: str, assistant_reply: str):
    history = sessions.get(session_id, [])
    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": assistant_reply})

    # Keep only the last N turns
    max_messages = MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        history = history[-max_messages:]

    sessions[session_id] = history


def clear_session(session_id: str):
    sessions.pop(session_id, None)


# --- Request / Response models ---

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None  # if None, stateless single-turn query

class QueryResponse(BaseModel):
    question: str
    rewritten_question: str
    answer: str
    sources: list[str]
    session_id: Optional[str]


class NewSessionResponse(BaseModel):
    session_id: str


# --- Endpoints ---

@app.post("/session/new", response_model=NewSessionResponse)
def new_session():
    """Create a new conversation session. Returns a session_id to use in /query."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = []
    return NewSessionResponse(session_id=session_id)


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    history = []
    if req.session_id:
        history = get_history(req.session_id)

    rewritten = rewrite_query(req.question, history, format_history)

    chunks = retrieve(rewritten, retriever)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    result = generate(req.question, chunks, history, format_history)

    if req.session_id:
        save_turn(req.session_id, req.question, result["answer"])

    return QueryResponse(
        question=req.question,
        rewritten_question=rewritten,
        answer=result["answer"],
        sources=result["sources"],
        session_id=req.session_id
    )


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Clear a session's history."""
    clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}


@app.get("/health")
def health():
    qdrant_ok = False
    ollama_ok = False

    try:
        httpx.get(f"{QDRANT_URL}/healthz", timeout=3)
        qdrant_ok = True
    except Exception:
        pass

    try:
        httpx.get(OLLAMA_BASE_URL, timeout=3)
        ollama_ok = True
    except Exception:
        pass

    return {
        "qdrant": qdrant_ok,
        "ollama": ollama_ok,
        "ready":  qdrant_ok and ollama_ok
    }