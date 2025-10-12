"""
FastAPI backend for Twilight Imperium AI Assistant

Endpoints:
- POST /chat         → chat with the assistant; persists session history
- POST /clear/{id}   → clear a conversation session
- GET  /health       → health check

Session storage:
- Uses Redis if configured via env; otherwise falls back to in-memory (dev only)

CORS:
- Restricted via ALLOWED_ORIGINS env (comma-separated). If unset, allows all (dev).
"""

import os
import json
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Redis (optional, with in-memory fallback)
try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Import chatbot (this will also validate your vector store & embeddings config)
from twilight_chatbot_langgraph_fixed import TwilightImperiumLangGraphBot


# -------------------------------
# Configuration helpers
# -------------------------------

def _parse_allowed_origins() -> List[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "").strip()
    if not raw:
        # Dev-friendly default; tighten in production by setting env
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _create_redis_client():
    """Create a Redis client if env vars are present; else return None.

    Supported env configurations:
    - REDIS_URL (rediss://...)
    - Or REDIS_HOST, REDIS_PORT, REDIS_PASSWORD
    """
    if redis is None:
        return None

    url = os.getenv("REDIS_URL")
    if url:
        try:
            return redis.from_url(url, decode_responses=True)
        except Exception:
            return None

    host = os.getenv("REDIS_HOST")
    port = os.getenv("REDIS_PORT")
    password = os.getenv("REDIS_PASSWORD")
    if host and port:
        try:
            return redis.Redis(
                host=host,
                port=int(port),
                password=password,
                decode_responses=True,
                ssl=os.getenv("REDIS_SSL", "false").lower() in ("1", "true", "yes"),
            )
        except Exception:
            return None

    return None


# -------------------------------
# Message (de)serialization helpers
# -------------------------------

def _serialize_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            serialized.append({"type": "human", "content": m.content})
        elif isinstance(m, AIMessage):
            serialized.append({"type": "ai", "content": m.content})
        else:
            # Fallback: store base message content only
            serialized.append({"type": "other", "content": getattr(m, "content", "")})
    return serialized


def _deserialize_messages(serialized: List[Dict[str, Any]]) -> List[BaseMessage]:
    messages: List[BaseMessage] = []
    for m in serialized:
        t = m.get("type")
        c = m.get("content", "")
        if t == "human":
            messages.append(HumanMessage(content=c))
        elif t == "ai":
            messages.append(AIMessage(content=c))
        else:
            # Ignore unknown types in history
            continue
    return messages


# -------------------------------
# Session storage abstraction
# -------------------------------

class SessionStore:
    """Abstracts storing conversation history.

    If Redis is available/configured, uses it; otherwise uses in-memory dict.
    """

    def __init__(self):
        self._redis = _create_redis_client()
        self._memory: Dict[str, List[Dict[str, Any]]] = {}
        self._ttl_seconds: Optional[int] = None
        ttl_env = os.getenv("SESSION_TTL_SECONDS")
        if ttl_env:
            try:
                self._ttl_seconds = int(ttl_env)
            except ValueError:
                self._ttl_seconds = None

    @staticmethod
    def _key(session_id: str) -> str:
        return f"ti:session:{session_id}"

    def load(self, session_id: str) -> List[BaseMessage]:
        key = self._key(session_id)
        if self._redis:
            raw = self._redis.get(key)
            if not raw:
                return []
            try:
                data = json.loads(raw)
            except Exception:
                return []
            return _deserialize_messages(data)

        # In-memory fallback
        data = self._memory.get(key, [])
        return _deserialize_messages(data)

    def save(self, session_id: str, messages: List[BaseMessage]) -> None:
        key = self._key(session_id)
        payload = json.dumps(_serialize_messages(messages))
        if self._redis:
            if self._ttl_seconds is not None:
                self._redis.setex(key, self._ttl_seconds, payload)
            else:
                self._redis.set(key, payload)
            return

        # In-memory fallback
        self._memory[key] = json.loads(payload)

    def clear(self, session_id: str) -> None:
        key = self._key(session_id)
        if self._redis:
            try:
                self._redis.delete(key)
                return
            except Exception:
                pass
        self._memory.pop(key, None)


# -------------------------------
# FastAPI app
# -------------------------------

app = FastAPI(title="Twilight Imperium AI Assistant API")

# CORS
allowed_origins = _parse_allowed_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session store (Redis or in-memory)
session_store = SessionStore()

# Lazily initialize the chatbot to reduce startup time
_chatbot: Optional[TwilightImperiumLangGraphBot] = None


def _get_chatbot() -> TwilightImperiumLangGraphBot:
    global _chatbot
    if _chatbot is None:
        _chatbot = TwilightImperiumLangGraphBot()
    return _chatbot


# -------------------------------
# Schemas
# -------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


# -------------------------------
# Routes
# -------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        chatbot = _get_chatbot()

        # Ensure a session id exists
        session_id = req.session_id or str(uuid.uuid4())

        # Load history
        history = session_store.load(session_id)

        # Get response
        response_text = chatbot.chat(req.message, history)

        # Update and persist history
        history.append(HumanMessage(content=req.message))
        history.append(AIMessage(content=response_text))

        # Keep last 20 messages
        if len(history) > 20:
            history = history[-20:]

        session_store.save(session_id, history)

        return ChatResponse(response=response_text, session_id=session_id)

    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear/{session_id}")
def clear(session_id: str) -> Dict[str, str]:
    session_store.clear(session_id)
    return {"message": "Session cleared"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


