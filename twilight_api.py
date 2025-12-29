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
from datetime import date

from fastapi import FastAPI, HTTPException, Depends, Header
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

# Supabase client for auth
from supabase import create_client, Client


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
# Supabase client for auth
# -------------------------------

def _create_supabase_client() -> Optional[Client]:
    """Create a Supabase client using service role key for backend auth.

    If env is not configured, returns None and the API works in dev mode
    without authentication (useful for local testing).
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")  # service_role key
    if not url or not key:
        print("⚠️  SUPABASE_URL/SUPABASE_SERVICE_KEY not set - auth disabled (dev mode)")
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        print(f"⚠️  Failed to create Supabase client: {e}")
        return None

def _create_supabase_anon_client() -> Optional[Client]:
    """Create a Supabase client using anon key for JWT token verification.

    This client is used specifically for verifying JWT tokens from the frontend.
    The anon key client can properly verify tokens issued by Supabase auth.
    """
    url = os.getenv("SUPABASE_URL")
    anon_key = os.getenv("SUPABASE_ANON_KEY")  # anon/public key
    if not url or not anon_key:
        print("⚠️  SUPABASE_URL/SUPABASE_ANON_KEY not set - token verification disabled")
        return None
    try:
        return create_client(url, anon_key)
    except Exception as e:
        print(f"⚠️  Failed to create Supabase anon client: {e}")
        return None

supabase_client: Optional[Client] = None
supabase_anon_client: Optional[Client] = None

def _get_supabase() -> Optional[Client]:
    """Get Supabase client with service role key (for database operations)."""
    global supabase_client
    if supabase_client is None:
        supabase_client = _create_supabase_client()
    return supabase_client

def _get_supabase_anon() -> Optional[Client]:
    """Get Supabase client with anon key (for JWT token verification)."""
    global supabase_anon_client
    if supabase_anon_client is None:
        supabase_anon_client = _create_supabase_anon_client()
    return supabase_anon_client

# Daily message limit for all users (disabled - no limit)
# DAILY_MESSAGE_LIMIT = 20


# -------------------------------
# Auth and usage helpers
# -------------------------------

async def verify_token(authorization: str = Header(None)) -> Dict[str, Any]:
    """Verify JWT (Supabase) and return minimal user info.

    Falls back to a dev user when Supabase is not configured.
    Uses anon key client for token verification (can verify JWT tokens from frontend).
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    token = authorization.replace("Bearer ", "")
    
    # Use anon key client for token verification (can verify JWT tokens)
    supabase_anon = _get_supabase_anon()
    if not supabase_anon:
        # Fallback to dev mode if anon key not configured
        return {"user_id": "dev-user", "email": "dev@localhost"}

    try:
        # Use anon key client to verify the JWT token
        user_response = supabase_anon.auth.get_user(token)
        if not user_response or not getattr(user_response, "user", None):
            raise HTTPException(status_code=401, detail="Invalid token")

        user_id = user_response.user.id  # type: ignore[attr-defined]
        user_email = getattr(user_response.user, "email", None)  # type: ignore[attr-defined]

        # Use service role client for database operations (has write permissions)
        supabase = _get_supabase()
        if not supabase:
            # If service role not configured, still return user info from token
            return {"user_id": user_id, "email": user_email}

        profile_response = (
            supabase
            .table("user_profiles")
            .select("*")
            .eq("id", user_id)
            .single()
            .execute()
        )
        if not getattr(profile_response, "data", None):
            # Profile doesn't exist - create it automatically (for OAuth users)
            try:
                supabase.table("user_profiles").insert({
                    "id": user_id,
                    "email": user_email,
                    "tier": "free"
                }).execute()
            except Exception as create_error:
                # If creation fails, still return user info (profile might exist but query failed)
                print(f"⚠️  Warning: Could not create user profile: {create_error}")
                return {"user_id": user_id, "email": user_email}
        
        return {"user_id": user_id, "email": user_email}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


async def check_and_increment_usage(user_info: Dict[str, Any]) -> None:
    """Track usage (no limit enforced).

    Skips when Supabase is not configured (dev mode).
    """
    supabase = _get_supabase()
    if not supabase:
        return

    user_id = user_info["user_id"]

    today = str(date.today())
    usage_response = (
        supabase
        .table("user_usage")
        .select("*")
        .eq("user_id", user_id)
        .eq("date", today)
        .execute()
    )

    if getattr(usage_response, "data", None):
        current_count = usage_response.data[0]["message_count"]  # type: ignore[index]
        # No limit check - just increment usage for tracking
        supabase.table("user_usage").update({"message_count": current_count + 1}).eq("user_id", user_id).eq("date", today).execute()
    else:
        supabase.table("user_usage").insert({"user_id": user_id, "date": today, "message_count": 1}).execute()

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

@app.get("/")
def root() -> Dict[str, Any]:
    """Root endpoint providing API information."""
    return {
        "service": "Twilight Imperium AI Assistant API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)",
            "me": "/me (GET)",
            "clear": "/clear/{session_id} (POST)"
        },
        "documentation": "See /health for service status"
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/me")
async def get_current_user(user_info: Dict[str, Any] = Depends(verify_token)) -> Dict[str, Any]:
    """Return current user info and today's usage summary."""
    supabase = _get_supabase()
    user_id = user_info["user_id"]

    used = 0
    if supabase:
        today = str(date.today())
        usage_response = supabase.table("user_usage").select("*").eq("user_id", user_id).eq("date", today).execute()
        if getattr(usage_response, "data", None):
            used = usage_response.data[0]["message_count"]

    return {
        "user_id": user_id,
        "email": user_info.get("email"),
        "usage": {"used": used, "limit": None, "remaining": None},  # No limit enforced
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, authorization: str = Header(None)) -> ChatResponse:
    try:
        # Check if user is authenticated
        is_authenticated = authorization and authorization.startswith("Bearer ")
        
        if is_authenticated:
            # Authenticated flow: verify token and enforce limits
            user_info = await verify_token(authorization)
            await check_and_increment_usage(user_info)
        
        chatbot = _get_chatbot()

        # Ensure a session id exists
        session_id = req.session_id or str(uuid.uuid4())

        # Load history (only for authenticated users)
        history = session_store.load(session_id) if is_authenticated else []

        # Get response
        response_text = chatbot.chat(req.message, history)

        if is_authenticated:
            # Update and persist history only for authenticated users
            history.append(HumanMessage(content=req.message))
            history.append(AIMessage(content=response_text))

            # Keep last 20 messages
            if len(history) > 20:
                history = history[-20:]

            session_store.save(session_id, history)

        return ChatResponse(response=response_text, session_id=session_id)

    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear/{session_id}")
def clear(session_id: str) -> Dict[str, str]:
    session_store.clear(session_id)
    return {"message": "Session cleared"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


