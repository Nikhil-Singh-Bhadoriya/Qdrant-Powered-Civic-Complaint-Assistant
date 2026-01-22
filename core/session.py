from __future__ import annotations
import json, time, uuid
from typing import Any, Dict, Optional
from .config import REDIS_URL, SESSION_TTL_SECONDS

class SessionStore:
    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError
    def set(self, session_id: str, data: Dict[str, Any], ttl_seconds: int = SESSION_TTL_SECONDS) -> None:
        raise NotImplementedError
    def new_session_id(self) -> str:
        return str(uuid.uuid4())

class InMemoryTTLSessionStore(SessionStore):
    def __init__(self):
        self._store: Dict[str, tuple[float, Dict[str, Any]]] = {}

    def _gc(self):
        now = time.time()
        expired = [k for k, (exp, _v) in self._store.items() if exp <= now]
        for k in expired:
            self._store.pop(k, None)

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        self._gc()
        if session_id not in self._store:
            return None
        exp, v = self._store[session_id]
        if exp <= time.time():
            self._store.pop(session_id, None)
            return None
        return v

    def set(self, session_id: str, data: Dict[str, Any], ttl_seconds: int = SESSION_TTL_SECONDS) -> None:
        self._gc()
        self._store[session_id] = (time.time() + ttl_seconds, data)

class RedisSessionStore(SessionStore):
    def __init__(self):
        import redis
        self.r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        raw = self.r.get(f"civicfix:sess:{session_id}")
        return json.loads(raw) if raw else None

    def set(self, session_id: str, data: Dict[str, Any], ttl_seconds: int = SESSION_TTL_SECONDS) -> None:
        self.r.setex(f"civicfix:sess:{session_id}", ttl_seconds, json.dumps(data))

def get_session_store() -> SessionStore:
    try:
        return RedisSessionStore()
    except Exception:
        return InMemoryTTLSessionStore()
