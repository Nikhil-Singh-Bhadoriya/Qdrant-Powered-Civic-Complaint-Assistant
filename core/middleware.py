from __future__ import annotations
import time, json, uuid
from typing import Callable, Optional
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class JsonLogger:
    def __init__(self, name: str = "civicfix"):
        self.name = name

    def log(self, **kwargs):
        print(json.dumps(kwargs, ensure_ascii=False))

class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, logger: Optional[JsonLogger] = None):
        super().__init__(app)
        self.logger = logger or JsonLogger()

    async def dispatch(self, request: Request, call_next: Callable):
        rid = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        start = time.time()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            dur_ms = int((time.time() - start) * 1000)
            self.logger.log(
                event="http",
                request_id=rid,
                method=request.method,
                path=request.url.path,
                status=(response.status_code if response else None),
                duration_ms=dur_ms,
                client_ip=(request.client.host if request.client else None),
                user_agent=request.headers.get("user-agent"),
            )
            if response is not None:
                response.headers["X-Request-Id"] = rid

class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Callable):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > self.max_bytes:
            raise HTTPException(status_code=413, detail="Request too large")
        return await call_next(request)

class RateLimiter:
    """Per-identity per-minute limiter with Redis fallback."""
    def __init__(self, per_minute: int, redis_url: Optional[str] = None):
        self.per_minute = per_minute
        self.redis = None
        self._mem = {}
        if redis_url:
            try:
                import redis
                self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
            except Exception:
                self.redis = None

    def _key(self, identity: str, window: int) -> str:
        return f"civicfix:rl:{identity}:{window}"

    def check(self, identity: str):
        now = int(time.time())
        window = now // 60
        if self.redis is not None:
            k = self._key(identity, window)
            try:
                v = self.redis.incr(k)
                if int(v) == 1:
                    self.redis.expire(k, 70)
                if int(v) > self.per_minute:
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                return
            except HTTPException:
                raise
            except Exception:
                pass
        # in-memory fallback
        key = (identity, window)
        self._mem[key] = self._mem.get(key, 0) + 1
        if self._mem[key] > self.per_minute:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
