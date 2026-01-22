from __future__ import annotations
import os

def env(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key)
    return v if v is not None else default

QDRANT_URL = env("QDRANT_URL", "http://localhost:6333")
REDIS_URL = env("REDIS_URL", "redis://localhost:6379/0")

CIVICFIX_API_KEY = env("CIVICFIX_API_KEY")
RATE_LIMIT_PER_MIN = int(env("RATE_LIMIT_PER_MIN", "60"))

OBJECT_STORE = env("OBJECT_STORE", "local")  # local|minio
LOCAL_STORE_DIR = env("LOCAL_STORE_DIR", "./object_store")

MINIO_ENDPOINT = env("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = env("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = env("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = env("MINIO_BUCKET", "civicfix")

OPENAI_API_KEY = env("OPENAI_API_KEY")
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-4.1-mini")
ENABLE_LLM_AGENT = env("ENABLE_LLM_AGENT", "1") == "1"

# Back-compat alias used by some modules.
# (This project uses ONLY a local open-source LLM via Transformers; no OpenAI calls are required.)
ENABLE_LLM = ENABLE_LLM_AGENT

ENABLE_HYBRID = env("ENABLE_HYBRID", "1") == "1"
ENABLE_RERANK = env("ENABLE_RERANK", "1") == "1"
RERANK_MODEL = env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

SESSION_TTL_SECONDS = int(env("SESSION_TTL_SECONDS", "1800"))


# Data directory for local stores (tickets, object store, etc.)
DATA_DIR = env('DATA_DIR', 'data')

MAX_UPLOAD_MB = int(env('MAX_UPLOAD_MB', '25'))
MAX_TEXT_CHARS = int(env('MAX_TEXT_CHARS', '4000'))
ENABLE_ASR = env('ENABLE_ASR', '0') == '1'
ENABLE_OCR = env('ENABLE_OCR', '1') == '1'
ENABLE_IMAGE_HINTS = env('ENABLE_IMAGE_HINTS', '1') == '1'
# Qdrant mode: 'remote' (default) uses QDRANT_URL; 'local' uses QDRANT_LOCAL_PATH.
QDRANT_MODE = env('QDRANT_MODE', 'remote')
QDRANT_LOCAL_PATH = env('QDRANT_LOCAL_PATH', 'data/qdrant_local')

LLM_MODEL = env('LLM_MODEL', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')