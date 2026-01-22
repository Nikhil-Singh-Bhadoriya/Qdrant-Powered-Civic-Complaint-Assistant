from __future__ import annotations
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from .config import QDRANT_URL, QDRANT_MODE, QDRANT_LOCAL_PATH

TEXT_DIM = 384
IMG_DIM = 512

COLLECTIONS = {
    "civic_kb": {"dense_text": qm.VectorParams(size=TEXT_DIM, distance=qm.Distance.COSINE)},
    "jurisdiction_directory": {"dense_text": qm.VectorParams(size=TEXT_DIM, distance=qm.Distance.COSINE)},
    "complaint_templates": {"dense_text": qm.VectorParams(size=TEXT_DIM, distance=qm.Distance.COSINE)},
    "case_library": {
        "dense_text": qm.VectorParams(size=TEXT_DIM, distance=qm.Distance.COSINE),
        "dense_image": qm.VectorParams(size=IMG_DIM, distance=qm.Distance.COSINE),
    },
    "user_memory": {"dense_text": qm.VectorParams(size=TEXT_DIM, distance=qm.Distance.COSINE)},
    "channel_status": {"dense_text": qm.VectorParams(size=TEXT_DIM, distance=qm.Distance.COSINE)},
}

def get_client() -> QdrantClient:
    # Local mode is useful for Jupyter environments without Docker.
    if (QDRANT_MODE or '').lower() == 'local':
        return QdrantClient(path=QDRANT_LOCAL_PATH)
    return QdrantClient(url=QDRANT_URL)

def ensure_collections(client: QdrantClient):
    existing = {c.name for c in client.get_collections().collections}
    for name, cfg in COLLECTIONS.items():
        if name not in existing:
            client.create_collection(collection_name=name, vectors_config=cfg, on_disk_payload=True)

def build_filter(**kwargs) -> Optional[qm.Filter]:
    must = []
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, list):
            must.append(qm.FieldCondition(key=k, match=qm.MatchAny(any=v)))
        else:
            must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
    return qm.Filter(must=must) if must else None
