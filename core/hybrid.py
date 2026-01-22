from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import re, time
from rank_bm25 import BM25Okapi
from qdrant_client.http import models as qm
from .qdrant_store import get_client, ensure_collections
from .embeddings import embed_text, embed_image

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]

class BM25Index:
    def __init__(self, docs: Optional[List[Dict[str, Any]]] = None):
        self.docs: List[Dict[str, Any]] = docs or []
        self.tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def build(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        self.tokens = [tokenize(d.get("text","")) for d in docs]
        self.bm25 = BM25Okapi(self.tokens) if self.tokens else None

    def search(self, query: str, topk: int = 10) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []
        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
        out = []
        for i in idx:
            d = dict(self.docs[i])
            d["bm25_score"] = float(scores[i])
            out.append(d)
        return out

def rrf_fuse(lists: List[List[Dict[str, Any]]], k: int = 60, topk: int = 10) -> List[Dict[str, Any]]:
    scores = {}
    items = {}
    for l in lists:
        for rank, d in enumerate(l, start=1):
            doc_id = d["id"]
            items[doc_id] = d
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = [dict(items[i], rrf_score=scores[i]) for i in scores]
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused[:topk]

def qdrant_dense_search(collection: str, query: str, flt: Optional[qm.Filter] = None, limit: int = 10) -> List[Dict[str, Any]]:
    client = get_client()
    ensure_collections(client)
    qvec = embed_text(query)[0].tolist()
    hits = client.search(
        collection_name=collection,
        query_vector=qm.NamedVector(name="dense_text", vector=qvec),
        query_filter=flt,
        limit=limit,
        with_payload=True,
    )
    return [{"id": h.id, "score": float(h.score), "text": h.payload.get("text",""), "payload": h.payload} for h in hits]

def qdrant_image_search(collection: str, pil_image, flt: Optional[qm.Filter] = None, limit: int = 10) -> List[Dict[str, Any]]:
    client = get_client()
    ensure_collections(client)
    ivec = embed_image(pil_image).tolist()
    hits = client.search(
        collection_name=collection,
        query_vector=qm.NamedVector(name="dense_image", vector=ivec),
        query_filter=flt,
        limit=limit,
        with_payload=True,
    )
    return [{"id": h.id, "score": float(h.score), "text": h.payload.get("text",""), "payload": h.payload} for h in hits]

class HybridRetriever:
    """Hybrid retrieval (dense + sparse BM25) with per-filter cache.
    BM25 is built from the same Qdrant subset (using the same filter) and rebuilt when point count changes.
    """
    def __init__(self):
        self._cache: Dict[Tuple[str,str], Dict[str, Any]] = {}

    def _key(self, collection: str, flt: Optional[qm.Filter]) -> Tuple[str, str]:
        return (collection, (flt.json() if flt else ""))

    def _count(self, collection: str, flt: Optional[qm.Filter]) -> int:
        client = get_client()
        ensure_collections(client)
        try:
            res = client.count(collection_name=collection, count_filter=flt, exact=False)
            return int(res.count)
        except Exception:
            # fallback via scroll
            points, _ = client.scroll(collection_name=collection, scroll_filter=flt, with_payload=False, limit=1)
            return 0 if not points else 1

    def build_bm25_from_qdrant(self, collection: str, flt: Optional[qm.Filter] = None, limit: int = 5000):
        client = get_client()
        ensure_collections(client)
        # Scroll may require multiple calls; we do a bounded loop for demo safety.
        docs = []
        offset = None
        fetched = 0
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                scroll_filter=flt,
                with_payload=True,
                limit=min(256, max(1, limit - fetched)),
                offset=offset
            )
            for p in points:
                docs.append({"id": p.id, "text": p.payload.get("text",""), "payload": p.payload})
            fetched += len(points)
            if not points or fetched >= limit or offset is None:
                break

        idx = BM25Index()
        idx.build(docs)
        k = self._key(collection, flt)
        self._cache[k] = {"idx": idx, "count": len(docs), "built_at": time.time()}

    def _get_or_build(self, collection: str, flt: Optional[qm.Filter]) -> Optional[BM25Index]:
        k = self._key(collection, flt)
        current = self._cache.get(k)
        # If missing or stale by count drift, rebuild
        if current is None:
            self.build_bm25_from_qdrant(collection, flt=flt)
            current = self._cache.get(k)
        else:
            qcount = self._count(collection, flt)
            if qcount > 0 and abs(qcount - int(current.get("count", 0))) >= max(3, int(0.1 * qcount)):
                self.build_bm25_from_qdrant(collection, flt=flt)
                current = self._cache.get(k)
        return current["idx"] if current else None

    def search(self, collection: str, query: str, flt: Optional[qm.Filter] = None, topk: int = 10) -> List[Dict[str, Any]]:
        dense = qdrant_dense_search(collection, query, flt=flt, limit=topk)
        bm25 = self._get_or_build(collection, flt)
        if bm25 is None:
            return dense
        sparse = bm25.search(query, topk=topk)
        return rrf_fuse([dense, sparse], topk=topk)
