from __future__ import annotations
from typing import Dict, Any, Optional
from qdrant_client.http import models as qm
from .qdrant_store import get_client, ensure_collections, build_filter
from .embeddings import embed_text

def get_channel_status(city: str, channel: str) -> Optional[Dict[str, Any]]:
    client = get_client()
    ensure_collections(client)
    flt = build_filter(city=city, channel=channel)
    qvec = embed_text(f"{channel} status")[0].tolist()
    hits = client.search("channel_status", query_vector=qm.NamedVector(name="dense_text", vector=qvec),
                         query_filter=flt, limit=1, with_payload=True)
    return hits[0].payload if hits else None

def score_channel(channel: str, urgency: str, portal_ok: bool, user_pref: Optional[str]) -> float:
    s = 0.0
    if urgency == "high":
        s += 2.0 if channel in ["helpline","email"] else 0.6
    elif urgency == "medium":
        s += 1.2 if channel in ["app","portal"] else 0.5
    else:
        s += 1.0 if channel in ["portal","app"] else 0.3
    if channel == "portal" and not portal_ok:
        s -= 2.5
    if user_pref and channel == user_pref:
        s += 1.5
    return s

def recommend_channel(kb_payload: Dict[str, Any], city: str, urgency: str, user_pref: Optional[str]):
    channels = kb_payload.get("channel_type", [])
    portal_status = get_channel_status(city, "portal")
    portal_ok = True if (portal_status is None or portal_status.get("status") == "up") else False
    scored = [(ch, score_channel(ch, urgency, portal_ok, user_pref)) for ch in channels]
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0][0] if scored else None
    backup = scored[1][0] if len(scored) > 1 else None
    return best, backup, {"scored": scored, "portal_ok": portal_ok}
