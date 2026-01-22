from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from qdrant_client.http import models as qm
from .qdrant_store import get_client, ensure_collections, build_filter
from .embeddings import embed_text

def now_utc():
    return datetime.now(timezone.utc)

def memory_get(user_id: str, limit: int = 10):
    client = get_client()
    ensure_collections(client)
    flt = build_filter(user_id=user_id, delete_flag=False)
    qvec = embed_text("user profile preferences history")[0].tolist()
    return client.search(
        collection_name="user_memory",
        query_vector=qm.NamedVector(name="dense_text", vector=qvec),
        query_filter=flt,
        limit=limit,
        with_payload=True
    )

def memory_upsert(user_id: str, memory_text: str, payload: Dict[str, Any], ttl_days: int = 180, version: int = 1, point_id: Optional[int] = None):
    client = get_client()
    ensure_collections(client)
    if point_id is None:
        point_id = int(now_utc().timestamp() * 1_000_000)
    full = {
        "user_id": user_id,
        "delete_flag": False,
        "ttl_days": str(ttl_days),
        "version": str(version),
        "last_updated": now_utc().isoformat(),
        "memory_text": memory_text,
        **payload
    }
    client.upsert(
        collection_name="user_memory",
        points=[qm.PointStruct(id=point_id, vector={"dense_text": embed_text(memory_text)[0].tolist()}, payload=full)]
    )
    return point_id

def memory_delete_user(user_id: str):
    client = get_client()
    ensure_collections(client)
    flt = build_filter(user_id=user_id)
    client.delete("user_memory", points_selector=qm.FilterSelector(filter=flt))
    return True

def memory_decay_cleanup(user_id: Optional[str] = None, now: Optional[datetime] = None) -> int:
    client = get_client()
    ensure_collections(client)
    now = now or now_utc()
    flt = build_filter(user_id=user_id) if user_id else None
    points, _ = client.scroll(collection_name="user_memory", scroll_filter=flt, with_payload=True, limit=1000)

    to_delete = []
    for p in points:
        try:
            ttl = int(p.payload.get("ttl_days", "180"))
        except Exception:
            ttl = 180
        lu = p.payload.get("last_updated")
        if not lu:
            continue
        try:
            ts = datetime.fromisoformat(lu)
        except Exception:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < now - timedelta(days=ttl):
            to_delete.append(p.id)

    if to_delete:
        client.delete("user_memory", points_selector=qm.PointIdsList(points=to_delete))
    return len(to_delete)

def reinforce_preference(user_id: str, channel: str, ttl_days: int = 180):
    hits = memory_get(user_id, limit=20)
    pref = None
    for h in hits:
        if h.payload.get("type") == "preference":
            pref = h
            break
    if pref is None:
        memory_upsert(user_id, f"Preference: user tends to use {channel}",
                      {"type":"preference","pref_channel":channel,"pref_weight":"1"},
                      ttl_days=ttl_days, version=1)
        return
    w = int(pref.payload.get("pref_weight","1")) + 1
    updated = dict(pref.payload)
    updated["pref_channel"] = channel
    updated["pref_weight"] = str(w)
    updated["last_updated"] = now_utc().isoformat()
    updated["memory_text"] = f"Preference: user tends to use {channel} (reinforced x{w})"
    client = get_client()
    ensure_collections(client)
    client.upsert(
        collection_name="user_memory",
        points=[qm.PointStruct(id=pref.id, vector={"dense_text": embed_text(updated["memory_text"])[0].tolist()}, payload=updated)]
    )
