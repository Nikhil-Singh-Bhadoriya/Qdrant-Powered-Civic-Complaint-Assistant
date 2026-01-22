from __future__ import annotations
import os, glob
from typing import List, Dict
from qdrant_client.http import models as qm

from core.qdrant_store import get_client, ensure_collections
from core.embeddings import embed_text
from core.preprocess import redact_pii

DATA_DIR = os.environ.get("KB_DIR", "data/kb_sources")

def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+max_chars])
        i += max_chars - overlap
    return chunks

def read_any(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".md"]:
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    if ext == ".pdf":
        try:
            import pypdf
            r = pypdf.PdfReader(path)
            return "\n".join((p.extract_text() or "") for p in r.pages)
        except Exception:
            return ""
    if ext in [".html", ".htm"]:
        try:
            from bs4 import BeautifulSoup
            html = open(path, "r", encoding="utf-8", errors="ignore").read()
            return BeautifulSoup(html, "html.parser").get_text("\n")
        except Exception:
            return ""
    return ""

def main():
    client = get_client()
    ensure_collections(client)

    paths = []
    for pat in ["*.txt","*.md","*.pdf","*.html","*.htm"]:
        paths.extend(glob.glob(os.path.join(DATA_DIR, pat)))

    if not paths:
        print(f"No documents found under {DATA_DIR}. Put civic procedure docs there and re-run.")
        return

    points = []
    pid = 10_000
    for path in paths:
        raw = read_any(path)
        raw = redact_pii(raw)
        for ch in chunk_text(raw):
            if len(ch.strip()) < 50:
                continue
            payload = {
                "city": os.environ.get("CITY", "DemoCity"),
                "state": os.environ.get("STATE", "DemoState"),
                "language": os.environ.get("LANG", "en"),
                "department": os.environ.get("DEPARTMENT", "General"),
                "category": os.environ.get("CATEGORY", "General"),
                "channel_type": ["portal","helpline","email"],
                "required_fields": ["location","landmark","date_time","photo"],
                "sla_days": int(os.environ.get("SLA_DAYS", "7")),
                "source": f"kb_file:{os.path.basename(path)}",
                "last_updated": os.environ.get("LAST_UPDATED", "2026-01-01"),
                "text": ch
            }
            vec = embed_text(ch)[0].tolist()
            points.append(qm.PointStruct(id=pid, vector={"dense_text": vec}, payload=payload))
            pid += 1

    client.upsert(collection_name="civic_kb", points=points)
    print(f"Ingested {len(points)} chunks into civic_kb from {len(paths)} files.")

if __name__ == "__main__":
    main()
