from __future__ import annotations
import os, sqlite3, time, uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from .config import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "tickets.db")

def _conn():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS tickets(
        ticket_id TEXT PRIMARY KEY,
        user_id TEXT,
        created_ts INTEGER,
        city TEXT,
        ward_id TEXT,
        category TEXT,
        department TEXT,
        channel TEXT,
        status TEXT,
        complaint_text TEXT,
        meta_json TEXT
    )
    """)
    return conn

def create_ticket(
    user_id: str,
    city: str,
    ward_id: str,
    category: str,
    department: str,
    channel: str,
    complaint_text: str,
    meta: Optional[Dict[str, Any]] = None
) -> str:
    ticket_id = "CF-" + uuid.uuid4().hex[:10].upper()
    meta_json = json_dumps(meta or {})
    with _conn() as c:
        c.execute(
            "INSERT INTO tickets(ticket_id,user_id,created_ts,city,ward_id,category,department,channel,status,complaint_text,meta_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (ticket_id, user_id, int(time.time()), city, ward_id, category, department, channel, "submitted", complaint_text, meta_json)
        )
    return ticket_id

def update_status(ticket_id: str, status: str):
    with _conn() as c:
        c.execute("UPDATE tickets SET status=? WHERE ticket_id=?", (status, ticket_id))

def get_ticket(ticket_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute("SELECT ticket_id,user_id,created_ts,city,ward_id,category,department,channel,status,complaint_text,meta_json FROM tickets WHERE ticket_id=?",
                        (ticket_id,)).fetchone()
    if not row:
        return None
    return {
        "ticket_id": row[0],
        "user_id": row[1],
        "created_ts": row[2],
        "city": row[3],
        "ward_id": row[4],
        "category": row[5],
        "department": row[6],
        "channel": row[7],
        "status": row[8],
        "complaint_text": row[9],
        "meta": json_loads(row[10]),
    }

def list_tickets(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT ticket_id,created_ts,city,ward_id,category,department,channel,status FROM tickets WHERE user_id=? ORDER BY created_ts DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
    return [
        {"ticket_id": r[0], "created_ts": r[1], "city": r[2], "ward_id": r[3], "category": r[4],
         "department": r[5], "channel": r[6], "status": r[7]}
        for r in rows
    ]

def json_dumps(x) -> str:
    import json
    return json.dumps(x, ensure_ascii=False)

def json_loads(s: str):
    import json
    try:
        return json.loads(s or "{}")
    except Exception:
        return {}
