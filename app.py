from __future__ import annotations
import io, json, logging, time
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pathlib import Path

from core.config import CIVICFIX_API_KEY, RATE_LIMIT_PER_MIN, SESSION_TTL_SECONDS, ENABLE_LLM_AGENT
from core.schemas import FeedbackRequest, ProcedureRequest, TrackRequest, EscalateRequest, MemoryDeleteRequest
from core.storage import get_object_store
from core.session import get_session_store
from core.agent_v2 import CivicFixAgent
from core.memory import memory_upsert, memory_decay_cleanup, memory_delete_user
from core.tickets import create_ticket, get_ticket
from core.config import ENABLE_ASR, MAX_UPLOAD_MB, MAX_TEXT_CHARS
from core.embeddings import transcribe_audio
from core.middleware import RequestContextMiddleware, BodySizeLimitMiddleware, RateLimiter, JsonLogger
from core.config import REDIS_URL

logger = logging.getLogger("civicfix")
logging.basicConfig(level=logging.INFO, format="%(message)s")

def log_json(**kwargs):
    logger.info(json.dumps(kwargs, ensure_ascii=False))

app = FastAPI(title="CivicFix API", version="1.2.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Request logging + basic safety limits (required for "end-to-end runnable / reproducible")
app.add_middleware(RequestContextMiddleware, logger=JsonLogger())
# Allow a bit more than MAX_UPLOAD_MB to account for multipart overhead and multiple small fields.
app.add_middleware(BodySizeLimitMiddleware, max_bytes=int((MAX_UPLOAD_MB + 5) * 1024 * 1024))

WEB_DIR = Path(__file__).resolve().parent / "clients" / "web"
# UI-heavy assets are optional per the hackathon spec. If the folder doesn't exist,
# we still want the API to start cleanly (so Streamlit / CLI / bots can use it).
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

store = get_object_store()
sessions = get_session_store()
agent = CivicFixAgent()

# Per-identity rate limiter (Redis-backed if available, otherwise in-memory)
limiter = RateLimiter(per_minute=RATE_LIMIT_PER_MIN, redis_url=REDIS_URL)

@app.get("/", response_class=HTMLResponse)
def home():
    p = WEB_DIR / "index.html"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return """
    <html><body style='font-family:system-ui;margin:2rem'>
      <h2>CivicFix API is running ✅</h2>
      <p>Open <a href='/docs'>/docs</a> for the interactive API docs.</p>
      <p>If you want a standalone web client, add <code>clients/web/index.html</code>.</p>
    </body></html>
    """

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()+"Z"}

def require_api_key(x_api_key: Optional[str]):
    if CIVICFIX_API_KEY and x_api_key != CIVICFIX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def validate_file(upload: UploadFile, allowed_prefix: str, max_mb: int = MAX_UPLOAD_MB) -> bytes:
    if not upload.content_type or not upload.content_type.startswith(allowed_prefix):
        raise HTTPException(status_code=400, detail=f"Invalid content-type for {upload.filename}")
    data = upload.file.read()
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large: {upload.filename}")
    upload.file.seek(0)
    return data

@app.post("/v1/complaints/assist")
async def assist(
    request: Request,
    x_api_key: Optional[str] = Header(None),
    user_id: str = Form(...),
    city: str = Form(...),
    ward_id: str = Form(...),
    landmark: str = Form(""),
    text: str = Form(...),
    preferred_channel: Optional[str] = Form(None),
    tone: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    transcript_text: Optional[str] = Form(None),
    issue_photo: Optional[UploadFile] = File(None),
    screenshot: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    intent: str = Form('new'),
    auto_submit: bool = Form(False),
    ticket_id: Optional[str] = Form(None),
    days_waited: int = Form(0),

    # Optional flag: force local open-source LLM generation on/off.
    # If omitted, defaults to ENABLE_LLM_AGENT from config.
    use_llm: Optional[bool] = Form(None),

):
    require_api_key(x_api_key)
    identity = x_api_key or (request.client.host if request.client else "anon")
    limiter.check(identity)

    if not session_id:
        session_id = sessions.new_session_id()
    sess = sessions.get(session_id) or {"messages": []}

    if text and len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail='Text too long')

    log_json(event="assist_request", user_id=user_id, city=city, ward_id=ward_id,
             has_photo=bool(issue_photo), has_screenshot=bool(screenshot), has_audio=bool(audio))

    photo_img = None
    screenshot_img = None

    if issue_photo:
        b = validate_file(issue_photo, "image/")
        obj = store.put_bytes(b, issue_photo.filename, issue_photo.content_type)
        photo_img = Image.open(io.BytesIO(b)).convert("RGB")
        sess["last_photo_uri"] = obj.uri

    if screenshot:
        b = validate_file(screenshot, "image/")
        obj = store.put_bytes(b, screenshot.filename, screenshot.content_type)
        screenshot_img = Image.open(io.BytesIO(b)).convert("RGB")
        sess["last_screenshot_uri"] = obj.uri

    if audio:
        b = validate_file(audio, "audio/", max_mb=25)
        obj = store.put_bytes(b, audio.filename, audio.content_type)
        sess["last_audio_uri"] = obj.uri
        # Optional ASR: if enabled and transcript not provided
        if ENABLE_ASR and not transcript_text:
            try:
                local_path = store.get_local_path(obj.uri)
                transcript_text = transcribe_audio(local_path)
            except Exception:
                pass

    sess["messages"].append({"role":"user","text": text, "time": time.time()})
    sess["messages"] = sess["messages"][-8:]
    sessions.set(session_id, sess, ttl_seconds=SESSION_TTL_SECONDS)

    try:
        memory_decay_cleanup(user_id=user_id)
    except Exception:
        pass

    # Intent routing
    if intent == 'procedure':
        out = agent.procedure_qa(city=city, text=text)
    elif intent == 'track':
        if not ticket_id:
            raise HTTPException(status_code=400, detail='ticket_id required for track')
        out = agent.track(ticket_id=ticket_id)
    elif intent == 'escalate':
        if not ticket_id:
            raise HTTPException(status_code=400, detail='ticket_id required for escalate')
        out = agent.escalate(city=city, ticket_id=ticket_id, days_waited=days_waited)
    else:
        force_llm_final = ENABLE_LLM_AGENT if use_llm is None else bool(use_llm)
        out = agent.assist_new(
        user_id=user_id,
        text=text,
        city=city,
        ward_id=ward_id,
        landmark=landmark,
        date_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
        preferred_channel=preferred_channel,
        tone=tone,
        issue_photo=photo_img,
        screenshot=screenshot_img,
        transcript_text=transcript_text,
        auto_submit=bool(auto_submit),
        force_llm=force_llm_final
    )
    out["session_id"] = session_id
    return JSONResponse(out)

@app.post("/v1/complaints/feedback")
async def feedback(req: FeedbackRequest, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    memory_upsert(
        user_id=req.user_id,
        memory_text=f"Feedback: {req.outcome}. Ticket: {req.ticket_id or ''}. Notes: {req.notes or ''}",
        payload={"type":"history","outcome":req.outcome,"ticket_id":req.ticket_id or "", "notes": req.notes or ""},
        ttl_days=365,
        version=1
    )
    log_json(event="feedback", user_id=req.user_id, outcome=req.outcome)
    return {"ok": True}


@app.post('/v1/memory/delete')
async def memory_delete(req: MemoryDeleteRequest, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    ok = memory_delete_user(req.user_id)
    log_json(event='memory_delete', user_id=req.user_id)
    return {'ok': bool(ok)}
