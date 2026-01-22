# CivicFix Complete Code Guide - Full Stack Setup & Execution

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Prerequisites & Installation](#prerequisites--installation)
4. [Complete Code Breakdown](#complete-code-breakdown)
5. [Running the Project](#running-the-project)
6. [Key Components](#key-components)

---

## 🎯 Project Overview

**CivicFix** is a full-stack civic complaint assistant that uses:
- **Qdrant**: Vector database for semantic search & memory
- **FastAPI**: Backend API server
- **Streamlit**: Interactive web frontend
- **Machine Learning**: Hybrid search, embeddings, LLM recommendations
- **Redis**: Session & cache management
- **MinIO**: Object storage for uploads

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB UI                              │
│              (streamlit_app.py - Port 8501)                      │
│  Tabs: New Complaint | Track Ticket | Escalate                  │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ├─────────────┬────────────────────────────┐
               │             │                            │
        (Mode A)      (Mode B: Optional)          Direct Module Calls
     In-Process       FastAPI Backend              (Local Development)
     Mode             (app.py - Port 8000)
               │             │
               └─────────────┴────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
      ┌─────────┐       ┌─────────┐      ┌──────────┐
      │ Qdrant  │       │ Redis   │      │ MinIO    │
      │ (6333)  │       │ (6379)  │      │ (9000)   │
      └─────────┘       └─────────┘      └──────────┘
```

---

## 📦 Prerequisites & Installation

### Step 1: System Requirements
```bash
# Python 3.10+
python --version

# Docker (for services)
docker --version
docker-compose --version
```

### Step 2: Clone & Install Dependencies
```bash
cd d:\Concine\FINAL\civic_fixed

# Install Python packages
pip install -r requirements.txt

# Key packages installed:
# - fastapi==0.115.0          (API framework)
# - streamlit==1.41.1         (Web UI)
# - qdrant-client==1.11.3     (Vector DB client)
# - redis==5.0.8              (Redis client)
# - sentence-transformers==3.0.1 (Embeddings)
# - torch==2.3.1              (ML framework)
# - transformers==4.44.2      (NLP models)
# - uvicorn[standard]==0.30.6 (ASGI server)
```

### Step 3: Start Infrastructure
```bash
# Start Docker services (Qdrant, Redis, MinIO)
docker compose up -d

# Verify services
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### Step 4: Seed Demo Data
```bash
# Set Python path to avoid import errors
$env:PYTHONPATH="d:\Concine\FINAL\civic_fixed"

# Run seeding script
python scripts/seed_demo_data.py
```

---

## 🔧 Complete Code Breakdown

### 1. MAIN ENTRY POINT: Streamlit Web Application

**File:** `streamlit_app.py` (497 lines)

```python
"""
Streamlit UI for CivicFix (Qdrant-powered Civic Complaint Assistant)

Two execution modes:
(A) In-process mode (default): Direct core module calls
(B) API mode: Calls FastAPI backend endpoints
"""

import streamlit as st
from PIL import Image
import requests
import json
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CivicFix – Civic Complaint Assistant",
    page_icon="🛠️",
    layout="wide",
)

st.title("🛠️ CivicFix – Civic Complaint Assistant")
st.caption("Qdrant-powered search + memory + recommendations for civic grievances")

# ═══════════════════════════════════════════════════════════════════
# EXECUTION MODE SETTINGS
# ═══════════════════════════════════════════════════════════════════
mode = "In-process (recommended)"
api_base_url = "http://localhost:8000"
api_key = ""
use_llm = True

# ═══════════════════════════════════════════════════════════════════
# LAZY IMPORTS & HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_agent():
    """Load CivicFixAgent only when needed (lazy loading)"""
    from core.agent_v2 import CivicFixAgent
    return CivicFixAgent()

def pil_from_upload(upload) -> Optional[Image.Image]:
    """Convert uploaded file to PIL Image"""
    if not upload:
        return None
    try:
        return Image.open(BytesIO(upload.getvalue())).convert("RGB")
    except Exception:
        return None

def pretty_json(obj: Any) -> str:
    """Format object as readable JSON"""
    return json.dumps(obj, ensure_ascii=False, indent=2)

def call_api_assist(form: Dict[str, Any], files: Dict[str, Any]) -> Dict[str, Any]:
    """POST to FastAPI /v1/complaints/assist endpoint"""
    import requests
    url = api_base_url.rstrip("/") + "/v1/complaints/assist"
    headers = {}
    if api_key.strip():
        headers["X-API-Key"] = api_key.strip()
    
    # Convert booleans for multipart encoding
    form["auto_submit"] = "true" if form.get("auto_submit") else "false"
    form["use_llm"] = "true" if form.get("use_llm") else "false"
    
    r = requests.post(url, data=form, files=files, headers=headers, timeout=120)
    try:
        return r.json()
    except Exception:
        return {"error": "Non-JSON response", "status_code": r.status_code}

def call_api_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST feedback to FastAPI /v1/complaints/feedback"""
    import requests
    url = api_base_url.rstrip("/") + "/v1/complaints/feedback"
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["X-API-Key"] = api_key.strip()
    r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=60)
    try:
        return r.json()
    except Exception:
        return {"ok": False, "status_code": r.status_code}

def call_api_memory_delete(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to delete memory /v1/memory/delete"""
    import requests
    url = api_base_url.rstrip("/") + "/v1/memory/delete"
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["X-API-Key"] = api_key.strip()
    r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=60)
    try:
        return r.json()
    except Exception:
        return {"ok": False}

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════
if "user_id" not in st.session_state:
    st.session_state.user_id = "demo_user"
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "last_ticket_id" not in st.session_state:
    st.session_state.last_ticket_id = ""

# ═══════════════════════════════════════════════════════════════════
# TAB 1: NEW COMPLAINT
# ═══════════════════════════════════════════════════════════════════
tab_new, tab_track, tab_escalate = st.tabs(
    ["🆕 New complaint", "📍 Track ticket", "⬆️ Escalate"]
)

with tab_new:
    st.subheader("New complaint")
    
    with st.form("new_complaint_form", clear_on_submit=False):
        # Input columns
        col1, col2, col3 = st.columns(3)
        with col1:
            user_id = st.text_input("User ID", value=st.session_state.user_id)
            city = st.text_input("City", value="DemoCity")
        with col2:
            ward_id = st.text_input("Ward / Zone ID", value="W-42")
            landmark = st.text_input("Landmark (optional)", value="")
        with col3:
            preferred_channel = st.selectbox(
                "Preferred channel (optional)", 
                options=["", "whatsapp", "phone", "email", "in_person"], 
                index=0
            )
            tone = st.selectbox(
                "Tone (optional)", 
                options=["", "formal", "polite", "urgent", "concise"], 
                index=0
            )
        
        # Complaint text
        text = st.text_area(
            "Complaint text", 
            height=140, 
            placeholder="Describe the issue clearly…"
        )
        
        # File uploads
        colA, colB = st.columns(2)
        with colA:
            issue_photo = st.file_uploader(
                "Issue photo (optional)", 
                type=["png", "jpg", "jpeg"]
            )
        
        # Submit buttons
        submit_col1, submit_col2 = st.columns(2)
        with submit_col1:
            submitted = st.form_submit_button("Generate recommendation")
        with submit_col2:
            direct_submit = st.form_submit_button("Direct submit (no recommendation)")
    
    # PROCESS FORM SUBMISSION
    if submitted or direct_submit:
        st.session_state.user_id = user_id
        
        if not text.strip():
            st.error("Please enter complaint text.")
        else:
            with st.spinner("Processing complaint…"):
                
                if direct_submit:
                    # OPTION A: Direct submit (no recommendation)
                    from core.tickets import create_ticket
                    try:
                        ticket_id = create_ticket(
                            user_id=user_id,
                            city=city,
                            ward_id=ward_id,
                            category="General",
                            department="Not specified",
                            channel=preferred_channel or "Not specified",
                            complaint_text=text,
                            meta={
                                "tone": tone or "Not specified",
                                "landmark": landmark or ""
                            }
                        )
                        st.session_state.last_ticket_id = ticket_id
                        st.success("✅ Complaint submitted successfully!")
                        st.info(f"**Ticket ID:** {ticket_id}")
                        
                    except Exception as e:
                        st.error(f"Failed to create ticket: {str(e)}")
                
                else:
                    # OPTION B: Generate recommendation using agent
                    if mode == "FastAPI backend (API mode)":
                        # Call FastAPI endpoint
                        files = {}
                        if issue_photo:
                            files["issue_photo"] = (
                                issue_photo.name, 
                                issue_photo.getvalue(), 
                                issue_photo.type
                            )
                        
                        form = {
                            "user_id": user_id,
                            "city": city,
                            "ward_id": ward_id,
                            "landmark": landmark or "",
                            "text": text,
                            "preferred_channel": preferred_channel or None,
                            "tone": tone or None,
                            "session_id": st.session_state.session_id,
                            "intent": "new",
                            "auto_submit": True,
                            "use_llm": use_llm,
                        }
                        out = call_api_assist(form, files)
                    
                    else:
                        # In-process mode: call agent directly
                        agent = get_agent()
                        photo_img = pil_from_upload(issue_photo)
                        
                        out = agent.assist_new(
                            user_id=user_id,
                            text=text,
                            city=city,
                            ward_id=ward_id,
                            landmark=landmark or "",
                            date_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
                            preferred_channel=preferred_channel or None,
                            tone=tone or None,
                            issue_photo=photo_img,
                            screenshot=None,
                            transcript_text=None,
                            auto_submit=True,
                            force_llm=bool(use_llm),
                        )
                    
                    # RENDER RESULTS
                    if isinstance(out, dict) and out.get("session_id"):
                        st.session_state.session_id = out["session_id"]
                    
                    if not isinstance(out, dict):
                        st.error("Unexpected response type.")
                    
                    elif out.get("need_more_info"):
                        st.warning("More info needed.")
                        st.write("Missing fields:", ", ".join(out.get("missing_fields", [])))
                    
                    elif out.get("error"):
                        st.error("Request failed.")
                        st.code(pretty_json(out))
                    
                    else:
                        # SUCCESS: Show recommendations
                        ra = out.get("recommended_action", {}) or {}
                        
                        # Display metrics
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Department", ra.get("department", "Sanitation"))
                        c2.metric("Category", ra.get("category", "General"))
                        c3.metric("Best channel", ra.get("best_channel", "helpline"))
                        c4.metric("Backup channel", ra.get("backup_channel", "email"))
                        
                        # Display complaint text
                        st.subheader("Complaint text (ready to paste)")
                        st.text_area(
                            "",
                            value=out.get("complaint_text_ready_to_paste", ""),
                            height=200
                        )
                        
                        # Show tips from similar cases
                        tips = out.get("tips_from_similar_cases", []) or []
                        if tips:
                            st.subheader("Tip from similar cases")
                            for t in tips:
                                st.info(t)
                        
                        # Save ticket
                        if out.get("ticket_id"):
                            st.session_state.last_ticket_id = out["ticket_id"]
                            st.success(f"Demo ticket created: **{out['ticket_id']}**")
                        
                        # Show LLM enhancement if available
                        if out.get("llm_markdown"):
                            st.markdown("### 💡 AI-Enhanced Recommendation")
                            st.markdown(out["llm_markdown"])
                        
                        # FEEDBACK SECTION
                        st.divider()
                        st.subheader("Feedback (improves memory)")
                        fb_col1, fb_col2 = st.columns([1, 2])
                        
                        with fb_col1:
                            outcome = st.selectbox(
                                "Outcome",
                                options=[
                                    "resolved", 
                                    "not_resolved", 
                                    "wrong_department",
                                    "portal_down", 
                                    "submitted"
                                ],
                                index=4
                            )
                        with fb_col2:
                            notes = st.text_input("Notes (optional)", value="")
                        
                        if st.button("Submit feedback"):
                            payload = {
                                "user_id": user_id,
                                "session_id": st.session_state.session_id,
                                "ticket_id": out.get("ticket_id") or st.session_state.last_ticket_id,
                                "outcome": outcome,
                                "notes": notes or None,
                            }
                            
                            if mode == "FastAPI backend (API mode)":
                                resp = call_api_feedback(payload)
                            else:
                                from core.memory import memory_upsert
                                memory_upsert(
                                    user_id=user_id,
                                    memory_text=f"Feedback: {outcome}. Ticket: {payload.get('ticket_id')}",
                                    payload={
                                        "type": "history",
                                        "outcome": outcome,
                                        "ticket_id": payload.get("ticket_id") or "",
                                        "notes": notes or ""
                                    },
                                    ttl_days=365,
                                    version=1,
                                )
                                resp = {"ok": True}
                            
                            if resp.get("ok"):
                                st.success("Thanks! Feedback saved.")
                            else:
                                st.error("Feedback failed.")

# ═══════════════════════════════════════════════════════════════════
# TAB 2: TRACK TICKET
# ═══════════════════════════════════════════════════════════════════
with tab_track:
    st.subheader("Track a ticket (demo store)")
    ticket_id = st.text_input(
        "Ticket ID", 
        value=st.session_state.last_ticket_id or ""
    )
    go = st.button("Track ticket")
    
    if go and ticket_id.strip():
        with st.spinner("Looking up ticket…"):
            if mode == "FastAPI backend (API mode)":
                out = call_api_assist(
                    form={
                        "user_id": st.session_state.user_id,
                        "city": "DemoCity",
                        "ward_id": "W-42",
                        "text": "track",
                        "intent": "track",
                        "ticket_id": ticket_id.strip(),
                        "use_llm": False,
                    },
                    files={},
                )
            else:
                agent = get_agent()
                out = agent.track(ticket_id.strip())
        
        if out.get("found"):
            st.success("✅ Ticket found!")
            
            # Display ticket info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ticket ID", out.get("ticket_id", "N/A"))
                st.metric("Status", out.get("status", "N/A").upper())
                st.metric("Category", out.get("category", "N/A"))
                st.metric("Channel", out.get("channel", "N/A"))
            
            with col2:
                st.metric("City", out.get("city", "N/A"))
                st.metric("Ward", out.get("ward_id", "N/A"))
                st.metric("Department", out.get("department", "N/A") or "Pending")
            
            st.divider()
            st.subheader("Complaint Details")
            st.write(out.get("complaint_text", "No description"))

# ═══════════════════════════════════════════════════════════════════
# TAB 3: ESCALATE COMPLAINT
# ═══════════════════════════════════════════════════════════════════
with tab_escalate:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 10px; color: white;">
        <h2>⬆️ Escalate Your Complaint</h2>
        <p>Escalate to senior authorities when not resolved.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        esc_ticket_id = st.text_input(
            "🎫 Ticket ID", 
            value=st.session_state.last_ticket_id or "", 
            key="esc_ticket"
        )
    with col2:
        days_waited = st.number_input(
            "⏱️ Days Waited", 
            min_value=0, 
            max_value=365, 
            value=7
        )
    
    esc_city = "DemoCity"
    
    go = st.button("🚀 Get Escalation Plan", use_container_width=True)
    
    if go and esc_ticket_id.strip():
        with st.spinner("🔄 Building your escalation strategy…"):
            if mode == "FastAPI backend (API mode)":
                out = call_api_assist(
                    form={
                        "user_id": st.session_state.user_id,
                        "city": esc_city,
                        "ward_id": "W-42",
                        "text": "escalate",
                        "intent": "escalate",
                        "ticket_id": esc_ticket_id.strip(),
                        "days_waited": int(days_waited),
                        "use_llm": False,
                    },
                    files={},
                )
            else:
                agent = get_agent()
                out = agent.escalate(
                    city=esc_city,
                    ticket_id=esc_ticket_id.strip(),
                    days_waited=int(days_waited)
                )
        
        if out.get("ok"):
            st.success("✅ Escalation plan created!")
            
            # Status cards
            st.markdown("### 📊 Current Status")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Current Status",
                    out.get('current_status', 'N/A').upper()
                )
            with col2:
                st.metric(
                    "Service Level Agreement",
                    f"{out.get('sla_days', 'N/A')} days"
                )
            
            st.divider()
            st.markdown("### 🪜 Recommended Escalation Steps")
            
            steps = out.get("recommended_escalation", []) or []
            for step_num, step in enumerate(steps, 1):
                with st.container():
                    col_num, col_content = st.columns([0.5, 9.5])
                    with col_num:
                        st.markdown(
                            f"<h3 style='text-align: center; color: #667eea;'>{step_num}</h3>",
                            unsafe_allow_html=True
                        )
                    with col_content:
                        st.markdown(f"**Step {step_num}:** {step}")
                st.divider()
```

---

### 2. BACKEND API: FastAPI Application

**File:** `app.py` (208 lines)

```python
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import json, logging, time
from PIL import Image
import io
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION & IMPORTS
# ═══════════════════════════════════════════════════════════════════
from core.config import (
    CIVICFIX_API_KEY, 
    RATE_LIMIT_PER_MIN, 
    SESSION_TTL_SECONDS,
    ENABLE_LLM_AGENT,
    ENABLE_ASR,
    MAX_UPLOAD_MB,
    MAX_TEXT_CHARS,
    REDIS_URL
)
from core.schemas import (
    FeedbackRequest, 
    ProcedureRequest, 
    TrackRequest,
    EscalateRequest, 
    MemoryDeleteRequest
)
from core.storage import get_object_store
from core.session import get_session_store
from core.agent_v2 import CivicFixAgent
from core.memory import memory_upsert, memory_decay_cleanup, memory_delete_user
from core.tickets import create_ticket, get_ticket
from core.embeddings import transcribe_audio
from core.middleware import (
    RequestContextMiddleware, 
    BodySizeLimitMiddleware, 
    RateLimiter, 
    JsonLogger
)

# ═══════════════════════════════════════════════════════════════════
# APP INITIALIZATION
# ═══════════════════════════════════════════════════════════════════
logger = logging.getLogger("civicfix")
logging.basicConfig(level=logging.INFO, format="%(message)s")

def log_json(**kwargs):
    logger.info(json.dumps(kwargs, ensure_ascii=False))

app = FastAPI(title="CivicFix API", version="1.2.1")

# CORS Middleware - Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Request logging & safety limits
app.add_middleware(RequestContextMiddleware, logger=JsonLogger())
app.add_middleware(
    BodySizeLimitMiddleware,
    max_bytes=int((MAX_UPLOAD_MB + 5) * 1024 * 1024)
)

# Initialize services
store = get_object_store()
sessions = get_session_store()
agent = CivicFixAgent()
limiter = RateLimiter(per_minute=RATE_LIMIT_PER_MIN, redis_url=REDIS_URL)

# ═══════════════════════════════════════════════════════════════════
# HEALTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def home():
    """Root endpoint - returns API info"""
    return """
    <html><body style='font-family:system-ui;margin:2rem'>
      <h2>CivicFix API is running ✅</h2>
      <p>Open <a href='/docs'>/docs</a> for interactive API docs.</p>
    </body></html>
    """

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "ok": True,
        "time": datetime.utcnow().isoformat() + "Z"
    }

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def require_api_key(x_api_key: Optional[str]):
    """Validate API key if configured"""
    if CIVICFIX_API_KEY and x_api_key != CIVICFIX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def validate_file(
    upload: UploadFile,
    allowed_prefix: str,
    max_mb: int = MAX_UPLOAD_MB
) -> bytes:
    """Validate uploaded file type and size"""
    if not upload.content_type or \
       not upload.content_type.startswith(allowed_prefix):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content-type for {upload.filename}"
        )
    
    data = upload.file.read()
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {upload.filename}"
        )
    
    upload.file.seek(0)
    return data

# ═══════════════════════════════════════════════════════════════════
# MAIN ENDPOINT: /v1/complaints/assist
# ═══════════════════════════════════════════════════════════════════

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
    use_llm: Optional[bool] = Form(None),
):
    """
    Main complaint processing endpoint
    
    Intents:
    - 'new': Generate recommendation for new complaint
    - 'track': Track existing ticket
    - 'escalate': Escalate unresolved ticket
    - 'procedure': Answer procedure questions
    """
    
    # Validate API key
    require_api_key(x_api_key)
    
    # Rate limiting
    identity = x_api_key or (request.client.host if request.client else "anon")
    limiter.check(identity)
    
    # Session management
    if not session_id:
        session_id = sessions.new_session_id()
    sess = sessions.get(session_id) or {"messages": []}
    
    # Validate text length
    if text and len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail='Text too long')
    
    log_json(
        event="assist_request",
        user_id=user_id,
        city=city,
        has_photo=bool(issue_photo),
        intent=intent
    )
    
    # ═══════════════════════════════════════════════════════════════
    # FILE PROCESSING
    # ═══════════════════════════════════════════════════════════════
    
    photo_img = None
    screenshot_img = None
    
    # Process photo upload
    if issue_photo:
        b = validate_file(issue_photo, "image/")
        obj = store.put_bytes(
            b,
            issue_photo.filename,
            issue_photo.content_type
        )
        photo_img = Image.open(io.BytesIO(b)).convert("RGB")
        sess["last_photo_uri"] = obj.uri
    
    # Process screenshot
    if screenshot:
        b = validate_file(screenshot, "image/")
        obj = store.put_bytes(
            b,
            screenshot.filename,
            screenshot.content_type
        )
        screenshot_img = Image.open(io.BytesIO(b)).convert("RGB")
        sess["last_screenshot_uri"] = obj.uri
    
    # Process audio & transcribe if needed
    if audio:
        b = validate_file(audio, "audio/", max_mb=25)
        obj = store.put_bytes(
            b,
            audio.filename,
            audio.content_type
        )
        sess["last_audio_uri"] = obj.uri
        
        # Optional ASR
        if ENABLE_ASR and not transcript_text:
            try:
                local_path = store.get_local_path(obj.uri)
                transcript_text = transcribe_audio(local_path)
            except Exception:
                pass
    
    # ═══════════════════════════════════════════════════════════════
    # UPDATE SESSION
    # ═══════════════════════════════════════════════════════════════
    
    sess["messages"].append({
        "role": "user",
        "text": text,
        "time": time.time()
    })
    sess["messages"] = sess["messages"][-8:]  # Keep last 8 messages
    sessions.set(session_id, sess, ttl_seconds=SESSION_TTL_SECONDS)
    
    # Cleanup old memories
    try:
        memory_decay_cleanup(user_id=user_id)
    except Exception:
        pass
    
    # ═══════════════════════════════════════════════════════════════
    # INTENT ROUTING
    # ═══════════════════════════════════════════════════════════════
    
    if intent == 'procedure':
        out = agent.procedure_qa(city=city, text=text)
    
    elif intent == 'track':
        if not ticket_id:
            raise HTTPException(
                status_code=400,
                detail='ticket_id required for track'
            )
        out = agent.track(ticket_id=ticket_id)
    
    elif intent == 'escalate':
        if not ticket_id:
            raise HTTPException(
                status_code=400,
                detail='ticket_id required for escalate'
            )
        out = agent.escalate(
            city=city,
            ticket_id=ticket_id,
            days_waited=days_waited
        )
    
    else:  # Default: 'new' complaint
        force_llm_final = (
            ENABLE_LLM_AGENT if use_llm is None 
            else bool(use_llm)
        )
        
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

# ═══════════════════════════════════════════════════════════════════
# FEEDBACK ENDPOINT: /v1/complaints/feedback
# ═══════════════════════════════════════════════════════════════════

@app.post("/v1/complaints/feedback")
async def feedback(
    req: FeedbackRequest,
    x_api_key: Optional[str] = Header(None)
):
    """
    Submit feedback on complaint outcome
    
    Feedback outcomes:
    - 'resolved': Problem was fixed
    - 'not_resolved': Problem persists
    - 'wrong_department': Sent to wrong department
    - 'portal_down': Portal was unavailable
    - 'submitted': Successfully submitted
    """
    
    require_api_key(x_api_key)
    
    # Store feedback in memory
    memory_upsert(
        user_id=req.user_id,
        memory_text=f"Feedback: {req.outcome}. Ticket: {req.ticket_id or ''}",
        payload={
            "type": "history",
            "outcome": req.outcome,
            "ticket_id": req.ticket_id or "",
            "notes": req.notes or ""
        },
        ttl_days=365,
        version=1
    )
    
    log_json(
        event="feedback",
        user_id=req.user_id,
        outcome=req.outcome
    )
    
    return {"ok": True}

# ═══════════════════════════════════════════════════════════════════
# MEMORY ENDPOINT: /v1/memory/delete
# ═══════════════════════════════════════════════════════════════════

@app.post('/v1/memory/delete')
async def memory_delete(
    req: MemoryDeleteRequest,
    x_api_key: Optional[str] = Header(None)
):
    """Delete all memory records for a user"""
    
    require_api_key(x_api_key)
    
    ok = memory_delete_user(req.user_id)
    
    log_json(
        event='memory_delete',
        user_id=req.user_id
    )
    
    return {'ok': bool(ok)}
```

---

### 3. CONFIGURATION

**File:** `core/config.py`

```python
import os

def env(key: str, default: str | None = None) -> str | None:
    """Read environment variable with default"""
    v = os.environ.get(key)
    return v if v is not None else default

# ═══════════════════════════════════════════════════════════════════
# SERVICE URLS
# ═══════════════════════════════════════════════════════════════════

QDRANT_URL = env("QDRANT_URL", "http://localhost:6333")
REDIS_URL = env("REDIS_URL", "redis://localhost:6379/0")

# ═══════════════════════════════════════════════════════════════════
# API SETTINGS
# ═══════════════════════════════════════════════════════════════════

CIVICFIX_API_KEY = env("CIVICFIX_API_KEY")
RATE_LIMIT_PER_MIN = int(env("RATE_LIMIT_PER_MIN", "60"))

# ═══════════════════════════════════════════════════════════════════
# OBJECT STORAGE
# ═══════════════════════════════════════════════════════════════════

OBJECT_STORE = env("OBJECT_STORE", "local")  # local|minio
LOCAL_STORE_DIR = env("LOCAL_STORE_DIR", "./object_store")

MINIO_ENDPOINT = env("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = env("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = env("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = env("MINIO_BUCKET", "civicfix")

# ═══════════════════════════════════════════════════════════════════
# LLM SETTINGS
# ═══════════════════════════════════════════════════════════════════

OPENAI_API_KEY = env("OPENAI_API_KEY")
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-4.1-mini")
ENABLE_LLM_AGENT = env("ENABLE_LLM_AGENT", "1") == "1"
ENABLE_LLM = ENABLE_LLM_AGENT

# ═══════════════════════════════════════════════════════════════════
# FEATURE FLAGS
# ═══════════════════════════════════════════════════════════════════

ENABLE_HYBRID = env("ENABLE_HYBRID", "1") == "1"  # BM25 + Dense search
ENABLE_RERANK = env("ENABLE_RERANK", "1") == "1"  # CrossEncoder reranking
ENABLE_OCR = env("ENABLE_OCR", "1") == "1"        # Extract text from images
ENABLE_ASR = env("ENABLE_ASR", "0") == "1"        # Audio transcription
ENABLE_IMAGE_HINTS = env("ENABLE_IMAGE_HINTS", "1") == "1"

# ═══════════════════════════════════════════════════════════════════
# QDRANT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

QDRANT_MODE = env('QDRANT_MODE', 'remote')  # remote|local
QDRANT_LOCAL_PATH = env('QDRANT_LOCAL_PATH', 'data/qdrant_local')

# ═══════════════════════════════════════════════════════════════════
# MODEL SETTINGS
# ═══════════════════════════════════════════════════════════════════

RERANK_MODEL = env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_MODEL = env('LLM_MODEL', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')

# ═══════════════════════════════════════════════════════════════════
# SESSION & DATA LIMITS
# ═══════════════════════════════════════════════════════════════════

SESSION_TTL_SECONDS = int(env("SESSION_TTL_SECONDS", "1800"))
MAX_UPLOAD_MB = int(env('MAX_UPLOAD_MB', '25'))
MAX_TEXT_CHARS = int(env('MAX_TEXT_CHARS', '4000'))
DATA_DIR = env('DATA_DIR', 'data')
```

---

## 🚀 Running the Project

### Complete Startup Commands

```bash
# 1. Navigate to project directory
cd d:\Concine\FINAL\civic_fixed

# 2. Set Python path (important!)
$env:PYTHONPATH="d:\Concine\FINAL\civic_fixed"

# 3. Start Docker services (Terminal 1)
docker compose up -d

# Wait for services to start
Start-Sleep -Seconds 5

# 4. Seed demo data (Terminal 2)
python scripts/seed_demo_data.py

# 5. Start FastAPI backend (Terminal 3)
python -m uvicorn app:app --reload --port 8000

# 6. Start Streamlit frontend (Terminal 4)
streamlit run streamlit_app.py --server.port 8501
```

### Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| **Streamlit UI** | http://localhost:8501 | Web interface |
| **FastAPI Docs** | http://localhost:8000/docs | Interactive API docs |
| **API Health** | http://localhost:8000/health | Health check |
| **Qdrant Console** | http://localhost:6333 | Vector DB admin |
| **MinIO Console** | http://localhost:9001 | Object storage UI |

---

## 📦 Key Core Components

### Agent Module
**File:** `core/agent_v2.py`

Handles all complaint processing:
- Text preprocessing & PII redaction
- Image OCR & vision hints
- Hybrid search (BM25 + semantic)
- Department & channel recommendations
- Memory-based learning

### Memory System
**File:** `core/memory.py`

Stores user feedback to improve recommendations:
- Feedback-based learning
- TTL-based decay
- Pattern recognition across complaints

### Vector Database
**File:** `core/qdrant_store.py`

Manages Qdrant connections:
- Semantic search
- Multimodal retrieval
- Hybrid filtering

### Ticket Management
**File:** `core/tickets.py`

Demo ticket storage:
- Create complaints
- Track status
- Store metadata

---

## 🎯 Workflow

1. **User submits complaint** via Streamlit UI
2. **Streamlit UI** calls agent (in-process) or FastAPI backend
3. **Agent processes** using:
   - Text cleanup & PII redaction
   - Image analysis if provided
   - Memory retrieval (past similar cases)
   - Hybrid search for best recommendations
   - LLM enhancement (optional)
4. **Results displayed** with:
   - Recommended department & category
   - Best communication channel
   - Pre-written complaint template
   - Tips from similar cases
   - Ticket ID for tracking
5. **User provides feedback** → Stored in memory for future improvements
6. **User can track** ticket status
7. **User can escalate** if not resolved

---

## 🔧 Advanced: Running in Different Modes

### Mode A: In-Process (Recommended for Development)
```bash
streamlit run streamlit_app.py
# Streamlit calls core modules directly (no API calls)
```

### Mode B: API Mode (Production-Ready)
```bash
# Terminal 1: Start backend
python -m uvicorn app:app --reload --port 8000

# Terminal 2: Start frontend
streamlit run streamlit_app.py
# Streamlit calls FastAPI endpoints
```

---

## 📊 Environment Variables

Create a `.env` file or set environment variables:

```bash
# Qdrant
QDRANT_URL=http://localhost:6333

# Redis
REDIS_URL=redis://localhost:6379/0

# Storage
OBJECT_STORE=local
LOCAL_STORE_DIR=./object_store

# LLM
ENABLE_LLM_AGENT=1
LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Features
ENABLE_HYBRID=1
ENABLE_RERANK=1
ENABLE_OCR=1
ENABLE_ASR=0

# Limits
MAX_UPLOAD_MB=25
MAX_TEXT_CHARS=4000
SESSION_TTL_SECONDS=1800
RATE_LIMIT_PER_MIN=60
```

---

## ✅ Verification Checklist

- [ ] Docker containers running (`docker ps`)
- [ ] Python dependencies installed (`pip list`)
- [ ] API responds to health check (`curl http://localhost:8000/health`)
- [ ] Qdrant collections created
- [ ] Demo data seeded
- [ ] Streamlit UI loads on http://localhost:8501
- [ ] Can submit new complaint
- [ ] Can track ticket
- [ ] Can escalate complaint

---

**This guide covers the entire code flow from starting services to running the full Streamlit website!** 🎉
