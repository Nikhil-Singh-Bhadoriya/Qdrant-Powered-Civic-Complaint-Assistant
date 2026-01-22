"""
Streamlit UI for CivicFix (Qdrant-powered Civic Complaint Assistant)

How to run (local):
1) Start dependencies (recommended):
   docker compose up -d

2) Seed demo data (once):
   python scripts/seed_demo_data.py

3) Run Streamlit:
   streamlit run streamlit_app.py

Notes:
- This UI can run in two modes:
  (A) In-process mode (default): calls core agent directly (no FastAPI required).
  (B) API mode: calls the FastAPI server (/v1/complaints/assist, /v1/complaints/feedback, /v1/memory/delete).
- Qdrant must be reachable via QDRANT_URL (default http://localhost:6333).
"""

from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Optional

import streamlit as st
from PIL import Image

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="CivicFix – Civic Complaint Assistant",
    page_icon="🛠️",
    layout="wide",
)

st.title("🛠️ CivicFix – Civic Complaint Assistant")
st.caption("Qdrant-powered search + memory + recommendations for civic grievances")


# ----------------------------
# Default settings
mode = "In-process (recommended)"
api_base_url = "http://localhost:8000"
api_key = ""
use_llm = True


# ----------------------------
# Lazy imports (after Streamlit init)
# ----------------------------
# We import core modules only when needed to keep Streamlit startup responsive.
@st.cache_resource(show_spinner=False)
def get_agent() -> Any:
    from core.agent_v2 import CivicFixAgent
    return CivicFixAgent()

def pil_from_upload(upload) -> Optional[Image.Image]:
    if not upload:
        return None
    try:
        return Image.open(BytesIO(upload.getvalue())).convert("RGB")
    except Exception:
        return None

def pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def call_api_assist(form: Dict[str, Any], files: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    url = api_base_url.rstrip("/") + "/v1/complaints/assist"
    headers = {}
    if api_key.strip():
        headers["X-API-Key"] = api_key.strip()

    # Convert booleans for multipart encoding
    if "auto_submit" in form:
        form["auto_submit"] = "true" if form["auto_submit"] else "false"
    if "use_llm" in form and form["use_llm"] is not None:
        form["use_llm"] = "true" if form["use_llm"] else "false"

    r = requests.post(url, data=form, files=files, headers=headers, timeout=120)
    try:
        out = r.json()
    except Exception:
        out = {"error": "Non-JSON response", "status_code": r.status_code, "text": r.text}
    return out

def call_api_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    url = api_base_url.rstrip("/") + "/v1/complaints/feedback"
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["X-API-Key"] = api_key.strip()
    r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=60)
    try:
        return r.json()
    except Exception:
        return {"ok": False, "status_code": r.status_code, "text": r.text}

def call_api_memory_delete(payload: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    url = api_base_url.rstrip("/") + "/v1/memory/delete"
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["X-API-Key"] = api_key.strip()
    r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=60)
    try:
        return r.json()
    except Exception:
        return {"ok": False, "status_code": r.status_code, "text": r.text}


# ----------------------------
# Shared state
# ----------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = "demo_user"
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "last_ticket_id" not in st.session_state:
    st.session_state.last_ticket_id = ""


# ----------------------------
# Tabs
# ----------------------------
tab_new, tab_track, tab_escalate = st.tabs(
    ["🆕 New complaint", "📍 Track ticket", "⬆️ Escalate"]
)

# ============================
# Tab: New Complaint
# ============================
with tab_new:
    st.subheader("New complaint")

    with st.form("new_complaint_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            user_id = st.text_input("User ID", value=st.session_state.user_id)
            city = st.text_input("City", value="DemoCity")
        with col2:
            ward_id = st.text_input("Ward / Zone ID", value="W-42")
            landmark = st.text_input("Landmark (optional)", value="")
        with col3:
            preferred_channel = st.selectbox("Preferred channel (optional)", options=["", "whatsapp", "phone", "email", "in_person"], index=0)
            tone = st.selectbox("Tone (optional)", options=["", "formal", "polite", "urgent", "concise"], index=0)

        text = st.text_area("Complaint text", height=140, placeholder="Describe the issue clearly…")

        colA, colB = st.columns(2)
        with colA:
            issue_photo = st.file_uploader("Issue photo (optional)", type=["png", "jpg", "jpeg"])
        with colB:
            pass

        submit_col1, submit_col2 = st.columns(2)
        with submit_col1:
            submitted = st.form_submit_button("Generate recommendation")
        with submit_col2:
            direct_submit = st.form_submit_button("Direct submit (no recommendation)")

    if submitted or direct_submit:
        st.session_state.user_id = user_id

        if not text.strip():
            st.error("Please enter complaint text.")
        else:
            with st.spinner("Processing complaint…"):
                if direct_submit:
                    # Direct submit - create ticket with original text without modification
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
                            meta={"tone": tone or "Not specified", "landmark": landmark or ""}
                        )
                        st.session_state.last_ticket_id = ticket_id
                        st.success("✅ Complaint submitted successfully!")
                        st.info(f"**Ticket ID:** {ticket_id}")
                        st.write(f"**Complaint Text:** {text}")
                        st.write(f"**Channel:** {preferred_channel or 'Not specified'}")
                        st.write(f"**Tone:** {tone or 'Not specified'}")
                    except Exception as e:
                        st.error(f"Failed to create ticket: {str(e)}")
                else:
                    # Generate recommendation
                    if mode == "FastAPI backend (API mode)":
                        files = {}
                        if issue_photo:
                            files["issue_photo"] = (issue_photo.name, issue_photo.getvalue(), issue_photo.type)

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
                        # Streamlit-local session id
                        if st.session_state.session_id is None:
                            st.session_state.session_id = f"st_{int(datetime.now().timestamp())}_{user_id}"

                    # Render output
                    if isinstance(out, dict) and out.get("session_id"):
                        st.session_state.session_id = out["session_id"]

                    if not isinstance(out, dict):
                        st.error("Unexpected response type.")
                        st.code(str(out))
                    elif out.get("need_more_info"):
                        st.warning("More info needed to generate a complete complaint draft.")
                        st.write("**Missing fields:**", ", ".join(out.get("missing_fields", [])))
                        qs = out.get("questions", [])
                        if qs:
                            st.write("**Questions:**")
                            for q in qs:
                                st.write(f"- {q}")
                    elif out.get("error"):
                        st.error("Request failed.")
                        st.code(pretty_json(out))
                    else:
                        ra = out.get("recommended_action", {}) or {}

                        # Extract values - these should now come from the agent
                        dept = ra.get("department") or "Sanitation"
                        cat = ra.get("category") or "General"
                        best_ch = ra.get("best_channel") or "helpline"
                        backup_ch = ra.get("backup_channel") or "email"

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Department", dept)
                        c2.metric("Category", cat)
                        c3.metric("Best channel", best_ch)
                        c4.metric("Backup channel", backup_ch)

                        st.subheader("Complaint text (ready to paste)")
                        st.text_area("", value=out.get("complaint_text_ready_to_paste", ""), height=200)

                        tips = out.get("tips_from_similar_cases", []) or []
                        if tips:
                            st.subheader("Tip from similar cases")
                            for t in tips:
                                st.info(t)

                        if out.get("ticket_id"):
                            st.session_state.last_ticket_id = out["ticket_id"]
                            st.success(f"Demo ticket created: **{out['ticket_id']}**")

                        if out.get("llm_markdown"):
                            st.markdown("### 💡 AI-Enhanced Recommendation")
                            st.markdown(out["llm_markdown"])

                        st.caption(out.get("safety_note", ""))

                        # Feedback UI
                        st.divider()
                        st.subheader("Feedback (improves memory)")
                        fb_col1, fb_col2 = st.columns([1, 2])
                        with fb_col1:
                            outcome = st.selectbox(
                                "Outcome",
                                options=["resolved", "not_resolved", "wrong_department", "portal_down", "submitted"],
                                index=4,
                            )
                        with fb_col2:
                            notes = st.text_input("Notes (optional)", value="")

                        if st.button("Submit feedback"):
                            payload = {
                                "user_id": user_id,
                                "session_id": st.session_state.session_id,
                                "ticket_id": out.get("ticket_id") or st.session_state.last_ticket_id or None,
                                "outcome": outcome,
                                "notes": notes or None,
                            }

                            if mode == "FastAPI backend (API mode)":
                                resp = call_api_feedback(payload)
                            else:
                                from core.memory import memory_upsert
                                memory_upsert(
                                    user_id=user_id,
                                    memory_text=f"Feedback: {outcome}. Ticket: {payload.get('ticket_id') or ''}. Notes: {notes or ''}",
                                    payload={"type": "history", "outcome": outcome, "ticket_id": payload.get("ticket_id") or "", "notes": notes or ""},
                                    ttl_days=365,
                                    version=1,
                                )
                                resp = {"ok": True}

                            if resp.get("ok"):
                                st.success("Thanks! Feedback saved.")
                            else:
                                st.error("Feedback failed.")
                                st.code(pretty_json(resp))


# ============================
# Tab: Track
# ============================
with tab_track:
    st.subheader("Track a ticket (demo store)")
    ticket_id = st.text_input("Ticket ID", value=st.session_state.last_ticket_id or "")
    go = st.button("Track ticket")

    if go and ticket_id.strip():
        with st.spinner("Looking up ticket…"):
            if mode == "FastAPI backend (API mode)":
                # API mode uses /assist with intent=track
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
            
            ticket = out
            # Format the ticket information in a user-friendly way
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Ticket ID", ticket.get("ticket_id", "N/A"))
                st.metric("Status", ticket.get("status", "N/A").upper())
                st.metric("Category", ticket.get("category", "N/A"))
                st.metric("Channel", ticket.get("channel", "N/A"))
            
            with col2:
                st.metric("City", ticket.get("city", "N/A"))
                st.metric("Ward", ticket.get("ward_id", "N/A"))
                st.metric("Department", ticket.get("department", "N/A") or "Pending")
                urgency = ticket.get("meta", {}).get("urgency", "N/A")
                st.metric("Urgency", urgency.upper() if urgency != "N/A" else "N/A")
            
            st.divider()
            st.subheader("Complaint Details")
            st.write(ticket.get("complaint_text", "No description"))
            
            st.divider()
            st.subheader("Ticket Information")
            from datetime import datetime
            created_ts = ticket.get("created_ts")
            if created_ts:
                created_date = datetime.fromtimestamp(created_ts).strftime("%B %d, %Y at %I:%M %p")
                st.info(f"📅 Created on: {created_date}")
            
            user_id = ticket.get("user_id")
            if user_id:
                st.info(f"👤 User ID: {user_id}")
        else:
            st.warning("❌ Ticket not found. Please check your Ticket ID and try again, or create a new complaint using the 'New Complaint' tab.")


# ============================
# Tab: Escalate
# ============================
with tab_escalate:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;">
        <h2>⬆️ Escalate Your Complaint</h2>
        <p>When your issue isn't resolved within the expected timeframe, escalate it to senior authorities.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    with st.container():
        st.markdown("### 📝 Enter Ticket Details")
        col1, col2 = st.columns(2)
        with col1:
            esc_ticket_id = st.text_input("🎫 Ticket ID", value=st.session_state.last_ticket_id or "", key="esc_ticket", help="Find this in your ticket tracking page")
        with col2:
            days_waited = st.number_input("⏱️ Days Waited", min_value=0, max_value=365, value=7, step=1, help="How many days since you filed the complaint?")
        
        # Set default city value
        esc_city = "DemoCity"

    col_submit, col_info = st.columns([1, 3])
    with col_submit:
        go = st.button("🚀 Get Escalation Plan", use_container_width=True)
    with col_info:
        st.info("💡 The system will analyze your complaint and suggest the best way to escalate it.")

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
                out = agent.escalate(city=esc_city, ticket_id=esc_ticket_id.strip(), days_waited=int(days_waited))

        if out.get("ok"):
            st.success("✅ Escalation plan created successfully!")
            
            st.divider()
            
            # Status Cards
            st.markdown("### 📊 Current Status")
            status_col1, status_col2 = st.columns(2)
            with status_col1:
                st.metric(
                    "Current Status",
                    out.get('current_status', 'N/A').upper(),
                    delta=None
                )
            with status_col2:
                st.metric(
                    "Service Level Agreement (SLA)",
                    f"{out.get('sla_days', 'N/A')} days",
                    delta=f"Waiting: {days_waited} days"
                )
            
            st.divider()
            
            # Escalation Steps
            st.markdown("### 🪜 Recommended Escalation Steps")
            steps = out.get("recommended_escalation", []) or []
            
            if steps:
                for step_num, step in enumerate(steps, 1):
                    # Create a numbered card for each step
                    with st.container():
                        col_num, col_content = st.columns([0.5, 9.5])
                        with col_num:
                            st.markdown(f"<h3 style='text-align: center; color: #667eea;'>{step_num}</h3>", unsafe_allow_html=True)
                        with col_content:
                            st.markdown(f"**Step {step_num}:** {step}")
                    st.divider()
            else:
                st.warning("No escalation steps available for this ticket.")
            
            st.divider()
            st.success("📧 Ready to escalate! Contact the recommended authority with your ticket ID and these details.")
        else:
            st.error("❌ Unable to create escalation plan")
            st.warning(out.get("message", "Failed to process escalation."))

