from __future__ import annotations
from typing import Dict, Any, Optional, List
from PIL import Image
from datetime import datetime

from .config import ENABLE_HYBRID, ENABLE_RERANK, RERANK_MODEL, ENABLE_OCR
from .preprocess import redact_pii, infer_urgency, detect_lang, ocr_image
from .vision_hints import infer_image_issue_hints
from .qdrant_store import build_filter
from .hybrid import HybridRetriever, qdrant_dense_search, qdrant_image_search
from .recommend import recommend_channel
from .response import snippet, fill_template, escalation_steps, confidence_from_score
from .memory import memory_get, reinforce_preference
from .rerank import Reranker, CrossEncoderReranker
from .tickets import create_ticket, get_ticket, list_tickets
from .config import ENABLE_LLM, LLM_MODEL
from .llm import LocalLLM, LLMConfig, build_llm_prompt

REQUIRED_BASE_FIELDS = ["city", "ward_id", "text"]

def _missing_fields(required: List[str], provided: Dict[str, Any]) -> List[str]:
    missing = []
    for f in required:
        if f.endswith("_optional"):
            continue
        v = provided.get(f)
        if v is None:
            missing.append(f)
        elif isinstance(v, str) and not v.strip():
            missing.append(f)
    return missing

def _questions_for(fields: List[str]) -> List[str]:
    qmap = {
        "location": "What is the exact location (street/area) of the issue?",
        "landmark": "Any nearby landmark (shop/school/bus stop) to help locate it?",
        "date_time": "When did you first notice it (date/time)?",
        "days_missed": "How many days has the issue continued (e.g., garbage not collected for N days)?",
        "photo": "Can you upload a clear photo of the issue?",
        "pole_number_optional": "If visible, what is the pole number?",
    }
    return [qmap.get(f, f"Please provide: {f}") for f in fields]

class CivicFixAgent:
    """Full architecture agent: intent routing + tool-order + slot filling."""
    def __init__(self):
        self.hybrid = HybridRetriever()
        self.reranker: Reranker = Reranker()
        if ENABLE_RERANK:
            try:
                self.reranker = CrossEncoderReranker(RERANK_MODEL)
            except Exception:
                self.reranker = Reranker()

    # -------- Core retrieval helpers --------
    def _kb_search(self, query: str, city: str, language: str):
        kb_flt = build_filter(city=city, language=language)
        if ENABLE_HYBRID:
            return self.hybrid.search("civic_kb", query, flt=kb_flt, topk=8)
        return qdrant_dense_search("civic_kb", query, flt=kb_flt, limit=8)

    def _rerank(self, query: str, hits: List[Dict[str, Any]]):
        docs = [{"id": h["id"], "text": h["text"], "payload": h["payload"], "score": h.get("score", h.get("rrf_score"))} for h in hits]
        return self.reranker.rerank(query, docs)

    def _template(self, category: str, tone: Optional[str], channel: Optional[str]):
        tpl_flt = build_filter(category=category)
        q = f"{category} complaint template {tone or ''} {channel or ''}"
        hits = qdrant_dense_search("complaint_templates", q, flt=tpl_flt, limit=6)
        docs = [{"id": h["id"], "text": h["text"], "payload": h["payload"], "score": h.get("score")} for h in hits]
        docs = self.reranker.rerank(f"{category} complaint template", docs)
        top = docs[0] if docs else None
        return (top["payload"].get("template") if top else None) or \
            "Please register a complaint for {category} at {location} near {landmark} observed on {date_time}. Attachments: {attachments}."

    # -------- Intents --------
    def assist_new(
        self,
        user_id: str,
        text: str,
        city: str,
        ward_id: str,
        landmark: str,
        date_time: str,
        preferred_channel: Optional[str],
        tone: Optional[str],
        issue_photo: Optional[Image.Image],
        screenshot: Optional[Image.Image],
        transcript_text: Optional[str],
        auto_submit: bool = False,
        force_llm: bool | None = None,
    ) -> Dict[str, Any]:
        language = detect_lang(text) if text else "en"
        urgency = infer_urgency(text)
        cleaned = redact_pii(text or "")

        # OCR
        if ENABLE_OCR and screenshot is not None:
            ocr = ocr_image(screenshot)
            if ocr:
                cleaned += "\n" + redact_pii(ocr)

        # Transcript
        if transcript_text:
            cleaned += "\n" + redact_pii(transcript_text)

        # Memory (prefs)
        mem_hits = memory_get(user_id, limit=20)
        mem_pref = preferred_channel
        for h in mem_hits:
            if h.payload.get("type") == "preference":
                mem_pref = mem_pref or h.payload.get("pref_channel")
                break

        # Image-based hints + similar case retrieval
        image_hints = []
        image_hits = []
        inferred_category = None
        if issue_photo is not None:
            image_hints = infer_image_issue_hints(issue_photo, topk=3)
            image_hits = qdrant_image_search("case_library", issue_photo, limit=3)
            if image_hits:
                inferred_category = image_hits[0]["payload"].get("category")

        # KB retrieval (hybrid)
        kb_hits = self._kb_search(cleaned, city=city, language=language)
        kb_docs = self._rerank(cleaned, kb_hits)
        kb_top = kb_docs[0] if kb_docs else None
        kb_payload = kb_top["payload"] if kb_top else {}

        category = kb_payload.get("category") or inferred_category or (image_hints[0]["label"] if image_hints else "General")
        department = kb_payload.get("department") or "Sanitation"  # Default to Sanitation for civic complaints

        # Directory routing
        dir_flt = build_filter(city=city, ward_id=ward_id)
        dir_hits = qdrant_dense_search("jurisdiction_directory", cleaned, flt=dir_flt, limit=3)

        # Similar cases (text)
        case_text_flt = build_filter(category=category) if category != "General" else None
        case_text_hits = qdrant_dense_search("case_library", cleaned, flt=case_text_flt, limit=3)

        # Recommend channel
        best, backup, details = recommend_channel(kb_payload, city=city, urgency=urgency, user_pref=mem_pref)
        # Provide defaults if recommend_channel returns None
        if not best:
            best = "helpline"
        if not backup:
            backup = "email"

        # Slot filling: minimal missing fields based on KB required_fields
        required_fields = kb_payload.get("required_fields", [])
        provided = {
            "city": city, "ward_id": ward_id, "landmark": landmark, "date_time": date_time, "text": text,
            "location": f"Ward {ward_id}, {city}",
            "photo": "yes" if issue_photo is not None else "",
            "days_missed": "",
        }
        missing = _missing_fields(required_fields, provided)
        if missing and not auto_submit:
            return {
                "need_more_info": True,
                "missing_fields": missing,
                "questions": _questions_for(missing),
                "inferred": {"category": category, "department": department, "urgency": urgency, "image_hints": image_hints},
                "evidence": [{
                    "collection": "civic_kb",
                    "score": float(kb_top.get("rerank_score", kb_top.get("score") or 0.0)) if kb_top else 0.0,
                    "source": kb_payload.get("source"),
                    "last_updated": kb_payload.get("last_updated"),
                    "snippet": snippet(kb_payload.get("text","")) if kb_payload else ""
                }] if kb_payload else [],
                "confidence": confidence_from_score(kb_top.get("rerank_score") if kb_top else None),
            }

        # Template
        template = self._template(category, tone=tone, channel=best or mem_pref)
        fields = {
            "category": category,
            "location": f"Ward {ward_id}, {city}",
            "landmark": landmark or "nearby landmark",
            "date_time": date_time,
            "attachments": "photo attached" if issue_photo is not None else ("screenshot attached" if screenshot is not None else "none"),
            "days_missed": "3",
            "pole_number_optional": "",
            "sender_name_optional": ""
        }
        complaint_text = fill_template(template, fields)
        sla_days = int(kb_payload.get("sla_days", 7)) if kb_payload.get("sla_days") else 7

        evidence = []
        if kb_top:
            evidence.append({
                "collection":"civic_kb",
                "score": float(kb_top.get("rerank_score", kb_top.get("score") or 0.0)),
                "source": kb_payload.get("source"),
                "last_updated": kb_payload.get("last_updated"),
                "snippet": snippet(kb_payload.get("text",""))
            })
        if dir_hits:
            evidence.append({
                "collection":"jurisdiction_directory",
                "score": float(dir_hits[0].get("score", 0.0)),
                "source":"directory",
                "last_updated": dir_hits[0]["payload"].get("last_updated"),
                "snippet": snippet(dir_hits[0]["payload"].get("text",""))
            })
        if image_hits:
            evidence.append({
                "collection":"case_library(image)",
                "score": float(image_hits[0].get("score", 0.0)),
                "source":"anonymized_case",
                "last_updated": image_hits[0]["payload"].get("last_updated"),
                "snippet": snippet(image_hits[0]["payload"].get("text",""))
            })

        trace = {
            "kb_top_ids": [(d["id"], d.get("rerank_score", d.get("score"))) for d in kb_docs[:5]],
            "dir_top_ids": [(d["id"], d.get("score")) for d in dir_hits[:3]],
            "case_text_ids": [(d["id"], d.get("score")) for d in case_text_hits[:3]],
            "case_image_ids": [(d["id"], d.get("score")) for d in image_hits[:3]],
            "image_hints": image_hints,
            "channel_scores": details["scored"],
            "portal_ok": details["portal_ok"],
            "hybrid_enabled": ENABLE_HYBRID,
            "rerank_enabled": ENABLE_RERANK
        }

        # Feedback loop: reinforce preference
        if best:
            reinforce_preference(user_id, best)

        result = {
            "need_more_info": False,
            "recommended_action": {"department": department, "category": category, "best_channel": best, "backup_channel": backup},
            "complaint_text_ready_to_paste": complaint_text,
            "checklist_required_fields": required_fields,
            "sla_days": sla_days,
            "escalation_steps": escalation_steps(sla_days),
            "tips_from_similar_cases": [snippet(h["payload"].get("text",""), 200) for h in case_text_hits[:1]],
            "evidence": evidence,
            "confidence": confidence_from_score(kb_top.get("rerank_score") if kb_top else None),
            "traceable_reasoning": trace,
            "safety_note": "If there is immediate danger (fire, exposed live wires, major accident risk), contact emergency services first."
        }


        # Optional LLM response generation (open-source local model)
        use_llm_final = ENABLE_LLM if force_llm is None else bool(force_llm)
        if use_llm_final:
            try:
                llm = LocalLLM(LLMConfig(model_id=LLM_MODEL))
                llm_prompt = build_llm_prompt(cleaned, result, evidence)
                result["llm_markdown"] = llm.generate(llm_prompt)
                result["llm_model"] = LLM_MODEL
            except Exception as _e:
                # Don't fail the request if LLM isn't available
                result["llm_markdown"] = ""
                result["llm_model"] = LLM_MODEL

        if auto_submit:
            ticket_id = create_ticket(
                user_id=user_id, city=city, ward_id=ward_id,
                category=category, department=department or "", channel=best or (mem_pref or ""),
                complaint_text=complaint_text, meta={"urgency": urgency}
            )
            result["ticket_id"] = ticket_id
            result["submission_status"] = "submitted"

        return result

    def procedure_qa(self, city: str, text: str) -> Dict[str, Any]:
        language = detect_lang(text) if text else "en"
        hits = self._kb_search(text, city=city, language=language)
        docs = self._rerank(text, hits)
        top = docs[:3]
        steps = []
        evidence = []
        for d in top:
            p = d["payload"]
            steps.append({
                "category": p.get("category"),
                "department": p.get("department"),
                "sla_days": p.get("sla_days"),
                "channels": p.get("channel_type"),
                "required_fields": p.get("required_fields"),
            })
            evidence.append({
                "collection":"civic_kb",
                "score": float(d.get("rerank_score", d.get("score") or 0.0)),
                "source": p.get("source"),
                "last_updated": p.get("last_updated"),
                "snippet": snippet(p.get("text",""))
            })
        return {"procedure_matches": steps, "evidence": evidence, "confidence": confidence_from_score(top[0].get("rerank_score") if top else None)}

    def track(self, ticket_id: str) -> Dict[str, Any]:
        t = get_ticket(ticket_id)
        if not t:
            return {"found": False, "ticket_id": ticket_id, "message": "Ticket not found (demo store)."}
        return {"found": True, **t}

    def list_user_tickets(self, user_id: str) -> Dict[str, Any]:
        return {"user_id": user_id, "tickets": list_tickets(user_id)}

    def escalate(self, city: str, ticket_id: str, days_waited: int) -> Dict[str, Any]:
        t = get_ticket(ticket_id)
        if not t:
            return {"ok": False, "message": "Ticket not found."}
        # Use KB to find escalation ladder
        q = f"escalation steps for {t.get('category')} complaint"
        proc = self.procedure_qa(city=city, text=q)
        sla = None
        if proc.get("procedure_matches"):
            sla = proc["procedure_matches"][0].get("sla_days")
        sla = int(sla) if sla else 7
        return {
            "ok": True,
            "ticket_id": ticket_id,
            "current_status": t.get("status"),
            "days_waited": days_waited,
            "sla_days": sla,
            "recommended_escalation": escalation_steps(sla, waited_days=days_waited),
            "evidence": proc.get("evidence", [])
        }
