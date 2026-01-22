from __future__ import annotations
from typing import Dict, Any, Optional
from PIL import Image

from .config import ENABLE_HYBRID, ENABLE_RERANK, RERANK_MODEL, ENABLE_LLM_AGENT, LLM_MODEL
from .llm import LocalLLM, LLMConfig, build_llm_prompt
from .preprocess import redact_pii, infer_urgency, ocr_image
from .qdrant_store import build_filter
from .hybrid import HybridRetriever, qdrant_dense_search, qdrant_image_search
from .recommend import recommend_channel
from .response import snippet, fill_template, escalation_steps, confidence_from_score
from .memory import memory_get, reinforce_preference
from .rerank import Reranker, CrossEncoderReranker

class DeterministicAgent:
    """Implements tool-order: Memory → Search → Recommend → Evidence → Response."""
    def __init__(self):
        self.hybrid = HybridRetriever()
        if ENABLE_HYBRID:
            try:
                self.hybrid.build_bm25_from_qdrant("civic_kb", limit=500)
            except Exception:
                pass
        self.reranker: Reranker = Reranker()
        if ENABLE_RERANK:
            try:
                self.reranker = CrossEncoderReranker(RERANK_MODEL)
            except Exception:
                self.reranker = Reranker()

    def assist(
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
    ) -> Dict[str, Any]:

        urgency = infer_urgency(text)
        language = "en"  # map from detect_lang() if you support multiple languages
        cleaned = redact_pii(text)

        if transcript_text:
            cleaned += "\n" + redact_pii(transcript_text)

        if screenshot is not None:
            ocr = ocr_image(screenshot)
            if ocr:
                cleaned += "\n" + redact_pii(ocr)

        # 1) Memory (prefs)
        mem_hits = memory_get(user_id, limit=20)
        mem_pref = preferred_channel
        for h in mem_hits:
            if h.payload.get("type") == "preference":
                mem_pref = mem_pref or h.payload.get("pref_channel")
                break

        # 2a) Multimodal search (photo → similar cases)
        image_hits = []
        inferred_category = None
        if issue_photo is not None:
            image_hits = qdrant_image_search("case_library", issue_photo, limit=3)
            if image_hits:
                inferred_category = image_hits[0]["payload"].get("category")

        # 2b) KB search (hybrid optional)
        kb_flt = build_filter(city=city, language=language)
        if ENABLE_HYBRID and self.hybrid.bm25 is not None:
            kb_hits = self.hybrid.search("civic_kb", cleaned, flt=kb_flt, topk=8)
        else:
            kb_hits = qdrant_dense_search("civic_kb", cleaned, flt=kb_flt, limit=8)

        # 3) Re-rank (optional)
        kb_docs = [{"id": h["id"], "text": h["text"], "payload": h["payload"], "score": h.get("score")} for h in kb_hits]
        kb_docs = self.reranker.rerank(cleaned, kb_docs)
        kb_top = kb_docs[0] if kb_docs else None
        kb_payload = kb_top["payload"] if kb_top else {}

        category = kb_payload.get("category") or inferred_category or "General"
        department = kb_payload.get("department")

        # 4) Directory routing (ward/zone → dept/channels)
        dir_flt = build_filter(city=city, ward_id=ward_id)
        dir_hits = qdrant_dense_search("jurisdiction_directory", cleaned, flt=dir_flt, limit=3)

        # 5) Templates
        tpl_flt = build_filter(category=category)
        tpl_hits = qdrant_dense_search("complaint_templates", f"{category} template {tone or ''} {mem_pref or ''}", flt=tpl_flt, limit=5)
        tpl_docs = [{"id": h["id"], "text": h["text"], "payload": h["payload"], "score": h.get("score")} for h in tpl_hits]
        tpl_docs = self.reranker.rerank(f"{category} complaint template", tpl_docs)
        tpl_top = tpl_docs[0] if tpl_docs else None
        template = (tpl_top["payload"].get("template") if tpl_top else None) or             "Please register a complaint for {category} at {location} near {landmark} observed on {date_time}. Attachments: {attachments}."

        # 6) Similar cases (text)
        case_text_flt = build_filter(category=category) if category != "General" else None
        case_text_hits = qdrant_dense_search("case_library", cleaned, flt=case_text_flt, limit=3)

        # 7) Recommend channel
        best, backup, details = recommend_channel(kb_payload, city=city, urgency=urgency, user_pref=mem_pref)

        # 8) Build response
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
        required_fields = kb_payload.get("required_fields", [])
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
            "tpl_top_ids": [(d["id"], d.get("rerank_score", d.get("score"))) for d in tpl_docs[:3]],
            "case_text_ids": [(d["id"], d.get("score")) for d in case_text_hits[:3]],
            "case_image_ids": [(d["id"], d.get("score")) for d in image_hits[:3]],
            "channel_scores": details["scored"],
            "portal_ok": details["portal_ok"],
            "hybrid_enabled": ENABLE_HYBRID,
            "rerank_enabled": ENABLE_RERANK
        }

        # Feedback loop: reinforce preference (in production, do on actual submission)
        if best:
            reinforce_preference(user_id, best)

        return {
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

class ToolCallingAgent:
    """Open-source "LLM orchestrator" wrapper.

    For this submission we keep tool execution deterministic (to guarantee reproducibility),
    then (optionally) use a **local open-source LLM** to render a grounded final response.

    This avoids any proprietary APIs and keeps the project end-to-end runnable offline
    once models are cached.
    """

    def __init__(self):
        self.fallback = DeterministicAgent()
        self.enabled = bool(ENABLE_LLM_AGENT)
        self._llm = None

    def assist(self, *args, **kwargs):
        out = self.fallback.assist(*args, **kwargs)
        if not self.enabled:
            return out

        # Best-effort: add an LLM-formatted answer grounded in evidence
        try:
            if self._llm is None:
                self._llm = LocalLLM(LLMConfig(model_id=LLM_MODEL))
            user_text = (kwargs.get("text") or "").strip()
            evidence = out.get("evidence", []) if isinstance(out, dict) else []
            prompt = build_llm_prompt(user_text, out, evidence)
            out["llm_markdown"] = self._llm.generate(prompt)
            out["llm_model"] = LLM_MODEL
        except Exception:
            out.setdefault("llm_markdown", "")
            out.setdefault("llm_model", LLM_MODEL)

        return out
