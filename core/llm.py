from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

SYSTEM_PROMPT = (
    "You are CivicFix Assistant. Your job is to help citizens file effective civic complaints. "
    "You MUST ground every claim in the provided EVIDENCE. If evidence is missing, say so. "
    "Do NOT invent helpline numbers, portal URLs, SLAs, or departments. "
    "Return a clear, structured answer with these sections:\n"
    "1) Recommended action (channel + department + category)\n"
    "2) Why (based on evidence)\n"
    "3) Ready-to-paste complaint text\n"
    "4) Checklist (required fields/attachments)\n"
    "5) Escalation steps (after SLA)\n"
    "6) Evidence (bulleted citations like: [source | last_updated])\n"
    "Keep it concise and formal."
)

@dataclass
class LLMConfig:
    model_id: str = DEFAULT_LLM_MODEL
    max_new_tokens: int = int(os.environ.get("LLM_MAX_NEW_TOKENS", "420"))
    temperature: float = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
    top_p: float = float(os.environ.get("LLM_TOP_P", "0.9"))
    repetition_penalty: float = float(os.environ.get("LLM_REP_PENALTY", "1.05"))

class LocalLLM:
    """Small, open-source local LLM response generator using Transformers.
    Designed to work on CPU. Uses a small chat model by default (TinyLlama).
    """
    _tokenizer = None
    _model = None
    _loaded_model_id = None

    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig()

    def _load(self):
        if LocalLLM._model is not None and LocalLLM._loaded_model_id == self.cfg.model_id:
            return
        model_id = self.cfg.model_id
        # Load tokenizer/model
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        # Some models don't have pad token; set to eos
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        LocalLLM._tokenizer = tok
        LocalLLM._model = model
        LocalLLM._loaded_model_id = model_id

    def _format_messages(self, system: str, user: str) -> torch.Tensor:
        tok = LocalLLM._tokenizer
        # Prefer chat template if available
        try:
            if hasattr(tok, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                return tok(prompt, return_tensors="pt").input_ids
        except Exception:
            pass
        # Fallback: plain prompt
        prompt = f"SYSTEM:\n{system}\n\nUSER:\n{user}\n\nASSISTANT:\n"
        return tok(prompt, return_tensors="pt").input_ids

    def generate(self, user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        self._load()
        model = LocalLLM._model
        tok = LocalLLM._tokenizer
        device = next(model.parameters()).device

        input_ids = self._format_messages(system_prompt, user_prompt).to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                repetition_penalty=self.cfg.repetition_penalty,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen = out[0][input_ids.shape[-1]:]
        text = tok.decode(gen, skip_special_tokens=True).strip()
        return text

def build_llm_prompt(
    user_text: str,
    structured: Dict[str, Any],
    evidence: List[Dict[str, Any]],
) -> str:
    # Evidence pack (limit for prompt size)
    ev_lines = []
    for e in evidence[:6]:
        src = e.get("source") or "unknown_source"
        lu = e.get("last_updated") or "unknown_date"
        snip = (e.get("snippet") or "").strip().replace("\n", " ")
        if len(snip) > 260:
            snip = snip[:260] + "…"
        ev_lines.append(f"- [{src} | {lu}] {snip}")

    rec = structured.get("recommended_action", {}) if isinstance(structured, dict) else {}
    checklist = structured.get("checklist_required_fields", []) if isinstance(structured, dict) else []
    esc = structured.get("escalation_steps", []) if isinstance(structured, dict) else []
    complaint_text = structured.get("complaint_text_ready_to_paste", "")

    prompt = (
        f"USER ISSUE (may include photo/screenshot/OCR):\n{user_text.strip()}\n\n"
        f"STRUCTURED OUTPUT (from tools):\n"
        f"- Department: {rec.get('department')}\n"
        f"- Category: {rec.get('category')}\n"
        f"- Best channel: {rec.get('best_channel')}\n"
        f"- Backup channel: {rec.get('backup_channel')}\n"
        f"- SLA days: {structured.get('sla_days')}\n\n"
        f"READY COMPLAINT TEXT (template draft):\n{complaint_text}\n\n"
        f"CHECKLIST REQUIRED FIELDS: {checklist}\n"
        f"ESCALATION STEPS: {esc}\n\n"
        "EVIDENCE (YOU MUST USE ONLY THIS):\n"
        + "\n".join(ev_lines)
    )
    return prompt
