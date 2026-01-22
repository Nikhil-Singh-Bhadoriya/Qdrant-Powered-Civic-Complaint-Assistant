from __future__ import annotations
import re
from langdetect import detect as _detect

PHONE_RE = re.compile(r"(\+?\d[\d\s\-]{8,}\d)")
EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")

def detect_lang(text: str) -> str:
    try:
        return _detect(text)
    except Exception:
        return "unknown"

def redact_pii(text: str) -> str:
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    return text

def infer_urgency(text: str) -> str:
    t = (text or "").lower()
    high = ["electric shock","exposed wire","sparking","fire","collapse","accident","injury"]
    if any(k in t for k in high):
        return "high"
    if any(k in t for k in ["danger","hazard","urgent","unsafe","immediate"]):
        return "high"
    if any(k in t for k in ["not working","broken","leak","overflow","garbage","pothole"]):
        return "medium"
    return "low"

def ocr_image(pil_image):
    try:
        import pytesseract
        return pytesseract.image_to_string(pil_image)
    except Exception:
        return ""
