from __future__ import annotations
from typing import Dict, Any

def snippet(txt: str, n: int = 240) -> str:
    if not txt:
        return ""
    return txt[:n] + ("..." if len(txt) > n else "")

def fill_template(template: str, fields: Dict[str, Any]) -> str:
    safe = {k: (v if v is not None else "") for k, v in fields.items()}
    for k in ["location","landmark","date_time","attachments","days_missed","pole_number_optional","sender_name_optional","category"]:
        safe.setdefault(k, "")
    return template.format(**safe)

def escalation_steps(sla_days: int, waited_days: int | None = None):
    return [
        f"Wait until the SLA ends (≈ {sla_days} days).",
        "If no action: escalate to ward/zonal officer with complaint reference and attachments.",
        "If still unresolved: submit on the grievance portal with evidence.",
        "Last resort: formal written complaint / RTI route as per local rules."
    ]

def confidence_from_score(score: float | None) -> str:
    if score is None:
        return "low"
    if score >= 0.35:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


# Override with waited-days aware escalation

def escalation_steps(sla_days: int, waited_days: int | None = None):
    waited = waited_days if waited_days is not None else sla_days
    base = [
        f"Day 0: Submit via recommended channel and save screenshot/ticket.",
        f"Day {min(1, waited)}: If no acknowledgement, re-submit with photo + landmark.",
        f"Day {min(sla_days, waited)}: If no resolution within SLA ({sla_days} days), escalate to ward/zone officer.",
        f"Day {min(sla_days+2, waited+2)}: If still unresolved, email commissioner/municipal grievance cell with prior ticket proof.",
        f"Day {min(sla_days+5, waited+5)}: File on state grievance portal / RTI if applicable (attach evidence).",
    ]
    return base
