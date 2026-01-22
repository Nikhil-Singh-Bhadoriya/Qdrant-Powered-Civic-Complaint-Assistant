from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional

class FeedbackRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    ticket_id: Optional[str] = None
    outcome: str = Field(..., description="resolved/not_resolved/wrong_department/portal_down/submitted")
    notes: Optional[str] = None


from pydantic import BaseModel

class SubmitRequest(BaseModel):
    user_id: str
    city: str
    ward_id: str
    landmark: str = ""
    text: str
    preferred_channel: str | None = None
    tone: str | None = None

class TrackRequest(BaseModel):
    ticket_id: str

class EscalateRequest(BaseModel):
    city: str
    ticket_id: str
    days_waited: int

class ProcedureRequest(BaseModel):
    city: str
    text: str

class MemoryDeleteRequest(BaseModel):
    user_id: str
