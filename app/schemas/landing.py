# app/schemas/landing.py - MODIFY THIS
from datetime import datetime
from typing import Literal, Optional  # ADD Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# =========================
# Landing Page Request
# =========================
class LandingStartRequest(BaseModel):
    token: Optional[str] = Field(None, min_length=1)  # CHANGE to Optional

    name: str = Field(..., min_length=3)
    email: EmailStr
    phone: str = Field(..., min_length=10)

    location: str
    position: str
    experience: Literal["Entry", "Mid", "Senior", "Lead"]

    # agreement checkbox is frontend-only (correctly excluded)


# =========================
# Landing Page Response
# =========================
class LandingStartResponse(BaseModel):
    attempt_id: UUID
    assessment_id: UUID


# =========================
# Internal (Optional, later use)
# =========================
class CandidateRead(BaseModel):
    id: UUID
    name: str
    email: EmailStr
    phone: str
    location: str
    position: str
    experience: str
    created_at: datetime

    class Config:
        from_attributes = True