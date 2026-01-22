# app/schemas/assessment.py

from pydantic import BaseModel
from typing import Dict
from datetime import datetime


class SimpleSubmission(BaseModel):
    """Simple schema that matches frontend format"""
    answers: Dict[str, str]  # {"1": "answer1", "2": "answer2", ...}
    
    class Config:
        extra = "allow"  # Allow extra fields


class SubmissionResponse(BaseModel):
    """Response after successful submission"""
    message: str
    score: float
    total_questions: int
    submitted_at: datetime
    status: str = "completed"