import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, String, Float, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.base import Base


# =========================
# Candidate (Landing Form)
# =========================
class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, index=True)
    phone = Column(String(50), nullable=False)

    location = Column(String(255), nullable=False)
    position = Column(String(100), nullable=False)
    experience = Column(String(50), nullable=False)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


# =========================
# Assessment Token
# =========================
class AssessmentToken(Base):
    __tablename__ = "assessment_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    token = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), nullable=True)

    assessment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("assessments.id"),
        nullable=False,
    )

    status = Column(String(50), nullable=False, default="issued")
    expires_at = Column(DateTime(timezone=True), nullable=True)
    used_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


# =========================
# Attempt - FIXED!
# =========================
class Attempt(Base):
    __tablename__ = "attempts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    assessment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("assessments.id"),
        nullable=False,
    )
    candidate_id = Column(
        UUID(as_uuid=True),
        ForeignKey("candidates.id"),
        nullable=False,
    )
    token_id = Column(
        UUID(as_uuid=True),
        ForeignKey("assessment_tokens.id"),
        nullable=False,
    )

    # ✅ FIXED: Changed from "started" to "in_progress"
    status = Column(String(50), nullable=False, default="in_progress")

    # ✅ FIXED: Added timezone=True
    started_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    submitted_at = Column(DateTime(timezone=True), nullable=True)
    
    # ✅ ADDED: Missing fields that your API needs!
    score = Column(Float, nullable=True)
    report = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)