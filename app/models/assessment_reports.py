from sqlalchemy import Column, String, Float, JSON, TIMESTAMP, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base
import uuid


class AssessmentReport(Base):
    __tablename__ = "assessment_reports"
    __table_args__ = {"schema": "public"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    assessment_id = Column(UUID(as_uuid=True), nullable=False)

    candidate_email = Column(String)
    candidate_phone = Column(String)

    total_score = Column(Float)
    max_score = Column(Float)
    percentage = Column(Float)

    level = Column(String)
    status = Column(String)

    section_summary = Column(JSON)
    pros = Column(JSON)
    cons = Column(JSON)
    scope_of_improvement = Column(JSON)

    # Per-question breakdown must be non-null (DB enforces NOT NULL)
    question_breakdown = Column(JSON, nullable=False)

    engine_version = Column(String)
    report_file_path = Column(String)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
