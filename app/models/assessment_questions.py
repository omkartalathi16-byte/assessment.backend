from sqlalchemy import Column, String, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.db.base import Base
import uuid


class AssessmentQuestion(Base):
    __tablename__ = "assessment_questions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    assessment_id = Column(UUID(as_uuid=True), nullable=False)
    section = Column(String(100), nullable=False)

    question_type = Column(String(20), nullable=False)
    question_text = Column(Text, nullable=False)

    options = Column(JSONB)
    correct_answer = Column(JSONB)

    rubric = Column(JSONB, nullable=False)
    max_score = Column(Integer, nullable=False)
