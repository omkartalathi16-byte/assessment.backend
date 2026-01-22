from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from app.db.session import get_db
from app.models.assessment_questions import AssessmentQuestion

router = APIRouter(
    prefix="/assessment",
    tags=["Assessment Questions"]
)


@router.get("/questions")
def get_assessment_questions(db: Session = Depends(get_db)):
    """
    FRONTEND-SAFE QUESTIONS API

    ✔ Returns only what UI needs
    ✔ No correct answers
    ✔ No rubrics
    ✔ No scoring data
    ✔ No engine metadata
    """

    questions = (
        db.query(AssessmentQuestion)
        .order_by(AssessmentQuestion.section, AssessmentQuestion.id)
        .all()
    )

    if not questions:
        raise HTTPException(status_code=404, detail="No questions found")

    response: List[Dict[str, Any]] = []

    for q in questions:
        response.append({
            "id": q.id,  # number (frontend expects number)
            "question": q.question_text,
            "type": q.question_type,  # mcq | one-line | detailed
            "options": q.options if q.question_type == "mcq" else None,
            "section_name": q.section
        })

    return response
