from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from typing import Dict, Any
import uuid
import os

from app.db.session import get_db
from app.models.landing import Attempt, Candidate
from app.models.assessment_questions import AssessmentQuestion
from app.models.assessment_reports import AssessmentReport

from app.engine.hybrid_engine import create_hybrid_engine
from app.reports.report_builder import generate_assessment_report
from app.reports.report_docx import generate_report_docx
from app.schemas.assessment import SimpleSubmission, SubmissionResponse

router = APIRouter(prefix="/assessment", tags=["Assessment"])

engine = create_hybrid_engine()

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# -------------------------------------------------
# POST: Submit Assessment (SINGLE SOURCE OF TRUTH)
# -------------------------------------------------

@router.post("/{attempt_id}/submit", response_model=SubmissionResponse)
def submit_assessment(
    attempt_id: str,
    submission: SimpleSubmission,
    db: Session = Depends(get_db),
):
    # -------------------------------
    # 1. Validate attempt
    # -------------------------------
    try:
        attempt_uuid = uuid.UUID(attempt_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid attempt ID")

    attempt = db.query(Attempt).filter(Attempt.id == attempt_uuid).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    if attempt.status != "in_progress":
        raise HTTPException(status_code=403, detail="Assessment already submitted")

    # -------------------------------
    # 2. Load questions
    # -------------------------------
    questions = db.query(AssessmentQuestion).all()
    if not questions:
        raise HTTPException(status_code=400, detail="No assessment questions found")

    # -------------------------------
    # 3. Prepare answers for engine
    # -------------------------------
    answers_for_engine: Dict[str, Any] = {}

    for q in questions:
        user_answer = submission.answers.get(str(q.id), "")

        answers_for_engine[str(q.id)] = {
            "type": q.question_type,
            "answer": user_answer,
            "meta": {
                "section": q.section,
                "correct_answer": q.correct_answer,
                "max_score": q.max_score,
                "rubric": q.rubric,
            },
        }

    # -------------------------------
    # 4. Run Hybrid Engine
    # -------------------------------
    engine_result = engine.evaluate_assessment(answers_for_engine)

    # -------------------------------
    # 5. Resolve candidate safely
    # -------------------------------
    candidate = None
    if attempt.candidate_id:
        candidate = (
            db.query(Candidate)
            .filter(Candidate.id == attempt.candidate_id)
            .first()
        )

    candidate_email = (
        getattr(candidate, "email", None)
        or getattr(attempt, "candidate_email", None)
        or ""
    )

    candidate_phone = (
        getattr(candidate, "phone", None)
        or getattr(attempt, "candidate_phone", None)
        or ""
    )

    # -------------------------------
    # 6. Generate structured report
    # -------------------------------
    report = generate_assessment_report(
        engine_result=engine_result,
        candidate_email=candidate_email,
        candidate_phone=candidate_phone,
    )

    # 🔒 HARD SAFETY: enforce NOT NULL DB fields
    report.setdefault("section_summary", {})
    report.setdefault("pros", [])
    report.setdefault("cons", [])
    report.setdefault("scope_of_improvement", [])
    report.setdefault("question_breakdown", {})

    # -------------------------------
    # 7. Generate DOCX report
    # -------------------------------
    file_name = f"assessment_report_{attempt.id}.docx"
    file_path = os.path.join(REPORTS_DIR, file_name)

    generate_report_docx(report, file_path)

    # -------------------------------
    # 8. Store report in DB (FIXED)
    # -------------------------------
    db_report = AssessmentReport(
        assessment_id=attempt.id,
        candidate_email=report["candidate"]["email"],
        candidate_phone=report["candidate"]["phone"],

        total_score=report["scores"]["total_score"],
        max_score=report["scores"]["max_score"],
        percentage=report["scores"]["percentage"],

        level=report["scores"]["level"],
        status=report["scores"]["status"],

        section_summary=report["section_summary"],
        pros=report["pros"],
        cons=report["cons"],
        scope_of_improvement=report["scope_of_improvement"],
        question_breakdown=report["question_breakdown"],

        engine_version=report["engine_version"],
        report_file_path=file_path,
    )

    db.add(db_report)

    # -------------------------------
    # 9. Update attempt
    # -------------------------------
    attempt.status = "completed"
    attempt.submitted_at = datetime.now(timezone.utc)
    attempt.score = report["scores"]["total_score"]

    db.commit()
    db.refresh(db_report)

    # -------------------------------
    # 10. Response for frontend
    # -------------------------------
    return SubmissionResponse(
        message="Assessment submitted successfully",
        score=report["scores"]["total_score"],
        total_questions=len(submission.answers),
        submitted_at=attempt.submitted_at,
        report_id=str(db_report.id),
    )


# -------------------------------------------------
# GET: Attempt info (frontend)
# -------------------------------------------------

@router.get("/{attempt_id}")
def get_attempt(
    attempt_id: str,
    db: Session = Depends(get_db),
):
    try:
        attempt_uuid = uuid.UUID(attempt_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid attempt ID")

    attempt = db.query(Attempt).filter(Attempt.id == attempt_uuid).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    candidate = None
    if attempt.candidate_id:
        candidate = (
            db.query(Candidate)
            .filter(Candidate.id == attempt.candidate_id)
            .first()
        )

    duration_minutes = getattr(attempt, "duration_minutes", 120)

    return {
        "attempt_id": str(attempt.id),
        "candidate_name": getattr(candidate, "name", None),
        "candidate_email": getattr(candidate, "email", None),
        "started_at": attempt.started_at.isoformat() if attempt.started_at else None,
        "duration_minutes": duration_minutes,
        "status": attempt.status,
        "message": "Attempt retrieved",
    }

