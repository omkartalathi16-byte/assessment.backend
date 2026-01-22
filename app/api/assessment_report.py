from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import os
import uuid

from app.db.session import get_db
from app.models.assessment_reports import AssessmentReport

router = APIRouter(prefix="/assessment/report", tags=["Assessment Report"])


@router.get("/{attempt_id}")
def get_or_download_report(
    attempt_id: str,
    download: bool = Query(False, description="Set true to download report"),
    db: Session = Depends(get_db)
):
    # ---------------------------------
    # 1. Validate attempt_id
    # ---------------------------------
    try:
        attempt_uuid = uuid.UUID(attempt_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid attempt ID")

    # ---------------------------------
    # 2. Fetch report from DB
    # ---------------------------------
    report = (
        db.query(AssessmentReport)
        .filter(AssessmentReport.assessment_id == attempt_uuid)
        .first()
    )

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # ---------------------------------
    # 3. DOWNLOAD MODE
    # ---------------------------------
    if download:
        if not report.report_file_path:
            raise HTTPException(status_code=404, detail="Report file not available")

        if not os.path.exists(report.report_file_path):
            raise HTTPException(status_code=404, detail="Report file missing on server")

        return FileResponse(
            path=report.report_file_path,
            filename=os.path.basename(report.report_file_path),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    # ---------------------------------
    # 4. JSON MODE (UI VIEW)
    # ---------------------------------
    return {
        "attempt_id": str(report.assessment_id),
        "scores": {
            "total_score": report.total_score,
            "max_score": report.max_score,
            "percentage": report.percentage,
            "level": report.level,
            "status": report.status
        },
        "section_summary": report.section_summary,
        "pros": report.pros,
        "cons": report.cons,
        "scope_of_improvement": report.scope_of_improvement,
        "engine_version": report.engine_version,
        "created_at": report.created_at
    }
