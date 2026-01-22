from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.deps import get_db
from app.schemas.landing import LandingStartRequest, LandingStartResponse
from app.services.landing import LandingService

router = APIRouter(prefix="/landing", tags=["Landing"])


@router.post(
    "/start",
    response_model=LandingStartResponse,
    status_code=status.HTTP_201_CREATED,
)
def start_assessment(
    payload: LandingStartRequest,
    db: Session = Depends(get_db),
):
    try:
        attempt = LandingService.start_assessment(db=db, payload=payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return LandingStartResponse(
        attempt_id=attempt.id,
        assessment_id=attempt.assessment_id,
    )
