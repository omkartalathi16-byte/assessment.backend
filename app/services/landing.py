from datetime import datetime, timezone
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.landing import Candidate, AssessmentToken, Attempt
from app.models.assessment import Assessment
from app.schemas.landing import LandingStartRequest


class LandingService:
    @staticmethod
    def start_assessment(
        db: Session,
        payload: LandingStartRequest,
    ) -> Attempt:
        """
        Start an assessment attempt from landing page submission.
        """

        # ==============================
        # 1️⃣ TOKEN HANDLING
        # ==============================
        if settings.DEV_BYPASS_TOKEN:
            print("⚠️ DEV MODE: Using dummy token...")
            
            # Get or create dummy token
            token = (
                db.query(AssessmentToken)
                .filter(AssessmentToken.token == "DEV_DUMMY_TOKEN")
                .first()
            )
            
            if not token:
                # First, get or create an assessment
                assessment = db.query(Assessment).first()
                if not assessment:
                    # Create default assessment
                    assessment = Assessment(
                        name="Default Assessment",
                        version="v1.0",
                        is_active=True,
                    )
                    db.add(assessment)
                    db.flush()
                
                # Create dummy token
                token = AssessmentToken(
                    token="DEV_DUMMY_TOKEN",
                    assessment_id=assessment.id,
                    status="issued"
                )
                db.add(token)
                db.flush()
            
            assessment_id = token.assessment_id
            
        else:
            # PRODUCTION MODE
            token = (
                db.query(AssessmentToken)
                .filter(AssessmentToken.token == payload.token)
                .first()
            )

            if not token:
                raise ValueError("Invalid assessment token")

            if token.expires_at and token.expires_at < datetime.utcnow():
                raise ValueError("Assessment token expired")

            if token.status != "issued":
                raise ValueError("Assessment token already used")

            assessment_id = token.assessment_id

        # ==============================
        # 2️⃣ CREATE CANDIDATE
        # ==============================
        candidate = Candidate(
            name=payload.name,
            email=payload.email,
            phone=payload.phone,
            location=payload.location,
            position=payload.position,
            experience=payload.experience,
        )

        db.add(candidate)
        db.flush()

        # ==============================
        # 3️⃣ CREATE ATTEMPT - FIXED!
        # ==============================
        attempt = Attempt(
            assessment_id=assessment_id,
            candidate_id=candidate.id,
            token_id=token.id,
            status="in_progress",              # ✅ FIXED: "in_progress" not "started"
            started_at=datetime.now(timezone.utc)  # ✅ FIXED: with timezone
        )

        db.add(attempt)

        # ==============================
        # 4️⃣ UPDATE TOKEN (PROD ONLY)
        # ==============================
        if not settings.DEV_BYPASS_TOKEN:
            token.status = "started"
            token.used_at = datetime.utcnow()

        db.commit()
        db.refresh(attempt)

        print(f"✅ Created attempt: {attempt.id}, status: {attempt.status}")
        return attempt