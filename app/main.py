# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers (router objects, not modules)
from app.api.landing import router as landing_router
from app.api.assessment import router as assessment_router
from app.api.assessment_questions import router as assessment_questions_router
from app.api.assessment_report import router as assessment_report_router

app = FastAPI(
    title="AI Assessment Platform",
    version="1.0.0"
)

# --------------------------------------------------
# CORS CONFIG
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # frontend
        "http://127.0.0.1:5173",
    ],
    # allow origin variations on localhost (ports) during development
    allow_origin_regex=r"http://localhost(:[0-9]+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# API ROUTES (SINGLE SOURCE OF TRUTH)
# --------------------------------------------------

# Landing (start attempt)
app.include_router(
    landing_router,
    prefix="/api/v1",
    tags=["Landing"]
)

# Assessment (get attempt, submit answers)
app.include_router(
    assessment_router,
    prefix="/api/v1",
    tags=["Assessment"]
)

# Assessment Questions (DB driven)
app.include_router(
    assessment_questions_router,
    prefix="/api/v1",
    tags=["Assessment Questions"]
)

# Backwards-compatibility: expose questions endpoint under /api as well
app.include_router(
    assessment_questions_router,
    prefix="/api",
    tags=["Assessment Questions"]
)

# Assessment Reports (view / download report)
app.include_router(
    assessment_report_router,
    prefix="/api/v1",
    tags=["Assessment Reports"]
)

# --------------------------------------------------
# ROOT HEALTH CHECK
# --------------------------------------------------
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "AI Assessment Platform",
        "version": "1.0.0"
    }
