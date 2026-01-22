from app.engine.hybrid_engine import create_hybrid_engine
from app.db.session import SessionLocal
from app.models.assessment_questions import AssessmentQuestion
from app.reports.report_builder import generate_assessment_report

# -------------------------
# Setup
# -------------------------
db = SessionLocal()
engine = create_hybrid_engine()

questions = db.query(AssessmentQuestion).all()

answers = {}

for q in questions:
    if q.question_type == "mcq":
        answers[str(q.id)] = {
            "type": "mcq",
            "answer": q.correct_answer,
            "meta": {
                "correct_answer": q.correct_answer,
                "max_score": q.max_score,
                "section": q.section
            }
        }
    else:
        answers[str(q.id)] = {
            "type": "detailed",
            "answer": (
                "preparation detection containment eradication recovery "
                "logs monitoring least privilege risk assets"
            ),
            "meta": {
                "rubric": q.rubric,
                "max_score": q.max_score,
                "section": q.section
            }
        }

# -------------------------
# Run engine
# -------------------------
engine_result = engine.evaluate_assessment(answers)

# -------------------------
# Build report
# -------------------------
report = generate_assessment_report(
    engine_result,
    candidate_email="test@example.com",
    candidate_phone="9999999999"
)

# -------------------------
# Print report
# -------------------------
print("\n=== REPORT SUMMARY ===")
print(report["summary"])

print("\n=== PROS ===")
for p in report["pros"]:
    print("-", p)

print("\n=== CONS ===")
for c in report["cons"]:
    print("-", c)
