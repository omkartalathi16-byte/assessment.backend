from app.engine.hybrid_engine import create_hybrid_engine
from app.db.session import SessionLocal
from app.models.assessment_questions import AssessmentQuestion

# ----------------------------
# Setup
# ----------------------------
db = SessionLocal()
engine = create_hybrid_engine()

questions = db.query(AssessmentQuestion).all()
print(f"\nFound {len(questions)} questions\n")

answers = {}

# ----------------------------
# Build answers dict for engine
# ----------------------------
for q in questions:
    if q.question_type == "mcq":
        answers[str(q.id)] = {
            "type": "mcq",
            "answer": q.correct_answer,   # simulate correct MCQ
            "meta": {
                "correct_answer": q.correct_answer,
                "max_score": q.max_score,
                "section": q.section
            }
        }

    elif q.question_type == "detailed":
        answers[str(q.id)] = {
            "type": "detailed",
            "answer": (
                "preparation detection containment eradication recovery "
                "logs monitoring least privilege risk assets access control"
            ),
            "meta": {
                "rubric": q.rubric,
                "max_score": q.max_score,
                "section": q.section
            }
        }

# ----------------------------
# Run hybrid evaluation
# ----------------------------
result = engine.evaluate_assessment(answers)

# ----------------------------
# Print per-question results
# ----------------------------
print("=" * 80)
print("PER QUESTION SCORES")
print("=" * 80)

for qid, qres in result["questions"].items():
    print(
        f"{qid} | "
        f"type={qres['type']} | "
        f"score={qres['final_score']}/{qres['max_score']}"
    )

# ----------------------------
# Print TOTAL SCORE
# ----------------------------
print("\n" + "=" * 80)
print("FINAL ASSESSMENT RESULT")
print("=" * 80)
print("TOTAL SCORE:", result["total_score"])
print("SECTION SCORES:", result["sections"])
print("ENGINE VERSION:", result["engine_version"])
