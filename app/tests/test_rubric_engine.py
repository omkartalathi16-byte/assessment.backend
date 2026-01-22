from app.engine.scorer import EnhancedScorer
from app.db.session import SessionLocal
from app.models.assessment_questions import AssessmentQuestion

db = SessionLocal()
scorer = EnhancedScorer()

questions = db.query(AssessmentQuestion).order_by(AssessmentQuestion.id).all()

print(f"\nFound {len(questions)} questions\n")

for q in questions:
    print("=" * 80)
    print(f"Q ID: {q.id}")
    print(f"Type: {q.question_type}")
    print(f"Question: {q.question_text}")

    # -------------------------
    # MCQ TEST
    # -------------------------
    if q.question_type == "mcq":
        candidate_answer = q.correct_answer  # simulate correct answer

        result = scorer.score_mcq(
            candidate_answer=candidate_answer,
            correct_answer=q.correct_answer,
            max_score=q.max_score
        )

        print("MCQ Result:", result)

    # -------------------------
    # DESCRIPTIVE TEST
    # -------------------------
    elif q.question_type == "descriptive":
        candidate_answer = (
            "This answer mentions preparation, detection, containment, "
            "eradication, recovery, logs, monitoring, and least privilege."
        )

        valid, msg = scorer.validate_rubric(q.rubric)
        print("Rubric valid:", valid, msg)

        if not valid:
            print("❌ Invalid rubric — skipping scoring")
            continue

        result = scorer.score_detailed(candidate_answer, q.rubric)

        print("Descriptive Result:")
        print("Score:", result["score"], "/", result["max_score"])
        print("Matched Keywords:", result["matched_keywords"])

    else:
        print("⚠️ Unknown question type")

print("\n✅ ALL QUESTIONS TESTED")
