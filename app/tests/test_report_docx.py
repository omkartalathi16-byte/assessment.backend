from app.reports.report_builder import generate_assessment_report
from app.reports.report_docx import generate_report_docx
from app.engine.hybrid_engine import create_hybrid_engine
from app.db.session import SessionLocal
from app.models.assessment_questions import AssessmentQuestion

db = SessionLocal()
engine = create_hybrid_engine()

questions = db.query(AssessmentQuestion).all()
answers = {}

for q in questions:
    answers[str(q.id)] = {
        "type": q.question_type,
        "answer": "preparation detection containment eradication recovery",
        "meta": {
            "rubric": q.rubric,
            "correct_answer": q.correct_answer,
            "max_score": q.max_score,
            "section": q.section
        }
    }

engine_result = engine.evaluate_assessment(answers)
report = generate_assessment_report(engine_result)

generate_report_docx(report, "assessment_report.docx")
print("Report generated: assessment_report.docx")
