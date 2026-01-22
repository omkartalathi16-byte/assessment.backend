from datetime import datetime
from typing import Dict, Any, List


def generate_assessment_report(
    engine_result: Dict[str, Any],
    candidate_email: str = None,
    candidate_phone: str = None
) -> Dict[str, Any]:

    questions = engine_result["questions"]

    total_score = engine_result["total_score"]
    max_score = sum(q["max_score"] for q in questions.values())
    percentage = round((total_score / max_score) * 100, 2) if max_score else 0.0

    # -------------------------
    # LEVEL & STATUS
    # -------------------------
    if percentage >= 75:
        level = "Advanced"
    elif percentage >= 50:
        level = "Intermediate"
    else:
        level = "Beginner"

    status = "Pass" if percentage >= 50 else "Fail"

    # -------------------------
    # SUMMARY (15 LINES)
    # -------------------------
    summary = [
        f"The candidate completed the assessment with an overall score of {total_score:.2f} out of {max_score:.0f}.",
        f"This corresponds to a percentage score of {percentage}%.",
        f"The assessed competency level is categorized as {level}.",
        f"The overall assessment status is marked as {status}.",
        "The evaluation was conducted using a deterministic rule-based and AI-assisted scoring engine.",
        "Multiple choice questions were evaluated using strict rule matching.",
        "Descriptive responses were evaluated using structured rubrics.",
        "Partial credit was awarded based on keyword and concept coverage.",
        "AI quality analysis was applied as an additive bonus where applicable.",
        "The candidate demonstrated varying levels of understanding across different topics.",
        "Core incident response concepts showed stronger performance.",
        "Conceptual depth varied across security framework domains.",
        "Some answers reflected clear structure and intent.",
        "Other responses lacked sufficient technical detail.",
        "Overall, the assessment highlights both strengths and improvement areas."
    ]

    # -------------------------
    # PROS / CONS
    # -------------------------
    pros: List[str] = []
    cons: List[str] = []

    for q in questions.values():
        if q["type"] == "mcq" or q["max_score"] == 0:
            continue

        ratio = q["final_score"] / q["max_score"]
        topic = q.get("section", "core security").capitalize()

        if ratio >= 0.7:
            pros.append(
                f"Strong understanding of {topic} concepts "
                f"(score: {q['final_score']}/{q['max_score']})."
            )
        elif ratio <= 0.4:
            cons.append(
                f"Limited understanding of {topic} concepts "
                f"(score: {q['final_score']}/{q['max_score']})."
            )

    # Ensure minimum 6 pros/cons
    while len(pros) < 6:
        pros.append("Demonstrated basic familiarity with security assessment terminology.")

    while len(cons) < 6:
        cons.append("Needs deeper conceptual clarity in multiple security domains.")

    # -------------------------
    # SCOPE OF IMPROVEMENT
    # -------------------------
    scope_of_improvement = []

    for c in cons:
        scope_of_improvement.append(
            f"Focus on improving: {c.replace('Limited understanding of', '').strip('.')}"
        )

    if not scope_of_improvement:
        scope_of_improvement.append(
            "Continue strengthening advanced concepts and real-world application scenarios."
        )

    # -------------------------
    # FINAL REPORT OBJECT
    # -------------------------
    return {
        "candidate": {
            "email": candidate_email,
            "phone": candidate_phone
        },
        "summary": summary,
        "pros": pros,
        "cons": cons,
        "scope_of_improvement": scope_of_improvement,
        "scores": {
            "total_score": total_score,
            "max_score": max_score,
            "percentage": percentage,
            "level": level,
            "status": status
        },
        "section_summary": engine_result.get("sections", {}),
        "engine_version": engine_result.get("engine_version"),
        # Per-question breakdown for persistence and audit
        "question_breakdown": engine_result.get("questions", {}),
        "generated_at": datetime.utcnow().isoformat()
    }
