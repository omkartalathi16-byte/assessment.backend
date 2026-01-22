from app.engine.hybrid_engine import create_hybrid_engine

engine = create_hybrid_engine()

answers = {
    "q1": {
        "type": "detailed",  # ✅ FIXED
        "answer": "preparation detection containment eradication recovery",
        "meta": {
            "rubric": {
                "max_score": 10,
                "criteria": [
                    {"keywords": ["preparation"], "weight": 0.2, "description": "prep"},
                    {"keywords": ["detection"], "weight": 0.2, "description": "detect"},
                    {"keywords": ["containment"], "weight": 0.2, "description": "contain"},
                    {"keywords": ["eradication"], "weight": 0.2, "description": "eradicate"},
                    {"keywords": ["recovery"], "weight": 0.2, "description": "recover"}
                ]
            },
            "section": "nist"
        }
    }
}

result = engine.evaluate_assessment(answers)

print("TOTAL SCORE:", result["total_score"])
print("QUESTIONS:", result["questions"])
