# app/engine/scorer.py

from typing import Dict, Any, List, Optional, Tuple
import re
from collections import defaultdict

from fastapi import logger


class EnhancedScorer:
    """
    PURE RULE-BASED SCORING ENGINE.
    
    This scorer is 100% deterministic and intentionally non-AI.
    No embeddings, no NLP, no ML - only rule-based keyword matching.
    
    Used as the rule-based component in the HybridAssessmentEngine.
    """
    
    def __init__(self, mode: str = "basic"):
        """
        Initialize the rule-based scorer.
        
        Args:
            mode: "basic" for standard scoring, "strict" for exact matching only
        """
        self.mode = mode
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for deterministic keyword matching.
        
        Rules:
        1. Convert to lowercase
        2. Remove all punctuation (preserve only alphanumeric and spaces)
        3. Collapse multiple spaces
        4. Trim whitespace
        """
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
        text = re.sub(r'\s+', ' ', text)      # Collapse multiple spaces
        return text.strip()
    
    def _strict_keyword_match(self, normalized_text: str, keyword: str) -> bool:
        """
        Strict keyword matching rule.
        
        Returns True if:
        1. Exact word match (keyword exists as separate word)
        2. For multi-word keywords: all words appear in correct order
        """
        normalized_keyword = self._normalize_text(keyword)
        
        # For multi-word keywords, check exact phrase
        if ' ' in normalized_keyword:
            return normalized_keyword in normalized_text
        
        # For single word, check as separate word
        words = normalized_text.split()
        return normalized_keyword in words
    
    def score_mcq(
        self,
        candidate_answer: str,
        correct_answer: str,
        max_score: float
    ) -> Dict[str, Any]:
        """
        Score multiple choice question - BINARY RULE ONLY.
        
        This is 100% deterministic:
        - Correct = full score
        - Incorrect = zero score
        - No partial credit
        
        Args:
            candidate_answer: Student's answer
            correct_answer: Expected correct answer
            max_score: Maximum possible score
            
        Returns:
            Dictionary with score, explanation, and matched status
        """
        # Simple exact string match (case-insensitive)
        is_correct = candidate_answer.strip().lower() == correct_answer.strip().lower()
        
        score = max_score if is_correct else 0.0
        
        return {
            "score": score,
            "max_score": max_score,
            "matched": is_correct,
            "explanation": "Full credit for exact match" if is_correct else "No credit for incorrect answer",
            "rule_engine": "binary_mcq"
        }
    
    def score_one_line(
        self,
        candidate_answer: str,
        expected_keywords: List[str],
        max_score: float
    ) -> Dict[str, Any]:
        """
        Score one-line answer with strict keyword matching.
        
        Rules:
        1. Normalize both answer and keywords
        2. Match keywords strictly (exact word/phrase)
        3. Score = (matched_keywords / total_keywords) * max_score
        4. No semantic understanding, no similarity measures
        
        Args:
            candidate_answer: Student's answer string
            expected_keywords: List of required keywords/concepts
            max_score: Maximum possible score
            
        Returns:
            Dictionary with score, matched keywords, and explanation
        """
        if not candidate_answer or not expected_keywords:
            return {
                "score": 0.0,
                "max_score": max_score,
                "matched_keywords": [],
                "coverage": 0.0,
                "explanation": "No keywords provided or empty answer",
                "rule_engine": "keyword_coverage"
            }
        
        # Normalize
        normalized_answer = self._normalize_text(candidate_answer)
        
        # Strict keyword matching
        matched_keywords = []
        for keyword in expected_keywords:
            if self._strict_keyword_match(normalized_answer, keyword):
                matched_keywords.append(keyword)
        
        # Calculate coverage
        total_keywords = len(expected_keywords)
        matched_count = len(matched_keywords)
        coverage = matched_count / total_keywords if total_keywords > 0 else 0.0
        
        # Final score (proportional to coverage)
        score = coverage * max_score
        
        # Build explanation
        if matched_count == total_keywords:
            explanation = f"All {matched_count} keywords matched perfectly"
        elif matched_count > 0:
            explanation = f"Matched {matched_count} of {total_keywords} keywords"
        else:
            explanation = "No keywords matched"
        
        return {
            "score": round(score, 2),
            "max_score": max_score,
            "matched_keywords": matched_keywords,
            "coverage": round(coverage, 3),
            "explanation": explanation,
            "rule_engine": "keyword_coverage"
        }
    
    def score_detailed(
        self,
        candidate_answer: str,
        rubric: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score detailed answer using pure rule-based rubric.
        
        Rubric format:
        {
            "max_score": 10.0,
            "criteria": [
                {
                    "keywords": ["jwt", "token"],
                    "weight": 0.4,
                    "description": "Mention JWT tokens"
                },
                {
                    "keywords": ["authentication", "auth"],
                    "weight": 0.6,
                    "description": "Discuss authentication"
                }
            ]
        }
        
        Rules:
        1. Each criterion scored independently
        2. Criterion score = keyword_coverage * weight * max_score
        3. No semantic understanding of concepts
        4. Weight validation (must sum to ~1.0)
        
        Args:
            candidate_answer: Student's detailed answer
            rubric: Dictionary with scoring criteria
            
        Returns:
            Dictionary with total score, criterion breakdown, and matched keywords
        """
        # Validate rubric
        if "criteria" not in rubric or not rubric["criteria"]:
            raise ValueError("Rubric must contain 'criteria' list")
        
        max_score = float(rubric.get("max_score", 10.0))
        
        # Normalize answer once
        normalized_answer = self._normalize_text(candidate_answer)
        
        # Score each criterion
        total_weight = 0.0
        total_score = 0.0
        criterion_results = []
        all_matched_keywords = []
        
        for i, criterion in enumerate(rubric["criteria"]):
            # Extract criterion data
            keywords = criterion.get("keywords", [])
            weight = float(criterion.get("weight", 0.0))
            description = criterion.get("description", f"Criterion {i+1}")
            
            # Validate
            if not keywords:
                continue
            
            # Calculate keyword coverage for this criterion
            matched_keywords = []
            for keyword in keywords:
                if self._strict_keyword_match(normalized_answer, keyword):
                    matched_keywords.append(keyword)
            
            coverage = len(matched_keywords) / len(keywords) if keywords else 0.0
            
            # Criterion score = coverage * weight * max_score
            criterion_score = coverage * weight * max_score
            
            # Track results
            total_weight += weight
            total_score += criterion_score
            all_matched_keywords.extend(matched_keywords)
            
            criterion_results.append({
                "criterion_index": i,
                "description": description,
                "keywords": keywords,
                "matched_keywords": matched_keywords,
                "coverage": round(coverage, 3),
                "weight": weight,
                "score": round(criterion_score, 2),
                "max_possible": weight * max_score
            })
        
        # Validate weights (allow small rounding errors)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Rubric weights sum to {total_weight}, expected ~1.0")
        
        # Ensure score doesn't exceed max
        final_score = min(total_score, max_score)
        
        # Build explanation
        matched_criteria = sum(1 for cr in criterion_results if cr["coverage"] > 0)
        total_criteria = len(criterion_results)
        
        explanation = (
            f"Matched {len(set(all_matched_keywords))} unique keywords across "
            f"{matched_criteria}/{total_criteria} criteria"
        )
        
        return {
            "score": round(final_score, 2),
            "max_score": max_score,
            "coverage": round(final_score / max_score, 3) if max_score > 0 else 0.0,
            "matched_keywords": list(set(all_matched_keywords)),  # Unique keywords
            "criteria": criterion_results,
            "explanation": explanation,
            "rule_engine": "rubric_based"
        }
    
    def validate_rubric(self, rubric: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate rubric structure for rule-based scoring.
        
        Args:
            rubric: Rubric dictionary to validate
            
        Returns:
            (is_valid, error_message)
        """
        try:
            # Check required fields
            if "max_score" not in rubric:
                return False, "Missing 'max_score' field"
            
            if "criteria" not in rubric or not rubric["criteria"]:
                return False, "Missing or empty 'criteria' list"
            
            max_score = float(rubric["max_score"])
            if max_score <= 0:
                return False, "max_score must be positive"
            
            # Validate each criterion
            total_weight = 0.0
            for i, criterion in enumerate(rubric["criteria"]):
                if "keywords" not in criterion or not criterion["keywords"]:
                    return False, f"Criterion {i+1} missing 'keywords' list"
                
                weight = float(criterion.get("weight", 0.0))
                if weight < 0 or weight > 1:
                    return False, f"Criterion {i+1} weight must be between 0 and 1"
                
                total_weight += weight
            
            # Check weight sum (allow small rounding errors)
            if abs(total_weight - 1.0) > 0.01:
                return False, f"Weights sum to {total_weight:.2f}, should be ~1.0"
            
            return True, "Rubric is valid"
            
        except (ValueError, TypeError) as e:
            return False, f"Rubric validation error: {str(e)}"


# -------------------------------------------------------------------
# FastAPI Integration Helper (Legacy - HybridAssessmentEngine replaces this)
# -------------------------------------------------------------------

def score_assessment(answers: dict) -> dict:
    """
    HYBRID ENGINE ENTRY POINT.
    Backward-compatible hook for UI / API.
    """
    # Lazy import to avoid circular dependencies
    from app.engine.engine_registry import hybrid_engine
    return hybrid_engine.evaluate_assessment(answers)


# -------------------------------------------------------------------
# Testing Helper
# -------------------------------------------------------------------

def create_sample_rubric() -> Dict[str, Any]:
    """Create a sample rubric for testing."""
    return {
        "max_score": 10.0,
        "criteria": [
            {
                "keywords": ["neural", "network", "nn"],
                "weight": 0.4,
                "description": "Mention neural networks"
            },
            {
                "keywords": ["backpropagation", "gradient", "descent"],
                "weight": 0.3,
                "description": "Discuss training algorithm"
            },
            {
                "keywords": ["activation", "function", "relu", "sigmoid"],
                "weight": 0.3,
                "description": "Cover activation functions"
            }
        ]
    }


if __name__ == "__main__":
    # Quick test
    scorer = EnhancedScorer()
    
    # Test MCQ
    mcq_result = scorer.score_mcq("Paris", "Paris", 1.0)
    print("MCQ Test:", mcq_result)
    
    # Test One-line
    one_line_result = scorer.score_one_line(
        "Neural networks use backpropagation for training",
        ["neural", "backpropagation", "training"],
        5.0
    )
    print("\nOne-line Test:", one_line_result)
    
    # Test Detailed
    detailed_result = scorer.score_detailed(
        "Neural networks are trained using backpropagation. "
        "They use activation functions like ReLU.",
        create_sample_rubric()
    )
    print("\nDetailed Test Score:", detailed_result["score"])
    print("Matched Keywords:", detailed_result["matched_keywords"])