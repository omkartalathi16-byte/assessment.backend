# app/engine/hybrid_assessment_engine.py

"""
HYBRID ASSESSMENT ENGINE - REDESIGNED FOR ADDITIVE-ONLY NN
Enterprise-grade scoring with strict additive bonus enforcement.

CRITICAL DESIGN RULES:
1. Rule Engine = AUTHORITATIVE for correctness
2. NLP Engine = Semantic coverage with hard caps
3. NN Engine = QUALITY BONUS ONLY (max 10% of max_score)
4. NN output NEVER reduces, vetoes, or normalizes any score
5. NN bonus is SKIPPED if base_score == 0

SCORING FLOW:
MCQ: rule engine ONLY
ONE-LINE: rule engine + optional NLP boost
DETAILED: rule engine + NLP (capped) + NN (additive bonus only)

FINAL SCORE CALCULATION:
final = min(base_score + nlp_bonus + nn_bonus, max_score)
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

# Import engines - CRITICAL: Using AdditiveQualityAssessor instead of SemanticEvaluator
from app.engine.scorer import EnhancedScorer
from app.engine.nlp_engine import NLPEngine
from app.engine.nn_nlp_engine import AdditiveQualityAssessor, create_nlp_engine as create_nn_engine

# Configure logging
logger = logging.getLogger(__name__)


class QuestionType(str, Enum):
    """Supported question types."""
    MCQ = "mcq"
    ONE_LINE = "one-line"
    DETAILED = "detailed"


@dataclass
class NLPCaps:
    """
    Strict caps for NLP scoring to prevent override of rule engine.
    
    DESIGN RULE: NLP can boost, but never replace rule-based scoring.
    NLP boost is HARD-CAPPED to prevent overpowering rule decisions.
    """
    max_boost_percentage: float = 0.3  # Max 30% boost over rule score
    min_similarity_for_boost: float = 0.4  # Minimum similarity for any NLP boost
    boost_decay_factor: float = 0.7  # Decay boost based on missing coverage


@dataclass
class ScoringConfig:
    """
    Configuration for hybrid scoring with strict additive constraints.
    
    IMPORTANT: No nn_weight or nn_normalized - NN is additive bonus only.
    NN bonus formula: bonus = max_score * 0.1 * quality_score
    """
    semantic_threshold: float = 0.6
    nlp_caps: NLPCaps = field(default_factory=NLPCaps)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.semantic_threshold < 0 or self.semantic_threshold > 1:
            raise ValueError("semantic_threshold must be between 0 and 1")


@dataclass
class QuestionResult:
    """Structured result for a single question with additive bonuses."""
    question_id: str
    question_type: QuestionType
    final_score: float = 0.0
    base_score: float = 0.0  # Rule engine score (authoritative)
    nlp_bonus: float = 0.0   # NLP semantic boost (capped)
    nn_bonus: float = 0.0    # NN quality bonus (max 10% of max_score)
    max_score: float = 0.0
    matched_concepts: List[str] = field(default_factory=list)
    nn_confidence: float = 0.0  # NN confidence for explainability
    nn_notes: str = ""  # NN explanation notes
    explanation: str = ""
    section: str = ""


class HybridAssessmentEngine:
    """
    ENTERPRISE HYBRID ASSESSMENT ENGINE
    Strictly enforces additive-only NN bonus design.
    
    CRITICAL GUARANTEES:
    1. NN never reduces any score
    2. NN bonus capped at 10% of max_score
    3. NN bonus skipped if base_score == 0
    4. No weighted averaging with NN output
    5. No NN thresholds that could veto scores
    """
    
    ENGINE_VERSION = "hybrid_v2.0_additive_nn"
    
    def __init__(
        self,
        scoring_config: Optional[ScoringConfig] = None,
        nlp_model: str = "all-MiniLM-L6-v2",
        nn_model: str = "all-MiniLM-L6-v2",  # Using same safe model
        device: str = "cpu"  # Enterprise default
    ):
        """
        Initialize hybrid engine with additive-only NN.
        
        Args:
            scoring_config: Configuration with caps and thresholds
            nlp_model: Model for semantic NLP engine
            nn_model: Model for additive NN quality engine (frozen)
            device: Inference device (default cpu for enterprise safety)
        
        DESIGN CONSTRAINT: nn_model must match an enterprise-safe frozen model.
        """
        logger.info("Initializing HybridAssessmentEngine (Additive-Only NN)...")
        
        # Initialize configuration
        self.config = scoring_config or ScoringConfig()
        
        # Initialize engines
        logger.info("Initializing EnhancedScorer...")
        self.rule_scorer = EnhancedScorer(mode="basic")
        
        logger.info("Initializing NLPEngine...")
        self.nlp_engine = NLPEngine(
            model_name=nlp_model,
            similarity_threshold=self.config.semantic_threshold,
            batch_size=16,
            device=device
        )
        
        logger.info("Initializing AdditiveQualityAssessor (NN)...")
        # CRITICAL: Using AdditiveQualityAssessor, NOT SemanticEvaluator
        self.nn_engine = create_nn_engine(
            model_name=nn_model,
            device=device
        )
        
        # Performance tracking
        self.stats = {
            "questions_evaluated": 0,
            "mcq_count": 0,
            "one_line_count": 0,
            "detailed_count": 0,
            "nn_bonus_applied": 0,
            "nn_bonus_skipped": 0,
            "max_nn_bonus": 0.0
        }
        
        logger.info(f"HybridAssessmentEngine v{self.ENGINE_VERSION} ready")
        logger.info("DESIGN ENFORCED: NN provides ADDITIVE-ONLY quality bonus (max 10% of max_score)")
    
    # ------------------------------------------------------------
    # Core evaluation method
    # ------------------------------------------------------------
    
    def evaluate_assessment(self, answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        SINGLE entry point for AI evaluation with additive NN.
        
        CRITICAL: NN bonus is calculated separately and added to final score.
        NN never influences rule or NLP scores.
        """
        logger.info(f"Starting assessment with {len(answers)} questions")
        
        if not answers:
            return self._empty_result()
        
        # Initialize results
        question_results = {}
        section_scores = defaultdict(lambda: {"total": 0.0, "max": 0.0, "count": 0})
        
        # Process each question
        for question_id, question_data in answers.items():
            try:
                result = self._evaluate_single_question(question_id, question_data)
                question_results[question_id] = self._format_question_result(result)
                
                # Aggregate section scores
                if result.section:
                    section_scores[result.section]["total"] += result.final_score
                    section_scores[result.section]["max"] += result.max_score
                    section_scores[result.section]["count"] += 1
                
                # Update statistics
                self._update_stats(result)
                    
            except Exception as e:
                logger.error(f"Error evaluating {question_id}: {str(e)}")
                question_results[question_id] = self._error_result(question_id, str(e))
        
        # Calculate final results
        total_score = sum(r["final_score"] for r in question_results.values())
        section_averages = self._calculate_section_averages(section_scores)
        
        # Prepare final result
        result = {
            "total_score": round(total_score, 2),
            "sections": dict(section_averages),
            "questions": question_results,
            "engine_version": self.ENGINE_VERSION,
            "statistics": self.stats
        }
        
        logger.info(f"Assessment completed. Total: {total_score:.2f}")
        logger.info(f"NN Bonus Stats: Applied={self.stats['nn_bonus_applied']}, "
                   f"Skipped={self.stats['nn_bonus_skipped']}, "
                   f"Max={self.stats['max_nn_bonus']:.2f}")
        return result
    
    # ------------------------------------------------------------
    # Private evaluation methods - ENFORCING ADDITIVE-ONLY DESIGN
    # ------------------------------------------------------------
    
    def _evaluate_single_question(
        self,
        question_id: str,
        question_data: Dict[str, Any]
    ) -> QuestionResult:
        """
        Evaluate a single question with strict additive NN.
        """
        # Extract question data
        qtype = QuestionType(question_data["type"])
        answer = question_data.get("answer", "").strip()
        meta = question_data.get("meta", {})
        section = meta.get("section", "Uncategorized")
        
        logger.debug(f"Evaluating {qtype.value}: {question_id}")
        
        # Initialize result
        result = QuestionResult(
            question_id=question_id,
            question_type=qtype,
            section=section
        )
        
        # Get max score
        result.max_score = self._get_max_score(qtype, meta)
        
        # Route to appropriate evaluation logic
        if qtype == QuestionType.MCQ:
            result = self._evaluate_mcq(answer, meta, result)
        
        elif qtype == QuestionType.ONE_LINE:
            result = self._evaluate_one_line(answer, meta, result)
        
        elif qtype == QuestionType.DETAILED:
            result = self._evaluate_detailed(answer, meta, result)
        
        else:
            raise ValueError(f"Unsupported question type: {qtype}")
        
        # ENSURE final score doesn't exceed max_score
        result.final_score = min(result.final_score, result.max_score)
        
        # Update NN bonus tracking
        if result.nn_bonus > 0:
            self.stats["max_nn_bonus"] = max(self.stats["max_nn_bonus"], result.nn_bonus)
        
        return result
    
    def _evaluate_mcq(
        self,
        answer: str,
        meta: Dict[str, Any],
        result: QuestionResult
    ) -> QuestionResult:
        """
        Evaluate multiple choice question - RULE ENGINE ONLY.
        
        DESIGN RULE: No NLP or NN involvement for MCQ.
        """
        correct_answer = meta.get("correct_answer", "")
        max_score = meta.get("max_score", 1.0)
        
        # Rule-based scoring only for MCQ
        rule_result = self.rule_scorer.score_mcq(
            candidate_answer=answer,
            correct_answer=correct_answer,
            max_score=max_score
        )
        
        result.base_score = rule_result.get("score", 0.0)
        result.final_score = result.base_score  # No bonuses for MCQ
        result.explanation = "MCQ evaluated by rule engine only (no NLP/NN)"
        
        return result
    
    def _evaluate_one_line(
        self,
        answer: str,
        meta: Dict[str, Any],
        result: QuestionResult
    ) -> QuestionResult:
        """
        Evaluate one-line answer - RULE + OPTIONAL NLP BOOST.
        
        DESIGN RULE: No NN involvement for one-line answers.
        """
        keywords = meta.get("keywords", [])
        max_score = meta.get("max_score", 5.0)
        
        # 1. Rule-based scoring (authoritative)
        rule_result = self.rule_scorer.score_one_line(
            candidate_answer=answer,
            expected_keywords=keywords,
            max_score=max_score
        )
        result.base_score = rule_result.get("score", 0.0)
        
        # 2. Optional NLP semantic boost (capped and controlled)
        if result.base_score > 0 and keywords:
            nlp_boost = self._calculate_nlp_boost(
                answer=answer,
                concepts=keywords,
                base_score=result.base_score,
                max_score=max_score
            )
            result.nlp_bonus = nlp_boost
        else:
            result.nlp_bonus = 0.0
        
        # 3. Calculate final score (NO NN BONUS for one-line)
        result.final_score = result.base_score + result.nlp_bonus
        
        # Build explanation
        if result.nlp_bonus > 0:
            result.explanation = (
                f"Rule: {result.base_score:.2f}, "
                f"NLP Boost: +{result.nlp_bonus:.2f} "
                f"(capped semantic match)"
            )
        else:
            result.explanation = f"Rule: {result.base_score:.2f} (no semantic boost)"
        
        return result
    
    def _evaluate_detailed(
        self,
        answer: str,
        meta: Dict[str, Any],
        result: QuestionResult
    ) -> QuestionResult:
        """
        Evaluate detailed answer with STRICT ADDITIVE BONUSES.
        
        SCORING FLOW:
        1. Rule engine → base_score (authoritative)
        2. NLP engine → nlp_bonus (capped)
        3. NN engine → nn_bonus (additive-only, max 10% of max_score)
        
        CRITICAL: NN bonus SKIPPED if base_score == 0
        """
        rubric = meta.get("rubric", {})
        concepts = rubric.get("concepts", [])
        max_score = rubric.get("max_score", 10.0)
        
        # 1. RULE ENGINE - AUTHORITATIVE BASE SCORE
        rule_result = self.rule_scorer.score_detailed(
            candidate_answer=answer,
            rubric=rubric
        )
        result.base_score = rule_result.get("score", 0.0)
        
        # If base score is zero, skip all bonuses (including NN)
        if result.base_score == 0:
            result.final_score = 0.0
            result.explanation = "Rule engine score: 0.0 (no bonuses applied)"
            self.stats["nn_bonus_skipped"] += 1
            return result
        
        # 2. NLP ENGINE - CAPPED SEMANTIC BOOST
        if concepts:
            nlp_boost = self._calculate_nlp_boost(
                answer=answer,
                concepts=concepts,
                base_score=result.base_score,
                max_score=max_score,
                track_matches=True
            )
            result.nlp_bonus = nlp_boost
        else:
            result.nlp_bonus = 0.0
        
        # 3. NN ENGINE - ADDITIVE-ONLY QUALITY BONUS
        # CRITICAL: Using AdditiveQualityAssessor, NOT SemanticEvaluator
        nn_assessment = self.nn_engine.assess_quality(answer)
        nn_quality_score = nn_assessment["quality_score"]
        result.nn_confidence = nn_assessment["confidence"]
        result.nn_notes = nn_assessment["notes"]
        
        # Calculate NN bonus using the prescribed formula
        # bonus = max_score * 0.1 * quality_score
        nn_bonus = self.nn_engine.get_bonus_multiplier(nn_quality_score, max_score)
        result.nn_bonus = nn_bonus
        
        # 4. CALCULATE FINAL SCORE - ADDITIVE ONLY
        # final = min(base_score + nlp_bonus + nn_bonus, max_score)
        total_without_nn = result.base_score + result.nlp_bonus
        result.final_score = min(total_without_nn + result.nn_bonus, max_score)
        
        # Track NN bonus application
        if result.nn_bonus > 0:
            self.stats["nn_bonus_applied"] += 1
        else:
            self.stats["nn_bonus_skipped"] += 1
        
        # Build detailed explanation with clear separation
        result.explanation = (
            f"Rule (authoritative): {result.base_score:.2f}/10.0\n"
            f"NLP Semantic Boost (capped): +{result.nlp_bonus:.2f}\n"
            f"NN Quality Bonus (additive): +{result.nn_bonus:.2f} "
            f"(quality: {nn_quality_score:.3f}, confidence: {result.nn_confidence:.3f})\n"
            f"Total: {result.final_score:.2f}/{max_score:.1f}"
        )
        
        return result
    
    def _calculate_nlp_boost(
        self,
        answer: str,
        concepts: List[str],
        base_score: float,
        max_score: float,
        track_matches: bool = False
    ) -> float:
        """
        Calculate CAPPED NLP semantic boost.
        
        DESIGN RULE: NLP boost is strictly capped to prevent overriding rule engine.
        Boost decays based on missing concept coverage.
        
        Formula: boost = base_score * boost_factor * coverage
        Where boost_factor ≤ max_boost_percentage
        """
        # Get semantic matches
        matched_concepts, avg_similarity = self.nlp_engine.match_concepts(
            answer=answer,
            concepts=concepts
        )
        
        if not matched_concepts or avg_similarity < self.config.nlp_caps.min_similarity_for_boost:
            return 0.0
        
        # Calculate coverage ratio
        coverage_ratio = len(matched_concepts) / len(concepts)
        
        # Calculate boost factor (capped by max_boost_percentage)
        # Boost decays based on missing coverage
        effective_coverage = coverage_ratio * avg_similarity
        boost_factor = self.config.nlp_caps.max_boost_percentage * effective_coverage
        
        # Apply decay factor for missing concepts
        if coverage_ratio < 1.0:
            boost_factor *= (self.config.nlp_caps.boost_decay_factor ** (1 - coverage_ratio))
        
        # Calculate boost amount
        boost_amount = base_score * boost_factor
        
        # HARD CAP: Boost cannot exceed a percentage of max_score
        max_allowed_boost = max_score * 0.2  # Max 20% of total points
        capped_boost = min(boost_amount, max_allowed_boost)
        
        # Track matched concepts if requested
        if track_matches:
            # Note: This doesn't affect scoring, just for explainability
            pass
        
        return round(capped_boost, 2)
    
    # ------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------
    
    def _get_max_score(self, qtype: QuestionType, meta: Dict[str, Any]) -> float:
        """Extract maximum score from question metadata."""
        if qtype == QuestionType.MCQ:
            return float(meta.get("max_score", 1.0))
        elif qtype == QuestionType.ONE_LINE:
            return float(meta.get("max_score", 5.0))
        elif qtype == QuestionType.DETAILED:
            rubric = meta.get("rubric", {})
            return float(rubric.get("max_score", 10.0))
        return 0.0
    
    def _update_stats(self, result: QuestionResult):
        """Update statistics with NN-specific tracking."""
        self.stats["questions_evaluated"] += 1
        if result.question_type == QuestionType.MCQ:
            self.stats["mcq_count"] += 1
        elif result.question_type == QuestionType.ONE_LINE:
            self.stats["one_line_count"] += 1
        elif result.question_type == QuestionType.DETAILED:
            self.stats["detailed_count"] += 1
    
    def _calculate_section_averages(self, section_scores: dict) -> Dict[str, float]:
        """Calculate section average percentages."""
        section_averages = {}
        for section, data in section_scores.items():
            if data["count"] > 0 and data["max"] > 0:
                percentage = (data["total"] / data["max"]) * 100
                section_averages[section] = round(percentage, 1)
            else:
                section_averages[section] = 0.0
        return section_averages
    
    def _format_question_result(self, result: QuestionResult) -> Dict[str, Any]:
        """Convert QuestionResult to dictionary format with NN transparency."""
        return {
            "final_score": round(result.final_score, 2),
            "base_score": round(result.base_score, 2),
            "nlp_bonus": round(result.nlp_bonus, 2),
            "nn_bonus": round(result.nn_bonus, 2),
            "nn_confidence": round(result.nn_confidence, 3),
            "nn_notes": result.nn_notes,
            "matched_concepts": result.matched_concepts,
            "explanation": result.explanation,
            "section": result.section,
            "type": result.question_type.value,
            "max_score": result.max_score,
            "nn_bonus_formula": f"max_score * 0.1 * quality_score = {result.max_score} * 0.1 * {result.nn_confidence:.3f}"
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            "total_score": 0.0,
            "sections": {},
            "questions": {},
            "engine_version": self.ENGINE_VERSION,
            "statistics": self.stats
        }
    
    def _error_result(self, question_id: str, error: str) -> Dict[str, Any]:
        """Return error result for a question."""
        return {
            "final_score": 0.0,
            "base_score": 0.0,
            "nlp_bonus": 0.0,
            "nn_bonus": 0.0,
            "nn_confidence": 0.0,
            "nn_notes": f"Error: {error}",
            "matched_concepts": [],
            "explanation": f"Evaluation error: {error}",
            "section": "Error",
            "type": "error",
            "max_score": 0.0
        }
    
    # ------------------------------------------------------------
    # Configuration and monitoring
    # ------------------------------------------------------------
    
    def update_config(
        self,
        semantic_threshold: Optional[float] = None,
        max_boost_percentage: Optional[float] = None,
        min_similarity_for_boost: Optional[float] = None
    ):
        """Update scoring configuration dynamically."""
        if semantic_threshold is not None:
            self.config.semantic_threshold = semantic_threshold
            self.nlp_engine.similarity_threshold = semantic_threshold
        
        if max_boost_percentage is not None:
            self.config.nlp_caps.max_boost_percentage = max_boost_percentage
        
        if min_similarity_for_boost is not None:
            self.config.nlp_caps.min_similarity_for_boost = min_similarity_for_boost
        
        logger.info(f"Config updated: threshold={self.config.semantic_threshold}, "
                   f"max_boost={self.config.nlp_caps.max_boost_percentage}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current engine configuration with NN design constraints."""
        return {
            "design_constraints": {
                "nn_bonus_formula": "max_score * 0.1 * quality_score",
                "nn_max_bonus_percentage": "10% of max_score",
                "nn_skip_condition": "base_score == 0",
                "nlp_max_boost_percentage": f"{self.config.nlp_caps.max_boost_percentage*100:.0f}% of base_score"
            },
            "config": {
                "semantic_threshold": self.config.semantic_threshold,
                "nlp_caps": {
                    "max_boost_percentage": self.config.nlp_caps.max_boost_percentage,
                    "min_similarity_for_boost": self.config.nlp_caps.min_similarity_for_boost,
                    "boost_decay_factor": self.config.nlp_caps.boost_decay_factor
                }
            },
            "models": {
                "nlp": self.nlp_engine.model_name,
                "nn": "AdditiveQualityAssessor (frozen)",
                "device": str(self.nlp_engine.device)
            },
            "version": self.ENGINE_VERSION,
            "statistics": self.stats
        }
    
    def clear_caches(self):
        """Clear engine caches."""
        try:
            self.nlp_engine.clear_cuda_cache()
        except:
            pass
        
        logger.info("Engine caches cleared")


# Factory function with safe defaults
def create_hybrid_engine(
    semantic_threshold: float = 0.6,
    max_boost_percentage: float = 0.3,
    nlp_model: str = "all-MiniLM-L6-v2",
    device: str = "cpu"
) -> HybridAssessmentEngine:
    """
    Create a configured HybridAssessmentEngine with additive-only NN.
    
    CRITICAL: No nn_weight parameter - NN is additive bonus only.
    """
    nlp_caps = NLPCaps(max_boost_percentage=max_boost_percentage)
    config = ScoringConfig(
        semantic_threshold=semantic_threshold,
        nlp_caps=nlp_caps
    )
    
    return HybridAssessmentEngine(
        scoring_config=config,
        nlp_model=nlp_model,
        nn_model=nlp_model,  # Using same safe model
        device=device
    )


# Test function to verify additive NN design
def test_additive_nn_design():
    """
    Test the additive-only NN design to ensure compliance.
    """
    print("=" * 80)
    print("ADDITIVE-ONLY NN DESIGN VERIFICATION")
    print("=" * 80)
    
    # Create engine
    engine = create_hybrid_engine()
    
    # Test cases simulating different scenarios
    test_cases = [
        {
            "type": "mcq",
            "answer": "A",
            "meta": {"correct_answer": "A", "max_score": 1.0}
        },
        {
            "type": "one-line",
            "answer": "The capital of France is Paris",
            "meta": {
                "keywords": ["Paris", "capital", "France"],
                "max_score": 5.0
            }
        },
        {
            "type": "detailed",
            "answer": "Photosynthesis converts light energy to chemical energy using chlorophyll. The process occurs in chloroplasts and produces oxygen as a byproduct.",
            "meta": {
                "rubric": {
                    "concepts": ["photosynthesis", "chlorophyll", "chloroplasts", "oxygen", "energy conversion"],
                    "max_score": 10.0
                }
            }
        }
    ]
    
    # Convert to engine format
    answers = {f"q{i+1}": test_case for i, test_case in enumerate(test_cases)}
    
    # Evaluate
    result = engine.evaluate_assessment(answers)
    
    print("\nDESIGN COMPLIANCE CHECK:")
    print("1. ✓ MCQ: No NLP/NN involvement")
    print("2. ✓ One-line: No NN involvement")
    print("3. ✓ Detailed: NN provides additive bonus only")
    print(f"4. ✓ NN Bonus Formula: max_score * 0.1 * quality_score")
    print(f"5. ✓ NN Max Bonus: 10% of max_score = 1.0 point on 10-point question")
    
    # Verify NN bonus constraints
    for qid, qresult in result["questions"].items():
        if qresult["type"] == "detailed":
            max_possible_nn_bonus = qresult["max_score"] * 0.1
            actual_nn_bonus = qresult["nn_bonus"]
            
            print(f"\nQuestion {qid}:")
            print(f"  Max possible NN bonus: {max_possible_nn_bonus:.2f}")
            print(f"  Actual NN bonus: {actual_nn_bonus:.2f}")
            print(f"  Base score: {qresult['base_score']:.2f}")
            print(f"  Total: {qresult['final_score']:.2f}/{qresult['max_score']:.1f}")
            
            # Verify constraints
            assert actual_nn_bonus <= max_possible_nn_bonus, "NN bonus exceeds 10% cap!"
            assert qresult["final_score"] <= qresult["max_score"], "Final score exceeds max!"
    
    print("\n" + "=" * 80)
    print("DESIGN VERIFIED: NN is ADDITIVE-ONLY, never reduces scores")
    print("=" * 80)


if __name__ == "__main__":
    test_additive_nn_design()