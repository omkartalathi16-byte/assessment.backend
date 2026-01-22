# app/engine/nlp_engine.py

from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple, Optional
import threading
import torch


class NLPEngine:
    """
    Highly optimized semantic NLP engine with:
    - Thread-safe model loading
    - Batch processing
    - GPU / CPU auto-detection
    - Explainability support
    """

    _model: Optional[SentenceTransformer] = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.6,
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: SentenceTransformer model name
            similarity_threshold: Minimum similarity for semantic match
            batch_size: Batch size for embedding
            device: cpu / cuda / mps (auto-detected if None)
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size

        # Auto-detect device
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device

        # Thread-safe singleton model loading
        if NLPEngine._model is None:
            with NLPEngine._lock:
                if NLPEngine._model is None:
                    NLPEngine._model = SentenceTransformer(
                        self.model_name,
                        device=self.device
                    )

        self.model = NLPEngine._model

    # -------------------------------------------------
    # Embedding helpers
    # -------------------------------------------------

    def embed_text(self, text: str) -> torch.Tensor:
        """Generate embedding for a single text."""
        return self.model.encode(
            text,
            convert_to_tensor=True,
            show_progress_bar=False
        )

    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        """Batch encode texts efficiently."""
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            batch_size=self.batch_size,
            show_progress_bar=False
        )

    # -------------------------------------------------
    # Semantic matching
    # -------------------------------------------------

    def match_concepts(
        self,
        answer: str,
        concepts: List[str]
    ) -> Tuple[List[str], float]:
        """
        Match concepts semantically against an answer.

        Returns:
            matched_concepts, avg_similarity
        """
        if not answer or not concepts:
            return [], 0.0

        all_texts = [answer] + concepts
        embeddings = self.embed_texts(all_texts)

        answer_embedding = embeddings[0].unsqueeze(0)
        concept_embeddings = embeddings[1:]

        similarities = util.cos_sim(
            answer_embedding,
            concept_embeddings
        )[0]

        mask = similarities >= self.similarity_threshold
        matched_indices = torch.nonzero(mask, as_tuple=False).flatten()

        matched_concepts = [
            concepts[i] for i in matched_indices.cpu().tolist()
        ]

        if len(matched_indices) > 0:
            avg_similarity = similarities[mask].mean().item()
        else:
            avg_similarity = 0.0  # IMPORTANT: no semantic match → no credit

        return matched_concepts, round(avg_similarity, 3)

    def match_concepts_batch(
        self,
        answers: List[str],
        concepts: List[str]
    ) -> List[Tuple[List[str], float]]:
        """
        Batch semantic matching for multiple answers.
        """
        if not answers or not concepts:
            return [([], 0.0) for _ in answers]

        all_texts = answers + concepts
        embeddings = self.embed_texts(all_texts)

        answer_embeddings = embeddings[:len(answers)]
        concept_embeddings = embeddings[len(answers):]

        similarity_matrix = util.cos_sim(
            answer_embeddings,
            concept_embeddings
        )

        results = []
        for i in range(len(answers)):
            similarities = similarity_matrix[i]
            mask = similarities >= self.similarity_threshold
            matched_indices = torch.nonzero(mask, as_tuple=False).flatten()

            matched_concepts = [
                concepts[j] for j in matched_indices.cpu().tolist()
            ]

            if len(matched_indices) > 0:
                avg_similarity = similarities[mask].mean().item()
            else:
                avg_similarity = 0.0

            results.append((matched_concepts, round(avg_similarity, 3)))

        return results

    # -------------------------------------------------
    # Scoring helpers
    # -------------------------------------------------

    def semantic_coverage_score(
        self,
        answer: str,
        concepts: List[str],
        max_score: float
    ) -> Dict[str, float]:
        """
        Compute semantic coverage score.
        """
        if not concepts:
            return {
                "score": 0.0,
                "coverage_ratio": 0.0,
                "matched_concepts": []
            }

        matched, avg_similarity = self.match_concepts(answer, concepts)

        coverage_ratio = len(matched) / len(concepts)
        score = max_score * coverage_ratio * avg_similarity

        return {
            "score": round(score, 3),
            "coverage_ratio": round(coverage_ratio, 3),
            "matched_concepts": matched
        }

    def semantic_coverage_scores_batch(
        self,
        answers: List[str],
        concepts: List[str],
        max_scores: List[float]
    ) -> List[Dict[str, float]]:
        """
        Batch semantic scoring.
        """
        batch_results = self.match_concepts_batch(answers, concepts)

        final_results = []
        for (matched, avg_similarity), max_score in zip(batch_results, max_scores):
            coverage_ratio = len(matched) / len(concepts)
            score = max_score * coverage_ratio * avg_similarity

            final_results.append({
                "score": round(score, 3),
                "coverage_ratio": round(coverage_ratio, 3),
                "matched_concepts": matched
            })

        return final_results

    # -------------------------------------------------
    # Explainability
    # -------------------------------------------------

    def explain_similarity(
        self,
        answer: str,
        concepts: List[str]
    ) -> Dict[str, float]:
        """
        Per-concept similarity explanation.
        """
        if not answer or not concepts:
            return {}

        all_texts = [answer] + concepts
        embeddings = self.embed_texts(all_texts)

        similarities = util.cos_sim(
            embeddings[0].unsqueeze(0),
            embeddings[1:]
        )[0]

        return {
            concept: round(sim.item(), 3)
            for concept, sim in zip(concepts, similarities)
        }

    def explain_similarity_batch(
        self,
        answers: List[str],
        concepts: List[str]
    ) -> List[Dict[str, float]]:
        """
        Batch explainability.
        """
        if not answers or not concepts:
            return [{} for _ in answers]

        all_texts = answers + concepts
        embeddings = self.embed_texts(all_texts)

        answer_embeddings = embeddings[:len(answers)]
        concept_embeddings = embeddings[len(answers):]

        similarity_matrix = util.cos_sim(
            answer_embeddings,
            concept_embeddings
        )

        results = []
        for i in range(len(answers)):
            results.append({
                concept: round(similarity_matrix[i][j].item(), 3)
                for j, concept in enumerate(concepts)
            })

        return results

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------

    def clear_cuda_cache(self):
        """Clear CUDA cache safely."""
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def to_device(self, device: str):
        """
        Safely reload model on a new device.
        """
        with NLPEngine._lock:
            NLPEngine._model = SentenceTransformer(
                self.model_name,
                device=device
            )
            self.model = NLPEngine._model
            self.device = device
