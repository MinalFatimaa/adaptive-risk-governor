from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ============================================================
# SEMANTIC COMPLAINT RESULT
# ============================================================

@dataclass
class SemanticComplaintResult:
    """
    Structured semantic signals extracted from a complaint.

    This layer does not make a risk decision.
    It only expands what the Governor can observe.
    """

    complaint_text: str

    normalized_claim_type: str

    embedding_dimension: int

    claim_confidence: float = 0.0

    similarity_to_previous: float = 0.0

    paraphrase_score: float = 0.0

    contradiction_score: float = 0.0

    semantic_claim_switch_score: float = 0.0

    evidence_consistency_score: float = 0.0

    metadata: dict[str, object] = field(
        default_factory=dict
    )


# ============================================================
# SEMANTIC COMPLAINT INTELLIGENCE
# ============================================================

class SemanticComplaintIntelligence:
    """
    Semantic intelligence layer for customer complaints.

    Responsibilities:

        1. Convert complaint text into embeddings.
        2. Infer a normalized claim type.
        3. Compare complaints semantically.
        4. Detect paraphrased complaints.
        5. Detect semantic claim switching.
        6. Detect semantic contradiction.
        7. Compare complaint text with evidence text.

    This component does NOT determine fraud.

    It produces observations for the Governor and
    future ML models.
    """

    CLAIM_PROTOTYPES = {
        "SHORTAGE_CLAIM": (
            "Some items were missing from the package."
        ),

        "WRONG_ITEM_CLAIM": (
            "The customer received a different item "
            "from the item that was ordered."
        ),

        "NON_DELIVERY_CLAIM": (
            "The customer says that the order "
            "was never delivered."
        ),

        "SUBSTITUTED_RETURN_CLAIM": (
            "The customer says that the returned item "
            "was replaced or substituted."
        ),
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        paraphrase_threshold: float = 0.80,
    ) -> None:

        if not 0.0 <= paraphrase_threshold <= 1.0:
            raise ValueError(
                "paraphrase_threshold must be between 0 and 1."
            )

        self.model_name = model_name

        self.paraphrase_threshold = (
            paraphrase_threshold
        )

        self._model = None

        self._prototype_embeddings = None

    # ========================================================
    # MODEL
    # ========================================================

    def _get_model(self):
        """
        Lazily load the sentence-transformer model.

        This avoids loading the model during module import.
        """

        if self._model is None:

            try:
                from sentence_transformers import (
                    SentenceTransformer,
                )

            except ImportError as exc:

                raise ImportError(
                    "sentence-transformers is required. "
                    "Install it with: "
                    "python -m pip install sentence-transformers"
                ) from exc

            self._model = SentenceTransformer(
                self.model_name
            )

        return self._model

    # ========================================================
    # EMBEDDING
    # ========================================================

    def embed(
        self,
        text: str,
    ) -> np.ndarray:

        text = self._validate_text(text)

        model = self._get_model()

        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        )

    # ========================================================
    # BATCH EMBEDDING
    # ========================================================

    def embed_many(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:

        if not texts:

            return np.empty(
                (0, 0),
                dtype=np.float32,
            )

        cleaned = [
            self._validate_text(text)
            for text in texts
        ]

        model = self._get_model()

        embeddings = model.encode(
            cleaned,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    # ========================================================
    # CLAIM TYPE
    # ========================================================

    def _get_prototype_embeddings(self) -> np.ndarray:

        if self._prototype_embeddings is None:

            prototype_texts = list(
                self.CLAIM_PROTOTYPES.values()
            )

            self._prototype_embeddings = (
                self.embed_many(
                    prototype_texts
                )
            )

        return self._prototype_embeddings

    def infer_claim_type(
        self,
        complaint_text: str,
    ) -> tuple[str, float]:

        complaint_embedding = self.embed(
            complaint_text
        )

        prototype_embeddings = (
            self._get_prototype_embeddings()
        )

        similarities = (
            prototype_embeddings
            @ complaint_embedding
        )

        prototype_names = list(
            self.CLAIM_PROTOTYPES.keys()
        )

        best_index = int(
            np.argmax(similarities)
        )

        confidence = float(
            similarities[best_index]
        )

        return (
            prototype_names[best_index],
            self._bounded(confidence),
        )

    # ========================================================
    # SIMILARITY
    # ========================================================

    def similarity(
        self,
        text_a: str,
        text_b: str,
    ) -> float:

        embedding_a = self.embed(
            text_a
        )

        embedding_b = self.embed(
            text_b
        )

        score = float(
            np.dot(
                embedding_a,
                embedding_b,
            )
        )

        return self._bounded(
            score
        )

    # ========================================================
    # HISTORY SIMILARITY
    # ========================================================

    def similarity_to_previous(
        self,
        complaint_text: str,
        previous_complaints: Sequence[str],
    ) -> float:

        if not previous_complaints:
            return 0.0

        current_embedding = self.embed(
            complaint_text
        )

        previous_embeddings = self.embed_many(
            previous_complaints
        )

        similarities = (
            previous_embeddings
            @ current_embedding
        )

        return self._bounded(
            float(
                np.max(similarities)
            )
        )

    # ========================================================
    # PARAPHRASE
    # ========================================================

    def paraphrase_score(
        self,
        complaint_text: str,
        previous_complaints: Sequence[str],
    ) -> float:

        if not previous_complaints:
            return 0.0

        similarity_score = (
            self.similarity_to_previous(
                complaint_text,
                previous_complaints,
            )
        )

        if (
            similarity_score
            >= self.paraphrase_threshold
        ):
            return similarity_score

        return 0.0

    # ========================================================
    # SEMANTIC CLAIM SWITCH
    # ========================================================

    def semantic_claim_switch_score(
        self,
        complaint_text: str,
        previous_complaint: str | None,
    ) -> float:

        if not previous_complaint:
            return 0.0

        current_type, current_confidence = (
            self.infer_claim_type(
                complaint_text
            )
        )

        previous_type, previous_confidence = (
            self.infer_claim_type(
                previous_complaint
            )
        )

        if current_type == previous_type:
            return 0.0

        score = (
            current_confidence
            * previous_confidence
        )

        return self._bounded(
            score
        )

    # ========================================================
    # CONTRADICTION
    # ========================================================

    def contradiction_score(
        self,
        complaint_text: str,
        previous_complaint: str | None,
    ) -> float:

        if not previous_complaint:
            return 0.0

        current_type, current_confidence = (
            self.infer_claim_type(
                complaint_text
            )
        )

        previous_type, previous_confidence = (
            self.infer_claim_type(
                previous_complaint
            )
        )

        if current_type == previous_type:
            return 0.0

        similarity = self.similarity(
            complaint_text,
            previous_complaint,
        )

        semantic_shift = (
            1.0 - similarity
        )

        confidence = (
            current_confidence
            * previous_confidence
        )

        score = (
            semantic_shift
            * confidence
        )

        return self._bounded(
            score
        )

    # ========================================================
    # EVIDENCE CONSISTENCY
    # ========================================================

    def evidence_consistency(
        self,
        complaint_text: str,
        evidence_texts: Sequence[str],
    ) -> float:

        if not evidence_texts:
            return 0.0

        complaint_embedding = self.embed(
            complaint_text
        )

        evidence_embeddings = self.embed_many(
            evidence_texts
        )

        similarities = (
            evidence_embeddings
            @ complaint_embedding
        )

        return self._bounded(
            float(
                np.mean(similarities)
            )
        )

    # ========================================================
    # FULL ANALYSIS
    # ========================================================

    def analyze(
        self,
        *,
        complaint_text: str,
        previous_complaints: Sequence[str] | None = None,
        evidence_texts: Sequence[str] | None = None,
    ) -> SemanticComplaintResult:

        previous_complaints = (
            previous_complaints or []
        )

        evidence_texts = (
            evidence_texts or []
        )

        claim_type, claim_confidence = (
            self.infer_claim_type(
                complaint_text
            )
        )

        similarity_score = (
            self.similarity_to_previous(
                complaint_text,
                previous_complaints,
            )
        )

        paraphrase = (
            self.paraphrase_score(
                complaint_text,
                previous_complaints,
            )
        )

        previous_complaint = (
            previous_complaints[-1]
            if previous_complaints
            else None
        )

        switch_score = (
            self.semantic_claim_switch_score(
                complaint_text,
                previous_complaint,
            )
        )

        contradiction = (
            self.contradiction_score(
                complaint_text,
                previous_complaint,
            )
        )

        evidence_consistency = (
            self.evidence_consistency(
                complaint_text,
                evidence_texts,
            )
        )

        embedding = self.embed(
            complaint_text
        )

        return SemanticComplaintResult(

            complaint_text=complaint_text,

            normalized_claim_type=claim_type,

            embedding_dimension=int(
                embedding.shape[0]
            ),

            claim_confidence=(
                claim_confidence
            ),

            similarity_to_previous=(
                similarity_score
            ),

            paraphrase_score=(
                paraphrase
            ),

            contradiction_score=(
                contradiction
            ),

            semantic_claim_switch_score=(
                switch_score
            ),

            evidence_consistency_score=(
                evidence_consistency
            ),

            metadata={
                "model_name": self.model_name,
                "previous_complaint_count": len(
                    previous_complaints
                ),
            },
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_text(
        text: str,
    ) -> str:

        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                "Complaint text must be a string."
            )

        cleaned = text.strip()

        if not cleaned:
            raise ValueError(
                "Complaint text cannot be empty."
            )

        return cleaned

    # ========================================================
    # BOUNDS
    # ========================================================

    @staticmethod
    def _bounded(
        value: float,
    ) -> float:

        if not np.isfinite(value):
            return 0.0

        return float(
            max(
                0.0,
                min(
                    1.0,
                    value,
                ),
            )
        )