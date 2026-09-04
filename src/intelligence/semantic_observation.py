from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

from src.intelligence.semantic_complaint import (
    SemanticComplaintIntelligence,
    SemanticComplaintResult,
)


class SemanticObservationAdapter:
    """
    Converts SemanticComplaintIntelligence output into
    Governor-compatible observation features.

    This adapter intentionally does NOT calculate risk.

    Its responsibility is only:

        complaint text
              ↓
        semantic intelligence
              ↓
        canonical observation features

    This keeps semantic interpretation separate from
    risk scoring and risk fusion.
    """

    def __init__(
        self,
        intelligence: SemanticComplaintIntelligence | None = None,
    ) -> None:

        self.intelligence = (
            intelligence
            if intelligence is not None
            else SemanticComplaintIntelligence()
        )

    # ========================================================
    # BUILD SEMANTIC OBSERVATION
    # ========================================================

    def build_observation(
        self,
        *,
        complaint_text: str,
        previous_complaints: Sequence[str] | None = None,
        evidence_texts: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze complaint text and return a structured
        observation suitable for the Governor.

        No risk score is produced here.
        """

        result = self.intelligence.analyze(
            complaint_text=complaint_text,
            previous_complaints=(
                previous_complaints or []
            ),
            evidence_texts=(
                evidence_texts or []
            ),
        )

        return self._result_to_observation(
            result
        )

    # ========================================================
    # CONVERT RESULT
    # ========================================================

    @staticmethod
    def _result_to_observation(
        result: SemanticComplaintResult,
    ) -> dict[str, Any]:
        """
        Convert semantic result into stable,
        flat observation fields.

        Flat numeric features make later ML integration
        straightforward.
        """

        return {
            "semantic_claim_type": (
                result.normalized_claim_type
            ),

            "semantic_claim_confidence": (
                float(
                    result.claim_confidence
                )
            ),

            "semantic_similarity_to_previous": (
                float(
                    result.similarity_to_previous
                )
            ),

            "semantic_paraphrase_score": (
                float(
                    result.paraphrase_score
                )
            ),

            "semantic_contradiction_score": (
                float(
                    result.contradiction_score
                )
            ),

            "semantic_claim_switch_score": (
                float(
                    result.semantic_claim_switch_score
                )
            ),

            "semantic_evidence_consistency": (
                float(
                    result.evidence_consistency_score
                )
            ),

            "semantic_embedding_dimension": (
                int(
                    result.embedding_dimension
                )
            ),
        }

    # ========================================================
    # MERGE WITH EXISTING OBSERVATION
    # ========================================================

    def enrich_observation(
        self,
        *,
        observation: dict[str, Any],
        complaint_text: str,
        previous_complaints: Sequence[str] | None = None,
        evidence_texts: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """
        Preserve the existing observation and add semantic
        intelligence fields.

        Existing fields are never overwritten.
        """

        semantic_observation = (
            self.build_observation(
                complaint_text=complaint_text,
                previous_complaints=(
                    previous_complaints or []
                ),
                evidence_texts=(
                    evidence_texts or []
                ),
            )
        )

        enriched = dict(
            observation
        )

        for key, value in (
            semantic_observation.items()
        ):

            if key in enriched:
                raise ValueError(
                    f"Semantic observation field "
                    f"already exists: {key}"
                )

            enriched[key] = value

        return enriched

    # ========================================================
    # CONVERT TO NUMERIC ML FEATURES
    # ========================================================

    @staticmethod
    def numeric_features(
        observation: dict[str, Any],
    ) -> dict[str, float]:
        """
        Extract only numeric semantic features.

        This is intentionally separated from the semantic
        claim label so that future ML models can consume
        these values directly.
        """

        numeric_keys = (
            "semantic_claim_confidence",
            "semantic_similarity_to_previous",
            "semantic_paraphrase_score",
            "semantic_contradiction_score",
            "semantic_claim_switch_score",
            "semantic_evidence_consistency",
        )

        return {
            key: float(
                observation[key]
            )
            for key in numeric_keys
            if key in observation
        }