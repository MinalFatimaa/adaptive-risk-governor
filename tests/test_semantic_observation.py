from __future__ import annotations

import pytest

from src.intelligence.semantic_observation import (
    SemanticObservationAdapter,
)


@pytest.fixture(scope="module")
def adapter():

    return SemanticObservationAdapter()


# ============================================================
# BASIC OBSERVATION
# ============================================================

def test_semantic_observation_is_created(
    adapter,
):

    observation = (
        adapter.build_observation(

            complaint_text=(
                "The product arrived damaged."
            )
        )
    )

    assert isinstance(
        observation,
        dict,
    )

    assert (
        "semantic_claim_type"
        in observation
    )

    assert (
        "semantic_claim_confidence"
        in observation
    )

    assert (
        "semantic_similarity_to_previous"
        in observation
    )

    assert (
        "semantic_paraphrase_score"
        in observation
    )

    assert (
        "semantic_contradiction_score"
        in observation
    )

    assert (
        "semantic_claim_switch_score"
        in observation
    )

    assert (
        "semantic_evidence_consistency"
        in observation
    )


# ============================================================
# SEMANTIC FEATURES ARE BOUNDED
# ============================================================

def test_semantic_features_are_bounded(
    adapter,
):

    observation = (
        adapter.build_observation(

            complaint_text=(
                "The item was broken when delivered."
            ),

            previous_complaints=[
                "The product arrived damaged."
            ],

            evidence_texts=[
                "Photo shows visible damage."
            ],
        )
    )

    numeric = (
        adapter.numeric_features(
            observation
        )
    )

    assert numeric

    for value in numeric.values():

        assert (
            0.0
            <= value
            <= 1.0
        )


# ============================================================
# PARAPHRASE PRESERVATION
# ============================================================

def test_semantic_paraphrase_signal_survives_adapter(
    adapter,
):

    observation = (
        adapter.build_observation(

            complaint_text=(
                "The product was broken when "
                "it arrived."
            ),

            previous_complaints=[
                "The item arrived damaged."
            ],
        )
    )

    assert (
        observation[
            "semantic_similarity_to_previous"
        ]
        > 0.50
    )


# ============================================================
# CLAIM SWITCH
# ============================================================

def test_semantic_claim_switch_survives_adapter(
    adapter,
):

    observation = (
        adapter.build_observation(

            complaint_text=(
                "My order never arrived."
            ),

            previous_complaints=[
                "One item was missing "
                "from my package."
            ],
        )
    )

    assert (
        observation[
            "semantic_claim_switch_score"
        ]
        >= 0.0
    )

    assert (
        observation[
            "semantic_claim_switch_score"
        ]
        <= 1.0
    )


# ============================================================
# EVIDENCE CONSISTENCY
# ============================================================

def test_evidence_signal_survives_adapter(
    adapter,
):

    observation = (
        adapter.build_observation(

            complaint_text=(
                "The package arrived damaged."
            ),

            evidence_texts=[
                (
                    "Photo shows damage to "
                    "the delivered package."
                )
            ],
        )
    )

    assert (
        0.0
        <= observation[
            "semantic_evidence_consistency"
        ]
        <= 1.0
    )


# ============================================================
# EXISTING OBSERVATION IS PRESERVED
# ============================================================

def test_existing_observation_is_preserved(
    adapter,
):

    existing = {
        "customer_id": "CUSTOMER_001",
        "requested_amount": 2500.0,
        "network_risk": 0.15,
        "agent_a_decision": "REQUEST_EVIDENCE",
    }

    enriched = (
        adapter.enrich_observation(

            observation=existing,

            complaint_text=(
                "The product arrived damaged."
            ),
        )
    )

    assert (
        enriched["customer_id"]
        == "CUSTOMER_001"
    )

    assert (
        enriched["requested_amount"]
        == 2500.0
    )

    assert (
        enriched["network_risk"]
        == 0.15
    )

    assert (
        enriched["agent_a_decision"]
        == "REQUEST_EVIDENCE"
    )

    assert (
        "semantic_claim_type"
        in enriched
    )


# ============================================================
# NO EXISTING FIELD IS OVERWRITTEN
# ============================================================

def test_existing_semantic_field_is_not_overwritten(
    adapter,
):

    existing = {
        "semantic_claim_type": "PREEXISTING_VALUE"
    }

    with pytest.raises(
        ValueError
    ):

        adapter.enrich_observation(

            observation=existing,

            complaint_text=(
                "The item arrived damaged."
            ),
        )


# ============================================================
# NUMERIC FEATURE EXTRACTION
# ============================================================

def test_numeric_features_exclude_claim_label(
    adapter,
):

    observation = (
        adapter.build_observation(

            complaint_text=(
                "The package never arrived."
            )
        )
    )

    numeric = (
        adapter.numeric_features(
            observation
        )
    )

    assert (
        "semantic_claim_type"
        not in numeric
    )

    assert (
        "semantic_claim_confidence"
        in numeric
    )

    assert (
        "semantic_similarity_to_previous"
        in numeric
    )


# ============================================================
# EMBEDDING DIMENSION
# ============================================================

def test_embedding_dimension_is_recorded(
    adapter,
):

    observation = (
        adapter.build_observation(

            complaint_text=(
                "One item was missing."
            )
        )
    )

    assert (
        observation[
            "semantic_embedding_dimension"
        ]
        > 0
    )


# ============================================================
# CANONICAL OBSERVATION COMPATIBILITY
# ============================================================

def test_semantic_fields_have_stable_names(
    adapter,
):

    observation = (
        adapter.build_observation(

            complaint_text=(
                "I received the wrong product."
            )
        )
    )

    expected_fields = {
        "semantic_claim_type",
        "semantic_claim_confidence",
        "semantic_similarity_to_previous",
        "semantic_paraphrase_score",
        "semantic_contradiction_score",
        "semantic_claim_switch_score",
        "semantic_evidence_consistency",
        "semantic_embedding_dimension",
    }

    assert expected_fields.issubset(
        observation.keys()
    )