from __future__ import annotations
import numpy as np
import pytest

from src.intelligence.semantic_complaint import (
    SemanticComplaintIntelligence,
)


@pytest.fixture(scope="module")
def intelligence():

    return SemanticComplaintIntelligence()


# ============================================================
# INITIALIZATION
# ============================================================

def test_semantic_intelligence_initializes():

    intelligence = (
        SemanticComplaintIntelligence()
    )

    assert (
        intelligence.model_name
        == "all-MiniLM-L6-v2"
    )


# ============================================================
# EMBEDDINGS
# ============================================================

def test_embedding_is_generated(
    intelligence,
):

    embedding = intelligence.embed(
        "The package arrived with one item missing."
    )

    assert embedding.ndim == 1

    assert embedding.shape[0] > 0


def test_embedding_is_normalized(
    intelligence,
):

    embedding = intelligence.embed(
        "The customer did not receive the order."
    )

    norm = float(
        np.linalg.norm(embedding)
    )

    assert norm == pytest.approx(
        1.0,
        abs=0.02,
    )


# ============================================================
# SIMILARITY
# ============================================================

def test_similar_complaints_have_high_similarity(
    intelligence,
):

    score = intelligence.similarity(

        "The item was broken when it arrived.",

        "The product arrived damaged and unusable.",
    )

    assert score > 0.60


def test_similarity_is_bounded(
    intelligence,
):

    score = intelligence.similarity(

        "The order never arrived.",

        "The order never arrived.",
    )

    assert (
        0.0
        <= score
        <= 1.0
    )


# ============================================================
# CLAIM TYPE
# ============================================================

@pytest.mark.parametrize(
    "text,expected",
    [

        (
            "One of the products was missing from my package.",
            "SHORTAGE_CLAIM",
        ),

        (
            "I received a completely different product.",
            "WRONG_ITEM_CLAIM",
        ),

        (
            "My order never arrived.",
            "NON_DELIVERY_CLAIM",
        ),

        (
            "The item I returned was substituted.",
            "SUBSTITUTED_RETURN_CLAIM",
        ),

    ],
)
def test_claim_type_inference(
    intelligence,
    text,
    expected,
):

    claim_type, confidence = (
        intelligence.infer_claim_type(
            text
        )
    )

    assert claim_type == expected

    assert (
        0.0
        <= confidence
        <= 1.0
    )


# ============================================================
# HISTORY
# ============================================================

def test_similarity_to_previous_complaints(
    intelligence,
):

    score = (
        intelligence.similarity_to_previous(

            "The product was broken when delivered.",

            [
                "The item arrived damaged.",
                "My package had one item missing.",
            ],
        )
    )

    assert (
        0.0
        <= score
        <= 1.0
    )

    assert score > 0.50


# ============================================================
# PARAPHRASE
# ============================================================

def test_paraphrase_score_is_bounded(
    intelligence,
):

    score = (
        intelligence.paraphrase_score(

            "The product arrived broken.",

            [
                "The item was damaged when it arrived.",
            ],
        )
    )

    assert (
        0.0
        <= score
        <= 1.0
    )


# ============================================================
# CLAIM SWITCH
# ============================================================

def test_semantic_claim_switch_detected(
    intelligence,
):

    score = (
        intelligence.semantic_claim_switch_score(

            "The order never arrived.",

            "One item was missing from the package.",
        )
    )

    assert (
        0.0
        <= score
        <= 1.0
    )

    assert score > 0.0


def test_same_claim_has_low_switch(
    intelligence,
):

    score = (
        intelligence.semantic_claim_switch_score(

            "One product was missing from my package.",

            "An item was missing from the shipment.",
        )
    )

    assert score < 0.30


# ============================================================
# CONTRADICTION
# ============================================================

def test_contradiction_is_bounded(
    intelligence,
):

    score = (
        intelligence.contradiction_score(

            "The order was never delivered.",

            "One item was missing from the package.",
        )
    )

    assert (
        0.0
        <= score
        <= 1.0
    )


# ============================================================
# EVIDENCE
# ============================================================

def test_evidence_consistency_is_bounded(
    intelligence,
):

    score = (
        intelligence.evidence_consistency(

            "The package arrived with a missing item.",

            [
                "Photo showing the opened package "
                "with one product missing.",
            ],
        )
    )

    assert (
        0.0
        <= score
        <= 1.0
    )


# ============================================================
# FULL ANALYSIS
# ============================================================

def test_full_analysis_returns_result(
    intelligence,
):

    result = intelligence.analyze(

        complaint_text=(
            "The product arrived damaged "
            "and could not be used."
        ),

        previous_complaints=[
            "The item was broken when delivered.",
        ],

        evidence_texts=[
            "Photo shows visible damage to the product.",
        ],
    )

    assert result.complaint_text != ""

    assert result.normalized_claim_type in (
        "SHORTAGE_CLAIM",
        "WRONG_ITEM_CLAIM",
        "NON_DELIVERY_CLAIM",
        "SUBSTITUTED_RETURN_CLAIM",
    )

    assert result.embedding_dimension > 0

    assert (
        0.0
        <= result.claim_confidence
        <= 1.0
    )

    assert (
        0.0
        <= result.similarity_to_previous
        <= 1.0
    )

    assert (
        0.0
        <= result.paraphrase_score
        <= 1.0
    )

    assert (
        0.0
        <= result.contradiction_score
        <= 1.0
    )

    assert (
        0.0
        <= result.semantic_claim_switch_score
        <= 1.0
    )

    assert (
        0.0
        <= result.evidence_consistency_score
        <= 1.0
    )


# ============================================================
# NO HISTORY
# ============================================================

def test_no_history_produces_zero_history_signals(
    intelligence,
):

    result = intelligence.analyze(

        complaint_text=(
            "The package was damaged."
        )
    )

    assert (
        result.similarity_to_previous
        == 0.0
    )

    assert (
        result.paraphrase_score
        == 0.0
    )

    assert (
        result.semantic_claim_switch_score
        == 0.0
    )

    assert (
        result.contradiction_score
        == 0.0
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def test_empty_complaint_is_rejected(
    intelligence,
):

    with pytest.raises(
        ValueError
    ):

        intelligence.embed(
            ""
        )


def test_non_string_complaint_is_rejected(
    intelligence,
):

    with pytest.raises(
        TypeError
    ):

        intelligence.embed(
            123
        )


# ============================================================
# BOUNDS
# ============================================================

def test_all_analysis_scores_are_bounded(
    intelligence,
):

    result = intelligence.analyze(

        complaint_text=(
            "Some items were missing."
        ),

        previous_complaints=[
            "Items were missing from my package."
        ],

        evidence_texts=[
            "The package photo shows missing items."
        ],
    )

    scores = [

        result.claim_confidence,

        result.similarity_to_previous,

        result.paraphrase_score,

        result.contradiction_score,

        result.semantic_claim_switch_score,

        result.evidence_consistency_score,

    ]

    for score in scores:

        assert (
            0.0
            <= score
            <= 1.0
        )