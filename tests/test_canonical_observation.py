from __future__ import annotations

import pytest

from src.schemas.governor_events import (
    SCHEMA_VERSION,
    AgentDecisionEvent,
    ClaimEvent,
    EventEnvelope,
    NetworkEvent,
    SemanticEvent,
    StrategicEvent,
    TemporalEvent,
    TransactionEvent,
    validate_event,
)

from src.schemas.governor_observation import (
    GovernorObservation,
    build_governor_observation,
)


# ============================================================
# TEST HELPERS
# ============================================================

def envelope(
    event_id: str,
    event_type: str,
    customer_id: str = "CUSTOMER_TEST",
):
    return EventEnvelope(
        event_id=event_id,
        event_type=event_type,
        timestamp="2026-08-26T10:00:00",
        customer_id=customer_id,
        order_id="ORDER_TEST",
        request_id="REQUEST_TEST",
        source="TEST",
    )


# ============================================================
# ENVELOPE TESTS
# ============================================================

def test_schema_version_is_defined():

    assert SCHEMA_VERSION == "1.0"


def test_event_envelope_contains_required_identity():

    event = envelope(
        "EVENT_1",
        "TEST",
    )

    assert event.event_id == "EVENT_1"

    assert event.customer_id == "CUSTOMER_TEST"

    assert event.schema_version == SCHEMA_VERSION


# ============================================================
# CLAIM
# ============================================================

def test_claim_event_preserves_raw_complaint_text():

    claim = ClaimEvent(
        envelope=envelope(
            "CLAIM_1",
            "CLAIM",
        ),
        claim_type="WRONG_ITEM_CLAIM",
        claim_text=(
            "The product I received was not "
            "the product I ordered."
        ),
        requested_amount=4500,
        evidence_available=[
            "package_photo"
        ],
    )

    validate_event(claim)

    assert (
        claim.claim_text
        == "The product I received was not "
           "the product I ordered."
    )

    assert claim.requested_amount == 4500


def test_claim_rejects_negative_amount():

    claim = ClaimEvent(
        envelope=envelope(
            "CLAIM_2",
            "CLAIM",
        ),
        claim_type="SHORTAGE_CLAIM",
        claim_text="Some items were missing.",
        requested_amount=-100,
    )

    with pytest.raises(ValueError):

        validate_event(claim)


# ============================================================
# TRANSACTION
# ============================================================

def test_transaction_event():

    transaction = TransactionEvent(
        envelope=envelope(
            "TX_1",
            "TRANSACTION",
        ),
        order_value=5000,
        item_count=3,
        payment_method="CARD",
        order_age_hours=48,
        previous_refund_count=2,
        previous_refund_amount=1200,
    )

    validate_event(transaction)

    assert transaction.order_value == 5000

    assert transaction.previous_refund_count == 2


# ============================================================
# AGENT A
# ============================================================

def test_agent_decision_is_first_class_event():

    decision = AgentDecisionEvent(
        envelope=envelope(
            "AGENT_1",
            "AGENT_DECISION",
        ),
        decision="REQUEST_EVIDENCE",
        reason_code="MISSING_EVIDENCE",
        approved_amount=0,
        evidence_requested=[
            "package_photo"
        ],
        evidence_used=[],
        decision_latency_seconds=4.2,
        confidence=0.91,
    )

    validate_event(decision)

    assert (
        decision.decision
        == "REQUEST_EVIDENCE"
    )

    assert (
        decision.decision_latency_seconds
        == 4.2
    )


# ============================================================
# NETWORK
# ============================================================

def test_network_event_contains_external_observability():

    network = NetworkEvent(
        envelope=envelope(
            "NETWORK_1",
            "NETWORK",
        ),
        ip_hash="IP_HASH_1",
        device_hash="DEVICE_HASH_1",
        payment_hash="PAYMENT_HASH_1",
        address_hash="ADDRESS_HASH_1",
        linked_customer_count=5,
        linked_order_count=12,
        ip_reuse_score=0.90,
        device_reuse_score=0.85,
        payment_reuse_score=0.80,
        address_reuse_score=0.70,
        network_abnormality_score=0.88,
    )

    validate_event(network)

    assert (
        network.linked_customer_count
        == 5
    )

    assert (
        network.network_abnormality_score
        == 0.88
    )


def test_network_scores_are_bounded():

    network = NetworkEvent(
        envelope=envelope(
            "NETWORK_2",
            "NETWORK",
        ),
        ip_reuse_score=1.5,
    )

    with pytest.raises(ValueError):

        validate_event(network)


# ============================================================
# TEMPORAL
# ============================================================

def test_temporal_event():

    temporal = TemporalEvent(
        envelope=envelope(
            "TEMPORAL_1",
            "TEMPORAL",
        ),
        requests_last_1h=6,
        requests_last_24h=10,
        refund_requests_last_24h=8,
        request_burst_score=1.0,
        claim_switch_rate=0.75,
        amount_acceleration=0.90,
        temporal_abnormality_score=0.95,
    )

    validate_event(temporal)

    assert temporal.requests_last_1h == 6

    assert (
        temporal.temporal_abnormality_score
        == 0.95
    )


# ============================================================
# SEMANTIC
# ============================================================

def test_semantic_event_supports_complaint_intelligence():

    semantic = SemanticEvent(
        envelope=envelope(
            "SEMANTIC_1",
            "SEMANTIC",
        ),
        normalized_intent="DAMAGED_ITEM",
        semantic_claim_type="DAMAGE_CLAIM",
        similarity_to_previous_claims=0.91,
        similarity_to_linked_customers=0.87,
        contradiction_score=0.20,
        paraphrase_score=0.94,
        semantic_abnormality_score=0.82,
        model_name="test-model",
    )

    validate_event(semantic)

    assert (
        semantic.normalized_intent
        == "DAMAGED_ITEM"
    )

    assert (
        semantic.paraphrase_score
        == 0.94
    )


# ============================================================
# STRATEGIC
# ============================================================

def test_strategic_event():

    strategic = StrategicEvent(
        envelope=envelope(
            "STRATEGIC_1",
            "STRATEGIC",
        ),
        evidence_sensitivity=0.95,
        amount_sensitivity=0.80,
        high_value_escalation=0.85,
        followup_sensitivity=0.90,
        prediction_accuracy=0.75,
        observed_interactions=20,
        strategic_adaptation_score=0.91,
    )

    validate_event(strategic)

    assert (
        strategic.strategic_adaptation_score
        == 0.91
    )

    assert (
        strategic.observed_interactions
        == 20
    )


# ============================================================
# COMPLETE OBSERVATION
# ============================================================

def test_complete_governor_observation():

    customer_id = "CUSTOMER_TEST"

    claim = ClaimEvent(
        envelope=envelope(
            "CLAIM_COMPLETE",
            "CLAIM",
            customer_id,
        ),
        claim_type="WRONG_ITEM_CLAIM",
        claim_text="Wrong product received.",
        requested_amount=6000,
        evidence_available=[
            "package_photo"
        ],
    )

    transaction = TransactionEvent(
        envelope=envelope(
            "TX_COMPLETE",
            "TRANSACTION",
            customer_id,
        ),
        order_value=6000,
        item_count=1,
        payment_method="CARD",
        order_age_hours=24,
        previous_refund_count=1,
        previous_refund_amount=1000,
    )

    agent = AgentDecisionEvent(
        envelope=envelope(
            "AGENT_COMPLETE",
            "AGENT_DECISION",
            customer_id,
        ),
        decision="REQUEST_EVIDENCE",
        reason_code="MISSING_EVIDENCE",
        decision_latency_seconds=3.2,
        confidence=0.88,
    )

    network = NetworkEvent(
        envelope=envelope(
            "NETWORK_COMPLETE",
            "NETWORK",
            customer_id,
        ),
        ip_hash="IP_1",
        device_hash="DEVICE_1",
        linked_customer_count=4,
        linked_order_count=7,
        ip_reuse_score=0.80,
        device_reuse_score=0.85,
        network_abnormality_score=0.90,
    )

    temporal = TemporalEvent(
        envelope=envelope(
            "TEMPORAL_COMPLETE",
            "TEMPORAL",
            customer_id,
        ),
        requests_last_1h=5,
        requests_last_24h=8,
        request_burst_score=0.90,
        claim_switch_rate=0.70,
        amount_acceleration=0.80,
        temporal_abnormality_score=0.88,
    )

    semantic = SemanticEvent(
        envelope=envelope(
            "SEMANTIC_COMPLETE",
            "SEMANTIC",
            customer_id,
        ),
        normalized_intent="WRONG_ITEM",
        semantic_claim_type="WRONG_ITEM_CLAIM",
        similarity_to_previous_claims=0.85,
        paraphrase_score=0.90,
        semantic_abnormality_score=0.80,
    )

    strategic = StrategicEvent(
        envelope=envelope(
            "STRATEGIC_COMPLETE",
            "STRATEGIC",
            customer_id,
        ),
        evidence_sensitivity=0.90,
        amount_sensitivity=0.80,
        high_value_escalation=0.85,
        followup_sensitivity=0.88,
        prediction_accuracy=0.75,
        observed_interactions=20,
        strategic_adaptation_score=0.90,
    )

    observation = build_governor_observation(
        observation_id="OBSERVATION_1",
        customer_id=customer_id,
        timestamp="2026-08-26T10:00:00",
        claim=claim,
        transaction=transaction,
        agent_decision=agent,
        network=network,
        temporal=temporal,
        semantic=semantic,
        strategic=strategic,
    )

    assert (
        observation.event_count()
        == 7
    )

    assert (
        observation.completeness()
        == 1.0
    )


# ============================================================
# CUSTOMER ID CONSISTENCY
# ============================================================

def test_observation_rejects_mismatched_customer():

    claim = ClaimEvent(
        envelope=envelope(
            "CLAIM_BAD",
            "CLAIM",
            "CUSTOMER_A",
        ),
        claim_type="SHORTAGE_CLAIM",
        claim_text="Missing item.",
        requested_amount=1000,
    )

    observation = GovernorObservation(
        observation_id="OBS_BAD",
        customer_id="CUSTOMER_B",
        timestamp="2026-08-26T10:00:00",
        claim=claim,
    )

    with pytest.raises(ValueError):

        observation.validate()


# ============================================================
# FEATURE FLATTENING
# ============================================================

def test_observation_can_be_flattened_for_ml():

    network = NetworkEvent(
        envelope=envelope(
            "NETWORK_FEATURE",
            "NETWORK",
        ),
        ip_reuse_score=0.80,
        device_reuse_score=0.70,
        network_abnormality_score=0.90,
    )

    observation = build_governor_observation(
        observation_id="OBS_FEATURE",
        customer_id="CUSTOMER_TEST",
        timestamp="2026-08-26T10:00:00",
        network=network,
    )

    features = (
        observation.to_feature_dict()
    )

    assert (
        features["ip_reuse_score"]
        == 0.80
    )

    assert (
        features["device_reuse_score"]
        == 0.70
    )

    assert (
        features["network_abnormality_score"]
        == 0.90
    )


# ============================================================
# MISSING EVENTS
# ============================================================

def test_partial_observation_is_allowed():

    observation = build_governor_observation(
        observation_id="OBS_PARTIAL",
        customer_id="CUSTOMER_TEST",
        timestamp="2026-08-26T10:00:00",
    )

    assert (
        observation.event_count()
        == 0
    )

    assert (
        observation.completeness()
        == 0.0
    )


# ============================================================
# SCHEMA VERSION
# ============================================================

def test_invalid_schema_version_is_rejected():

    event = envelope(
        "VERSION_BAD",
        "TEST",
    )

    event.schema_version = "999.0"

    with pytest.raises(ValueError):

        validate_event(event)