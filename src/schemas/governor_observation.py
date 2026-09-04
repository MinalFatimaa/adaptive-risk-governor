from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .governor_events import (
    SCHEMA_VERSION,
    AgentDecisionEvent,
    ClaimEvent,
    NetworkEvent,
    SemanticEvent,
    StrategicEvent,
    TemporalEvent,
    TransactionEvent,
    validate_event,
)


# ============================================================
# CANONICAL GOVERNOR OBSERVATION
# ============================================================

@dataclass
class GovernorObservation:
    """
    Canonical assembled observation presented to the Governor.

    This is the main contract between the observation layer
    and the risk layer.

    It preserves the distinction between:

        1. What happened.
        2. What was derived from what happened.

    The Governor can therefore evolve from heuristic rules
    to ML without changing the upstream observation contract.
    """

    observation_id: str

    customer_id: str

    timestamp: str

    claim: ClaimEvent | None = None

    transaction: TransactionEvent | None = None

    agent_decision: AgentDecisionEvent | None = None

    network: NetworkEvent | None = None

    temporal: TemporalEvent | None = None

    semantic: SemanticEvent | None = None

    strategic: StrategicEvent | None = None

    schema_version: str = SCHEMA_VERSION

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(self) -> None:
        """
        Validate the complete observation.
        """

        if not self.observation_id:
            raise ValueError(
                "observation_id cannot be empty."
            )

        if not self.customer_id:
            raise ValueError(
                "customer_id cannot be empty."
            )

        if not self.timestamp:
            raise ValueError(
                "timestamp cannot be empty."
            )

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: "
                f"{self.schema_version}"
            )

        events = (
            self.claim,
            self.transaction,
            self.agent_decision,
            self.network,
            self.temporal,
            self.semantic,
            self.strategic,
        )

        present_events = [
            event
            for event in events
            if event is not None
        ]

        for event in present_events:

            validate_event(event)

            if event.envelope.customer_id != self.customer_id:

                raise ValueError(
                    "Event customer_id does not match "
                    "GovernorObservation customer_id."
                )

    # ========================================================
    # EVENT COUNT
    # ========================================================

    def event_count(self) -> int:
        """
        Number of populated event categories.
        """

        return sum(
            event is not None
            for event in (
                self.claim,
                self.transaction,
                self.agent_decision,
                self.network,
                self.temporal,
                self.semantic,
                self.strategic,
            )
        )

    # ========================================================
    # COMPLETENESS
    # ========================================================

    def completeness(self) -> float:
        """
        Fraction of the seven observation categories that
        are currently populated.
        """

        return self.event_count() / 7.0

    # ========================================================
    # FLATTENING
    # ========================================================

    def to_feature_dict(self) -> dict[str, float]:
        """
        Produce a numeric feature representation suitable for
        the future ML risk model.

        Missing event categories are represented with zeroes.

        Raw text and identifiers are deliberately NOT converted
        into numeric features here. The semantic/network
        processing layers own that transformation.
        """

        features: dict[str, float] = {}

        # ----------------------------------------------------
        # Claim
        # ----------------------------------------------------

        if self.claim is not None:

            features["requested_amount"] = float(
                self.claim.requested_amount
            )

            features["evidence_count"] = float(
                len(
                    self.claim.evidence_available
                )
            )

        else:

            features["requested_amount"] = 0.0

            features["evidence_count"] = 0.0

        # ----------------------------------------------------
        # Transaction
        # ----------------------------------------------------

        if self.transaction is not None:

            features["order_value"] = float(
                self.transaction.order_value
            )

            features["item_count"] = float(
                self.transaction.item_count
            )

            features["order_age_hours"] = float(
                self.transaction.order_age_hours
            )

            features["previous_refund_count"] = float(
                self.transaction.previous_refund_count
            )

            features["previous_refund_amount"] = float(
                self.transaction.previous_refund_amount
            )

        else:

            features["order_value"] = 0.0

            features["item_count"] = 0.0

            features["order_age_hours"] = 0.0

            features["previous_refund_count"] = 0.0

            features["previous_refund_amount"] = 0.0

        # ----------------------------------------------------
        # Agent A
        # ----------------------------------------------------

        if self.agent_decision is not None:

            features["agent_approved_amount"] = float(
                self.agent_decision.approved_amount
            )

            features["agent_decision_latency_seconds"] = float(
                self.agent_decision.decision_latency_seconds
            )

            features["agent_evidence_requested_count"] = float(
                len(
                    self.agent_decision.evidence_requested
                )
            )

            features["agent_evidence_used_count"] = float(
                len(
                    self.agent_decision.evidence_used
                )
            )

            features["agent_confidence"] = float(
                self.agent_decision.confidence
                if self.agent_decision.confidence is not None
                else 0.0
            )

        else:

            features["agent_approved_amount"] = 0.0

            features["agent_decision_latency_seconds"] = 0.0

            features["agent_evidence_requested_count"] = 0.0

            features["agent_evidence_used_count"] = 0.0

            features["agent_confidence"] = 0.0

        # ----------------------------------------------------
        # Network
        # ----------------------------------------------------

        if self.network is not None:

            features["linked_customer_count"] = float(
                self.network.linked_customer_count
            )

            features["linked_order_count"] = float(
                self.network.linked_order_count
            )

            features["ip_reuse_score"] = float(
                self.network.ip_reuse_score
            )

            features["device_reuse_score"] = float(
                self.network.device_reuse_score
            )

            features["payment_reuse_score"] = float(
                self.network.payment_reuse_score
            )

            features["address_reuse_score"] = float(
                self.network.address_reuse_score
            )

            features["network_abnormality_score"] = float(
                self.network.network_abnormality_score
            )

        else:

            for key in (
                "linked_customer_count",
                "linked_order_count",
                "ip_reuse_score",
                "device_reuse_score",
                "payment_reuse_score",
                "address_reuse_score",
                "network_abnormality_score",
            ):

                features[key] = 0.0

        # ----------------------------------------------------
        # Temporal
        # ----------------------------------------------------

        if self.temporal is not None:

            features["requests_last_1h"] = float(
                self.temporal.requests_last_1h
            )

            features["requests_last_24h"] = float(
                self.temporal.requests_last_24h
            )

            features["refund_requests_last_24h"] = float(
                self.temporal.refund_requests_last_24h
            )

            features["request_burst_score"] = float(
                self.temporal.request_burst_score
            )

            features["claim_switch_rate"] = float(
                self.temporal.claim_switch_rate
            )

            features["amount_acceleration"] = float(
                self.temporal.amount_acceleration
            )

            features["temporal_abnormality_score"] = float(
                self.temporal.temporal_abnormality_score
            )

        else:

            for key in (
                "requests_last_1h",
                "requests_last_24h",
                "refund_requests_last_24h",
                "request_burst_score",
                "claim_switch_rate",
                "amount_acceleration",
                "temporal_abnormality_score",
            ):

                features[key] = 0.0

        # ----------------------------------------------------
        # Semantic
        # ----------------------------------------------------

        if self.semantic is not None:

            features["similarity_to_previous_claims"] = float(
                self.semantic.similarity_to_previous_claims
            )

            features["similarity_to_linked_customers"] = float(
                self.semantic.similarity_to_linked_customers
            )

            features["contradiction_score"] = float(
                self.semantic.contradiction_score
            )

            features["paraphrase_score"] = float(
                self.semantic.paraphrase_score
            )

            features["semantic_abnormality_score"] = float(
                self.semantic.semantic_abnormality_score
            )

        else:

            for key in (
                "similarity_to_previous_claims",
                "similarity_to_linked_customers",
                "contradiction_score",
                "paraphrase_score",
                "semantic_abnormality_score",
            ):

                features[key] = 0.0

        # ----------------------------------------------------
        # Strategic
        # ----------------------------------------------------

        if self.strategic is not None:

            features["evidence_sensitivity"] = float(
                self.strategic.evidence_sensitivity
            )

            features["amount_sensitivity"] = float(
                self.strategic.amount_sensitivity
            )

            features["high_value_escalation"] = float(
                self.strategic.high_value_escalation
            )

            features["followup_sensitivity"] = float(
                self.strategic.followup_sensitivity
            )

            features["prediction_accuracy"] = float(
                self.strategic.prediction_accuracy
            )

            features["observed_interactions"] = float(
                self.strategic.observed_interactions
            )

            features["strategic_adaptation_score"] = float(
                self.strategic.strategic_adaptation_score
            )

        else:

            features["evidence_sensitivity"] = 0.5

            features["amount_sensitivity"] = 0.5

            features["high_value_escalation"] = 0.5

            features["followup_sensitivity"] = 0.5

            features["prediction_accuracy"] = 0.0

            features["observed_interactions"] = 0.0

            features["strategic_adaptation_score"] = 0.0

        return features


# ============================================================
# FACTORY
# ============================================================

def build_governor_observation(
    *,
    observation_id: str,
    customer_id: str,
    timestamp: str,
    claim: ClaimEvent | None = None,
    transaction: TransactionEvent | None = None,
    agent_decision: AgentDecisionEvent | None = None,
    network: NetworkEvent | None = None,
    temporal: TemporalEvent | None = None,
    semantic: SemanticEvent | None = None,
    strategic: StrategicEvent | None = None,
    metadata: dict[str, Any] | None = None,
) -> GovernorObservation:
    """
    Construct and validate a canonical Governor observation.
    """

    observation = GovernorObservation(
        observation_id=observation_id,
        customer_id=customer_id,
        timestamp=timestamp,
        claim=claim,
        transaction=transaction,
        agent_decision=agent_decision,
        network=network,
        temporal=temporal,
        semantic=semantic,
        strategic=strategic,
        metadata=metadata or {},
    )

    observation.validate()

    return observation