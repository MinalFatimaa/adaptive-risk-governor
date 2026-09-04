from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ============================================================
# CANONICAL SCHEMA VERSION
# ============================================================

SCHEMA_VERSION = "1.0"


# ============================================================
# COMMON EVENT ENVELOPE
# ============================================================

@dataclass
class EventEnvelope:
    """
    Common metadata shared by every Governor observation event.

    This is the identity/lineage layer.

    Raw identifiers such as IP/device/payment identifiers should
    be represented using safe hashed/pseudonymous values.
    """

    event_id: str

    event_type: str

    timestamp: str

    customer_id: str

    order_id: str | None = None

    request_id: str | None = None

    source: str = "UNKNOWN"

    schema_version: str = SCHEMA_VERSION

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# CLAIM EVENT
# ============================================================

@dataclass
class ClaimEvent:
    """
    Customer's refund/complaint request.

    The original claim text is deliberately preserved.

    Semantic interpretation will be added later without
    replacing the original text.
    """

    envelope: EventEnvelope

    claim_type: str

    claim_text: str

    requested_amount: float

    evidence_available: list[str] = field(
        default_factory=list
    )

    submitted_at: str | None = None


# ============================================================
# TRANSACTION EVENT
# ============================================================

@dataclass
class TransactionEvent:
    """
    Commercial/order context associated with the claim.
    """

    envelope: EventEnvelope

    order_value: float = 0.0

    item_count: int = 0

    payment_method: str | None = None

    order_age_hours: float = 0.0

    previous_refund_count: int = 0

    previous_refund_amount: float = 0.0

    transaction_timestamp: str | None = None


# ============================================================
# AGENT A DECISION EVENT
# ============================================================

@dataclass
class AgentDecisionEvent:
    """
    First-class representation of Agent A's behaviour.

    This is critical to the architecture.

    The Governor does not only observe the customer's behaviour.
    It also observes how Agent A responded to that behaviour.
    """

    envelope: EventEnvelope

    decision: str

    reason_code: str | None = None

    approved_amount: float = 0.0

    evidence_requested: list[str] = field(
        default_factory=list
    )

    evidence_used: list[str] = field(
        default_factory=list
    )

    decision_latency_seconds: float = 0.0

    confidence: float | None = None


# ============================================================
# NETWORK EVENT
# ============================================================

@dataclass
class NetworkEvent:
    """
    External observability/network information.

    Identifiers should be hashed/pseudonymous rather than storing
    raw sensitive identifiers in the Governor feature layer.
    """

    envelope: EventEnvelope

    ip_hash: str | None = None

    device_hash: str | None = None

    payment_hash: str | None = None

    address_hash: str | None = None

    linked_customer_count: int = 0

    linked_order_count: int = 0

    ip_reuse_score: float = 0.0

    device_reuse_score: float = 0.0

    payment_reuse_score: float = 0.0

    address_reuse_score: float = 0.0

    network_abnormality_score: float = 0.0


# ============================================================
# TEMPORAL EVENT
# ============================================================

@dataclass
class TemporalEvent:
    """
    Behavioural patterns over time.
    """

    envelope: EventEnvelope

    requests_last_1h: int = 0

    requests_last_24h: int = 0

    refund_requests_last_24h: int = 0

    request_burst_score: float = 0.0

    claim_switch_rate: float = 0.0

    amount_acceleration: float = 0.0

    temporal_abnormality_score: float = 0.0


# ============================================================
# SEMANTIC EVENT
# ============================================================

@dataclass
class SemanticEvent:
    """
    Semantic interpretation of customer complaint text.

    This schema intentionally supports the future LLM/embedding
    implementation without coupling the rest of the Governor
    to a particular model provider.
    """

    envelope: EventEnvelope

    normalized_intent: str | None = None

    semantic_claim_type: str | None = None

    similarity_to_previous_claims: float = 0.0

    similarity_to_linked_customers: float = 0.0

    contradiction_score: float = 0.0

    paraphrase_score: float = 0.0

    semantic_abnormality_score: float = 0.0

    model_name: str | None = None


# ============================================================
# STRATEGIC EVENT
# ============================================================

@dataclass
class StrategicEvent:
    """
    Strategic adaptation observed from the customer/Agent B.

    These values represent what the strategic customer has
    learned about Agent A's behaviour.
    """

    envelope: EventEnvelope

    evidence_sensitivity: float = 0.5

    amount_sensitivity: float = 0.5

    high_value_escalation: float = 0.5

    followup_sensitivity: float = 0.5

    prediction_accuracy: float = 0.0

    observed_interactions: int = 0

    strategic_adaptation_score: float = 0.0


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_probability(
    value: float,
    field_name: str,
) -> None:
    """
    Validate a value expected to be in [0, 1].
    """

    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1."
        )


def validate_non_negative(
    value: float,
    field_name: str,
) -> None:
    """
    Validate a value that cannot be negative.
    """

    if float(value) < 0.0:
        raise ValueError(
            f"{field_name} cannot be negative."
        )


# ============================================================
# EVENT VALIDATION
# ============================================================

def validate_event(
    event: Any,
) -> None:
    """
    Lightweight runtime validation.

    We intentionally use explicit validation instead of
    introducing a new dependency such as Pydantic at this stage.
    """

    if not hasattr(event, "envelope"):
        raise ValueError(
            "Canonical event must contain an envelope."
        )

    envelope = event.envelope

    if not envelope.event_id:
        raise ValueError(
            "event_id cannot be empty."
        )

    if not envelope.event_type:
        raise ValueError(
            "event_type cannot be empty."
        )

    if not envelope.customer_id:
        raise ValueError(
            "customer_id cannot be empty."
        )

    if envelope.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version: "
            f"{envelope.schema_version}"
        )

    # --------------------------------------------------------
    # Claim
    # --------------------------------------------------------

    if isinstance(event, ClaimEvent):

        validate_non_negative(
            event.requested_amount,
            "requested_amount",
        )

        if not event.claim_type:
            raise ValueError(
                "claim_type cannot be empty."
            )

        if not event.claim_text:
            raise ValueError(
                "claim_text cannot be empty."
            )

    # --------------------------------------------------------
    # Transaction
    # --------------------------------------------------------

    elif isinstance(event, TransactionEvent):

        validate_non_negative(
            event.order_value,
            "order_value",
        )

        validate_non_negative(
            event.previous_refund_amount,
            "previous_refund_amount",
        )

        if event.item_count < 0:
            raise ValueError(
                "item_count cannot be negative."
            )

        if event.order_age_hours < 0:
            raise ValueError(
                "order_age_hours cannot be negative."
            )

        if event.previous_refund_count < 0:
            raise ValueError(
                "previous_refund_count cannot be negative."
            )

    # --------------------------------------------------------
    # Agent A
    # --------------------------------------------------------

    elif isinstance(event, AgentDecisionEvent):

        if not event.decision:
            raise ValueError(
                "Agent decision cannot be empty."
            )

        validate_non_negative(
            event.approved_amount,
            "approved_amount",
        )

        validate_non_negative(
            event.decision_latency_seconds,
            "decision_latency_seconds",
        )

        if event.confidence is not None:

            validate_probability(
                event.confidence,
                "confidence",
            )

    # --------------------------------------------------------
    # Network
    # --------------------------------------------------------

    elif isinstance(event, NetworkEvent):

        for field_name in (
            "ip_reuse_score",
            "device_reuse_score",
            "payment_reuse_score",
            "address_reuse_score",
            "network_abnormality_score",
        ):

            validate_probability(
                getattr(event, field_name),
                field_name,
            )

        if event.linked_customer_count < 0:
            raise ValueError(
                "linked_customer_count cannot be negative."
            )

        if event.linked_order_count < 0:
            raise ValueError(
                "linked_order_count cannot be negative."
            )

    # --------------------------------------------------------
    # Temporal
    # --------------------------------------------------------

    elif isinstance(event, TemporalEvent):

        for field_name in (
            "request_burst_score",
            "claim_switch_rate",
            "amount_acceleration",
            "temporal_abnormality_score",
        ):

            validate_probability(
                getattr(event, field_name),
                field_name,
            )

        for field_name in (
            "requests_last_1h",
            "requests_last_24h",
            "refund_requests_last_24h",
        ):

            if getattr(event, field_name) < 0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )

    # --------------------------------------------------------
    # Semantic
    # --------------------------------------------------------

    elif isinstance(event, SemanticEvent):

        for field_name in (
            "similarity_to_previous_claims",
            "similarity_to_linked_customers",
            "contradiction_score",
            "paraphrase_score",
            "semantic_abnormality_score",
        ):

            validate_probability(
                getattr(event, field_name),
                field_name,
            )

    # --------------------------------------------------------
    # Strategic
    # --------------------------------------------------------

    elif isinstance(event, StrategicEvent):

        for field_name in (
            "evidence_sensitivity",
            "amount_sensitivity",
            "high_value_escalation",
            "followup_sensitivity",
            "prediction_accuracy",
            "strategic_adaptation_score",
        ):

            validate_probability(
                getattr(event, field_name),
                field_name,
            )

        if event.observed_interactions < 0:
            raise ValueError(
                "observed_interactions cannot be negative."
            )