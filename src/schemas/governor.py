from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .customer import CustomerObservableState
from .order import OrderState
from .refund import RefundRequest, RefundState, Evidence
from .interaction import SupportDecision
from .graph import GraphSnapshot


# ============================================================
# 1. WHAT THE GOVERNOR IS ALLOWED TO SEE
# ============================================================

class GovernorObservation(BaseModel):
    """
    The complete information boundary exposed to the Governor.

    IMPORTANT:
    This object must NEVER contain ground-truth labels,
    counterparty type, private customer-agent state,
    or the Governor's own previous decisions.
    """

    timestamp: datetime

    merchant_id: str

    # Current customer information visible to merchant
    customer: CustomerObservableState

    # Current transaction/order
    current_order: OrderState

    # Current refund request
    current_refund_request: RefundRequest

    # Historical refunds visible to merchant
    refund_history: list[RefundState] = Field(default_factory=list)

    # Previous interaction events visible to merchant
    interaction_history: list[dict] = Field(default_factory=list)

    # Current support-agent decision
    support_decision: SupportDecision

    # Evidence attached to current request
    evidence: list[Evidence] = Field(default_factory=list)

    # Relationship information visible to merchant
    relationship_graph: GraphSnapshot

    # Financial exposure
    current_exposure: float = Field(ge=0)

    historical_refund_exposure: float = Field(ge=0)


# ============================================================
# 2. GOVERNOR SIGNAL
# ============================================================

class GovernorSignal(BaseModel):
    """
    One piece of evidence used by the Governor.
    """

    signal_name: str

    value: str

    severity: Literal[
        "LOW",
        "MEDIUM",
        "HIGH",
    ]

    direction: Literal[
        "POSITIVE_RISK",
        "NEGATIVE_RISK",
        "NEUTRAL",
    ]

    explanation: str


# ============================================================
# 3. CONFLICTING EVIDENCE
# ============================================================

class EvidenceConflict(BaseModel):
    """
    Represents situations where different signals point
    in different directions.

    Example:

    Current evidence looks legitimate,
    but behavioral trajectory looks suspicious.
    """

    signal_a: str

    signal_b: str

    conflict_description: str

    resolution: str


# ============================================================
# 4. EXPECTED-LOSS ESTIMATE
# ============================================================

class ExpectedLoss(BaseModel):
    """
    Estimated financial consequences of each possible action.
    """

    allow: float = Field(ge=0)

    review: float = Field(ge=0)

    block: float = Field(ge=0)


# ============================================================
# 5. GOVERNOR DECISION
# ============================================================

class GovernorDecision(BaseModel):
    """
    Final action taken by the Governor.
    """

    decision_id: str

    episode_id: str

    refund_request_id: str

    decision: Literal[
        "ALLOW",
        "REVIEW",
        "BLOCK",
    ]

    risk_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    expected_loss: ExpectedLoss

    reason_codes: list[str] = Field(
        default_factory=list
    )

    timestamp: datetime


# ============================================================
# 6. GOVERNOR REASONING TRACE
# ============================================================

class GovernorTrace(BaseModel):
    """
    Human-readable audit trail explaining how the Governor
    reached its decision.

    This is important for the hackathon demo because judges
    should be able to see the Governor weigh conflicting
    evidence instead of simply displaying a probability.
    """

    decision_id: str

    observations: list[GovernorSignal] = Field(
        default_factory=list
    )

    conflicts: list[EvidenceConflict] = Field(
        default_factory=list
    )

    decision_summary: str

    key_factors: list[str] = Field(
        default_factory=list
    )


# ============================================================
# 7. COMPLETE GOVERNOR OUTPUT
# ============================================================

class GovernorResult(BaseModel):
    """
    Complete result returned by the Governor after evaluating
    a refund request.
    """

    decision: GovernorDecision

    trace: GovernorTrace