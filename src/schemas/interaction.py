from datetime import datetime

from pydantic import BaseModel, Field


class SupportAgentState(BaseModel):
    """
    State/configuration of a support agent.
    """

    agent_id: str

    policy_version: str

    customer_satisfaction_weight: float
    merchant_loss_weight: float
    review_cost_weight: float


class SupportDecision(BaseModel):
    """
    Decision made by a support agent for a refund request.
    """

    decision_id: str

    agent_id: str

    refund_request_id: str

    decision: str

    proposed_amount: float

    reason_code: str

    requested_evidence: list[str] = Field(
        default_factory=list
    )

    timestamp: datetime


class InteractionEvent(BaseModel):
    """
    Observable event recorded in the simulation.

    This schema intentionally has no dependency on
    EnvironmentState. The environment imports this schema,
    so importing EnvironmentState here would create a
    circular dependency.
    """

    event_id: str

    episode_id: str

    timestamp: datetime

    sequence_number: int

    actor: str

    event_type: str

    visibility: str

    payload: dict