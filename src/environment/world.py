from pydantic import BaseModel, Field

from ..schemas.customer import (
    CustomerObservableState,
    CustomerPrivateState,
    CustomerGroundTruth,
)
from ..schemas.merchant import MerchantState
from ..schemas.order import OrderState
from ..schemas.refund import RefundState
from ..schemas.interaction import InteractionEvent
from ..schemas.graph import GraphNode, GraphEdge


class EnvironmentState(BaseModel):
    """
    Complete internal state of the simulation.

    IMPORTANT:
    This object is the simulator's private world.
    Individual agents must never receive this entire object.
    """

    merchant: MerchantState

    customers: dict[str, CustomerObservableState] = Field(
        default_factory=dict
    )

    customer_private: dict[str, CustomerPrivateState] = Field(
        default_factory=dict
    )

    ground_truth: dict[str, CustomerGroundTruth] = Field(
        default_factory=dict
    )

    orders: dict[str, OrderState] = Field(
        default_factory=dict
    )

    refunds: dict[str, RefundState] = Field(
        default_factory=dict
    )

    events: list[InteractionEvent] = Field(
        default_factory=list
    )

    graph_nodes: list[GraphNode] = Field(
        default_factory=list
    )

    graph_edges: list[GraphEdge] = Field(
        default_factory=list
    )