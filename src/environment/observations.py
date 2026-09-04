from datetime import datetime

from ..schemas.customer import CustomerObservableState
from ..schemas.refund import RefundRequest
from ..schemas.interaction import SupportDecision
from ..schemas.governor import GovernorObservation
from ..schemas.graph import GraphSnapshot

from .world import EnvironmentState


def build_governor_observation(
    env: EnvironmentState,
    customer_id: str,
    refund_request: RefundRequest,
    support_decision: SupportDecision,
    current_time: datetime,
) -> GovernorObservation:

    customer = env.customers[customer_id]

    current_order = env.orders[refund_request.order_id]

    refund_history = [
        refund
        for refund in env.refunds.values()
        if refund.customer_id == customer_id
    ]

    interaction_history = [
        event.model_dump()
        for event in env.events
        if event.visibility in {
            "MERCHANT",
            "GOVERNOR",
            "PUBLIC",
        }
        and (
            event.payload.get("customer_id") == customer_id
            or event.payload.get("refund_request_id")
            == refund_request.refund_request_id
        )
    ]

    graph = GraphSnapshot(
        nodes=env.graph_nodes,
        edges=env.graph_edges,
        customer_component=[],
        relationship_counts={},
    )

    historical_refund_exposure = sum(
        refund.approved_amount
        for refund in refund_history
    )

    return GovernorObservation(
        timestamp=current_time,
        merchant_id=env.merchant.merchant_id,
        customer=customer,
        current_order=current_order,
        current_refund_request=refund_request,
        refund_history=refund_history,
        interaction_history=interaction_history,
        support_decision=support_decision,
        evidence=[],
        relationship_graph=graph,
        current_exposure=refund_request.requested_amount,
        historical_refund_exposure=historical_refund_exposure,
    )


def build_customer_observation(
    env: EnvironmentState,
    customer_id: str,
    current_time: datetime,
) -> dict:

    customer = env.customers[customer_id]

    private_state = env.customer_private[customer_id]

    # Only information that the customer is legitimately
    # allowed to know.
    return {
        "timestamp": current_time,
        "customer": customer.model_dump(),

        "private_state": private_state.model_dump(),

        "own_orders": [
            order.model_dump()
            for order in env.orders.values()
            if order.customer_id == customer_id
        ],

        "own_refunds": [
            refund.model_dump()
            for refund in env.refunds.values()
            if refund.customer_id == customer_id
        ],

        "support_responses": [
            event.model_dump()
            for event in env.events
            if event.actor == "SUPPORT_AGENT"
            and (
                event.payload.get("customer_id")
                == customer_id
            )
        ],
    }