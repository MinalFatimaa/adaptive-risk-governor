from datetime import datetime

from src.agents.support_agent import SupportAgent
from src.environment.world_generator import create_world
from src.schemas.support import RefundRequest


def test_support_agent_exposes_learnable_policy_signal():

    world = create_world(seed=42)

    customer = next(iter(world.customers.values()))

    # Find an order worth enough to test both sides of
    # the ₹2,000 evidence boundary.
    order = next(
        order
        for order in world.orders.values()
        if order.customer_id == customer.customer_id
        and order.order_amount >= 4000
    )

    agent = SupportAgent(world.merchant)

    # --------------------------------------------------
    # Attempt 1: ₹4,000
    # --------------------------------------------------

    request_1 = RefundRequest(
        request_id="TEST_REQUEST_001",
        customer_id=customer.customer_id,
        order_id=order.order_id,
        claim_type="SHORTAGE_CLAIM",
        claim_text="Some items were missing from my delivery.",
        requested_amount=4000,
        submitted_at=datetime.now(),
        evidence_available=["delivery_photo"],
    )

    decision_1 = agent.decide(
        customer=customer,
        order=order,
        request=request_1,
    )

    # ₹4,000 is above the evidence threshold
    # but below the human-review threshold.
    assert decision_1.decision == "APPROVE"

    # --------------------------------------------------
    # Attempt 2: ₹1,800
    # --------------------------------------------------

    request_2 = RefundRequest(
        request_id="TEST_REQUEST_002",
        customer_id=customer.customer_id,
        order_id=order.order_id,
        claim_type="SHORTAGE_CLAIM",
        claim_text="Some items were missing from my delivery.",
        requested_amount=1800,
        submitted_at=datetime.now(),
        evidence_available=["delivery_photo"],
    )

    decision_2 = agent.decide(
        customer=customer,
        order=order,
        request=request_2,
    )

    assert decision_2.decision == "APPROVE"

    # --------------------------------------------------
    # The customer can observe that both requests
    # were approved, but the policy still exposes
    # different behavior when evidence is absent.
    # --------------------------------------------------

    request_3 = RefundRequest(
        request_id="TEST_REQUEST_003",
        customer_id=customer.customer_id,
        order_id=order.order_id,
        claim_type="SHORTAGE_CLAIM",
        claim_text="Some items were missing from my delivery.",
        requested_amount=1800,
        submitted_at=datetime.now(),
        evidence_available=[],
    )

    decision_3 = agent.decide(
        customer=customer,
        order=order,
        request=request_3,
    )

    assert decision_3.decision == "REQUEST_EVIDENCE"

    assert "delivery_photo" in decision_3.requested_evidence