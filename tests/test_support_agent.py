from datetime import datetime

from src.agents.support_agent import SupportAgent
from src.environment.world_generator import create_world
from src.schemas.support import RefundRequest


def build_world():
    return create_world(
        seed=42,
    )


def build_request(
    customer_id,
    order_id,
    claim_type="SHORTAGE_CLAIM",
    amount=1000,
    evidence=None,
):
    return RefundRequest(
        request_id="REQUEST_TEST_001",
        customer_id=customer_id,
        order_id=order_id,
        claim_type=claim_type,
        claim_text="I have a problem with my order.",
        requested_amount=amount,
        submitted_at=datetime.now(),
        evidence_available=evidence or [],
    )


def test_support_agent_can_approve_valid_claim():

    world = build_world()

    customer = next(
        iter(world.customers.values())
    )

    order_id = customer.current_order_ids[0]

    order = world.orders[order_id]

    agent = SupportAgent(
        merchant=world.merchant
    )

    request = build_request(
        customer_id=customer.customer_id,
        order_id=order.order_id,
        claim_type="SHORTAGE_CLAIM",
        amount=min(1000, order.order_amount),
        evidence=["delivery_photo"],
    )

    decision = agent.decide(
        customer=customer,
        order=order,
        request=request,
    )

    assert decision.decision == "APPROVE"

    assert decision.approved_amount == request.requested_amount


def test_support_agent_requests_evidence():

    world = build_world()

    customer = next(
        iter(world.customers.values())
    )

    order_id = customer.current_order_ids[0]

    order = world.orders[order_id]

    agent = SupportAgent(
        merchant=world.merchant
    )

    request = build_request(
        customer_id=customer.customer_id,
        order_id=order.order_id,
        claim_type="SHORTAGE_CLAIM",
        amount=1000,
    )

    decision = agent.decide(
        customer=customer,
        order=order,
        request=request,
    )

    assert decision.decision == "REQUEST_EVIDENCE"

    assert "delivery_photo" in decision.requested_evidence


def test_support_agent_rejects_excessive_amount():

    world = build_world()

    customer = next(
        iter(world.customers.values())
    )

    order_id = customer.current_order_ids[0]

    order = world.orders[order_id]

    agent = SupportAgent(
        merchant=world.merchant
    )

    request = build_request(
        customer_id=customer.customer_id,
        order_id=order.order_id,
        claim_type="SHORTAGE_CLAIM",
        amount=order.order_amount + 1,
        evidence=["delivery_photo"],
    )

    decision = agent.decide(
        customer=customer,
        order=order,
        request=request,
    )

    assert decision.decision == "DENY"


def test_support_agent_escalates_high_value_case():

    world = build_world()

    customer = next(
        iter(world.customers.values())
    )

    order_id = customer.current_order_ids[0]

    order = world.orders[order_id]

    agent = SupportAgent(
        merchant=world.merchant
    )

    amount = min(
        order.order_amount,
        6000,
    )

    request = build_request(
        customer_id=customer.customer_id,
        order_id=order.order_id,
        claim_type="SHORTAGE_CLAIM",
        amount=amount,
        evidence=["delivery_photo"],
    )

    decision = agent.decide(
        customer=customer,
        order=order,
        request=request,
    )

    if amount > 5000:
        assert decision.decision == "ESCALATE"