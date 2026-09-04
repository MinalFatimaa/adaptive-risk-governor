from src.agents.adaptive_customer import (
    AdaptiveCustomerAgent,
    AdaptiveCustomerState,
)


class MockDecision:

    def __init__(
        self,
        decision,
        reason_code="",
        requested_evidence=None,
        approved_amount=0.0,
        requires_followup=False,
    ):

        self.decision = decision
        self.reason_code = reason_code
        self.requested_evidence = (
            requested_evidence or []
        )
        self.approved_amount = approved_amount
        self.requires_followup = (
            requires_followup
        )


def build_agent():

    state = AdaptiveCustomerState(
        customer_id="TEST_CUSTOMER",
        objective="maximize_illegitimate_refund_value",
    )

    return AdaptiveCustomerAgent(
        state=state,
        seed=42,
    )


def test_evidence_request_increases_evidence_sensitivity():

    agent = build_agent()

    before = agent.get_inferred_policy()[
        "evidence_sensitivity"
    ]

    decision = MockDecision(
        decision="REQUEST_EVIDENCE",
        reason_code="INSUFFICIENT_EVIDENCE",
        requested_evidence=[
            "delivery_photo"
        ],
    )

    agent.infer_support_policy(
        decision=decision,
        claim_type="SHORTAGE_CLAIM",
        requested_amount=1500,
        evidence_available=[],
    )

    after = agent.get_inferred_policy()[
        "evidence_sensitivity"
    ]

    assert after > before


def test_high_value_escalation_is_learned():

    agent = build_agent()

    before = agent.get_inferred_policy()[
        "high_value_escalation"
    ]

    decision = MockDecision(
        decision="ESCALATE",
        reason_code="HIGH_VALUE",
    )

    agent.infer_support_policy(
        decision=decision,
        claim_type="SHORTAGE_CLAIM",
        requested_amount=7000,
        evidence_available=[
            "delivery_photo"
        ],
    )

    after = agent.get_inferred_policy()[
        "high_value_escalation"
    ]

    assert after > before


def test_policy_observation_is_conditioned_on_claim():

    agent = build_agent()

    decision = MockDecision(
        decision="REQUEST_EVIDENCE",
        reason_code="INSUFFICIENT_EVIDENCE",
        requested_evidence=[
            "package_photo"
        ],
    )

    agent.infer_support_policy(
        decision=decision,
        claim_type="WRONG_ITEM_CLAIM",
        requested_amount=1200,
        evidence_available=[],
    )

    assert (
        "WRONG_ITEM_CLAIM"
        in agent.state.observations_by_claim
    )

    assert len(
        agent.state.observations_by_claim[
            "WRONG_ITEM_CLAIM"
        ]
    ) == 1


def test_policy_values_remain_bounded():

    agent = build_agent()

    for _ in range(100):

        decision = MockDecision(
            decision="REQUEST_EVIDENCE",
            requested_evidence=[
                "delivery_photo"
            ],
        )

        agent.infer_support_policy(
            decision=decision,
            claim_type="SHORTAGE_CLAIM",
            requested_amount=1000,
            evidence_available=[],
        )

    policy = agent.get_inferred_policy()

    for value in policy.values():

        assert 0.0 <= value <= 1.0