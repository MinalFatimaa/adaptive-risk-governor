from src.governor.risk_governor import (
    AdaptiveRiskGovernor,
)


def test_governor_low_risk():

    governor = AdaptiveRiskGovernor()

    decision = governor.evaluate(
        customer_history=[],
        current_claim_type="SHORTAGE_CLAIM",
        requested_amount=1000,
        evidence_available=[
            "delivery_photo"
        ],
    )

    assert 0.0 <= decision.risk_score <= 1.0
    assert decision.risk_level == "LOW"
    assert (
        decision.action
        == "ALLOW_AGENT_A_DECISION"
    )


def test_governor_detects_repeated_denials():

    governor = AdaptiveRiskGovernor()

    history = [
        {
            "claim_type": "SHORTAGE_CLAIM",
            "support_decision": "DENY",
        },
        {
            "claim_type": "WRONG_ITEM_CLAIM",
            "support_decision": "DENY",
        },
        {
            "claim_type": "NON_DELIVERY_CLAIM",
            "support_decision": "DENY",
        },
    ]

    decision = governor.evaluate(
        customer_history=history,
        current_claim_type="SHORTAGE_CLAIM",
        requested_amount=1000,
    )

    assert (
        "REPEATED_DENIALS"
        in decision.reason_codes
    )

    assert decision.risk_score > 0.0


def test_governor_detects_claim_switching():

    governor = AdaptiveRiskGovernor()

    history = [
        {
            "claim_type": "SHORTAGE_CLAIM",
            "support_decision": "DENY",
        },
        {
            "claim_type": "WRONG_ITEM_CLAIM",
            "support_decision": "DENY",
        },
        {
            "claim_type": "NON_DELIVERY_CLAIM",
            "support_decision": "DENY",
        },
        {
            "claim_type": "SUBSTITUTED_RETURN_CLAIM",
            "support_decision": "DENY",
        },
        {
            "claim_type": "SHORTAGE_CLAIM",
            "support_decision": "DENY",
        },
    ]

    decision = governor.evaluate(
        customer_history=history,
        current_claim_type="SHORTAGE_CLAIM",
        requested_amount=1000,
    )

    assert (
        "FREQUENT_CLAIM_SWITCHING"
        in decision.reason_codes
    )


def test_governor_detects_strategic_customer():

    governor = AdaptiveRiskGovernor()

    class FakeStrategicState:

        interaction_history = [
            {"support_decision": "REQUEST_EVIDENCE"}
            for _ in range(20)
        ]

        inferred_policy = {
            "evidence_sensitivity": 0.95,
            "amount_sensitivity": 0.80,
            "high_value_escalation": 0.85,
            "followup_sensitivity": 0.90,
        }

    decision = governor.evaluate(
        customer_history=[],
        current_claim_type="WRONG_ITEM_CLAIM",
        requested_amount=6000,
        evidence_available=[
            "package_photo"
        ],
        strategic_state=FakeStrategicState(),
    )

    assert (
        decision.features[
            "strategic_adaptation_score"
        ] > 0.70
    )

    assert (
        "STRATEGIC_ADAPTATION"
        in decision.reason_codes
    )

    assert decision.risk_level == "HIGH"


def test_governor_score_is_bounded():

    governor = AdaptiveRiskGovernor()

    history = [
        {
            "claim_type": "SHORTAGE_CLAIM",
            "support_decision": "DENY",
        },
        {
            "claim_type": "WRONG_ITEM_CLAIM",
            "support_decision": "DENY",
        },
        {
            "claim_type": "NON_DELIVERY_CLAIM",
            "support_decision": "DENY",
        },
        {
            "claim_type": "SUBSTITUTED_RETURN_CLAIM",
            "support_decision": "DENY",
        },
        {
            "claim_type": "SHORTAGE_CLAIM",
            "support_decision": "DENY",
        },
    ]

    decision = governor.evaluate(
        customer_history=history,
        current_claim_type="SHORTAGE_CLAIM",
        requested_amount=10000,
    )

    assert 0.0 <= decision.risk_score <= 1.0