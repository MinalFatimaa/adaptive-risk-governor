from src.agents.strategic_customer import (
    StrategicCustomerAgent,
    StrategicCustomerState,
)


def test_strategic_agent_initializes():

    state = StrategicCustomerState(
        customer_id="CUSTOMER_TEST",
    )

    agent = StrategicCustomerAgent(
        state=state,
        seed=42,
    )

    policy = agent.get_policy()

    assert len(policy) == 4

    for claim_type in policy:

        assert len(policy[claim_type]) == 6


def test_strategic_agent_can_choose_action():

    state = StrategicCustomerState(
        customer_id="CUSTOMER_TEST",
    )

    agent = StrategicCustomerAgent(
        state=state,
        seed=42,
    )

    action = agent.choose_action(
        claim_type="SHORTAGE_CLAIM",
        requested_amount=1500,
        evidence_available=[],
    )

    assert action in {
        "DIRECT_REQUEST",
        "EVIDENCE_FIRST",
        "LOWER_AMOUNT",
        "FOLLOWUP_RESPONSE",
        "CLAIM_SWITCH",
        "WAIT_AND_RETRY",
    }


def test_strategic_agent_updates_after_response():

    state = StrategicCustomerState(
        customer_id="CUSTOMER_TEST",
        objective="maximize_illegitimate_refund_value",
    )

    agent = StrategicCustomerAgent(
        state=state,
        seed=42,
        exploration_rate=0.0,
    )

    action = "DIRECT_REQUEST"

    initial = (
        state.action_values[
            "SHORTAGE_CLAIM"
        ][action]
    )

    reward = agent.observe_support_response(
        claim_type="SHORTAGE_CLAIM",
        action=action,
        requested_amount=1000,
        support_decision="APPROVE",
        approved_amount=1000,
        evidence_available=[],
    )

    updated = (
        state.action_values[
            "SHORTAGE_CLAIM"
        ][action]
    )

    assert reward > 0

    assert updated > initial

    assert agent.observation_count() == 1


def test_strategic_agent_learns_from_denial():

    state = StrategicCustomerState(
        customer_id="CUSTOMER_TEST",
        objective="maximize_illegitimate_refund_value",
    )

    agent = StrategicCustomerAgent(
        state=state,
        seed=42,
        exploration_rate=0.0,
    )

    action = "DIRECT_REQUEST"

    initial = (
        state.action_values[
            "SHORTAGE_CLAIM"
        ][action]
    )

    reward = agent.observe_support_response(
        claim_type="SHORTAGE_CLAIM",
        action=action,
        requested_amount=4000,
        support_decision="DENY",
        approved_amount=0,
        evidence_available=[],
    )

    updated = (
        state.action_values[
            "SHORTAGE_CLAIM"
        ][action]
    )

    assert reward < 0

    assert updated < initial


def test_strategic_agent_infers_evidence_sensitivity():

    state = StrategicCustomerState(
        customer_id="CUSTOMER_TEST",
    )

    agent = StrategicCustomerAgent(
        state=state,
        seed=42,
    )

    initial = (
        state.inferred_policy[
            "evidence_sensitivity"
        ]
    )

    agent.observe_support_response(
        claim_type="WRONG_ITEM_CLAIM",
        action="DIRECT_REQUEST",
        requested_amount=1500,
        support_decision="REQUEST_EVIDENCE",
        evidence_available=[],
    )

    updated = (
        state.inferred_policy[
            "evidence_sensitivity"
        ]
    )

    assert updated > initial


def test_strategic_agent_records_history():

    state = StrategicCustomerState(
        customer_id="CUSTOMER_TEST",
    )

    agent = StrategicCustomerAgent(
        state=state,
        seed=42,
    )

    agent.observe_support_response(
        claim_type="NON_DELIVERY_CLAIM",
        action="DIRECT_REQUEST",
        requested_amount=1000,
        support_decision="APPROVE",
        approved_amount=1000,
    )

    assert len(
        state.interaction_history
    ) == 1

    observation = (
        state.interaction_history[0]
    )

    assert observation[
        "claim_type"
    ] == "NON_DELIVERY_CLAIM"

    assert observation[
        "support_decision"
    ] == "APPROVE"