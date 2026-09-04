from src.agents.adaptive_customer import (
    AdaptiveCustomerAgent,
    AdaptiveCustomerState,
)


def build_agent():

    state = AdaptiveCustomerState(
        customer_id="CUSTOMER_TEST",
        objective="maximize_successful_legitimate_resolution",
    )

    return AdaptiveCustomerAgent(
        state=state,
        seed=42,
    )


def test_initial_beliefs_are_normalized():

    agent = build_agent()

    beliefs = agent.get_belief_summary()

    assert len(beliefs) == 4

    assert abs(
        sum(beliefs.values()) - 1.0
    ) < 1e-9


def test_successful_action_changes_belief():

    agent = build_agent()

    before = agent.get_belief_summary()

    agent.update_beliefs(
        action="SUBMIT_EVIDENCE",
        outcome="APPROVED",
        reward=1.0,
    )

    after = agent.get_belief_summary()

    assert (
        after["EVIDENCE_FIRST"]
        > before["EVIDENCE_FIRST"]
    )


def test_failed_action_changes_belief():

    agent = build_agent()

    before = agent.get_belief_summary()

    agent.update_beliefs(
        action="SUBMIT_REQUEST",
        outcome="DENIED",
        reward=-1.0,
    )

    after = agent.get_belief_summary()

    assert (
        after["DIRECT_REQUEST"]
        < before["DIRECT_REQUEST"]
    )


def test_action_selection_returns_valid_action():

    agent = build_agent()

    available = [
        "SUBMIT_REQUEST",
        "SUBMIT_EVIDENCE",
        "ADJUST_AMOUNT",
    ]

    action = agent.select_action(
        available_actions=available,
        exploration_rate=0.0,
    )

    assert action in available


def test_observation_is_recorded():

    agent = build_agent()

    class MockDecision:

        decision = "REQUEST_EVIDENCE"
        reason_code = "INSUFFICIENT_EVIDENCE"
        requested_evidence = ["delivery_photo"]
        approved_amount = 0.0
        requires_followup = True

    agent.observe_decision(
        MockDecision()
    )

    assert len(
        agent.state.support_observations
    ) == 1

    observation = (
        agent.state.support_observations[0]
    )

    assert (
        observation["decision"]
        == "REQUEST_EVIDENCE"
    )

    assert (
        "delivery_photo"
        in observation["requested_evidence"]
    )