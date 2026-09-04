from src.agents.policy_learner import (
    PolicyObservation,
    SupportPolicyLearner,
)


def make_observations():

    observations = []

    # Low-value requests
    for _ in range(20):

        observations.append(
            PolicyObservation(
                claim_type="SHORTAGE_CLAIM",
                requested_amount=800,
                evidence_available=True,
                evidence_count=1,
                support_decision="APPROVE",
            )
        )

    # Medium-value requests
    for _ in range(20):

        observations.append(
            PolicyObservation(
                claim_type="SHORTAGE_CLAIM",
                requested_amount=1500,
                evidence_available=False,
                evidence_count=0,
                support_decision="REQUEST_EVIDENCE",
            )
        )

    # High-value requests
    for _ in range(20):

        observations.append(
            PolicyObservation(
                claim_type="NON_DELIVERY_CLAIM",
                requested_amount=6000,
                evidence_available=True,
                evidence_count=1,
                support_decision="ESCALATE",
            )
        )

    return observations


def test_policy_learner_can_train():

    observations = make_observations()

    learner = SupportPolicyLearner()

    learner.fit(observations)

    assert learner.is_fitted


def test_policy_learner_predicts():

    observations = make_observations()

    learner = SupportPolicyLearner()

    learner.fit(observations)

    observation = PolicyObservation(
        claim_type="SHORTAGE_CLAIM",
        requested_amount=800,
        evidence_available=True,
        evidence_count=1,
    )

    prediction = learner.predict(
        observation
    )

    assert prediction in {
        "APPROVE",
        "REQUEST_EVIDENCE",
        "ESCALATE",
        "REJECT",
    }


def test_policy_learner_returns_probabilities():

    observations = make_observations()

    learner = SupportPolicyLearner()

    learner.fit(observations)

    observation = PolicyObservation(
        claim_type="SHORTAGE_CLAIM",
        requested_amount=800,
        evidence_available=True,
        evidence_count=1,
    )

    probabilities = (
        learner.predict_proba(
            observation
        )
    )

    assert len(probabilities) >= 2

    assert abs(
        sum(probabilities.values()) - 1.0
    ) < 1e-6


def test_policy_learner_evaluation():

    observations = make_observations()

    learner = SupportPolicyLearner()

    learner.fit(observations)

    result = learner.evaluate(
        observations
    )

    assert "accuracy" in result

    assert "log_loss" in result

    assert 0.0 <= result["accuracy"] <= 1.0