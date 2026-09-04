from src.evaluation.economic_evaluation import (
    EconomicCase,
    EconomicModel,
    risk_based_policy,
)


def make_fraud_case(
    *,
    amount=1000.0,
    recovery=1000.0,
):
    return EconomicCase(
        case_id="FRAUD_001",
        requested_amount=amount,
        is_legitimate=False,
        fraud_loss_if_allowed=amount,
        friction_cost_if_verified=25.0,
        review_cost_if_escalated=100.0,
        recovery_if_escalated=recovery,
    )


def make_legitimate_case(
    *,
    amount=1000.0,
):
    return EconomicCase(
        case_id="LEGIT_001",
        requested_amount=amount,
        is_legitimate=True,
        fraud_loss_if_allowed=0.0,
        friction_cost_if_verified=25.0,
        review_cost_if_escalated=100.0,
        recovery_if_escalated=0.0,
    )


# ============================================================
# LOSS PREVENTION
# ============================================================

def test_loss_prevented_cannot_exceed_fraud_exposure():

    model = EconomicModel()

    case = make_fraud_case(
        amount=1000.0,
        recovery=5000.0,
    )

    outcome = model.evaluate_case(
        case,
        "ESCALATE_TO_HUMAN_REVIEW",
    )

    assert outcome.loss_prevented == 1000.0
    assert outcome.recovery == 1000.0


def test_remaining_fraud_loss_is_zero_when_fully_recovered():

    model = EconomicModel()

    case = make_fraud_case(
        amount=1000.0,
        recovery=1000.0,
    )

    _, metrics = model.evaluate_dataset(
        [case],
        ["ESCALATE_TO_HUMAN_REVIEW"],
    )

    assert metrics.fraud_loss_exposure == 1000.0
    assert metrics.total_loss_prevented == 1000.0
    assert metrics.fraud_loss_remaining == 0.0


def test_remaining_fraud_loss_is_never_negative():

    model = EconomicModel()

    case = make_fraud_case(
        amount=1000.0,
        recovery=999999.0,
    )

    _, metrics = model.evaluate_dataset(
        [case],
        ["ESCALATE_TO_HUMAN_REVIEW"],
    )

    assert metrics.fraud_loss_remaining >= 0.0


# ============================================================
# LEGITIMATE REFUNDS
# ============================================================

def test_legitimate_refund_is_preserved_when_allowed():

    model = EconomicModel()

    case = make_legitimate_case(
        amount=1500.0,
    )

    outcome = model.evaluate_case(
        case,
        "ALLOW_AGENT_A_DECISION",
    )

    assert (
        outcome.legitimate_refund_preserved
        == 1500.0
    )

    assert not outcome.false_intervention


def test_legitimate_refund_is_preserved_after_verification():

    model = EconomicModel()

    case = make_legitimate_case(
        amount=1500.0,
    )

    outcome = model.evaluate_case(
        case,
        "REQUEST_ADDITIONAL_EVIDENCE",
    )

    assert (
        outcome.legitimate_refund_preserved
        == 1500.0
    )

    assert outcome.friction_cost == 25.0
    assert not outcome.false_intervention


def test_legitimate_escalation_is_false_intervention():

    model = EconomicModel()

    case = make_legitimate_case()

    outcome = model.evaluate_case(
        case,
        "ESCALATE_TO_HUMAN_REVIEW",
    )

    assert outcome.false_intervention
    assert outcome.review_cost == 100.0


# ============================================================
# NET ECONOMIC BENEFIT
# ============================================================

def test_net_benefit_is_negative_net_cost():

    model = EconomicModel()

    case = make_fraud_case(
        amount=1000.0,
        recovery=1000.0,
    )

    outcome = model.evaluate_case(
        case,
        "ESCALATE_TO_HUMAN_REVIEW",
    )

    assert (
        outcome.net_benefit
        == -outcome.net_cost
    )


def test_full_recovery_produces_positive_net_benefit():

    model = EconomicModel()

    case = make_fraud_case(
        amount=1000.0,
        recovery=1000.0,
    )

    outcome = model.evaluate_case(
        case,
        "ESCALATE_TO_HUMAN_REVIEW",
    )

    assert outcome.net_benefit == 900.0


# ============================================================
# AGGREGATE METRICS
# ============================================================

def test_loss_prevention_rate_is_bounded():

    model = EconomicModel()

    fraud = make_fraud_case(
        amount=1000.0,
        recovery=1000.0,
    )

    _, metrics = model.evaluate_dataset(
        [fraud],
        ["ESCALATE_TO_HUMAN_REVIEW"],
    )

    assert (
        0.0
        <= metrics.loss_prevention_rate
        <= 1.0
    )


def test_false_intervention_rate_is_bounded():

    model = EconomicModel()

    legitimate = make_legitimate_case()

    _, metrics = model.evaluate_dataset(
        [legitimate],
        ["ESCALATE_TO_HUMAN_REVIEW"],
    )

    assert (
        0.0
        <= metrics.false_intervention_rate
        <= 1.0
    )


def test_prevented_loss_cannot_exceed_exposure():

    model = EconomicModel()

    fraud = make_fraud_case(
        amount=1000.0,
        recovery=999999.0,
    )

    _, metrics = model.evaluate_dataset(
        [fraud],
        ["ESCALATE_TO_HUMAN_REVIEW"],
    )

    assert (
        metrics.total_loss_prevented
        <= metrics.fraud_loss_exposure
    )


# ============================================================
# POLICY
# ============================================================

def test_risk_policy_thresholds():

    actions = risk_based_policy(
        [0.1, 0.5, 0.9],
        verify_threshold=0.30,
        escalate_threshold=0.70,
    )

    assert actions == [
        "ALLOW_AGENT_A_DECISION",
        "REQUEST_ADDITIONAL_EVIDENCE",
        "ESCALATE_TO_HUMAN_REVIEW",
    ]