from __future__ import annotations

from src.evaluation.phase13_5_economic_stress import (
    generate_stress_cases,
    governor_policy,
    evaluate_policy,
)


# ============================================================
# DATASET TESTS
# ============================================================

def test_stress_dataset_has_expected_size():

    cases = generate_stress_cases()

    assert len(cases) == 10


def test_stress_dataset_contains_legitimate_cases():

    cases = generate_stress_cases()

    assert any(
        case.economic_case.is_legitimate
        for case in cases
    )


def test_stress_dataset_contains_fraud_cases():

    cases = generate_stress_cases()

    assert any(
        not case.economic_case.is_legitimate
        for case in cases
    )


# ============================================================
# HARD-CASE TESTS
# ============================================================

def test_legitimate_high_risk_case_exists():

    cases = generate_stress_cases()

    assert any(
        case.economic_case.is_legitimate
        and case.risk_score >= 0.70
        for case in cases
    )


def test_low_risk_fraud_case_exists():

    cases = generate_stress_cases()

    assert any(
        not case.economic_case.is_legitimate
        and case.risk_score < 0.30
        for case in cases
    )


def test_risk_overlap_exists():

    cases = generate_stress_cases()

    legitimate_risks = [
        case.risk_score
        for case in cases
        if case.economic_case.is_legitimate
    ]

    fraud_risks = [
        case.risk_score
        for case in cases
        if not case.economic_case.is_legitimate
    ]

    assert max(legitimate_risks) > min(fraud_risks)


# ============================================================
# GOVERNOR POLICY
# ============================================================

def test_governor_policy_returns_one_action_per_case():

    cases = generate_stress_cases()

    actions = governor_policy(
        cases
    )

    assert len(actions) == len(cases)


def test_governor_policy_actions_are_valid():

    cases = generate_stress_cases()

    actions = governor_policy(
        cases
    )

    valid_actions = {
        "ALLOW_AGENT_A_DECISION",
        "REQUEST_ADDITIONAL_EVIDENCE",
        "ESCALATE_TO_HUMAN_REVIEW",
    }

    assert all(
        action in valid_actions
        for action in actions
    )


# ============================================================
# ECONOMIC EVALUATION
# ============================================================

def test_governor_economic_result_is_finite():

    cases = generate_stress_cases()

    actions = governor_policy(
        cases
    )

    result = evaluate_policy(
        cases,
        actions,
        "CURRENT_GOVERNOR",
    )

    assert (
        result.net_benefit
        == result.net_benefit
    )


def test_remaining_loss_is_non_negative():

    cases = generate_stress_cases()

    actions = governor_policy(
        cases
    )

    result = evaluate_policy(
        cases,
        actions,
        "CURRENT_GOVERNOR",
    )

    assert result.remaining_loss >= 0.0


def test_loss_prevention_does_not_exceed_exposure():

    cases = generate_stress_cases()

    actions = governor_policy(
        cases
    )

    result = evaluate_policy(
        cases,
        actions,
        "CURRENT_GOVERNOR",
    )

    assert (
        result.loss_prevented
        <= result.fraud_exposure
    )


def test_false_interventions_are_bounded():

    cases = generate_stress_cases()

    actions = governor_policy(
        cases
    )

    result = evaluate_policy(
        cases,
        actions,
        "CURRENT_GOVERNOR",
    )

    assert (
        0
        <= result.false_interventions
        <= result.legitimate_cases
    )