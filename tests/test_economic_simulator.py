from __future__ import annotations

import math

import pytest

from src.evaluation.economic_simulator import (
    CaseTruth,
    EconomicCase,
    EconomicCosts,
    EconomicOutcome,
    EconomicSummary,
    GovernorAction,
    SupportDecision,
    evaluate_policy,
    normalize_governor_action,
    normalize_support_decision,
    resolve_effective_decision,
    simulate_economic_outcome,
    summarize_economic_outcomes,
)


# ============================================================
# CASE FACTORIES
# ============================================================


def fraud_case(
    amount: float = 1000.0,
    case_id: str = "FRAUD_001",
    evidence_resolution: SupportDecision | None = None,
    human_review_resolution: SupportDecision | None = None,
) -> EconomicCase:

    return EconomicCase(
        case_id=case_id,
        truth=CaseTruth.FRAUD,
        requested_amount=amount,
        fraud_loss_exposure=amount,
        legitimate_refund_amount=0.0,
        evidence_resolution=evidence_resolution,
        human_review_resolution=human_review_resolution,
    )


def legitimate_case(
    amount: float = 1000.0,
    case_id: str = "LEGIT_001",
    evidence_resolution: SupportDecision | None = None,
    human_review_resolution: SupportDecision | None = None,
) -> EconomicCase:

    return EconomicCase(
        case_id=case_id,
        truth=CaseTruth.LEGITIMATE,
        requested_amount=amount,
        fraud_loss_exposure=0.0,
        legitimate_refund_amount=amount,
        evidence_resolution=evidence_resolution,
        human_review_resolution=human_review_resolution,
    )


# ============================================================
# ACTION SPACE
# ============================================================


def test_governor_action_space_is_canonical():

    assert set(GovernorAction) == {
        GovernorAction.ALLOW_AGENT_A_DECISION,
        GovernorAction.REQUEST_ADDITIONAL_EVIDENCE,
        GovernorAction.ESCALATE_TO_HUMAN_REVIEW,
    }


def test_support_decision_space_is_canonical():

    assert set(SupportDecision) == {
        SupportDecision.APPROVE,
        SupportDecision.REQUEST_EVIDENCE,
        SupportDecision.ESCALATE,
        SupportDecision.DENY,
    }


# ============================================================
# NORMALIZATION
# ============================================================


def test_governor_action_normalization():

    assert (
        normalize_governor_action(
            "ALLOW_AGENT_A_DECISION"
        )
        == GovernorAction.ALLOW_AGENT_A_DECISION
    )

    assert (
        normalize_governor_action(
            GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
        )
        == GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
    )


def test_support_decision_normalization():

    assert (
        normalize_support_decision("APPROVE")
        == SupportDecision.APPROVE
    )

    assert (
        normalize_support_decision(
            SupportDecision.DENY
        )
        == SupportDecision.DENY
    )


def test_invalid_governor_action_rejected():

    with pytest.raises(ValueError):
        normalize_governor_action(
            "INVALID_ACTION"
        )


def test_invalid_support_decision_rejected():

    with pytest.raises(ValueError):
        normalize_support_decision(
            "INVALID_DECISION"
        )


# ============================================================
# ACTION RESOLUTION
# ============================================================


def test_allow_preserves_agent_approve():

    result = resolve_effective_decision(
        governor_action=(
            GovernorAction.ALLOW_AGENT_A_DECISION
        ),
        agent_decision=SupportDecision.APPROVE,
    )

    assert result == SupportDecision.APPROVE


def test_allow_preserves_agent_deny():

    result = resolve_effective_decision(
        governor_action=(
            GovernorAction.ALLOW_AGENT_A_DECISION
        ),
        agent_decision=SupportDecision.DENY,
    )

    assert result == SupportDecision.DENY


def test_allow_preserves_agent_evidence_request():

    result = resolve_effective_decision(
        governor_action=(
            GovernorAction.ALLOW_AGENT_A_DECISION
        ),
        agent_decision=SupportDecision.REQUEST_EVIDENCE,
    )

    assert result == SupportDecision.REQUEST_EVIDENCE


def test_allow_preserves_agent_escalation():

    result = resolve_effective_decision(
        governor_action=(
            GovernorAction.ALLOW_AGENT_A_DECISION
        ),
        agent_decision=SupportDecision.ESCALATE,
    )

    assert result == SupportDecision.ESCALATE


def test_evidence_override():

    result = resolve_effective_decision(
        governor_action=(
            GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
        ),
        agent_decision=SupportDecision.APPROVE,
    )

    assert result == SupportDecision.REQUEST_EVIDENCE


def test_evidence_overrides_deny():

    result = resolve_effective_decision(
        governor_action=(
            GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
        ),
        agent_decision=SupportDecision.DENY,
    )

    assert result == SupportDecision.REQUEST_EVIDENCE


def test_escalation_override():

    result = resolve_effective_decision(
        governor_action=(
            GovernorAction.ESCALATE_TO_HUMAN_REVIEW
        ),
        agent_decision=SupportDecision.APPROVE,
    )

    assert result == SupportDecision.ESCALATE


def test_escalation_overrides_deny():

    result = resolve_effective_decision(
        governor_action=(
            GovernorAction.ESCALATE_TO_HUMAN_REVIEW
        ),
        agent_decision=SupportDecision.DENY,
    )

    assert result == SupportDecision.ESCALATE


# ============================================================
# APPROVAL
# ============================================================


def test_fraud_approval_creates_fraud_loss():

    outcome = simulate_economic_outcome(
        case=fraud_case(1000.0),
        governor_action=(
            GovernorAction.ALLOW_AGENT_A_DECISION
        ),
        agent_decision=SupportDecision.APPROVE,
    )

    assert outcome.loss_prevented == 0.0

    assert outcome.remaining_fraud_loss == 1000.0

    assert (
        outcome.legitimate_refund_preserved
        == 0.0
    )

    assert outcome.false_intervention is False

    assert outcome.net_benefit == -1000.0


def test_legitimate_approval_preserves_refund():

    outcome = simulate_economic_outcome(
        case=legitimate_case(1000.0),
        governor_action=(
            GovernorAction.ALLOW_AGENT_A_DECISION
        ),
        agent_decision=SupportDecision.APPROVE,
    )

    assert outcome.loss_prevented == 0.0

    assert outcome.remaining_fraud_loss == 0.0

    assert (
        outcome.legitimate_refund_preserved
        == 1000.0
    )

    assert outcome.false_intervention is False

    assert outcome.net_benefit == 1000.0


# ============================================================
# DENIAL
# ============================================================


def test_fraud_denial_prevents_full_loss():

    outcome = simulate_economic_outcome(
        case=fraud_case(1000.0),
        governor_action=(
            GovernorAction.ALLOW_AGENT_A_DECISION
        ),
        agent_decision=SupportDecision.DENY,
    )

    assert outcome.loss_prevented == 1000.0

    assert outcome.remaining_fraud_loss == 0.0

    assert (
        outcome.legitimate_refund_preserved
        == 0.0
    )

    assert outcome.net_benefit == 1000.0


def test_legitimate_denial_is_false_intervention():

    outcome = simulate_economic_outcome(
        case=legitimate_case(1000.0),
        governor_action=(
            GovernorAction.ALLOW_AGENT_A_DECISION
        ),
        agent_decision=SupportDecision.DENY,
    )

    assert outcome.false_intervention is True

    assert (
        outcome.legitimate_refund_preserved
        == 0.0
    )

    assert outcome.net_benefit == -1000.0


# ============================================================
# EVIDENCE
# ============================================================


def test_fraud_evidence_request_prevents_loss():

    outcome = simulate_economic_outcome(
        case=fraud_case(
            1000.0,
            evidence_resolution=SupportDecision.DENY,
        ),
        governor_action=(
            GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
        ),
        agent_decision=SupportDecision.APPROVE,
    )

    assert (
        outcome.effective_decision
        == SupportDecision.DENY
    )

    assert outcome.loss_prevented == 1000.0

    assert outcome.remaining_fraud_loss == 0.0

    assert outcome.friction_cost == 25.0

    assert outcome.net_benefit == 975.0


def test_legitimate_evidence_request_preserves_refund():

    outcome = simulate_economic_outcome(
        case=legitimate_case(
            1000.0,
            evidence_resolution=SupportDecision.APPROVE,
        ),
        governor_action=(
            GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
        ),
        agent_decision=SupportDecision.APPROVE,
    )

    assert (
        outcome.effective_decision
        == SupportDecision.APPROVE
    )

    assert (
        outcome.legitimate_refund_preserved
        == 1000.0
    )

    assert outcome.friction_cost == 25.0

    assert outcome.false_intervention is True

    assert outcome.net_benefit == 975.0


def test_evidence_request_requires_evidence_resolution():

    with pytest.raises(ValueError):

        simulate_economic_outcome(
            case=fraud_case(1000.0),
            governor_action=(
                GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
            ),
            agent_decision=SupportDecision.APPROVE,
        )


# ============================================================
# ESCALATION
# ============================================================


def test_fraud_escalation_prevents_loss():

    outcome = simulate_economic_outcome(
        case=fraud_case(
            1000.0,
            human_review_resolution=SupportDecision.DENY,
        ),
        governor_action=(
            GovernorAction.ESCALATE_TO_HUMAN_REVIEW
        ),
        agent_decision=SupportDecision.APPROVE,
    )

    assert (
        outcome.effective_decision
        == SupportDecision.DENY
    )

    assert outcome.loss_prevented == 1000.0

    assert outcome.remaining_fraud_loss == 0.0

    assert outcome.human_review_cost == 50.0

    assert outcome.net_benefit == 950.0


def test_legitimate_escalation_preserves_refund():

    outcome = simulate_economic_outcome(
        case=legitimate_case(
            1000.0,
            human_review_resolution=SupportDecision.APPROVE,
        ),
        governor_action=(
            GovernorAction.ESCALATE_TO_HUMAN_REVIEW
        ),
        agent_decision=SupportDecision.APPROVE,
    )

    assert (
        outcome.effective_decision
        == SupportDecision.APPROVE
    )

    assert (
        outcome.legitimate_refund_preserved
        == 1000.0
    )

    assert outcome.human_review_cost == 50.0

    assert outcome.false_intervention is True

    assert outcome.net_benefit == 950.0


def test_human_review_requires_human_review_resolution():

    with pytest.raises(ValueError):

        simulate_economic_outcome(
            case=fraud_case(1000.0),
            governor_action=(
                GovernorAction.ESCALATE_TO_HUMAN_REVIEW
            ),
            agent_decision=SupportDecision.APPROVE,
        )


# ============================================================
# ALLOW AGENT DECISION — ALL SUPPORT OUTCOMES
# ============================================================


@pytest.mark.parametrize(
    "decision",
    [
        SupportDecision.APPROVE,
        SupportDecision.REQUEST_EVIDENCE,
        SupportDecision.ESCALATE,
        SupportDecision.DENY,
    ],
)
def test_allow_action_preserves_agent_decision(
    decision,
):

    result = resolve_effective_decision(
        governor_action=(
            GovernorAction.ALLOW_AGENT_A_DECISION
        ),
        agent_decision=decision,
    )

    assert result == decision


# ============================================================
# ECONOMIC INVARIANTS
# ============================================================


@pytest.mark.parametrize(
    "truth",
    [
        CaseTruth.FRAUD,
        CaseTruth.LEGITIMATE,
    ],
)
@pytest.mark.parametrize(
    "action",
    list(GovernorAction),
)
def test_economic_values_are_finite(
    truth,
    action,
):

    if truth == CaseTruth.FRAUD:

        case = (
            fraud_case(
                1000.0,
                evidence_resolution=SupportDecision.DENY,
                human_review_resolution=SupportDecision.DENY,
            )
            if action
            != GovernorAction.ALLOW_AGENT_A_DECISION
            else fraud_case(1000.0)
        )

    else:

        case = (
            legitimate_case(
                1000.0,
                evidence_resolution=SupportDecision.APPROVE,
                human_review_resolution=SupportDecision.APPROVE,
            )
            if action
            != GovernorAction.ALLOW_AGENT_A_DECISION
            else legitimate_case(1000.0)
        )

    outcome = simulate_economic_outcome(
        case=case,
        governor_action=action,
        agent_decision=SupportDecision.APPROVE,
    )

    assert math.isfinite(
        outcome.net_benefit
    )

    assert math.isfinite(
        outcome.loss_prevented
    )

    assert math.isfinite(
        outcome.remaining_fraud_loss
    )

    assert math.isfinite(
        outcome.legitimate_refund_preserved
    )

    assert math.isfinite(
        outcome.friction_cost
    )

    assert math.isfinite(
        outcome.human_review_cost
    )


@pytest.mark.parametrize(
    "truth",
    [
        CaseTruth.FRAUD,
        CaseTruth.LEGITIMATE,
    ],
)
@pytest.mark.parametrize(
    "action",
    list(GovernorAction),
)
def test_loss_prevention_is_bounded(
    truth,
    action,
):

    if truth == CaseTruth.FRAUD:

        case = (
            fraud_case(
                1000.0,
                evidence_resolution=SupportDecision.DENY,
                human_review_resolution=SupportDecision.DENY,
            )
            if action
            != GovernorAction.ALLOW_AGENT_A_DECISION
            else fraud_case(1000.0)
        )

    else:

        case = (
            legitimate_case(
                1000.0,
                evidence_resolution=SupportDecision.APPROVE,
                human_review_resolution=SupportDecision.APPROVE,
            )
            if action
            != GovernorAction.ALLOW_AGENT_A_DECISION
            else legitimate_case(1000.0)
        )

    outcome = simulate_economic_outcome(
        case=case,
        governor_action=action,
        agent_decision=SupportDecision.APPROVE,
    )

    assert (
        0.0
        <= outcome.loss_prevented
        <= outcome.fraud_loss_exposure
    )

    assert (
        outcome.remaining_fraud_loss
        >= 0.0
    )


def test_fraud_exposure_is_zero_for_legitimate_case():

    case = legitimate_case(1000.0)

    assert case.fraud_loss_exposure == 0.0


def test_legitimate_refund_is_zero_for_fraud_case():

    case = fraud_case(1000.0)

    assert case.legitimate_refund_amount == 0.0


# ============================================================
# CUSTOM COSTS
# ============================================================


def test_custom_costs_are_respected():

    costs = EconomicCosts(
        evidence_request_cost=40.0,
        human_review_cost=100.0,
    )

    evidence_outcome = simulate_economic_outcome(
        case=fraud_case(
            1000.0,
            evidence_resolution=SupportDecision.DENY,
        ),
        governor_action=(
            GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
        ),
        agent_decision=SupportDecision.APPROVE,
        costs=costs,
    )

    assert (
        evidence_outcome.friction_cost
        == 40.0
    )

    escalation_outcome = simulate_economic_outcome(
        case=fraud_case(
            1000.0,
            human_review_resolution=SupportDecision.DENY,
        ),
        governor_action=(
            GovernorAction.ESCALATE_TO_HUMAN_REVIEW
        ),
        agent_decision=SupportDecision.APPROVE,
        costs=costs,
    )

    assert (
        escalation_outcome.human_review_cost
        == 100.0
    )


def test_false_intervention_cost_is_applied():

    costs = EconomicCosts(
        false_intervention_cost=75.0,
    )

    outcome = simulate_economic_outcome(
        case=legitimate_case(
            1000.0,
            evidence_resolution=SupportDecision.APPROVE,
        ),
        governor_action=(
            GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
        ),
        agent_decision=SupportDecision.APPROVE,
        costs=costs,
    )

    assert outcome.false_intervention is True

    assert outcome.net_benefit == pytest.approx(
        1000.0
        - 25.0
        - 75.0
    )


# ============================================================
# CASE VALIDATION
# ============================================================


def test_empty_case_id_is_rejected():

    with pytest.raises(ValueError):

        EconomicCase(
            case_id="",
            truth=CaseTruth.FRAUD,
            requested_amount=1000.0,
            fraud_loss_exposure=1000.0,
            legitimate_refund_amount=0.0,
        )


def test_negative_requested_amount_is_rejected():

    with pytest.raises(ValueError):

        EconomicCase(
            case_id="FRAUD_001",
            truth=CaseTruth.FRAUD,
            requested_amount=-1.0,
            fraud_loss_exposure=1000.0,
            legitimate_refund_amount=0.0,
        )


def test_negative_fraud_exposure_is_rejected():

    with pytest.raises(ValueError):

        EconomicCase(
            case_id="FRAUD_001",
            truth=CaseTruth.FRAUD,
            requested_amount=1000.0,
            fraud_loss_exposure=-1.0,
            legitimate_refund_amount=0.0,
        )


def test_negative_legitimate_refund_is_rejected():

    with pytest.raises(ValueError):

        EconomicCase(
            case_id="LEGIT_001",
            truth=CaseTruth.LEGITIMATE,
            requested_amount=1000.0,
            fraud_loss_exposure=0.0,
            legitimate_refund_amount=-1.0,
        )


def test_fraud_case_cannot_have_legitimate_refund():

    with pytest.raises(ValueError):

        EconomicCase(
            case_id="FRAUD_001",
            truth=CaseTruth.FRAUD,
            requested_amount=1000.0,
            fraud_loss_exposure=1000.0,
            legitimate_refund_amount=100.0,
        )


def test_legitimate_case_cannot_have_fraud_exposure():

    with pytest.raises(ValueError):

        EconomicCase(
            case_id="LEGIT_001",
            truth=CaseTruth.LEGITIMATE,
            requested_amount=1000.0,
            fraud_loss_exposure=100.0,
            legitimate_refund_amount=1000.0,
        )


# ============================================================
# SUMMARY
# ============================================================


def test_summary_counts_cases():

    outcomes = [
        simulate_economic_outcome(
            case=fraud_case(
                1000.0,
                case_id="FRAUD_001",
            ),
            governor_action=(
                GovernorAction.ALLOW_AGENT_A_DECISION
            ),
            agent_decision=SupportDecision.APPROVE,
        ),
        simulate_economic_outcome(
            case=legitimate_case(
                1000.0,
                case_id="LEGIT_001",
            ),
            governor_action=(
                GovernorAction.ALLOW_AGENT_A_DECISION
            ),
            agent_decision=SupportDecision.APPROVE,
        ),
    ]

    summary = summarize_economic_outcomes(
        outcomes
    )

    assert isinstance(
        summary,
        EconomicSummary,
    )

    assert summary.total_cases == 2

    assert summary.fraud_cases == 1

    assert summary.legitimate_cases == 1


def test_summary_fraud_exposure():

    outcomes = [
        simulate_economic_outcome(
            case=fraud_case(
                1000.0,
                case_id="FRAUD_001",
            ),
            governor_action=(
                GovernorAction.ALLOW_AGENT_A_DECISION
            ),
            agent_decision=SupportDecision.APPROVE,
        ),
        simulate_economic_outcome(
            case=fraud_case(
                2000.0,
                case_id="FRAUD_002",
            ),
            governor_action=(
                GovernorAction.ALLOW_AGENT_A_DECISION
            ),
            agent_decision=SupportDecision.APPROVE,
        ),
    ]

    summary = summarize_economic_outcomes(
        outcomes
    )

    assert summary.fraud_exposure == 3000.0

    assert summary.remaining_fraud_loss == 3000.0

    assert summary.loss_prevented == 0.0

    assert summary.loss_prevention_rate == 0.0


def test_summary_loss_prevention_rate():

    outcomes = [
        simulate_economic_outcome(
            case=fraud_case(
                1000.0,
                case_id="FRAUD_001",
                evidence_resolution=SupportDecision.DENY,
            ),
            governor_action=(
                GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
            ),
            agent_decision=SupportDecision.APPROVE,
        ),
        simulate_economic_outcome(
            case=fraud_case(
                2000.0,
                case_id="FRAUD_002",
            ),
            governor_action=(
                GovernorAction.ALLOW_AGENT_A_DECISION
            ),
            agent_decision=SupportDecision.APPROVE,
        ),
    ]

    summary = summarize_economic_outcomes(
        outcomes
    )

    assert summary.fraud_exposure == 3000.0

    assert summary.loss_prevented == 1000.0

    assert (
        summary.loss_prevention_rate
        == pytest.approx(1.0 / 3.0)
    )


def test_summary_false_intervention_rate():

    outcomes = [
        simulate_economic_outcome(
            case=legitimate_case(
                1000.0,
                case_id="LEGIT_001",
                evidence_resolution=SupportDecision.APPROVE,
            ),
            governor_action=(
                GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
            ),
            agent_decision=SupportDecision.APPROVE,
        ),
        simulate_economic_outcome(
            case=legitimate_case(
                1000.0,
                case_id="LEGIT_002",
            ),
            governor_action=(
                GovernorAction.ALLOW_AGENT_A_DECISION
            ),
            agent_decision=SupportDecision.APPROVE,
        ),
    ]

    summary = summarize_economic_outcomes(
        outcomes
    )

    assert summary.false_interventions == 1

    assert (
        summary.false_intervention_rate
        == pytest.approx(0.5)
    )


def test_summary_average_benefit():

    outcomes = [
        simulate_economic_outcome(
            case=fraud_case(
                1000.0,
                case_id="FRAUD_001",
                evidence_resolution=SupportDecision.DENY,
            ),
            governor_action=(
                GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
            ),
            agent_decision=SupportDecision.APPROVE,
        ),
        simulate_economic_outcome(
            case=legitimate_case(
                1000.0,
                case_id="LEGIT_001",
            ),
            governor_action=(
                GovernorAction.ALLOW_AGENT_A_DECISION
            ),
            agent_decision=SupportDecision.APPROVE,
        ),
    ]

    summary = summarize_economic_outcomes(
        outcomes
    )

    assert (
        summary.average_benefit_per_case
        == pytest.approx(
            summary.net_economic_benefit / 2
        )
    )


def test_empty_summary_is_rejected():

    with pytest.raises(ValueError):
        summarize_economic_outcomes([])


# ============================================================
# POLICY EVALUATION
# ============================================================


def test_evaluate_policy():

    cases = [
        fraud_case(
            1000.0,
            case_id="FRAUD_001",
            evidence_resolution=SupportDecision.DENY,
        ),
        legitimate_case(
            1000.0,
            case_id="LEGIT_001",
            evidence_resolution=SupportDecision.APPROVE,
        ),
    ]

    summary = evaluate_policy(
        cases=cases,
        governor_action=(
            GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
        ),
        agent_decision=SupportDecision.APPROVE,
    )

    assert summary.total_cases == 2

    assert summary.fraud_cases == 1

    assert summary.legitimate_cases == 1

    assert summary.loss_prevented == 1000.0

    assert summary.remaining_fraud_loss == 0.0

    assert summary.friction_cost == 50.0

    assert summary.false_interventions == 1


# ============================================================
# OUTCOME TYPE
# ============================================================


def test_economic_outcome_type():

    outcome = simulate_economic_outcome(
        case=fraud_case(1000.0),
        governor_action=(
            GovernorAction.ALLOW_AGENT_A_DECISION
        ),
        agent_decision=SupportDecision.APPROVE,
    )

    assert isinstance(
        outcome,
        EconomicOutcome,
    )


# ============================================================
# NO HIDDEN ACTIONS
# ============================================================


def test_every_governor_action_resolves_to_valid_support_decision():

    for action in GovernorAction:

        result = resolve_effective_decision(
            governor_action=action,
            agent_decision=SupportDecision.APPROVE,
        )

        assert result in set(
            SupportDecision
        )


# ============================================================
# GOVERNOR ACTIONS MUST NOT USE GROUND TRUTH
# ============================================================


def test_same_action_flow_is_deterministic_for_same_case():

    case = fraud_case(
        1500.0,
        human_review_resolution=SupportDecision.DENY,
    )

    first = simulate_economic_outcome(
        case=case,
        governor_action=(
            GovernorAction.ESCALATE_TO_HUMAN_REVIEW
        ),
        agent_decision=SupportDecision.APPROVE,
    )

    second = simulate_economic_outcome(
        case=case,
        governor_action=(
            GovernorAction.ESCALATE_TO_HUMAN_REVIEW
        ),
        agent_decision=SupportDecision.APPROVE,
    )

    assert first == second


# ============================================================
# FINAL INTEGRITY CHECK
# ============================================================


def test_all_canonical_actions_have_economic_outcomes():

    cases = [
        fraud_case(
            1000.0,
            case_id="FRAUD_001",
            evidence_resolution=SupportDecision.DENY,
            human_review_resolution=SupportDecision.DENY,
        ),
        legitimate_case(
            1000.0,
            case_id="LEGIT_001",
            evidence_resolution=SupportDecision.APPROVE,
            human_review_resolution=SupportDecision.APPROVE,
        ),
    ]

    for action in GovernorAction:

        for case in cases:

            outcome = simulate_economic_outcome(
                case=case,
                governor_action=action,
                agent_decision=SupportDecision.APPROVE,
            )

            assert outcome.governor_action == action

            assert (
                outcome.effective_decision
                in set(SupportDecision)
            )

            assert (
                outcome.remaining_fraud_loss
                >= 0.0
            )

            assert math.isfinite(
                outcome.net_benefit
            )