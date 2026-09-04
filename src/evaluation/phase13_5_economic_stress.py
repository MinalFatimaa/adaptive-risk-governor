from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.evaluation.economic_evaluation import (
    ACTIONS,
    EconomicCase,
    EconomicModel,
    risk_based_policy,
)


# ============================================================
# CONFIGURATION
# ============================================================

VERIFY_THRESHOLD = 0.30
ESCALATE_THRESHOLD = 0.70


# ============================================================
# STRESS CASE
# ============================================================

@dataclass(frozen=True)
class StressCase:
    """
    Hard economic evaluation case.

    risk_score represents what the current Governor believes.
    is_legitimate represents the hidden ground truth.

    The Governor does NOT receive is_legitimate.
    """

    economic_case: EconomicCase
    risk_score: float
    scenario: str


# ============================================================
# HARD-CASE DATASET
# ============================================================

def generate_stress_cases() -> list[StressCase]:
    """
    Construct deliberately difficult cases.

    These cases are NOT intended to represent production
    probabilities. They test whether the economic policy
    behaves sensibly when risk estimates are imperfect.
    """

    cases: list[StressCase] = []

    # --------------------------------------------------------
    # 1. NORMAL LEGITIMATE CASE
    # --------------------------------------------------------

    cases.append(
        StressCase(
            economic_case=EconomicCase(
                case_id="STRESS_001",
                requested_amount=1200.0,
                is_legitimate=True,
                fraud_loss_if_allowed=0.0,
                friction_cost_if_verified=25.0,
                review_cost_if_escalated=100.0,
                recovery_if_escalated=0.0,
            ),
            risk_score=0.08,
            scenario="LEGITIMATE_LOW_RISK",
        )
    )

    # --------------------------------------------------------
    # 2. HIGH-RISK LEGITIMATE CUSTOMER
    # --------------------------------------------------------

    cases.append(
        StressCase(
            economic_case=EconomicCase(
                case_id="STRESS_002",
                requested_amount=7000.0,
                is_legitimate=True,
                fraud_loss_if_allowed=0.0,
                friction_cost_if_verified=25.0,
                review_cost_if_escalated=100.0,
                recovery_if_escalated=0.0,
            ),
            risk_score=0.91,
            scenario="LEGITIMATE_HIGH_RISK",
        )
    )

    # --------------------------------------------------------
    # 3. LOW-RISK FRAUD THAT SLIPS THROUGH
    # --------------------------------------------------------

    cases.append(
        StressCase(
            economic_case=EconomicCase(
                case_id="STRESS_003",
                requested_amount=6500.0,
                is_legitimate=False,
                fraud_loss_if_allowed=6500.0,
                friction_cost_if_verified=25.0,
                review_cost_if_escalated=100.0,
                recovery_if_escalated=5200.0,
            ),
            risk_score=0.18,
            scenario="LOW_RISK_FRAUD",
        )
    )

    # --------------------------------------------------------
    # 4. MEDIUM-RISK FRAUD
    # --------------------------------------------------------

    cases.append(
        StressCase(
            economic_case=EconomicCase(
                case_id="STRESS_004",
                requested_amount=4500.0,
                is_legitimate=False,
                fraud_loss_if_allowed=4500.0,
                friction_cost_if_verified=25.0,
                review_cost_if_escalated=100.0,
                recovery_if_escalated=3600.0,
            ),
            risk_score=0.52,
            scenario="MEDIUM_RISK_FRAUD",
        )
    )

    # --------------------------------------------------------
    # 5. HIGH-RISK FRAUD
    # --------------------------------------------------------

    cases.append(
        StressCase(
            economic_case=EconomicCase(
                case_id="STRESS_005",
                requested_amount=8000.0,
                is_legitimate=False,
                fraud_loss_if_allowed=8000.0,
                friction_cost_if_verified=25.0,
                review_cost_if_escalated=100.0,
                recovery_if_escalated=6400.0,
            ),
            risk_score=0.94,
            scenario="HIGH_RISK_FRAUD",
        )
    )

    # --------------------------------------------------------
    # 6. HIGH-VALUE LEGITIMATE CASE
    # --------------------------------------------------------

    cases.append(
        StressCase(
            economic_case=EconomicCase(
                case_id="STRESS_006",
                requested_amount=7800.0,
                is_legitimate=True,
                fraud_loss_if_allowed=0.0,
                friction_cost_if_verified=40.0,
                review_cost_if_escalated=150.0,
                recovery_if_escalated=0.0,
            ),
            risk_score=0.76,
            scenario="HIGH_VALUE_LEGITIMATE",
        )
    )

    # --------------------------------------------------------
    # 7. LOW-VALUE FRAUD
    # --------------------------------------------------------

    cases.append(
        StressCase(
            economic_case=EconomicCase(
                case_id="STRESS_007",
                requested_amount=600.0,
                is_legitimate=False,
                fraud_loss_if_allowed=600.0,
                friction_cost_if_verified=25.0,
                review_cost_if_escalated=100.0,
                recovery_if_escalated=480.0,
            ),
            risk_score=0.38,
            scenario="LOW_VALUE_FRAUD",
        )
    )

    # --------------------------------------------------------
    # 8. LEGITIMATE SHARED-INFRASTRUCTURE CUSTOMER
    # --------------------------------------------------------

    cases.append(
        StressCase(
            economic_case=EconomicCase(
                case_id="STRESS_008",
                requested_amount=2500.0,
                is_legitimate=True,
                fraud_loss_if_allowed=0.0,
                friction_cost_if_verified=25.0,
                review_cost_if_escalated=100.0,
                recovery_if_escalated=0.0,
            ),
            risk_score=0.48,
            scenario="LEGITIMATE_SHARED_DEVICE",
        )
    )

    # --------------------------------------------------------
    # 9. COORDINATED FRAUD WITH LOW BEHAVIORAL RISK
    # --------------------------------------------------------

    cases.append(
        StressCase(
            economic_case=EconomicCase(
                case_id="STRESS_009",
                requested_amount=7200.0,
                is_legitimate=False,
                fraud_loss_if_allowed=7200.0,
                friction_cost_if_verified=25.0,
                review_cost_if_escalated=100.0,
                recovery_if_escalated=5760.0,
            ),
            risk_score=0.28,
            scenario="COORDINATED_LOW_BEHAVIORAL_RISK",
        )
    )

    # --------------------------------------------------------
    # 10. STRATEGIC BUT LEGITIMATE CUSTOMER
    # --------------------------------------------------------

    cases.append(
        StressCase(
            economic_case=EconomicCase(
                case_id="STRESS_010",
                requested_amount=5000.0,
                is_legitimate=True,
                fraud_loss_if_allowed=0.0,
                friction_cost_if_verified=25.0,
                review_cost_if_escalated=100.0,
                recovery_if_escalated=0.0,
            ),
            risk_score=0.63,
            scenario="LEGITIMATE_STRATEGIC",
        )
    )

    return cases


# ============================================================
# POLICY BUILDERS
# ============================================================

def allow_all_policy(
    cases: list[StressCase],
) -> list[str]:

    return [
        "ALLOW_AGENT_A_DECISION"
        for _ in cases
    ]


def verify_all_policy(
    cases: list[StressCase],
) -> list[str]:

    return [
        "REQUEST_ADDITIONAL_EVIDENCE"
        for _ in cases
    ]


def escalate_all_policy(
    cases: list[StressCase],
) -> list[str]:

    return [
        "ESCALATE_TO_HUMAN_REVIEW"
        for _ in cases
    ]


def governor_policy(
    cases: list[StressCase],
) -> list[str]:

    return risk_based_policy(
        [
            case.risk_score
            for case in cases
        ],
        verify_threshold=VERIFY_THRESHOLD,
        escalate_threshold=ESCALATE_THRESHOLD,
    )


# ============================================================
# POLICY RESULT
# ============================================================

@dataclass(frozen=True)
class StressMetrics:

    policy_name: str

    total_cases: int

    fraud_cases: int

    legitimate_cases: int

    fraud_exposure: float

    loss_prevented: float

    remaining_loss: float

    legitimate_refunds_preserved: float

    friction_cost: float

    review_cost: float

    false_interventions: int

    net_benefit: float

    missed_fraud_cases: int


# ============================================================
# EVALUATION
# ============================================================

def evaluate_policy(
    cases: list[StressCase],
    actions: list[str],
    policy_name: str,
) -> StressMetrics:

    model = EconomicModel()

    economic_cases = [
        case.economic_case
        for case in cases
    ]

    outcomes, metrics = model.evaluate_dataset(
        economic_cases,
        actions,
    )

    missed_fraud_cases = 0

    for case, outcome in zip(
        cases,
        outcomes,
    ):

        if (
            not case.economic_case.is_legitimate
            and outcome.loss_prevented == 0.0
        ):
            missed_fraud_cases += 1

    return StressMetrics(
        policy_name=policy_name,
        total_cases=metrics.total_cases,
        fraud_cases=metrics.fraud_cases,
        legitimate_cases=metrics.legitimate_cases,
        fraud_exposure=metrics.fraud_loss_exposure,
        loss_prevented=metrics.total_loss_prevented,
        remaining_loss=metrics.fraud_loss_remaining,
        legitimate_refunds_preserved=(
            metrics.total_legitimate_refunds_preserved
        ),
        friction_cost=metrics.total_friction_cost,
        review_cost=metrics.total_review_cost,
        false_interventions=metrics.false_interventions,
        net_benefit=-metrics.total_net_cost,
        missed_fraud_cases=missed_fraud_cases,
    )


# ============================================================
# PRINTING
# ============================================================

def print_policy_result(
    result: StressMetrics,
) -> None:

    false_intervention_rate = (
        result.false_interventions
        / result.legitimate_cases
        if result.legitimate_cases > 0
        else 0.0
    )

    loss_prevention_rate = (
        result.loss_prevented
        / result.fraud_exposure
        if result.fraud_exposure > 0
        else 0.0
    )

    print(
        f"{result.policy_name:<24}"
        f"₹ {result.net_benefit:>11,.2f}"
        f"{loss_prevention_rate * 100:>15.2f}%"
        f"{false_intervention_rate * 100:>18.2f}%"
        f"{result.missed_fraud_cases:>15}"
    )


# ============================================================
# CASE-LEVEL ANALYSIS
# ============================================================

def print_case_analysis(
    cases: list[StressCase],
    actions: list[str],
) -> None:

    model = EconomicModel()

    print()
    print("=" * 95)
    print("CASE-LEVEL ECONOMIC STRESS ANALYSIS")
    print("=" * 95)

    print(
        f"{'CASE':<13}"
        f"{'SCENARIO':<34}"
        f"{'TRUTH':<10}"
        f"{'RISK':>8}"
        f"{'ACTION':<32}"
        f"{'BENEFIT':>12}"
    )

    print("-" * 95)

    for case, action in zip(
        cases,
        actions,
    ):

        outcome = model.evaluate_case(
            case.economic_case,
            action,
        )

        truth = (
            "LEGIT"
            if case.economic_case.is_legitimate
            else "FRAUD"
        )

        print(
            f"{case.economic_case.case_id:<13}"
            f"{case.scenario:<34}"
            f"{truth:<10}"
            f"{case.risk_score:>8.2f}"
            f"{action:<32}"
            f"₹ {(-outcome.net_cost):>9.2f}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    cases = generate_stress_cases()

    policies: dict[str, Callable] = {
        "ALLOW_ALL": allow_all_policy,
        "VERIFY_ALL": verify_all_policy,
        "ESCALATE_ALL": escalate_all_policy,
        "CURRENT_GOVERNOR": governor_policy,
    }

    results: list[StressMetrics] = []

    for name, builder in policies.items():

        actions = builder(cases)

        result = evaluate_policy(
            cases,
            actions,
            name,
        )

        results.append(result)

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()
    print("=" * 95)
    print("PHASE 13.5 — ECONOMIC STRESS / HARD-CASE EVALUATION")
    print("=" * 95)

    print()
    print(
        "Purpose:"
    )

    print(
        "Evaluate the Governor when risk estimates are imperfect "
        "and legitimate/fraud cases deliberately overlap."
    )

    print()
    print(
        f"Stress cases : {len(cases)}"
    )

    # --------------------------------------------------------
    # POLICY COMPARISON
    # --------------------------------------------------------

    print()
    print("=" * 95)
    print("POLICY COMPARISON")
    print("=" * 95)

    print(
        f"{'POLICY':<24}"
        f"{'NET BENEFIT':>16}"
        f"{'LOSS PREVENTED':>18}"
        f"{'FALSE INTERVENTION':>22}"
        f"{'MISSED FRAUD':>15}"
    )

    print("-" * 95)

    for result in results:

        print_policy_result(
            result
        )

    # --------------------------------------------------------
    # GOVERNOR
    # --------------------------------------------------------

    governor_result = next(
        result
        for result in results
        if result.policy_name
        == "CURRENT_GOVERNOR"
    )

    print()
    print("=" * 95)
    print("CURRENT GOVERNOR ECONOMIC OUTCOME")
    print("=" * 95)

    print(
        f"Fraud exposure              : "
        f"₹ {governor_result.fraud_exposure:,.2f}"
    )

    print(
        f"Loss prevented               : "
        f"₹ {governor_result.loss_prevented:,.2f}"
    )

    print(
        f"Remaining fraud loss         : "
        f"₹ {governor_result.remaining_loss:,.2f}"
    )

    print(
        f"Legitimate refunds preserved : "
        f"₹ {governor_result.legitimate_refunds_preserved:,.2f}"
    )

    print(
        f"Friction cost                : "
        f"₹ {governor_result.friction_cost:,.2f}"
    )

    print(
        f"Human review cost            : "
        f"₹ {governor_result.review_cost:,.2f}"
    )

    print(
        f"False interventions          : "
        f"{governor_result.false_interventions}"
    )

    print(
        f"Missed fraud cases           : "
        f"{governor_result.missed_fraud_cases}"
    )

    print(
        f"Net economic benefit         : "
        f"₹ {governor_result.net_benefit:,.2f}"
    )

    # --------------------------------------------------------
    # CASE ANALYSIS
    # --------------------------------------------------------

    current_actions = governor_policy(
        cases
    )

    print_case_analysis(
        cases,
        current_actions,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print("=" * 95)
    print("VALIDATION")
    print("=" * 95)

    governor_loss_rate = (
        governor_result.loss_prevented
        / governor_result.fraud_exposure
        if governor_result.fraud_exposure > 0
        else 0.0
    )

    governor_false_rate = (
        governor_result.false_interventions
        / governor_result.legitimate_cases
        if governor_result.legitimate_cases > 0
        else 0.0
    )

    checks = {

        # Dataset exists.
        "stress_cases_generated": (
            len(cases) == 10
        ),

        # Both classes exist.
        "legitimate_cases_present": (
            governor_result.legitimate_cases > 0
        ),

        "fraud_cases_present": (
            governor_result.fraud_cases > 0
        ),

        # Risk overlap exists.
        "risk_overlap_exists": (
            any(
                case.economic_case.is_legitimate
                and case.risk_score >= 0.70
                for case in cases
            )
            and any(
                not case.economic_case.is_legitimate
                and case.risk_score < 0.30
                for case in cases
            )
        ),

        # The stress dataset contains a legitimate
        # false-positive opportunity.
        "false_positive_opportunity_exists": (
            any(
                case.economic_case.is_legitimate
                and case.risk_score >= 0.70
                for case in cases
            )
        ),

        # The stress dataset contains a missed-fraud
        # opportunity.
        "missed_fraud_opportunity_exists": (
            any(
                not case.economic_case.is_legitimate
                and case.risk_score < 0.30
                for case in cases
            )
        ),

        # Economic quantities remain valid.
        "loss_prevention_bounded": (
            0.0
            <= governor_loss_rate
            <= 1.0
        ),

        "false_intervention_bounded": (
            0.0
            <= governor_false_rate
            <= 1.0
        ),

        "remaining_loss_non_negative": (
            governor_result.remaining_loss >= 0.0
        ),

        "benefit_is_finite": (
            governor_result.net_benefit
            == governor_result.net_benefit
        ),
    }

    all_passed = True

    for name, passed in checks.items():

        status = (
            "PASSED"
            if passed
            else "FAILED"
        )

        print(
            f"{name:<55}: {status}"
        )

        if not passed:
            all_passed = False

    print()

    print(
        "Overall validation       : "
        + (
            "PASSED"
            if all_passed
            else "FAILED"
        )
    )

    print()

    if not all_passed:

        raise RuntimeError(
            "Phase 13.5 validation failed."
        )

    print(
        "PHASE 13.5 ECONOMIC STRESS VALIDATION COMPLETE"
    )


if __name__ == "__main__":
    main()