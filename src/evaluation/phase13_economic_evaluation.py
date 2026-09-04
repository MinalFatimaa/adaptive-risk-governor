from __future__ import annotations

import random

from src.evaluation.economic_evaluation import (
    EconomicCase,
    EconomicModel,
    compare_policies,
    risk_based_policy,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
NUM_CASES = 1000


# ============================================================
# SYNTHETIC ECONOMIC DATASET
# ============================================================

def generate_cases(
    *,
    seed: int = SEED,
    count: int = NUM_CASES,
) -> list[EconomicCase]:

    rng = random.Random(seed)

    cases: list[EconomicCase] = []

    for index in range(count):

        is_legitimate = (
            rng.random() >= 0.20
        )

        amount = round(
            rng.uniform(
                500.0,
                8000.0,
            ),
            2,
        )

        if is_legitimate:

            fraud_loss = 0.0

        else:

            fraud_loss = amount

        cases.append(
            EconomicCase(
                case_id=(
                    f"CASE_{index:05d}"
                ),
                requested_amount=amount,
                is_legitimate=is_legitimate,
                fraud_loss_if_allowed=fraud_loss,
                friction_cost_if_verified=25.0,
                review_cost_if_escalated=100.0,
                recovery_if_escalated=fraud_loss,
            )
        )

    return cases


# ============================================================
# SYNTHETIC GOVERNOR RISK
# ============================================================

def generate_demo_risk_scores(
    cases: list[EconomicCase],
    *,
    seed: int = SEED,
) -> list[float]:

    rng = random.Random(seed)

    scores: list[float] = []

    for case in cases:

        if case.is_legitimate:

            # Deliberate overlap creates difficult legitimate
            # cases rather than perfect separation.

            score = rng.uniform(
                0.02,
                0.55,
            )

        else:

            # Fraud receives higher risk on average,
            # but overlap is deliberately retained.

            score = rng.uniform(
                0.35,
                0.98,
            )

        scores.append(
            round(
                max(
                    0.0,
                    min(
                        score,
                        1.0,
                    ),
                ),
                4,
            )
        )

    return scores


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results,
) -> None:

    print()
    print("=" * 78)
    print(
        "PHASE 13 — ECONOMIC CALIBRATION & "
        "FINANCIAL EVALUATION"
    )
    print("=" * 78)

    print()

    print(
        f"Evaluation cases : {NUM_CASES}"
    )

    print()

    print(
        "POLICY COMPARISON"
    )

    print("-" * 78)

    print(
        f"{'POLICY':<24}"
        f"{'NET BENEFIT':>16}"
        f"{'LOSS PREVENTED':>18}"
        f"{'FALSE INTERVENTION':>20}"
    )

    print("-" * 78)

    for result in results:

        print(
            f"{result.policy_name:<24}"
            f"₹ {result.net_benefit:>11,.2f}"
            f"{result.loss_prevention_rate * 100:>15.2f}%"
            f"{result.false_intervention_rate * 100:>17.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    cases = generate_cases()

    risk_scores = (
        generate_demo_risk_scores(
            cases
        )
    )

    model = EconomicModel()

    # ========================================================
    # BASELINES
    # ========================================================

    allow_all = [
        "ALLOW_AGENT_A_DECISION"
        for _ in cases
    ]

    verify_all = [
        "REQUEST_ADDITIONAL_EVIDENCE"
        for _ in cases
    ]

    escalate_all = [
        "ESCALATE_TO_HUMAN_REVIEW"
        for _ in cases
    ]

    # ========================================================
    # CURRENT RISK-THRESHOLD POLICY
    # ========================================================

    governor_policy = risk_based_policy(
        risk_scores,
        verify_threshold=0.30,
        escalate_threshold=0.70,
    )

    policies = {
        "ALLOW_ALL": allow_all,
        "VERIFY_ALL": verify_all,
        "ESCALATE_ALL": escalate_all,
        "RISK_THRESHOLD_POLICY": governor_policy,
    }

    results = compare_policies(
        cases,
        policies,
    )

    print_results(
        results
    )

    # ========================================================
    # DETAILED GOVERNOR POLICY
    # ========================================================

    _, metrics = model.evaluate_dataset(
        cases,
        governor_policy,
    )

    print()
    print("=" * 78)
    print("GOVERNOR ECONOMIC OUTCOME")
    print("=" * 78)

    print(
        f"Total cases                  : "
        f"{metrics.total_cases}"
    )

    print(
        f"Requested amount             : "
        f"₹ {metrics.total_requested_amount:,.2f}"
    )

    print(
        f"Fraud loss exposure          : "
        f"₹ {metrics.fraud_loss_exposure:,.2f}"
    )

    print(
        f"Loss prevented               : "
        f"₹ {metrics.total_loss_prevented:,.2f}"
    )

    print(
        f"Remaining fraud loss         : "
        f"₹ {metrics.fraud_loss_remaining:,.2f}"
    )

    print(
        f"Loss prevention rate         : "
        f"{metrics.loss_prevention_rate * 100:.2f}%"
    )

    print(
        f"Legitimate refunds preserved : "
        f"₹ {metrics.total_legitimate_refunds_preserved:,.2f}"
    )

    print(
        f"Friction cost                : "
        f"₹ {metrics.total_friction_cost:,.2f}"
    )

    print(
        f"Human review cost            : "
        f"₹ {metrics.total_review_cost:,.2f}"
    )

    print(
        f"False interventions          : "
        f"{metrics.false_interventions}"
    )

    print(
        f"False intervention rate      : "
        f"{metrics.false_intervention_rate * 100:.2f}%"
    )

    print(
        f"Net economic benefit         : "
        f"₹ {metrics.total_net_benefit:,.2f}"
    )

    print(
        f"Average benefit / case       : "
        f"₹ {metrics.average_benefit_per_case:,.2f}"
    )

    # ========================================================
    # ECONOMIC INVARIANTS
    # ========================================================

    print()
    print("=" * 78)
    print("ECONOMIC INVARIANTS")
    print("=" * 78)

    invariant_checks = {
        "remaining_loss_non_negative": (
            metrics.fraud_loss_remaining >= 0.0
        ),

        "prevented_not_above_exposure": (
            metrics.total_loss_prevented
            <= metrics.fraud_loss_exposure
            + 1e-9
        ),

        "legitimate_refunds_non_negative": (
            metrics.total_legitimate_refunds_preserved
            >= 0.0
        ),

        "friction_cost_non_negative": (
            metrics.total_friction_cost
            >= 0.0
        ),

        "review_cost_non_negative": (
            metrics.total_review_cost
            >= 0.0
        ),

        "loss_prevention_bounded": (
            0.0
            <= metrics.loss_prevention_rate
            <= 1.0
        ),

        "false_intervention_bounded": (
            0.0
            <= metrics.false_intervention_rate
            <= 1.0
        ),

        "economic_benefit_finite": (
            metrics.total_net_benefit
            == metrics.total_net_benefit
        ),
    }

    all_invariants_passed = True

    for name, passed in invariant_checks.items():

        status = (
            "PASSED"
            if passed
            else "FAILED"
        )

        print(
            f"{name:<45}: {status}"
        )

        if not passed:
            all_invariants_passed = False

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)

    checks = {

        "cases_generated": (
            len(cases) == NUM_CASES
        ),

        "risk_scores_generated": (
            len(risk_scores)
            == NUM_CASES
        ),

        "loss_prevention_bounded": (
            0.0
            <= metrics.loss_prevention_rate
            <= 1.0
        ),

        "false_intervention_bounded": (
            0.0
            <= metrics.false_intervention_rate
            <= 1.0
        ),

        "economic_cost_finite": (
            metrics.total_net_cost
            == metrics.total_net_cost
        ),

        "legitimate_cases_present": (
            metrics.legitimate_cases > 0
        ),

        "fraud_cases_present": (
            metrics.fraud_cases > 0
        ),

        "remaining_loss_non_negative": (
            metrics.fraud_loss_remaining
            >= 0.0
        ),

        "prevented_loss_not_above_exposure": (
            metrics.total_loss_prevented
            <= metrics.fraud_loss_exposure
            + 1e-9
        ),

        "net_benefit_available": (
            metrics.total_net_benefit
            == -metrics.total_net_cost
        ),
    }

    all_passed = (
        all_invariants_passed
        and all(checks.values())
    )

    for name, passed in checks.items():

        status = (
            "PASSED"
            if passed
            else "FAILED"
        )

        print(
            f"{name:<45}: {status}"
        )

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

    if all_passed:

        print(
            "PHASE 13 ECONOMIC VALIDATION COMPLETE"
        )

    else:

        raise RuntimeError(
            "Phase 13 economic validation failed."
        )


if __name__ == "__main__":
    main()