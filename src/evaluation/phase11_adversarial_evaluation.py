from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from src.governor.risk_governor import (
    AdaptiveRiskGovernor,
)

from src.governor.risk_fusion import (
    RiskFusionEngine,
)

from src.governor.temporal_features import (
    TemporalFeatureBuilder,
    TemporalObservation,
)


# ============================================================
# STRATEGIC STATE
# ============================================================

@dataclass
class StrategicState:

    interaction_history: list[dict[str, Any]]

    inferred_policy: dict[str, float]


# ============================================================
# CASE
# ============================================================

@dataclass
class EvaluationCase:

    name: str

    category: str

    history: list[dict[str, Any]]

    temporal_observations: list[TemporalObservation]

    strategic_state: StrategicState

    network_features: dict[str, float]

    requested_amount: float

    evidence_available: list[str]


# ============================================================
# NORMAL NETWORK
# ============================================================

def normal_network() -> dict[str, float]:

    return {
        "ip_reuse_score": 0.05,
        "device_reuse_score": 0.04,
        "payment_reuse_score": 0.03,
        "address_reuse_score": 0.05,
        "refund_velocity_score": 0.04,
        "claim_similarity_score": 0.05,
        "network_abnormality_score": 0.04,
    }


# ============================================================
# COORDINATED NETWORK
# ============================================================

def coordinated_network() -> dict[str, float]:

    return {
        "ip_reuse_score": 0.95,
        "device_reuse_score": 0.92,
        "payment_reuse_score": 0.94,
        "address_reuse_score": 0.90,
        "refund_velocity_score": 0.91,
        "claim_similarity_score": 0.93,
        "network_abnormality_score": 0.95,
    }


# ============================================================
# NORMAL STRATEGIC STATE
# ============================================================

def normal_strategic_state() -> StrategicState:

    return StrategicState(
        interaction_history=[],
        inferred_policy={
            "evidence_sensitivity": 0.50,
            "amount_sensitivity": 0.50,
            "high_value_escalation": 0.50,
            "followup_sensitivity": 0.50,
        },
    )


# ============================================================
# STRATEGIC STATE
# ============================================================

def strategic_state() -> StrategicState:

    return StrategicState(
        interaction_history=[
            {
                "support_decision": "REQUEST_EVIDENCE",
                "claim_type": "WRONG_ITEM_CLAIM",
            }
            for _ in range(20)
        ],
        inferred_policy={
            "evidence_sensitivity": 0.90,
            "amount_sensitivity": 0.80,
            "high_value_escalation": 0.85,
            "followup_sensitivity": 0.90,
        },
    )


# ============================================================
# NORMAL TEMPORAL PATTERN
# ============================================================

def normal_temporal_observations() -> list[
    TemporalObservation
]:

    base = datetime(
        2026,
        8,
        1,
        12,
        0,
        0,
    )

    return [
        TemporalObservation(
            customer_id="NORMAL_001",
            timestamp=base,
            claim_type="WRONG_ITEM_CLAIM",
            requested_amount=1200,
            support_decision="APPROVE",
        ),

        TemporalObservation(
            customer_id="NORMAL_001",
            timestamp=(
                base
                + timedelta(days=7)
            ),
            claim_type="SHORTAGE_CLAIM",
            requested_amount=1500,
            support_decision="APPROVE",
        ),

        TemporalObservation(
            customer_id="NORMAL_001",
            timestamp=(
                base
                + timedelta(days=15)
            ),
            claim_type="WRONG_ITEM_CLAIM",
            requested_amount=1800,
            support_decision="APPROVE",
        ),
    ]


# ============================================================
# BURST / ADVERSARIAL TEMPORAL PATTERN
# ============================================================

def adversarial_temporal_observations() -> list[
    TemporalObservation
]:

    base = datetime(
        2026,
        8,
        26,
        12,
        0,
        0,
    )

    observations = []

    amounts = [
        1000,
        1200,
        1600,
        2200,
        3500,
        5000,
    ]

    claims = [
        "SHORTAGE_CLAIM",
        "WRONG_ITEM_CLAIM",
        "NON_DELIVERY_CLAIM",
        "SUBSTITUTED_RETURN_CLAIM",
        "SHORTAGE_CLAIM",
        "WRONG_ITEM_CLAIM",
    ]

    for index, (
        amount,
        claim,
    ) in enumerate(
        zip(
            amounts,
            claims,
        )
    ):

        observations.append(
            TemporalObservation(
                customer_id="ADV_001",
                timestamp=(
                    base
                    + timedelta(
                        minutes=index * 10
                    )
                ),
                claim_type=claim,
                requested_amount=amount,
                support_decision="DENY",
            )
        )

    return observations


# ============================================================
# EVALUATE CASE
# ============================================================

def evaluate_case(
    case: EvaluationCase,
) -> dict[str, Any]:

    governor = AdaptiveRiskGovernor()

    temporal_builder = (
        TemporalFeatureBuilder()
    )

    temporal_features = (
        temporal_builder.build(
            case.temporal_observations
        )
    )

    governor_decision = governor.evaluate(
        customer_history=case.history,
        current_claim_type="WRONG_ITEM_CLAIM",
        requested_amount=case.requested_amount,
        evidence_available=(
            case.evidence_available
        ),
        strategic_state=case.strategic_state,
    )

    fusion = RiskFusionEngine()

    # --------------------------------------------------------
    # Add temporal abnormality to the network signal.
    #
    # This keeps the existing RiskFusionEngine unchanged.
    # --------------------------------------------------------

    network_features = dict(
        case.network_features
    )

    temporal_abnormality = (
        temporal_features[
            "temporal_abnormality_score"
        ]
    )

    network_features[
        "network_abnormality_score"
    ] = max(
        network_features.get(
            "network_abnormality_score",
            0.0,
        ),
        temporal_abnormality,
    )

    fused = fusion.fuse(
        governor_decision=(
            governor_decision
        ),
        network_features=(
            network_features
        ),
    )

    return {
        "case": case,
        "governor": governor_decision,
        "temporal": temporal_features,
        "fused": fused,
    }


# ============================================================
# BUILD CASES
# ============================================================

def build_cases() -> list[
    EvaluationCase
]:

    normal_case = EvaluationCase(
        name="NORMAL_CUSTOMER",
        category="NORMAL",
        history=[],
        temporal_observations=(
            normal_temporal_observations()
        ),
        strategic_state=(
            normal_strategic_state()
        ),
        network_features=(
            normal_network()
        ),
        requested_amount=1500,
        evidence_available=[
            "package_photo"
        ],
    )

    strategic_case = EvaluationCase(
        name="STRATEGIC_CUSTOMER",
        category="STRATEGIC",
        history=[],
        temporal_observations=(
            normal_temporal_observations()
        ),
        strategic_state=(
            strategic_state()
        ),
        network_features=(
            normal_network()
        ),
        requested_amount=6500,
        evidence_available=[
            "package_photo"
        ],
    )

    coordinated_case = EvaluationCase(
        name="COORDINATED_CUSTOMER",
        category="COORDINATED",
        history=[],
        temporal_observations=(
            normal_temporal_observations()
        ),
        strategic_state=(
            normal_strategic_state()
        ),
        network_features=(
            coordinated_network()
        ),
        requested_amount=1500,
        evidence_available=[
            "package_photo"
        ],
    )

    temporal_case = EvaluationCase(
        name="TEMPORAL_ABUSE",
        category="TEMPORAL",
        history=[],
        temporal_observations=(
            adversarial_temporal_observations()
        ),
        strategic_state=(
            normal_strategic_state()
        ),
        network_features=(
            normal_network()
        ),
        requested_amount=5000,
        evidence_available=[],
    )

    combined_case = EvaluationCase(
        name="STRATEGIC_NETWORK_TEMPORAL",
        category="COMBINED",
        history=[
            {
                "support_decision": "DENY",
                "claim_type": "SHORTAGE_CLAIM",
            },
            {
                "support_decision": "REQUEST_EVIDENCE",
                "claim_type": "NON_DELIVERY_CLAIM",
            },
            {
                "support_decision": "DENY",
                "claim_type": "WRONG_ITEM_CLAIM",
            },
        ],
        temporal_observations=(
            adversarial_temporal_observations()
        ),
        strategic_state=(
            strategic_state()
        ),
        network_features=(
            coordinated_network()
        ),
        requested_amount=8500,
        evidence_available=[],
    )

    return [
        normal_case,
        strategic_case,
        coordinated_case,
        temporal_case,
        combined_case,
    ]


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results: list[dict[str, Any]],
) -> None:

    print()

    print("=" * 70)
    print(
        "PHASE 11 — ADVERSARIAL / TEMPORAL EVALUATION"
    )
    print("=" * 70)

    print()

    print(
        f"{'CASE':35}"
        f"{'TEMPORAL':>12}"
        f"{'NETWORK':>12}"
        f"{'FUSED':>12}"
        f"{'LEVEL':>10}"
    )

    print(
        "-" * 70
    )

    for result in results:

        case = result["case"]
        temporal = result["temporal"]
        fused = result["fused"]

        print(
            f"{case.name:35}"
            f"{temporal['temporal_abnormality_score']:>12.3f}"
            f"{fused.network_risk:>12.3f}"
            f"{fused.fused_risk_score:>12.3f}"
            f"{fused.risk_level:>10}"
        )


# ============================================================
# ABLATION
# ============================================================

def run_ablation(
    case: EvaluationCase,
    *,
    remove_temporal: bool = False,
    remove_network: bool = False,
    remove_strategic: bool = False,
) -> float:

    modified_temporal = (
        []
        if remove_temporal
        else case.temporal_observations
    )

    modified_network = dict(
        case.network_features
    )

    if remove_network:

        for key in modified_network:

            modified_network[key] = 0.0

    modified_strategy = (
        normal_strategic_state()
        if remove_strategic
        else case.strategic_state
    )

    modified_case = EvaluationCase(
        name=case.name,
        category=case.category,
        history=(
            []
            if remove_strategic
            else case.history
        ),
        temporal_observations=(
            modified_temporal
        ),
        strategic_state=(
            modified_strategy
        ),
        network_features=(
            modified_network
        ),
        requested_amount=(
            case.requested_amount
        ),
        evidence_available=(
            case.evidence_available
        ),
    )

    result = evaluate_case(
        modified_case
    )

    return float(
        result["fused"].fused_risk_score
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    cases = build_cases()

    results = [
        evaluate_case(case)
        for case in cases
    ]

    print_results(
        results
    )

    # ========================================================
    # DETAILED TEMPORAL RESULTS
    # ========================================================

    print()

    print("=" * 70)
    print("TEMPORAL SIGNALS")
    print("=" * 70)

    for result in results:

        case = result["case"]
        temporal = result["temporal"]

        print()

        print(
            f"{case.name}"
        )

        print(
            f"Requests last 24h      : "
            f"{temporal['requests_last_24h']:.0f}"
        )

        print(
            f"Request burst score    : "
            f"{temporal['request_burst_score']:.3f}"
        )

        print(
            f"Claim switch rate      : "
            f"{temporal['claim_switch_rate']:.3f}"
        )

        print(
            f"Amount acceleration    : "
            f"{temporal['amount_acceleration_score']:.3f}"
        )

        print(
            f"Temporal abnormality   : "
            f"{temporal['temporal_abnormality_score']:.3f}"
        )

    # ========================================================
    # ABLATION
    # ========================================================

    combined_case = cases[-1]

    full_score = (
        results[-1][
            "fused"
        ].fused_risk_score
    )

    without_temporal = run_ablation(
        combined_case,
        remove_temporal=True,
    )

    without_network = run_ablation(
        combined_case,
        remove_network=True,
    )

    without_strategic = run_ablation(
        combined_case,
        remove_strategic=True,
    )

    without_all = run_ablation(
        combined_case,
        remove_temporal=True,
        remove_network=True,
        remove_strategic=True,
    )

    print()

    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    print(
        f"Full system risk       : "
        f"{full_score:.3f}"
    )

    print(
        f"Without temporal       : "
        f"{without_temporal:.3f}"
    )

    print(
        f"Without network        : "
        f"{without_network:.3f}"
    )

    print(
        f"Without strategic      : "
        f"{without_strategic:.3f}"
    )

    print(
        f"Without all three      : "
        f"{without_all:.3f}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    validation = {}

    normal_result = results[0]

    strategic_result = results[1]

    coordinated_result = results[2]

    temporal_result = results[3]

    combined_result = results[4]

    validation[
        "normal_low_risk"
    ] = (
        normal_result[
            "fused"
        ].risk_level
        != "HIGH"
    )

    validation[
        "strategic_detected"
    ] = (
        strategic_result[
            "fused"
        ].fused_risk_score
        >
        normal_result[
            "fused"
        ].fused_risk_score
    )

    validation[
        "network_detected"
    ] = (
        coordinated_result[
            "fused"
        ].risk_level
        == "HIGH"
    )

    validation[
        "temporal_signal_detected"
    ] = (
        temporal_result[
            "temporal"
        ]["temporal_abnormality_score"]
        > 0.50
    )

    validation[
        "temporal_increases_risk"
    ] = (
        temporal_result[
            "fused"
        ].fused_risk_score
        >
        normal_result[
            "fused"
        ].fused_risk_score
    )

    validation[
        "combined_high_risk"
    ] = (
        combined_result[
            "fused"
        ].risk_level
        == "HIGH"
    )

    validation[
        "temporal_ablation_changes_result"
    ] = (
        without_temporal
        < full_score
    )

    validation[
        "network_ablation_changes_result"
    ] = (
        without_network
        < full_score
    )

    validation[
        "strategic_ablation_changes_result"
    ] = (
        without_strategic
        < full_score
    )

    validation[
        "combined_ablation_changes_result"
    ] = (
        without_all
        < full_score
    )

    validation[
        "scores_bounded"
    ] = all(
        0.0
        <= result[
            "fused"
        ].fused_risk_score
        <= 1.0
        for result in results
    )

    print()

    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    for name, passed in validation.items():

        print(
            f"{name:40}"
            f": {'PASSED' if passed else 'FAILED'}"
        )

    overall = all(
        validation.values()
    )

    print()

    print(
        f"Overall validation       : "
        f"{'PASSED' if overall else 'FAILED'}"
    )

    if not overall:

        raise AssertionError(
            "Phase 11 validation failed."
        )

    print()

    print("=" * 70)
    print(
        "PHASE 11 VALIDATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()