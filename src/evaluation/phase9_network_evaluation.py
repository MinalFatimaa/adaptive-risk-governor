from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from random import Random

from src.governor.risk_fusion import RiskFusionEngine
from src.governor.risk_governor import AdaptiveRiskGovernor


# ============================================================
# EVALUATION RESULT
# ============================================================


@dataclass
class EvaluationCase:

    case_id: str

    scenario: str

    governor_risk: float

    network_risk: float

    fused_risk: float

    risk_level: str

    coordination_detected: bool

    reason_codes: list[str]


# ============================================================
# NETWORK FEATURE GENERATION
# ============================================================


def normal_network_features() -> dict[str, float]:
    """
    Represents a normal customer/network.

    The customer may share infrastructure with other
    legitimate customers occasionally, but there is no
    strong coordinated pattern.
    """

    return {
        "ip_reuse_score": 0.05,
        "device_reuse_score": 0.08,
        "payment_reuse_score": 0.04,
        "address_reuse_score": 0.10,
        "refund_velocity_score": 0.12,
        "claim_similarity_score": 0.08,
        "network_abnormality_score": 0.05,
    }


def coordinated_network_features() -> dict[str, float]:
    """
    Represents a coordinated/refund-abuse network.

    Multiple customers share infrastructure and exhibit
    correlated behavior.
    """

    return {
        "ip_reuse_score": 0.95,
        "device_reuse_score": 0.92,
        "payment_reuse_score": 0.88,
        "address_reuse_score": 0.90,
        "refund_velocity_score": 0.86,
        "claim_similarity_score": 0.91,
        "network_abnormality_score": 0.93,
    }


# ============================================================
# STRATEGIC STATE
# ============================================================


class NormalStrategicState:
    """
    Weak/no evidence of strategic adaptation.
    """

    interaction_history: list[dict[str, Any]] = []

    inferred_policy = {
        "evidence_sensitivity": 0.52,
        "amount_sensitivity": 0.50,
        "high_value_escalation": 0.51,
        "followup_sensitivity": 0.49,
    }


class CoordinatedStrategicState:
    """
    Strong learned evidence that the customer has adapted
    to the support policy.
    """

    interaction_history = [
        {
            "support_decision": "REQUEST_EVIDENCE",
            "claim_type": "WRONG_ITEM_CLAIM",
        }
        for _ in range(20)
    ]

    inferred_policy = {
        "evidence_sensitivity": 0.95,
        "amount_sensitivity": 0.82,
        "high_value_escalation": 0.88,
        "followup_sensitivity": 0.91,
    }


# ============================================================
# EVALUATION ENGINE
# ============================================================


class NetworkEvaluationEngine:

    def __init__(
        self,
        *,
        seed: int = 42,
    ):

        self.rng = Random(seed)

        self.governor = AdaptiveRiskGovernor()

        self.fusion = RiskFusionEngine()

    # ========================================================
    # SINGLE CASE
    # ========================================================

    def evaluate_case(
        self,
        *,
        case_id: str,
        scenario: str,
        customer_history: list[dict[str, Any]],
        claim_type: str,
        requested_amount: float,
        evidence_available: list[str],
        strategic_state: Any,
        network_features: dict[str, float],
    ) -> EvaluationCase:

        # ----------------------------------------------------
        # Agent A / behavioral governor
        # ----------------------------------------------------

        governor_decision = self.governor.evaluate(

            customer_history=customer_history,

            current_claim_type=claim_type,

            requested_amount=requested_amount,

            evidence_available=evidence_available,

            strategic_state=strategic_state,
        )

        # ----------------------------------------------------
        # Risk fusion
        # ----------------------------------------------------

        fused = self.fusion.fuse(

            governor_decision=governor_decision,

            network_features=network_features,
        )

        # ----------------------------------------------------
        # Coordination detection
        # ----------------------------------------------------

        coordination_detected = bool(
            fused.metadata.get(
                "coordination_detected",
                False,
            )
        )

        return EvaluationCase(

            case_id=case_id,

            scenario=scenario,

            governor_risk=float(
                governor_decision.risk_score
            ),

            network_risk=float(
                fused.network_risk
            ),

            fused_risk=float(
                fused.fused_risk_score
            ),

            risk_level=fused.risk_level,

            coordination_detected=(
                coordination_detected
            ),

            reason_codes=list(
                fused.reason_codes
            ),
        )


# ============================================================
# DATASET GENERATION
# ============================================================


def generate_evaluation_cases(
    *,
    n_normal: int = 25,
    n_coordinated: int = 25,
    seed: int = 42,
) -> list[EvaluationCase]:

    engine = NetworkEvaluationEngine(
        seed=seed
    )

    cases: list[EvaluationCase] = []

    # ========================================================
    # NORMAL CUSTOMERS
    # ========================================================

    for i in range(
        n_normal
    ):

        case = engine.evaluate_case(

            case_id=f"NORMAL_{i + 1:04d}",

            scenario="NORMAL",

            customer_history=[],

            claim_type="WRONG_ITEM_CLAIM",

            requested_amount=(
                1200.0 + (i % 5) * 300.0
            ),

            evidence_available=[
                "package_photo"
            ],

            strategic_state=(
                NormalStrategicState()
            ),

            network_features=(
                normal_network_features()
            ),
        )

        cases.append(case)

    # ========================================================
    # COORDINATED CUSTOMERS
    # ========================================================

    for i in range(
        n_coordinated
    ):

        case = engine.evaluate_case(

            case_id=(
                f"COORDINATED_{i + 1:04d}"
            ),

            scenario="COORDINATED",

            customer_history=[],

            claim_type="WRONG_ITEM_CLAIM",

            requested_amount=6000.0,

            evidence_available=[
                "package_photo"
            ],

            strategic_state=(
                CoordinatedStrategicState()
            ),

            network_features=(
                coordinated_network_features()
            ),
        )

        cases.append(case)

    return cases


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    cases: list[EvaluationCase],
) -> dict[str, float]:

    normal = [
        x
        for x in cases
        if x.scenario == "NORMAL"
    ]

    coordinated = [
        x
        for x in cases
        if x.scenario == "COORDINATED"
    ]

    if not normal:
        raise ValueError(
            "No normal evaluation cases."
        )

    if not coordinated:
        raise ValueError(
            "No coordinated evaluation cases."
        )

    normal_high_rate = (
        sum(
            x.risk_level == "HIGH"
            for x in normal
        )
        / len(normal)
    )

    coordinated_high_rate = (
        sum(
            x.risk_level == "HIGH"
            for x in coordinated
        )
        / len(coordinated)
    )

    coordination_detection_rate = (
        sum(
            x.coordination_detected
            for x in coordinated
        )
        / len(coordinated)
    )

    false_coordination_rate = (
        sum(
            x.coordination_detected
            for x in normal
        )
        / len(normal)
    )

    normal_mean_fused = (
        sum(
            x.fused_risk
            for x in normal
        )
        / len(normal)
    )

    coordinated_mean_fused = (
        sum(
            x.fused_risk
            for x in coordinated
        )
        / len(coordinated)
    )

    return {

        "normal_high_risk_rate": (
            normal_high_rate
        ),

        "coordinated_high_risk_rate": (
            coordinated_high_rate
        ),

        "coordination_detection_rate": (
            coordination_detection_rate
        ),

        "false_coordination_rate": (
            false_coordination_rate
        ),

        "normal_mean_fused_risk": (
            normal_mean_fused
        ),

        "coordinated_mean_fused_risk": (
            coordinated_mean_fused
        ),
    }


# ============================================================
# VALIDATION
# ============================================================


def validate_evaluation(
    cases: list[EvaluationCase],
) -> dict[str, bool]:

    metrics = calculate_metrics(
        cases
    )

    return {

        "cases_generated": (
            len(cases) > 0
        ),

        "normal_cases_present": (
            any(
                x.scenario == "NORMAL"
                for x in cases
            )
        ),

        "coordinated_cases_present": (
            any(
                x.scenario == "COORDINATED"
                for x in cases
            )
        ),

        "coordinated_network_detected": (
            metrics[
                "coordination_detection_rate"
            ]
            >= 0.90
        ),

        "normal_false_coordination_low": (
            metrics[
                "false_coordination_rate"
            ]
            <= 0.10
        ),

        "coordinated_risk_higher": (
            metrics[
                "coordinated_mean_fused_risk"
            ]
            > metrics[
                "normal_mean_fused_risk"
            ]
        ),

        "coordinated_high_risk_high": (
            metrics[
                "coordinated_high_risk_rate"
            ]
            >= 0.90
        ),
    }


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=" * 70
    )

    print(
        "PHASE 9 — NORMAL vs COORDINATED NETWORK EVALUATION"
    )

    print(
        "=" * 70
    )

    cases = generate_evaluation_cases(
        n_normal=25,
        n_coordinated=25,
        seed=42,
    )

    metrics = calculate_metrics(
        cases
    )

    validation = validate_evaluation(
        cases
    )

    print()

    print(
        "EVALUATION DATASET"
    )

    print(
        "-" * 70
    )

    print(
        f"Total cases             : {len(cases)}"
    )

    print(
        "Normal cases             : "
        f"{sum(x.scenario == 'NORMAL' for x in cases)}"
    )

    print(
        "Coordinated cases        : "
        f"{sum(x.scenario == 'COORDINATED' for x in cases)}"
    )

    print()

    print(
        "RISK COMPARISON"
    )

    print(
        "-" * 70
    )

    print(
        "Normal mean fused risk   : "
        f"{metrics['normal_mean_fused_risk']:.3f}"
    )

    print(
        "Coordinated mean risk    : "
        f"{metrics['coordinated_mean_fused_risk']:.3f}"
    )

    print()

    print(
        "DETECTION"
    )

    print(
        "-" * 70
    )

    print(
        "Normal HIGH-risk rate    : "
        f"{metrics['normal_high_risk_rate']:.1%}"
    )

    print(
        "Coordinated HIGH-risk    : "
        f"{metrics['coordinated_high_risk_rate']:.1%}"
    )

    print(
        "Coordination detection   : "
        f"{metrics['coordination_detection_rate']:.1%}"
    )

    print(
        "False coordination rate  : "
        f"{metrics['false_coordination_rate']:.1%}"
    )

    print()

    print(
        "VALIDATION"
    )

    print(
        "-" * 70
    )

    for name, passed in validation.items():

        print(
            f"{name:<35}: "
            f"{'PASSED' if passed else 'FAILED'}"
        )

    print()

    all_passed = all(
        validation.values()
    )

    print(
        "Overall validation       : "
        f"{'PASSED' if all_passed else 'FAILED'}"
    )

    print()

    # --------------------------------------------------------
    # Example cases
    # --------------------------------------------------------

    print(
        "EXAMPLE NORMAL CASE"
    )

    print(
        "-" * 70
    )

    normal_example = next(
        x
        for x in cases
        if x.scenario == "NORMAL"
    )

    print(
        f"Governor risk            : "
        f"{normal_example.governor_risk:.3f}"
    )

    print(
        f"Network risk             : "
        f"{normal_example.network_risk:.3f}"
    )

    print(
        f"Fused risk               : "
        f"{normal_example.fused_risk:.3f}"
    )

    print(
        f"Risk level               : "
        f"{normal_example.risk_level}"
    )

    print(
        f"Coordination detected    : "
        f"{normal_example.coordination_detected}"
    )

    print()

    print(
        "EXAMPLE COORDINATED CASE"
    )

    print(
        "-" * 70
    )

    coordinated_example = next(
        x
        for x in cases
        if x.scenario == "COORDINATED"
    )

    print(
        f"Governor risk            : "
        f"{coordinated_example.governor_risk:.3f}"
    )

    print(
        f"Network risk             : "
        f"{coordinated_example.network_risk:.3f}"
    )

    print(
        f"Fused risk               : "
        f"{coordinated_example.fused_risk:.3f}"
    )

    print(
        f"Risk level               : "
        f"{coordinated_example.risk_level}"
    )

    print(
        f"Coordination detected    : "
        f"{coordinated_example.coordination_detected}"
    )

    print(
        "Reason codes             : "
        f"{coordinated_example.reason_codes}"
    )

    print()

    if not all_passed:

        raise RuntimeError(
            "Phase 9 validation failed."
        )


if __name__ == "__main__":
    main()