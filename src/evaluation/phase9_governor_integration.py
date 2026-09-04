from __future__ import annotations

from dataclasses import dataclass


from src.governor.risk_governor import (
    AdaptiveRiskGovernor,
)

from src.governor.risk_fusion import (
    RiskFusionEngine,
)


# ============================================================
# SCENARIO
# ============================================================

@dataclass
class Scenario:

    name: str

    network_features: dict[str, float]

    requested_amount: float = 2000.0

    evidence_available: list[str] | None = None

    strategic_state: object | None = None


# ============================================================
# NETWORK FEATURES
# ============================================================

NORMAL_NETWORK = {

    "ip_reuse_score": 0.10,

    "device_reuse_score": 0.08,

    "payment_reuse_score": 0.05,

    "address_reuse_score": 0.20,

    "refund_velocity_score": 0.10,

    "claim_similarity_score": 0.15,

    "network_abnormality_score": 0.05,
}


COORDINATED_NETWORK = {

    "ip_reuse_score": 0.92,

    "device_reuse_score": 0.94,

    "payment_reuse_score": 0.86,

    "address_reuse_score": 0.82,

    "refund_velocity_score": 0.91,

    "claim_similarity_score": 0.88,

    "network_abnormality_score": 0.93,
}


# ============================================================
# SCENARIOS
# ============================================================

def create_scenarios():

    return [

        Scenario(
            name="NORMAL_CUSTOMER",
            network_features=(
                NORMAL_NETWORK.copy()
            ),
            requested_amount=2000.0,
            evidence_available=[
                "package_photo"
            ],
        ),

        Scenario(
            name="COORDINATED_CUSTOMER",
            network_features=(
                COORDINATED_NETWORK.copy()
            ),
            requested_amount=2000.0,
            evidence_available=[],
        ),
    ]


# ============================================================
# RUN ONE SCENARIO
# ============================================================

def evaluate_scenario(
    scenario: Scenario,
):

    governor = AdaptiveRiskGovernor()

    fusion = RiskFusionEngine()

    governor_decision = governor.evaluate(

        customer_history=[],

        current_claim_type=(
            "WRONG_ITEM_CLAIM"
        ),

        requested_amount=(
            scenario.requested_amount
        ),

        evidence_available=(
            scenario.evidence_available
            or []
        ),

        strategic_state=(
            scenario.strategic_state
        ),
    )

    fused = fusion.fuse(

        governor_decision=(
            governor_decision
        ),

        network_features=(
            scenario.network_features
        ),
    )

    return governor_decision, fused


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PHASE 9 — GOVERNOR + EXTERNAL OBSERVABILITY"
    )
    print("=" * 70)

    scenarios = create_scenarios()

    results = []

    for scenario in scenarios:

        governor_decision, fused = (
            evaluate_scenario(
                scenario
            )
        )

        results.append(
            (
                scenario,
                governor_decision,
                fused,
            )
        )

    # ========================================================
    # RESULTS
    # ========================================================

    for (
        scenario,
        governor_decision,
        fused,
    ) in results:

        print()
        print("=" * 70)

        print(
            scenario.name
        )

        print("-" * 70)

        print(
            f"Agent A / Governor risk : "
            f"{governor_decision.risk_score:.3f}"
        )

        print(
            f"Strategic risk          : "
            f"{fused.strategic_risk:.3f}"
        )

        print(
            f"Network risk             : "
            f"{fused.network_risk:.3f}"
        )

        print(
            f"Fused risk               : "
            f"{fused.fused_risk_score:.3f}"
        )

        print(
            f"Risk level               : "
            f"{fused.risk_level}"
        )

        print(
            f"Action                   : "
            f"{fused.action}"
        )

        print()

        print(
            "Network reasons:"
        )

        if fused.reason_codes:

            for reason in fused.reason_codes:

                print(
                    f"  - {reason}"
                )

        else:

            print(
                "  None"
            )

    # ========================================================
    # COMPARISON
    # ========================================================

    normal = results[0][2]

    coordinated = results[1][2]

    print()
    print("=" * 70)
    print(
        "NORMAL vs COORDINATED"
    )
    print("=" * 70)

    print(
        f"Normal network risk      : "
        f"{normal.network_risk:.3f}"
    )

    print(
        f"Coordinated network risk : "
        f"{coordinated.network_risk:.3f}"
    )

    print(
        f"Normal fused risk        : "
        f"{normal.fused_risk_score:.3f}"
    )

    print(
        f"Coordinated fused risk   : "
        f"{coordinated.fused_risk_score:.3f}"
    )

    print()

    if (
        coordinated.fused_risk_score
        > normal.fused_risk_score
    ):

        print(
            "Network integration: DETECTED"
        )

    else:

        print(
            "Network integration: FAILED"
        )

        raise AssertionError(
            "Coordinated scenario did not "
            "produce higher risk."
        )

    if (
        coordinated.risk_level
        == "HIGH"
    ):

        print(
            "Coordinated customer detection: PASSED"
        )

    else:

        print(
            "Coordinated customer detection: FAILED"
        )

        raise AssertionError(
            "Coordinated customer was not "
            "classified as HIGH risk."
        )

    if (
        normal.risk_level
        != "HIGH"
    ):

        print(
            "Normal customer false-positive check: PASSED"
        )

    else:

        print(
            "Normal customer false-positive check: FAILED"
        )

        raise AssertionError(
            "Normal customer incorrectly "
            "classified as HIGH risk."
        )

    print()
    print("=" * 70)
    print(
        "VALIDATION"
    )
    print("=" * 70)

    print(
        "Risk fusion        : PASSED"
    )

    print(
        "Network integration: PASSED"
    )

    print(
        "False-positive check: PASSED"
    )

    print()
    print(
        "Validation: PASSED"
    )


if __name__ == "__main__":

    main()