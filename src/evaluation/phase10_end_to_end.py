from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.governor.risk_governor import (
    AdaptiveRiskGovernor,
)

from src.governor.risk_fusion import (
    RiskFusionEngine,
)


# ============================================================
# PHASE 10
# END-TO-END ADAPTIVE RISK EVALUATION
# ============================================================

print("=" * 70)
print("PHASE 10 — END-TO-END ADAPTIVE RISK EVALUATION")
print("=" * 70)


# ============================================================
# SYNTHETIC STRATEGIC STATE
# ============================================================

@dataclass
class StrategicState:

    interaction_history: list[dict[str, Any]]

    inferred_policy: dict[str, float]


# ============================================================
# NETWORK PATTERN GENERATORS
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


def suspicious_network() -> dict[str, float]:

    return {
        "ip_reuse_score": 0.60,
        "device_reuse_score": 0.55,
        "payment_reuse_score": 0.30,
        "address_reuse_score": 0.65,
        "refund_velocity_score": 0.70,
        "claim_similarity_score": 0.45,
        "network_abnormality_score": 0.60,
    }


# ============================================================
# STRATEGIC STATES
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


def adaptive_strategic_state() -> StrategicState:

    history = [
        {
            "support_decision": "REQUEST_EVIDENCE"
        }
        for _ in range(20)
    ]

    return StrategicState(
        interaction_history=history,
        inferred_policy={
            "evidence_sensitivity": 0.90,
            "amount_sensitivity": 0.80,
            "high_value_escalation": 0.85,
            "followup_sensitivity": 0.90,
        },
    )


# ============================================================
# GOVERNOR
# ============================================================

def evaluate_governor(
    *,
    customer_history: list[dict[str, Any]],
    requested_amount: float,
    evidence_available: list[str],
    strategic_state: StrategicState,
):

    governor = AdaptiveRiskGovernor()

    return governor.evaluate(
        customer_history=customer_history,
        current_claim_type="WRONG_ITEM_CLAIM",
        requested_amount=requested_amount,
        evidence_available=evidence_available,
        strategic_state=strategic_state,
    )


# ============================================================
# FUSION ENGINE
# ============================================================

def evaluate_fusion(
    *,
    governor_decision,
    network_features,
):

    engine = RiskFusionEngine()

    return engine.fuse(
        governor_decision=governor_decision,
        network_features=network_features,
    )


# ============================================================
# CASE RUNNER
# ============================================================

def run_case(
    *,
    name: str,
    customer_history: list[dict[str, Any]],
    requested_amount: float,
    evidence_available: list[str],
    strategic_state: StrategicState,
    network_features: dict[str, float],
):

    governor_decision = evaluate_governor(
        customer_history=customer_history,
        requested_amount=requested_amount,
        evidence_available=evidence_available,
        strategic_state=strategic_state,
    )

    fused_decision = evaluate_fusion(
        governor_decision=governor_decision,
        network_features=network_features,
    )

    return {
        "name": name,
        "governor": governor_decision,
        "fused": fused_decision,
    }


# ============================================================
# CASE 1
# NORMAL CUSTOMER
# ============================================================

normal_case = run_case(
    name="NORMAL",
    customer_history=[],
    requested_amount=1200,
    evidence_available=[
        "package_photo"
    ],
    strategic_state=normal_strategic_state(),
    network_features=normal_network(),
)


# ============================================================
# CASE 2
# STRATEGIC CUSTOMER
# ============================================================

strategic_case = run_case(
    name="STRATEGIC",
    customer_history=[],
    requested_amount=6500,
    evidence_available=[
        "package_photo"
    ],
    strategic_state=adaptive_strategic_state(),
    network_features=normal_network(),
)


# ============================================================
# CASE 3
# COORDINATED NETWORK
# ============================================================

coordinated_case = run_case(
    name="COORDINATED_NETWORK",
    customer_history=[],
    requested_amount=1200,
    evidence_available=[
        "package_photo"
    ],
    strategic_state=normal_strategic_state(),
    network_features=coordinated_network(),
)


# ============================================================
# CASE 4
# STRATEGIC + COORDINATED
# ============================================================

strategic_coordinated_case = run_case(
    name="STRATEGIC_AND_COORDINATED",
    customer_history=[
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
    requested_amount=8500,
    evidence_available=[
        "package_photo"
    ],
    strategic_state=adaptive_strategic_state(),
    network_features=coordinated_network(),
)


# ============================================================
# CASE 5
# PARTIALLY SUSPICIOUS
# ============================================================

suspicious_case = run_case(
    name="SUSPICIOUS",
    customer_history=[
        {
            "support_decision": "REQUEST_EVIDENCE",
            "claim_type": "SHORTAGE_CLAIM",
        },
        {
            "support_decision": "REQUEST_EVIDENCE",
            "claim_type": "SHORTAGE_CLAIM",
        },
    ],
    requested_amount=4500,
    evidence_available=[],
    strategic_state=normal_strategic_state(),
    network_features=suspicious_network(),
)


# ============================================================
# DISPLAY
# ============================================================

cases = [
    normal_case,
    strategic_case,
    coordinated_case,
    strategic_coordinated_case,
    suspicious_case,
]


print()
print("=" * 70)
print("END-TO-END RISK COMPARISON")
print("=" * 70)

print(
    f"{'CASE':30}"
    f"{'GOVERNOR':>12}"
    f"{'NETWORK':>12}"
    f"{'FUSED':>12}"
    f"{'LEVEL':>10}"
)

print("-" * 70)

for case in cases:

    governor = case["governor"]
    fused = case["fused"]

    print(
        f"{case['name']:30}"
        f"{governor.risk_score:>12.3f}"
        f"{fused.network_risk:>12.3f}"
        f"{fused.fused_risk_score:>12.3f}"
        f"{fused.risk_level:>10}"
    )


# ============================================================
# DETAILED RESULTS
# ============================================================

print()
print("=" * 70)
print("DETAILED RESULTS")
print("=" * 70)


for case in cases:

    governor = case["governor"]
    fused = case["fused"]

    print()
    print("-" * 70)

    print(
        f"CASE: {case['name']}"
    )

    print(
        f"Governor risk        : "
        f"{governor.risk_score:.3f}"
    )

    print(
        f"Network risk         : "
        f"{fused.network_risk:.3f}"
    )

    print(
        f"Fused risk           : "
        f"{fused.fused_risk_score:.3f}"
    )

    print(
        f"Risk level           : "
        f"{fused.risk_level}"
    )

    print(
        f"Coordination         : "
        f"{fused.metadata.get('coordination_detected', False)}"
    )

    print(
        f"Action               : "
        f"{getattr(fused, 'action', 'N/A')}"
    )

    print(
        f"Reasons              : "
        f"{fused.reason_codes}"
    )


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("VALIDATION")
print("=" * 70)


validation = {}


# ------------------------------------------------------------
# 1. All cases generated
# ------------------------------------------------------------

validation[
    "cases_generated"
] = (
    len(cases) == 5
)


# ------------------------------------------------------------
# 2. Normal should remain low
# ------------------------------------------------------------

validation[
    "normal_case_low_risk"
] = (
    normal_case["fused"].risk_level
    == "LOW"
)


# ------------------------------------------------------------
# 3. Strategic behavior should increase
# ------------------------------------------------------------

validation[
    "strategic_risk_detected"
] = (
    strategic_case["fused"].fused_risk_score
    > normal_case["fused"].fused_risk_score
)


# ------------------------------------------------------------
# 4. Coordinated network should be detected
# ------------------------------------------------------------

validation[
    "coordination_detected"
] = (
    coordinated_case["fused"].metadata.get(
        "coordination_detected",
        False,
    )
)


# ------------------------------------------------------------
# 5. Coordinated network should be high risk
# ------------------------------------------------------------

validation[
    "coordinated_case_high_risk"
] = (
    coordinated_case["fused"].risk_level
    == "HIGH"
)


# ------------------------------------------------------------
# 6. Strategic + coordinated should be high
# ------------------------------------------------------------

validation[
    "strategic_coordinated_high_risk"
] = (
    strategic_coordinated_case[
        "fused"
    ].risk_level
    == "HIGH"
)


# ------------------------------------------------------------
# 7. Combined case should exceed normal
# ------------------------------------------------------------

validation[
    "combined_risk_higher_than_normal"
] = (
    strategic_coordinated_case[
        "fused"
    ].fused_risk_score
    > normal_case["fused"].fused_risk_score
)


# ------------------------------------------------------------
# 8. Scores bounded
# ------------------------------------------------------------

validation[
    "scores_bounded"
] = all(
    0.0
    <= case["fused"].fused_risk_score
    <= 1.0
    for case in cases
)


# ============================================================
# VALIDATION OUTPUT
# ============================================================

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
        "Phase 10 validation failed."
    )


print()
print("=" * 70)
print("PHASE 10 VALIDATION COMPLETE")
print("=" * 70)