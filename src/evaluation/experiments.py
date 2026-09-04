from src.agents.adaptive_customer import (
    AdaptiveCustomerAgent,
    AdaptiveCustomerState,
)


# ============================================================
# MOCK OBSERVABLE RESPONSE FROM AGENT A
# ============================================================

class Decision:

    def __init__(
        self,
        decision: str,
        reason_code: str,
        requested_evidence=None,
        approved_amount: float = 0.0,
        requires_followup: bool = False,
    ):

        self.decision = decision

        self.reason_code = reason_code

        self.requested_evidence = (
            requested_evidence or []
        )

        self.approved_amount = approved_amount

        self.requires_followup = (
            requires_followup
        )


# ============================================================
# CREATE ADAPTIVE CUSTOMER
# ============================================================

state = AdaptiveCustomerState(
    customer_id="ADAPTIVE_TEST",
    objective="maximize_illegitimate_refund_value",
)

agent = AdaptiveCustomerAgent(
    state=state,
    seed=42,
)


# ============================================================
# INITIAL STATE
# ============================================================

print("=" * 60)
print("INITIAL AGENT B POLICY BELIEFS")
print("=" * 60)

print(
    agent.get_inferred_policy()
)


# ============================================================
# OBSERVATION 1
# Low-value request + no evidence
# Agent A approves
# ============================================================

print("\n" + "=" * 60)
print("OBSERVATION 1")
print("=" * 60)

decision = Decision(
    decision="APPROVE",
    reason_code="LOW_RISK",
    approved_amount=800,
)

agent.infer_support_policy(
    decision=decision,
    claim_type="SHORTAGE_CLAIM",
    requested_amount=800,
    evidence_available=[],
)

print("A response:", decision.decision)
print("B inference:")
print(agent.get_inferred_policy())


# ============================================================
# OBSERVATION 2
# Medium-value request + no evidence
# Agent A requests evidence
# ============================================================

print("\n" + "=" * 60)
print("OBSERVATION 2")
print("=" * 60)

decision = Decision(
    decision="REQUEST_EVIDENCE",
    reason_code="INSUFFICIENT_EVIDENCE",
    requested_evidence=[
        "delivery_photo"
    ],
)

agent.infer_support_policy(
    decision=decision,
    claim_type="SHORTAGE_CLAIM",
    requested_amount=1500,
    evidence_available=[],
)

print("A response:", decision.decision)
print("B inference:")
print(agent.get_inferred_policy())


# ============================================================
# OBSERVATION 3
# Higher-value request + no evidence
# Agent A requests evidence
# ============================================================

print("\n" + "=" * 60)
print("OBSERVATION 3")
print("=" * 60)

decision = Decision(
    decision="REQUEST_EVIDENCE",
    reason_code="INSUFFICIENT_EVIDENCE",
    requested_evidence=[
        "delivery_photo",
        "package_photo",
    ],
)

agent.infer_support_policy(
    decision=decision,
    claim_type="WRONG_ITEM_CLAIM",
    requested_amount=2500,
    evidence_available=[],
)

print("A response:", decision.decision)
print("B inference:")
print(agent.get_inferred_policy())


# ============================================================
# OBSERVATION 4
# High-value request + evidence
# Agent A escalates
# ============================================================

print("\n" + "=" * 60)
print("OBSERVATION 4")
print("=" * 60)

decision = Decision(
    decision="ESCALATE",
    reason_code="HIGH_VALUE",
)

agent.infer_support_policy(
    decision=decision,
    claim_type="NON_DELIVERY_CLAIM",
    requested_amount=5500,
    evidence_available=[
        "delivery_photo"
    ],
)

print("A response:", decision.decision)
print("B inference:")
print(agent.get_inferred_policy())


# ============================================================
# OBSERVATION 5
# Low-value request + evidence
# Agent A approves
# ============================================================

print("\n" + "=" * 60)
print("OBSERVATION 5")
print("=" * 60)

decision = Decision(
    decision="APPROVE",
    reason_code="EVIDENCE_SUFFICIENT",
    approved_amount=800,
)

agent.infer_support_policy(
    decision=decision,
    claim_type="SHORTAGE_CLAIM",
    requested_amount=800,
    evidence_available=[
        "delivery_photo"
    ],
)

print("A response:", decision.decision)
print("B inference:")
print(agent.get_inferred_policy())


# ============================================================
# OBSERVATION 6
# High-value request + no evidence
# Agent A escalates
# ============================================================

print("\n" + "=" * 60)
print("OBSERVATION 6")
print("=" * 60)

decision = Decision(
    decision="ESCALATE",
    reason_code="HIGH_VALUE",
)

agent.infer_support_policy(
    decision=decision,
    claim_type="SUBSTITUTED_RETURN",
    requested_amount=5500,
    evidence_available=[],
)

print("A response:", decision.decision)
print("B inference:")
print(agent.get_inferred_policy())


# ============================================================
# FINAL LEARNED STATE
# ============================================================

print("\n" + "=" * 60)
print("FINAL INFERRED POLICY")
print("=" * 60)

policy = agent.get_inferred_policy()

for key, value in policy.items():

    print(
        f"{key:25s}: {value:.3f}"
    )


# ============================================================
# CLAIM-SPECIFIC OBSERVATIONS
# ============================================================

print("\n" + "=" * 60)
print("CLAIM-SPECIFIC OBSERVATIONS")
print("=" * 60)

for claim_type in (
    "SHORTAGE_CLAIM",
    "WRONG_ITEM_CLAIM",
    "NON_DELIVERY_CLAIM",
    "SUBSTITUTED_RETURN",
):

    observations = (
        agent.get_claim_observations(
            claim_type
        )
    )

    print(
        f"{claim_type:25s}: "
        f"{len(observations)} observation(s)"
    )


# ============================================================
# TOTAL OBSERVATIONS
# ============================================================

print("\n" + "=" * 60)
print("TOTAL OBSERVATIONS")
print("=" * 60)

print(
    len(agent.state.support_observations)
)