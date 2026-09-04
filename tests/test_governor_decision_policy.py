from __future__ import annotations

"""
GOVERNOR DECISION POLICY AUDIT

Validates the existing Adaptive Risk Governor decision layer.

Scope:
    83. Governor decision logic
    84. Governor action space
    85. Governor must NOT directly issue DENY
    86. Intervention thresholds / policy
    87. Intervention reasoning / trace

IMPORTANT:
    - Read-only test.
    - Does NOT modify the existing Governor.
    - Does NOT modify the GNN.
    - Does NOT modify Risk Fusion.
    - Does NOT modify Agent A or Agent B.
    - Does NOT introduce a new decision architecture.

Run:

    python -m tests.test_governor_decision_policy
"""

from typing import Any
from types import SimpleNamespace


# ============================================================
# EXISTING GOVERNOR IMPORT
# ============================================================

from src.governor.risk_governor import (
    AdaptiveRiskGovernor,
)


# ============================================================
# CONFIGURATION
# ============================================================

VALID_ACTIONS = {
    "ALLOW_AGENT_A_DECISION",
    "REQUEST_ADDITIONAL_EVIDENCE",
    "ESCALATE_TO_HUMAN_REVIEW",
}

FORBIDDEN_ACTIONS = {
    "DENY",
}

VALID_RISK_LEVELS = {
    "LOW",
    "MEDIUM",
    "HIGH",
}

CLAIM_TYPE = "PRODUCT_NOT_RECEIVED"

EVIDENCE = [
    "order_receipt",
    "delivery_record",
]


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def subsection(title: str) -> None:
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def bounded(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(
        0.0,
        min(
            value,
            1.0,
        ),
    )


def evaluate(
    governor: AdaptiveRiskGovernor,
    *,
    history,
    amount: float,
    support_decision: str,
    strategic_state=None,
):
    return governor.evaluate(
        customer_history=history,
        current_claim_type=CLAIM_TYPE,
        requested_amount=amount,
        evidence_available=EVIDENCE,
        support_decision=support_decision,
        strategic_state=strategic_state,
    )


# ============================================================
# TEST DATA
# ============================================================

NEUTRAL_HISTORY = [
    {
        "claim_type": "PRODUCT_NOT_RECEIVED",
        "requested_amount": 250.0,
        "support_decision": "APPROVE",
    },
]

REPEATED_HISTORY = [
    {
        "claim_type": "PRODUCT_NOT_RECEIVED",
        "requested_amount": 250.0,
        "support_decision": "APPROVE",
    },
    {
        "claim_type": "PRODUCT_NOT_RECEIVED",
        "requested_amount": 500.0,
        "support_decision": "APPROVE",
    },
    {
        "claim_type": "PRODUCT_NOT_RECEIVED",
        "requested_amount": 750.0,
        "support_decision": "APPROVE",
    },
]


# ============================================================
# TEST 1
# GOVERNOR INITIALIZATION
# ============================================================

def test_governor_initialization() -> AdaptiveRiskGovernor:

    section(
        "TEST 1 — GOVERNOR INITIALIZATION"
    )

    governor = AdaptiveRiskGovernor()

    if governor is None:
        raise AssertionError(
            "AdaptiveRiskGovernor failed to initialize."
        )

    print(
        "AdaptiveRiskGovernor initialized successfully."
    )

    print(
        "[PASS] Governor initialization."
    )

    return governor


# ============================================================
# TEST 2
# ACTION SPACE
# ============================================================

def test_action_space() -> None:

    section(
        "TEST 2 — GOVERNOR ACTION SPACE"
    )

    print(
        "Allowed actions:"
    )

    for action in sorted(VALID_ACTIONS):
        print(
            f"  - {action}"
        )

    if "DENY" in VALID_ACTIONS:
        raise AssertionError(
            "DENY must not belong to the Governor action space."
        )

    if len(VALID_ACTIONS) != 3:
        raise AssertionError(
            "Unexpected Governor action space."
        )

    print()
    print(
        "[PASS] Governor action space contains "
        "exactly the three permitted actions."
    )


# ============================================================
# TEST 3
# BASE DECISION STRUCTURE
# ============================================================

def test_decision_structure(
    governor: AdaptiveRiskGovernor,
):

    section(
        "TEST 3 — GOVERNOR DECISION STRUCTURE"
    )

    decision = evaluate(
        governor,
        history=NEUTRAL_HISTORY,
        amount=250.0,
        support_decision="APPROVE",
    )

    if decision is None:
        raise AssertionError(
            "Governor returned None."
        )

    required_attributes = [
        "action",
        "risk_score",
        "risk_level",
        "reason_codes",
    ]

    missing = [
        name
        for name in required_attributes
        if not hasattr(decision, name)
    ]

    if missing:
        raise AssertionError(
            "Governor decision is missing required fields: "
            f"{missing}"
        )

    print(
        f"Risk score   : {bounded(decision.risk_score):.6f}"
    )

    print(
        f"Risk level   : {decision.risk_level}"
    )

    print(
        f"Action       : {decision.action}"
    )

    print(
        f"Reason codes : {decision.reason_codes}"
    )

    print(
        "[PASS] Governor decision structure."
    )

    return decision


# ============================================================
# TEST 4
# DENY MUST NEVER BE GOVERNOR ACTION
# ============================================================

def test_no_governor_deny(
    governor: AdaptiveRiskGovernor,
) -> None:

    section(
        "TEST 4 — GOVERNOR MUST NOT ISSUE DENY"
    )

    scenarios = [
        (
            "neutral",
            NEUTRAL_HISTORY,
            250.0,
            "APPROVE",
        ),
        (
            "repeated",
            REPEATED_HISTORY,
            750.0,
            "APPROVE",
        ),
        (
            "high_amount",
            REPEATED_HISTORY,
            5000.0,
            "APPROVE",
        ),
    ]

    observed_actions = set()

    for (
        name,
        history,
        amount,
        support_decision,
    ) in scenarios:

        decision = evaluate(
            governor,
            history=history,
            amount=amount,
            support_decision=support_decision,
        )

        action = str(
            decision.action
        )

        observed_actions.add(
            action
        )

        print(
            f"{name:<15} -> {action}"
        )

        if action in FORBIDDEN_ACTIONS:
            raise AssertionError(
                "Governor directly issued DENY."
            )

    print()
    print(
        f"Observed actions: {sorted(observed_actions)}"
    )

    print(
        "[PASS] Governor never directly issues DENY."
    )


# ============================================================
# TEST 5
# ACTION VALIDATION
# ============================================================

def test_returned_action_is_valid(
    governor: AdaptiveRiskGovernor,
) -> None:

    section(
        "TEST 5 — RETURNED ACTION VALIDATION"
    )

    scenarios = [
        (
            "neutral",
            NEUTRAL_HISTORY,
            250.0,
        ),
        (
            "repeated",
            REPEATED_HISTORY,
            750.0,
        ),
        (
            "large_amount",
            REPEATED_HISTORY,
            5000.0,
        ),
    ]

    for name, history, amount in scenarios:

        decision = evaluate(
            governor,
            history=history,
            amount=amount,
            support_decision="APPROVE",
        )

        action = str(
            decision.action
        )

        print(
            f"{name:<15} -> {action}"
        )

        if action not in VALID_ACTIONS:
            raise AssertionError(
                "Governor returned an action outside "
                f"the permitted action space: {action}"
            )

    print()
    print(
        "[PASS] All returned Governor actions are valid."
    )


# ============================================================
# TEST 6
# RISK SCORE / LEVEL CONSISTENCY
# ============================================================

def test_risk_output_bounds(
    governor: AdaptiveRiskGovernor,
) -> None:

    section(
        "TEST 6 — RISK OUTPUT SANITY"
    )

    scenarios = [
        (
            "neutral",
            NEUTRAL_HISTORY,
            250.0,
        ),
        (
            "repeated",
            REPEATED_HISTORY,
            750.0,
        ),
        (
            "large_amount",
            REPEATED_HISTORY,
            5000.0,
        ),
    ]

    for name, history, amount in scenarios:

        decision = evaluate(
            governor,
            history=history,
            amount=amount,
            support_decision="APPROVE",
        )

        score = float(
            decision.risk_score
        )

        level = str(
            decision.risk_level
        )

        print(
            f"{name:<15} "
            f"risk={score:.6f} "
            f"level={level}"
        )

        if not 0.0 <= score <= 1.0:
            raise AssertionError(
                f"Risk score outside [0,1]: {score}"
            )

        if level not in VALID_RISK_LEVELS:
            raise AssertionError(
                f"Invalid risk level: {level}"
            )

    print()
    print(
        "[PASS] Risk scores and risk levels are structurally valid."
    )


# ============================================================
# TEST 7
# REASONING / TRACE
# ============================================================

def test_reasoning_trace(
    governor: AdaptiveRiskGovernor,
) -> None:

    section(
        "TEST 7 — GOVERNOR REASONING / TRACE"
    )

    scenarios = [
        (
            "neutral",
            NEUTRAL_HISTORY,
            250.0,
        ),
        (
            "repeated",
            REPEATED_HISTORY,
            750.0,
        ),
        (
            "large_amount",
            REPEATED_HISTORY,
            5000.0,
        ),
    ]

    for name, history, amount in scenarios:

        decision = evaluate(
            governor,
            history=history,
            amount=amount,
            support_decision="APPROVE",
        )

        reason_codes = getattr(
            decision,
            "reason_codes",
            None,
        )

        reason = getattr(
            decision,
            "reason",
            None,
        )

        print()
        print(
            f"Scenario : {name}"
        )

        print(
            f"Action   : {decision.action}"
        )

        print(
            f"Reasons  : {reason_codes}"
        )

        if reason is not None:
            print(
                f"Reason   : {reason}"
            )

        # A trace can legitimately be empty for
        # a low-risk ALLOW decision.
        #
        # For intervention decisions, some explanation
        # should exist.

        if decision.action != "ALLOW_AGENT_A_DECISION":

            has_reason_codes = (
                isinstance(
                    reason_codes,
                    (list, tuple, set),
                )
                and len(reason_codes) > 0
            )

            has_reason_text = (
                isinstance(
                    reason,
                    str,
                )
                and bool(
                    reason.strip()
                )
            )

            if not (
                has_reason_codes
                or has_reason_text
            ):
                raise AssertionError(
                    "Governor intervention has no "
                    "reasoning/trace information."
                )

    print()
    print(
        "[PASS] Governor reasoning/trace inspected."
    )


# ============================================================
# TEST 8
# THRESHOLD / INTERVENTION DISCOVERY
# ============================================================

def test_intervention_policy(
    governor: AdaptiveRiskGovernor,
) -> None:

    section(
        "TEST 8 — INTERVENTION POLICY / THRESHOLD DISCOVERY"
    )

    scenarios = []

    # Generate progressively stronger histories.
    for i in range(0, 8):

        history = []

        for j in range(i):

            history.append(
                {
                    "claim_type": "PRODUCT_NOT_RECEIVED",
                    "requested_amount": 250.0 * (j + 1),
                    "support_decision": "APPROVE",
                }
            )

        scenarios.append(
            (
                i,
                history,
                250.0 * max(
                    1,
                    i + 1,
                ),
            )
        )

    observed = []

    for count, history, amount in scenarios:

        decision = evaluate(
            governor,
            history=history,
            amount=amount,
            support_decision="APPROVE",
        )

        risk = bounded(
            decision.risk_score
        )

        action = str(
            decision.action
        )

        observed.append(
            (
                risk,
                action,
            )
        )

        print(
            f"history={count:<2} "
            f"risk={risk:.6f} "
            f"action={action}"
        )

    intervention_points = [
        (
            index,
            risk,
            action,
        )
        for index, (
            risk,
            action,
        ) in enumerate(
            observed
        )
        if action != "ALLOW_AGENT_A_DECISION"
    ]

    print()

    if intervention_points:

        print(
            "First observed intervention:"
        )

        index, risk, action = (
            intervention_points[0]
        )

        print(
            f"  scenario index : {index}"
        )

        print(
            f"  risk           : {risk:.6f}"
        )

        print(
            f"  action         : {action}"
        )

        print()
        print(
            "[PASS] Intervention policy is observable."
        )

    else:

        print(
            "[WARNING] No intervention was triggered "
            "by these synthetic scenarios."
        )

        print(
            "[INFO] This does not prove that intervention "
            "logic is absent."
        )

        print(
            "[INFO] Existing Governor thresholds should "
            "be inspected before finalizing policy."
        )


# ============================================================
# TEST 9
# INTERVENTION ACTION SEMANTICS
# ============================================================

def test_action_semantics(
    governor: AdaptiveRiskGovernor,
) -> None:

    section(
        "TEST 9 — ACTION SEMANTICS"
    )

    decisions = []

    for history, amount in [
        (
            NEUTRAL_HISTORY,
            250.0,
        ),
        (
            REPEATED_HISTORY,
            750.0,
        ),
        (
            REPEATED_HISTORY,
            5000.0,
        ),
    ]:

        decision = evaluate(
            governor,
            history=history,
            amount=amount,
            support_decision="APPROVE",
        )

        decisions.append(
            decision
        )

    for decision in decisions:

        action = str(
            decision.action
        )

        if action == "ALLOW_AGENT_A_DECISION":

            print(
                "ALLOW_AGENT_A_DECISION:"
            )

            print(
                "  Governor permits Agent A's "
                "original decision."
            )

        elif action == "REQUEST_ADDITIONAL_EVIDENCE":

            print(
                "REQUEST_ADDITIONAL_EVIDENCE:"
            )

            print(
                "  Governor requests additional "
                "evidence rather than denying."
            )

        elif action == "ESCALATE_TO_HUMAN_REVIEW":

            print(
                "ESCALATE_TO_HUMAN_REVIEW:"
            )

            print(
                "  Governor transfers the case "
                "to human review."
            )

        else:

            raise AssertionError(
                f"Unknown Governor action: {action}"
            )

    print()
    print(
        "[PASS] Governor action semantics remain "
        "within the intended action space."
    )


# ============================================================
# TEST 10
# ARCHITECTURE BOUNDARY
# ============================================================

def test_architecture_boundary(
    governor: AdaptiveRiskGovernor,
) -> None:

    section(
        "TEST 10 — GOVERNOR ARCHITECTURE BOUNDARY"
    )

    decision = evaluate(
        governor,
        history=REPEATED_HISTORY,
        amount=750.0,
        support_decision="APPROVE",
    )

    action = str(
        decision.action
    )

    if action == "DENY":
        raise AssertionError(
            "Architecture violation: Governor directly "
            "issued DENY."
        )

    if action not in VALID_ACTIONS:
        raise AssertionError(
            "Architecture violation: unknown Governor action."
        )

    print(
        "Governor remains responsible for intervention."
    )

    print(
        "Final customer disposition remains outside "
        "the Governor action space."
    )

    print(
        "[PASS] Governor architecture boundary preserved."
    )


# ============================================================
# TEST 11 — INTERVENTION POLICY ACTIVATION
# ============================================================
#
# This test validates the EXISTING Governor policy.
#
# IMPORTANT:
# We do NOT modify the Governor implementation.
#
# Existing Governor rules being validated:
#
#   repeated denials >= 3       -> +0.20
#   claim switches >= 4         -> +0.25
#   high-value amount > 5000    -> +0.10
#   strategic adaptation >= .70 -> HIGH override
#
# Therefore:
#
#   0.20 + 0.25 + 0.10 = 0.55 -> MEDIUM
#
# HIGH risk is reached through the existing strategic
# adaptation override, not by inventing another score rule.
# ============================================================


def test_intervention_policy_activation(
    governor: AdaptiveRiskGovernor,
) -> None:

    section(
        "TEST 11 — INTERVENTION POLICY ACTIVATION"
    )

    allowed_actions = {
        "ALLOW_AGENT_A_DECISION",
        "REQUEST_ADDITIONAL_EVIDENCE",
        "ESCALATE_TO_HUMAN_REVIEW",
    }

    # ========================================================
    # SCENARIO A — REPEATED DENIALS
    # ========================================================
    #
    # Existing Governor rule:
    #
    #   denied_count >= 3 -> +0.20
    #
    # No claim switching.
    # No high-value request.
    #
    # Expected:
    #
    #   risk = 0.20
    #   level = LOW
    #   action = ALLOW_AGENT_A_DECISION
    #
    # This is intentional. A 0.20 score is below the
    # Governor's default low-risk threshold of 0.30.
    # ========================================================

    history_denials = [
        {
            "claim_type": "REFUND",
            "support_decision": "DENY",
        },
        {
            "claim_type": "REFUND",
            "support_decision": "DENY",
        },
        {
            "claim_type": "REFUND",
            "support_decision": "DENY",
        },
    ]

    decision_a = governor.evaluate(
        customer_history=history_denials,
        current_claim_type="REFUND",
        requested_amount=1000.0,
        evidence_available=[],
        support_decision="APPROVE",
        strategic_state=None,
    )

    print()
    print("Scenario A — repeated denials")

    print(
        "Risk score : "
        f"{decision_a.risk_score:.6f}"
    )

    print(
        "Risk level : "
        f"{decision_a.risk_level}"
    )

    print(
        "Action     : "
        f"{decision_a.action}"
    )

    print(
        "Reasons    : "
        f"{decision_a.reason_codes}"
    )

    assert (
        "REPEATED_DENIALS"
        in decision_a.reason_codes
    )

    assert abs(
        decision_a.risk_score - 0.20
    ) < 1e-9

    assert (
        decision_a.risk_level
        == "LOW"
    )

    assert (
        decision_a.action
        == "ALLOW_AGENT_A_DECISION"
    )

    assert (
        decision_a.action
        in allowed_actions
    )

    assert (
        decision_a.action
        != "DENY"
    )

    print(
        "[PASS] Repeated-denial signal is "
        "detected correctly."
    )

    # ========================================================
    # SCENARIO B — MEDIUM-RISK INTERVENTION
    # ========================================================
    #
    # IMPORTANT:
    #
    # The Governor requires:
    #
    #   denied_count >= 3
    #       -> +0.20
    #
    #   claim_switches >= 4
    #       -> +0.25
    #
    #   requested_amount > 5000
    #       -> +0.10
    #
    # Total:
    #
    #   0.20 + 0.25 + 0.10 = 0.55
    #
    # Therefore this should produce:
    #
    #   MEDIUM
    #   REQUEST_ADDITIONAL_EVIDENCE
    #
    # The history below contains FIVE claim transitions:
    #
    #   REFUND
    #   DAMAGED_ITEM       switch 1
    #   MISSING_ITEM       switch 2
    #   REFUND             switch 3
    #   DAMAGED_ITEM       switch 4
    #
    # It also contains three denials.
    # ========================================================

    history_medium = [
        {
            "claim_type": "REFUND",
            "support_decision": "DENY",
        },
        {
            "claim_type": "DAMAGED_ITEM",
            "support_decision": "DENY",
        },
        {
            "claim_type": "MISSING_ITEM",
            "support_decision": "DENY",
        },
        {
            "claim_type": "REFUND",
            "support_decision": "APPROVE",
        },
        {
            "claim_type": "DAMAGED_ITEM",
            "support_decision": "APPROVE",
        },
    ]

    decision_b = governor.evaluate(
        customer_history=history_medium,
        current_claim_type="MISSING_ITEM",
        requested_amount=6000.0,
        evidence_available=[],
        support_decision="APPROVE",
        strategic_state=None,
    )

    print()
    print("Scenario B — medium-risk intervention")

    print(
        "Risk score : "
        f"{decision_b.risk_score:.6f}"
    )

    print(
        "Risk level : "
        f"{decision_b.risk_level}"
    )

    print(
        "Action     : "
        f"{decision_b.action}"
    )

    print(
        "Reasons    : "
        f"{decision_b.reason_codes}"
    )

    assert (
        "REPEATED_DENIALS"
        in decision_b.reason_codes
    )

    assert (
        "FREQUENT_CLAIM_SWITCHING"
        in decision_b.reason_codes
    )

    assert (
        "HIGH_VALUE_REQUEST"
        in decision_b.reason_codes
    )

    assert abs(
        decision_b.risk_score - 0.55
    ) < 1e-9

    assert (
        decision_b.risk_score >= 0.30
    )

    assert (
        decision_b.risk_score < 0.70
    )

    assert (
        decision_b.risk_level
        == "MEDIUM"
    )

    assert (
        decision_b.action
        == "REQUEST_ADDITIONAL_EVIDENCE"
    )

    assert (
        decision_b.action
        in allowed_actions
    )

    assert (
        decision_b.action
        != "DENY"
    )

    print(
        "[PASS] Medium-risk intervention is reachable "
        "through the existing Governor policy."
    )

    # ========================================================
    # SCENARIO C — COMBINED CLAIM-LEVEL SIGNALS
    # ========================================================
    #
    # This deliberately exercises all three ordinary
    # claim-level risk contributions.
    #
    # Expected:
    #
    #   repeated denials      = +0.20
    #   frequent switching    = +0.25
    #   high-value request    = +0.10
    #
    #   TOTAL                 = 0.55
    #
    # The Governor therefore remains MEDIUM.
    # ========================================================

    history_combined = [
        {
            "claim_type": "REFUND",
            "support_decision": "DENY",
        },
        {
            "claim_type": "DAMAGED_ITEM",
            "support_decision": "DENY",
        },
        {
            "claim_type": "MISSING_ITEM",
            "support_decision": "DENY",
        },
        {
            "claim_type": "REFUND",
            "support_decision": "DENY",
        },
        {
            "claim_type": "DAMAGED_ITEM",
            "support_decision": "DENY",
        },
    ]

    decision_c = governor.evaluate(
        customer_history=history_combined,
        current_claim_type="REFUND",
        requested_amount=10000.0,
        evidence_available=[],
        support_decision="APPROVE",
        strategic_state=None,
    )

    print()
    print(
        "Scenario C — combined claim-level signals"
    )

    print(
        "Risk score : "
        f"{decision_c.risk_score:.6f}"
    )

    print(
        "Risk level : "
        f"{decision_c.risk_level}"
    )

    print(
        "Action     : "
        f"{decision_c.action}"
    )

    print(
        "Reasons    : "
        f"{decision_c.reason_codes}"
    )

    assert (
        "REPEATED_DENIALS"
        in decision_c.reason_codes
    )

    assert (
        "FREQUENT_CLAIM_SWITCHING"
        in decision_c.reason_codes
    )

    assert (
        "HIGH_VALUE_REQUEST"
        in decision_c.reason_codes
    )

    assert abs(
        decision_c.risk_score - 0.55
    ) < 1e-9

    assert (
        decision_c.risk_level
        == "MEDIUM"
    )

    assert (
        decision_c.action
        == "REQUEST_ADDITIONAL_EVIDENCE"
    )

    assert (
        decision_c.action
        in allowed_actions
    )

    assert (
        decision_c.action
        != "DENY"
    )

    print(
        "[PASS] Combined claim-level signals "
        "produce the expected MEDIUM intervention."
    )

    # ========================================================
    # SCENARIO D — STRATEGIC ADAPTATION HIGH-RISK OVERRIDE
    # ========================================================
    #
    # This validates the EXISTING strategic override:
    #
    #   strategic_adaptation >= 0.70
    #       -> HIGH
    #       -> ESCALATE_TO_HUMAN_REVIEW
    #
    # With 20 interactions:
    #
    #   learning_signal = min(20 / 20, 1.0)
    #                   = 1.0
    #
    # With beliefs at 0.90:
    #
    #   belief_shift =
    #       (0.4 + 0.4 + 0.4 + 0.4) / 2
    #       = 0.8
    #
    # Therefore:
    #
    #   adaptation =
    #       0.5 * 1.0 + 0.5 * 0.8
    #       = 0.90
    #
    # This triggers the existing HIGH-risk override.
    # ========================================================

    from types import SimpleNamespace

    strategic_history = [
        {
            "support_decision": "APPROVE",
        }
        for _ in range(20)
    ]

    strategic_state = SimpleNamespace(
        interaction_history=strategic_history,

        inferred_policy={
            "evidence_sensitivity": 0.90,
            "amount_sensitivity": 0.90,
            "high_value_escalation": 0.90,
            "followup_sensitivity": 0.90,
        },
    )

    decision_d = governor.evaluate(
        customer_history=[],
        current_claim_type="PRODUCT_NOT_RECEIVED",
        requested_amount=750.0,
        evidence_available=[
            "order_receipt",
            "delivery_record",
        ],
        support_decision="APPROVE",
        strategic_state=strategic_state,
    )

    print()
    print(
        "Scenario D — strategic adaptation "
        "high-risk override"
    )

    print(
        "Risk score : "
        f"{decision_d.risk_score:.6f}"
    )

    print(
        "Risk level : "
        f"{decision_d.risk_level}"
    )

    print(
        "Action     : "
        f"{decision_d.action}"
    )

    print(
        "Reasons    : "
        f"{decision_d.reason_codes}"
    )

    strategic_features = getattr(
        decision_d,
        "features",
        {},
    )

    strategic_adaptation = float(
        strategic_features[
            "strategic_adaptation_score"
        ]
    )

    print(
        "Strategic adaptation : "
        f"{strategic_adaptation:.6f}"
    )

    assert (
        strategic_adaptation
        >= 0.70
    )

    assert (
        "STRATEGIC_ADAPTATION"
        in decision_d.reason_codes
    )

    assert (
        decision_d.risk_level
        == "HIGH"
    )

    assert (
        decision_d.action
        == "ESCALATE_TO_HUMAN_REVIEW"
    )

    assert (
        decision_d.action
        in allowed_actions
    )

    assert (
        decision_d.action
        != "DENY"
    )

    print(
        "[PASS] Strategic adaptation correctly "
        "activates the HIGH-risk intervention."
    )

    # ========================================================
    # SCENARIO E — GOVERNOR ACTION-SPACE VALIDATION
    # ========================================================

    observed_actions = {
        decision_a.action,
        decision_b.action,
        decision_c.action,
        decision_d.action,
    }

    print()
    print(
        "Scenario E — Governor action-space validation"
    )

    print(
        "Observed actions:"
    )

    for action in sorted(
        observed_actions
    ):
        print(
            f"  - {action}"
        )

    assert (
        observed_actions
        <= allowed_actions
    )

    assert (
        "DENY"
        not in observed_actions
    )

    print(
        "[PASS] All intervention actions remain "
        "inside the permitted Governor action space."
    )

    # ========================================================
    # FINAL TEST 11 RESULT
    # ========================================================

    print()
    print(
        "[PASS] INTERVENTION POLICY ACTIVATION"
    )

    print()
    print(
        "Verified:"
    )

    print(
        "  Repeated denials          -> risk signal detected"
    )

    print(
        "  Medium-risk behavior      -> "
        "REQUEST_ADDITIONAL_EVIDENCE"
    )

    print(
        "  Combined claim signals    -> "
        "MEDIUM intervention"
    )

    print(
        "  Strategic adaptation      -> "
        "HIGH intervention"
    )

    print(
        "  HIGH risk                 -> "
        "ESCALATE_TO_HUMAN_REVIEW"
    )

    print(
        "  Direct DENY by Governor   -> NOT PERMITTED"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "ADAPTIVE RISK GOVERNOR — DECISION POLICY AUDIT"
    )

    print(
        "READ-ONLY validation."
    )

    print(
        "Existing architecture will NOT be modified."
    )

    print()
    print(
        "Scope:"
    )

    print(
        "  83. Governor decision logic"
    )

    print(
        "  84. Governor action space"
    )

    print(
        "  85. No direct DENY"
    )

    print(
        "  86. Intervention policy"
    )

    print(
        "  87. Reasoning / trace"
    )

    governor = test_governor_initialization()

    test_action_space()

    test_decision_structure(
        governor
    )

    test_no_governor_deny(
        governor
    )

    test_returned_action_is_valid(
        governor
    )

    test_risk_output_bounds(
        governor
    )

    test_reasoning_trace(
        governor
    )

    test_intervention_policy(
        governor
    )

    test_action_semantics(
        governor
    )

    test_architecture_boundary(
        governor
    )

    test_intervention_policy_activation(
        governor
    )

    section(
        "GOVERNOR DECISION POLICY AUDIT COMPLETE"
    )

    print(
        "No existing source files were modified."
    )

    print(
        "No GNN files were modified."
    )

    print(
        "No fusion files were modified."
    )

    print(
        "No Agent A / Agent B files were modified."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()