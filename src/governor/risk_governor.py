from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ============================================================
# RISK LEVEL
# ============================================================

RISK_LEVELS = (
    "LOW",
    "MEDIUM",
    "HIGH",
)


# ============================================================
# GOVERNOR DECISION
# ============================================================

@dataclass
class GovernorDecision:

    risk_score: float

    risk_level: str

    action: str

    reason_codes: list[str] = field(
        default_factory=list
    )

    features: dict[str, float] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# ADAPTIVE RISK GOVERNOR
# ============================================================

class AdaptiveRiskGovernor:

    """
    Adaptive risk layer sitting above Agent A.

    Responsibilities:

        1. Build behavioral risk features.
        2. Detect strategic adaptation.
        3. Produce an adaptive risk score.
        4. Convert the score into a risk level.
        5. Recommend an intervention.

    The governor does NOT directly modify Agent A's
    private policy.

    Later, an ML model can replace the heuristic
    risk-score calculation while preserving this
    interface.
    """

    def __init__(
        self,
        low_risk_threshold: float = 0.30,
        high_risk_threshold: float = 0.70,
    ):

        if not (
            0.0
            <= low_risk_threshold
            < high_risk_threshold
            <= 1.0
        ):
            raise ValueError(
                "Risk thresholds must satisfy "
                "0 <= low < high <= 1."
            )

        self.low_risk_threshold = (
            low_risk_threshold
        )

        self.high_risk_threshold = (
            high_risk_threshold
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def evaluate(
        self,
        *,
        customer_history: list[dict[str, Any]] | None = None,
        current_claim_type: str,
        requested_amount: float,
        evidence_available: list[str] | None = None,
        support_decision: str | None = None,
        strategic_state: Any | None = None,
    ) -> GovernorDecision:

        customer_history = (
            customer_history or []
        )

        evidence_available = (
            evidence_available or []
        )

        features = self.build_features(
            customer_history=customer_history,
            current_claim_type=current_claim_type,
            requested_amount=requested_amount,
            evidence_available=evidence_available,
            support_decision=support_decision,
            strategic_state=strategic_state,
        )

        risk_score, reason_codes = (
            self._calculate_risk(
                features
            )
        )

        risk_level = self._risk_level(
            score=risk_score,
            features=features,
        )

        action = self._recommended_action(
            risk_level=risk_level,
            support_decision=support_decision,
        )

        return GovernorDecision(
            risk_score=risk_score,
            risk_level=risk_level,
            action=action,
            reason_codes=reason_codes,
            features=features,
        )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    def build_features(
        self,
        *,
        customer_history: list[dict[str, Any]],
        current_claim_type: str,
        requested_amount: float,
        evidence_available: list[str],
        support_decision: str | None,
        strategic_state: Any | None,
    ) -> dict[str, float]:

        history_count = len(
            customer_history
        )

        denied_count = sum(
            1
            for x in customer_history
            if x.get("support_decision") == "DENY"
        )

        approved_count = sum(
            1
            for x in customer_history
            if x.get("support_decision") == "APPROVE"
        )

        evidence_requests = sum(
            1
            for x in customer_history
            if x.get("support_decision")
            == "REQUEST_EVIDENCE"
        )

        escalation_count = sum(
            1
            for x in customer_history
            if x.get("support_decision")
            == "ESCALATE"
        )

        claim_switches = self._claim_switch_count(
            customer_history
        )

        amount = max(
            float(requested_amount),
            0.0,
        )

        high_value = (
            1.0
            if amount > 5000
            else 0.0
        )

        evidence_present = (
            1.0
            if evidence_available
            else 0.0
        )

        # ----------------------------------------------------
        # Strategic-agent features
        # ----------------------------------------------------

        evidence_sensitivity = 0.5
        amount_sensitivity = 0.5
        high_value_escalation = 0.5
        followup_sensitivity = 0.5

        if strategic_state is not None:

            beliefs = getattr(
                strategic_state,
                "inferred_policy",
                {},
            )

            evidence_sensitivity = float(
                beliefs.get(
                    "evidence_sensitivity",
                    0.5,
                )
            )

            amount_sensitivity = float(
                beliefs.get(
                    "amount_sensitivity",
                    0.5,
                )
            )

            high_value_escalation = float(
                beliefs.get(
                    "high_value_escalation",
                    0.5,
                )
            )

            followup_sensitivity = float(
                beliefs.get(
                    "followup_sensitivity",
                    0.5,
                )
            )

        # ----------------------------------------------------
        # Strategic adaptation
        # ----------------------------------------------------

        strategic_adaptation = (
            self._strategic_adaptation_score(
                strategic_state
            )
        )

        return {
            "history_count": float(
                history_count
            ),

            "denied_count": float(
                denied_count
            ),

            "approved_count": float(
                approved_count
            ),

            "evidence_request_count": float(
                evidence_requests
            ),

            "escalation_count": float(
                escalation_count
            ),

            "claim_switch_count": float(
                claim_switches
            ),

            "requested_amount": amount,

            "high_value_flag": high_value,

            "evidence_present": evidence_present,

            "evidence_sensitivity": (
                evidence_sensitivity
            ),

            "amount_sensitivity": (
                amount_sensitivity
            ),

            "high_value_escalation": (
                high_value_escalation
            ),

            "followup_sensitivity": (
                followup_sensitivity
            ),

            "strategic_adaptation_score": (
                strategic_adaptation
            ),
        }

    # ========================================================
    # CLAIM SWITCHES
    # ========================================================

    @staticmethod
    def _claim_switch_count(
        history: list[dict[str, Any]],
    ) -> int:

        if len(history) < 2:
            return 0

        switches = 0

        previous = history[0].get(
            "claim_type"
        )

        for interaction in history[1:]:

            current = interaction.get(
                "claim_type"
            )

            if (
                current is not None
                and previous is not None
                and current != previous
            ):

                switches += 1

            previous = current

        return switches

    # ========================================================
    # STRATEGIC ADAPTATION
    # ========================================================

    @staticmethod
    def _strategic_adaptation_score(
        strategic_state: Any | None,
    ) -> float:

        if strategic_state is None:
            return 0.0

        history = getattr(
            strategic_state,
            "interaction_history",
            [],
        )

        if not history:
            return 0.0

        # ----------------------------------------------------
        # Learning signal
        # ----------------------------------------------------
        #
        # More interactions mean Agent B has had more
        # opportunity to observe Agent A.
        #

        learning_signal = min(
            len(history) / 20.0,
            1.0,
        )

        # ----------------------------------------------------
        # Policy-belief shift
        # ----------------------------------------------------

        beliefs = getattr(
            strategic_state,
            "inferred_policy",
            {},
        )

        belief_values = [
            float(
                beliefs.get(
                    "evidence_sensitivity",
                    0.5,
                )
            ),

            float(
                beliefs.get(
                    "amount_sensitivity",
                    0.5,
                )
            ),

            float(
                beliefs.get(
                    "high_value_escalation",
                    0.5,
                )
            ),

            float(
                beliefs.get(
                    "followup_sensitivity",
                    0.5,
                )
            ),
        ]

        belief_shift = sum(
            abs(value - 0.5)
            for value in belief_values
        ) / 2.0

        belief_shift = min(
            belief_shift,
            1.0,
        )

        # ----------------------------------------------------
        # Final adaptation score
        # ----------------------------------------------------

        return min(
            1.0,
            0.5 * learning_signal
            + 0.5 * belief_shift,
        )

    # ========================================================
    # RISK CALCULATION
    # ========================================================

    def _calculate_risk(
        self,
        features: dict[str, float],
    ) -> tuple[float, list[str]]:

        score = 0.0

        reasons: list[str] = []

        # ----------------------------------------------------
        # Repeated denials
        # ----------------------------------------------------

        denied_count = features[
            "denied_count"
        ]

        if denied_count >= 3:

            score += 0.20

            reasons.append(
                "REPEATED_DENIALS"
            )

        elif denied_count >= 1:

            score += 0.05

        # ----------------------------------------------------
        # Claim switching
        # ----------------------------------------------------

        claim_switches = features[
            "claim_switch_count"
        ]

        if claim_switches >= 4:

            score += 0.25

            reasons.append(
                "FREQUENT_CLAIM_SWITCHING"
            )

        elif claim_switches >= 2:

            score += 0.10

        # ----------------------------------------------------
        # High-value request
        # ----------------------------------------------------

        if features[
            "high_value_flag"
        ]:

            score += 0.10

            reasons.append(
                "HIGH_VALUE_REQUEST"
            )

        # ----------------------------------------------------
        # Strategic adaptation
        # ----------------------------------------------------

        adaptation = features[
            "strategic_adaptation_score"
        ]

        if adaptation >= 0.70:

            # Strong strategic adaptation is a major
            # behavioral-risk signal.

            score += 0.30

            reasons.append(
                "STRATEGIC_ADAPTATION"
            )

        elif adaptation >= 0.40:

            score += 0.15

        # ----------------------------------------------------
        # Learned policy exploitation
        # ----------------------------------------------------

        evidence_sensitivity = features[
            "evidence_sensitivity"
        ]

        if (
            evidence_sensitivity >= 0.85
            and features[
                "evidence_present"
            ] == 1.0
        ):

            score += 0.10

            reasons.append(
                "POLICY_AWARE_EVIDENCE_BEHAVIOR"
            )

        # ----------------------------------------------------
        # Bound score
        # ----------------------------------------------------

        score = max(
            0.0,
            min(
                score,
                1.0,
            ),
        )

        return score, reasons

    # ========================================================
    # RISK LEVEL
    # ========================================================

    def _risk_level(
        self,
        score: float,
        features: dict[str, float] | None = None,
    ) -> str:

        features = features or {}

        strategic_adaptation = features.get(
            "strategic_adaptation_score",
            0.0,
        )

        # ----------------------------------------------------
        # STRATEGIC HIGH-RISK OVERRIDE
        # ----------------------------------------------------
        #
        # This is deliberately separate from the numerical
        # score threshold.
        #
        # A customer who has strongly learned Agent A's
        # behavior is itself a high-risk behavioral signal.
        #
        # This prevents a strong strategic customer from
        # being classified as MEDIUM merely because the
        # ordinary claim-level signals are weak.
        #

        if strategic_adaptation >= 0.70:

            return "HIGH"

        # ----------------------------------------------------
        # Standard score-based classification
        # ----------------------------------------------------

        if score < self.low_risk_threshold:

            return "LOW"

        if score < self.high_risk_threshold:

            return "MEDIUM"

        return "HIGH"

    # ========================================================
    # RECOMMENDED ACTION
    # ========================================================

    @staticmethod
    def _recommended_action(
        *,
        risk_level: str,
        support_decision: str | None,
    ) -> str:

        if risk_level == "LOW":

            return (
                "ALLOW_AGENT_A_DECISION"
            )

        if risk_level == "MEDIUM":

            return (
                "REQUEST_ADDITIONAL_EVIDENCE"
            )

        return (
            "ESCALATE_TO_HUMAN_REVIEW"
        )