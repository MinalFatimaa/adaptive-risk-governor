from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


# ============================================================
# ACTIONS
# ============================================================

ACTIONS = (
    "ALLOW_AGENT_A_DECISION",
    "REQUEST_ADDITIONAL_EVIDENCE",
    "ESCALATE_TO_HUMAN_REVIEW",
)


# ============================================================
# CASE OUTCOME
# ============================================================

@dataclass(frozen=True)
class EconomicCase:
    """
    Ground-truth economic information for one refund case.

    This represents what actually happened, independently
    from what the Governor predicted.
    """

    case_id: str

    requested_amount: float

    is_legitimate: bool

    fraud_loss_if_allowed: float

    friction_cost_if_verified: float

    review_cost_if_escalated: float

    recovery_if_escalated: float = 0.0


# ============================================================
# ECONOMIC RESULT
# ============================================================

@dataclass(frozen=True)
class EconomicOutcome:
    case_id: str

    action: str

    gross_loss: float

    friction_cost: float

    review_cost: float

    recovery: float

    net_cost: float

    loss_prevented: float

    legitimate_refund_preserved: float

    false_intervention: bool

    @property
    def net_benefit(self) -> float:
        """
        Positive means economically beneficial.

        Example:

            net_cost = -500

        means:

            net_benefit = +500
        """

        return -self.net_cost


# ============================================================
# AGGREGATE METRICS
# ============================================================

@dataclass
class EconomicMetrics:

    total_cases: int = 0

    total_requested_amount: float = 0.0

    total_net_cost: float = 0.0

    total_loss_prevented: float = 0.0

    total_legitimate_refunds_preserved: float = 0.0

    total_friction_cost: float = 0.0

    total_review_cost: float = 0.0

    false_interventions: int = 0

    fraud_cases: int = 0

    legitimate_cases: int = 0

    fraud_loss_exposure: float = 0.0

    fraud_loss_remaining: float = 0.0

    @property
    def total_net_benefit(self) -> float:
        """
        Positive = better economic outcome.
        """

        return -self.total_net_cost

    @property
    def false_intervention_rate(self) -> float:

        if self.legitimate_cases == 0:
            return 0.0

        return max(
            0.0,
            min(
                self.false_interventions
                / self.legitimate_cases,
                1.0,
            ),
        )

    @property
    def loss_prevention_rate(self) -> float:

        if self.fraud_loss_exposure <= 0.0:
            return 0.0

        rate = (
            self.total_loss_prevented
            / self.fraud_loss_exposure
        )

        return max(
            0.0,
            min(rate, 1.0),
        )

    @property
    def average_cost_per_case(self) -> float:

        if self.total_cases == 0:
            return 0.0

        return (
            self.total_net_cost
            / self.total_cases
        )

    @property
    def average_benefit_per_case(self) -> float:

        if self.total_cases == 0:
            return 0.0

        return (
            self.total_net_benefit
            / self.total_cases
        )


# ============================================================
# ECONOMIC MODEL
# ============================================================

class EconomicModel:
    """
    Converts Governor actions into financial outcomes.

    Architecture:

        Risk model
             ↓
        Governor action
             ↓
        Economic model
             ↓
        Financial outcome

    The economic model does NOT determine risk.

    It defines the financial consequences of the
    Governor's decision.

    This later becomes the reward function for
    the contextual bandit / RL policy.
    """

    @staticmethod
    def _non_negative(
        value: float,
    ) -> float:

        return max(
            float(value),
            0.0,
        )

    @staticmethod
    def _bounded(
        value: float,
        lower: float,
        upper: float,
    ) -> float:

        return max(
            lower,
            min(
                float(value),
                upper,
            ),
        )

    # ========================================================
    # SINGLE CASE
    # ========================================================

    def evaluate_case(
        self,
        case: EconomicCase,
        action: str,
    ) -> EconomicOutcome:

        if action not in ACTIONS:
            raise ValueError(
                f"Unknown action: {action}"
            )

        amount = self._non_negative(
            case.requested_amount
        )

        fraud_loss = self._non_negative(
            case.fraud_loss_if_allowed
        )

        friction_cost = self._non_negative(
            case.friction_cost_if_verified
        )

        review_cost = self._non_negative(
            case.review_cost_if_escalated
        )

        recovery_if_escalated = self._bounded(
            case.recovery_if_escalated,
            0.0,
            fraud_loss,
        )

        gross_loss = 0.0
        actual_friction_cost = 0.0
        actual_review_cost = 0.0
        recovery = 0.0
        loss_prevented = 0.0
        legitimate_refund_preserved = 0.0
        false_intervention = False

        # ====================================================
        # ALLOW
        # ====================================================

        if action == "ALLOW_AGENT_A_DECISION":

            if case.is_legitimate:

                legitimate_refund_preserved = amount

            else:

                gross_loss = fraud_loss

                loss_prevented = 0.0

        # ====================================================
        # REQUEST ADDITIONAL EVIDENCE
        # ====================================================

        elif action == "REQUEST_ADDITIONAL_EVIDENCE":

            actual_friction_cost = friction_cost

            if case.is_legitimate:

                # Legitimate customer eventually receives
                # the refund, but experiences friction.

                legitimate_refund_preserved = amount

            else:

                # In this economic environment, verification
                # successfully prevents the fraudulent refund.

                loss_prevented = fraud_loss

        # ====================================================
        # ESCALATE TO HUMAN
        # ====================================================

        elif action == "ESCALATE_TO_HUMAN_REVIEW":

            actual_review_cost = review_cost

            if case.is_legitimate:

                # Legitimate refund remains preserved,
                # but human review introduces intervention.

                legitimate_refund_preserved = amount

                false_intervention = True

            else:

                # Human review recovers some or all of the
                # potential fraudulent loss.

                recovery = recovery_if_escalated

                loss_prevented = recovery

        # ====================================================
        # HARD ECONOMIC INVARIANTS
        # ====================================================

        # A case can never prevent more loss than was exposed.

        loss_prevented = self._bounded(
            loss_prevented,
            0.0,
            fraud_loss,
        )

        # Recovery can never exceed fraud exposure.

        recovery = self._bounded(
            recovery,
            0.0,
            fraud_loss,
        )

        # Legitimate refund preservation cannot exceed
        # the requested amount.

        legitimate_refund_preserved = self._bounded(
            legitimate_refund_preserved,
            0.0,
            amount,
        )

        # Gross loss is bounded as well.

        gross_loss = self._bounded(
            gross_loss,
            0.0,
            fraud_loss,
        )

        # ====================================================
        # NET COST
        # ====================================================

        net_cost = (
            gross_loss
            + actual_friction_cost
            + actual_review_cost
            - recovery
        )

        return EconomicOutcome(
            case_id=case.case_id,
            action=action,
            gross_loss=gross_loss,
            friction_cost=actual_friction_cost,
            review_cost=actual_review_cost,
            recovery=recovery,
            net_cost=net_cost,
            loss_prevented=loss_prevented,
            legitimate_refund_preserved=(
                legitimate_refund_preserved
            ),
            false_intervention=(
                false_intervention
            ),
        )

    # ========================================================
    # DATASET EVALUATION
    # ========================================================

    def evaluate_dataset(
        self,
        cases: Iterable[EconomicCase],
        actions: Iterable[str],
    ) -> tuple[list[EconomicOutcome], EconomicMetrics]:

        cases = list(cases)
        actions = list(actions)

        if len(cases) != len(actions):
            raise ValueError(
                "Number of cases and actions "
                "must be identical."
            )

        outcomes: list[EconomicOutcome] = []

        metrics = EconomicMetrics()

        for case, action in zip(
            cases,
            actions,
        ):

            outcome = self.evaluate_case(
                case=case,
                action=action,
            )

            outcomes.append(outcome)

            metrics.total_cases += 1

            metrics.total_requested_amount += (
                self._non_negative(
                    case.requested_amount
                )
            )

            metrics.total_net_cost += (
                outcome.net_cost
            )

            metrics.total_loss_prevented += (
                outcome.loss_prevented
            )

            metrics.total_legitimate_refunds_preserved += (
                outcome.legitimate_refund_preserved
            )

            metrics.total_friction_cost += (
                outcome.friction_cost
            )

            metrics.total_review_cost += (
                outcome.review_cost
            )

            if outcome.false_intervention:
                metrics.false_interventions += 1

            if case.is_legitimate:

                metrics.legitimate_cases += 1

            else:

                metrics.fraud_cases += 1

                fraud_exposure = self._non_negative(
                    case.fraud_loss_if_allowed
                )

                metrics.fraud_loss_exposure += (
                    fraud_exposure
                )

                # Remaining loss is exposure minus the
                # amount actually prevented/recovered.

                remaining_loss = max(
                    fraud_exposure
                    - outcome.loss_prevented,
                    0.0,
                )

                metrics.fraud_loss_remaining += (
                    remaining_loss
                )

        # ====================================================
        # FINAL AGGREGATE SAFETY BOUNDS
        # ====================================================

        # Total prevented loss can never exceed total exposure.

        metrics.total_loss_prevented = min(
            max(
                metrics.total_loss_prevented,
                0.0,
            ),
            metrics.fraud_loss_exposure,
        )

        # Remaining fraud loss is always:

        # exposure - prevented

        metrics.fraud_loss_remaining = max(
            metrics.fraud_loss_exposure
            - metrics.total_loss_prevented,
            0.0,
        )

        # Legitimate refund preservation cannot be negative.

        metrics.total_legitimate_refunds_preserved = max(
            metrics.total_legitimate_refunds_preserved,
            0.0,
        )

        return outcomes, metrics


# ============================================================
# BASELINE ACTION POLICIES
# ============================================================

def allow_everything_policy(
    cases: Iterable[EconomicCase],
) -> list[str]:

    return [
        "ALLOW_AGENT_A_DECISION"
        for _ in cases
    ]


def verify_everything_policy(
    cases: Iterable[EconomicCase],
) -> list[str]:

    return [
        "REQUEST_ADDITIONAL_EVIDENCE"
        for _ in cases
    ]


def escalate_everything_policy(
    cases: Iterable[EconomicCase],
) -> list[str]:

    return [
        "ESCALATE_TO_HUMAN_REVIEW"
        for _ in cases
    ]


# ============================================================
# RISK-BASED POLICY
# ============================================================

def risk_based_policy(
    risk_scores: Iterable[float],
    *,
    verify_threshold: float = 0.30,
    escalate_threshold: float = 0.70,
) -> list[str]:

    if not (
        0.0
        <= verify_threshold
        < escalate_threshold
        <= 1.0
    ):
        raise ValueError(
            "Thresholds must satisfy "
            "0 <= verify < escalate <= 1."
        )

    actions: list[str] = []

    for raw_score in risk_scores:

        score = max(
            0.0,
            min(
                float(raw_score),
                1.0,
            ),
        )

        if score < verify_threshold:

            actions.append(
                "ALLOW_AGENT_A_DECISION"
            )

        elif score < escalate_threshold:

            actions.append(
                "REQUEST_ADDITIONAL_EVIDENCE"
            )

        else:

            actions.append(
                "ESCALATE_TO_HUMAN_REVIEW"
            )

    return actions


# ============================================================
# ECONOMIC COMPARISON
# ============================================================

@dataclass(frozen=True)
class PolicyComparison:

    policy_name: str

    # Kept for backward compatibility.
    total_cost: float

    loss_prevention_rate: float

    false_intervention_rate: float

    legitimate_refunds_preserved: float

    @property
    def net_benefit(self) -> float:
        """
        Positive = economically better.
        """

        return -self.total_cost


def compare_policies(
    cases: Iterable[EconomicCase],
    policies: dict[str, list[str]],
) -> list[PolicyComparison]:

    cases = list(cases)

    model = EconomicModel()

    comparisons: list[PolicyComparison] = []

    for policy_name, actions in policies.items():

        _, metrics = model.evaluate_dataset(
            cases,
            actions,
        )

        comparisons.append(
            PolicyComparison(
                policy_name=policy_name,
                total_cost=metrics.total_net_cost,
                loss_prevention_rate=(
                    metrics.loss_prevention_rate
                ),
                false_intervention_rate=(
                    metrics.false_intervention_rate
                ),
                legitimate_refunds_preserved=(
                    metrics
                    .total_legitimate_refunds_preserved
                ),
            )
        )

    return comparisons