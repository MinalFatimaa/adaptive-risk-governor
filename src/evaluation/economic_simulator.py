from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import math


# ============================================================
# CANONICAL GOVERNOR ACTION SPACE
# ============================================================


class GovernorAction(str, Enum):

    ALLOW_AGENT_A_DECISION = (
        "ALLOW_AGENT_A_DECISION"
    )

    REQUEST_ADDITIONAL_EVIDENCE = (
        "REQUEST_ADDITIONAL_EVIDENCE"
    )

    ESCALATE_TO_HUMAN_REVIEW = (
        "ESCALATE_TO_HUMAN_REVIEW"
    )


# ============================================================
# SUPPORT AGENT DECISION SPACE
# ============================================================


class SupportDecision(str, Enum):

    APPROVE = "APPROVE"

    REQUEST_EVIDENCE = "REQUEST_EVIDENCE"

    ESCALATE = "ESCALATE"

    DENY = "DENY"


# ============================================================
# CASE TRUTH
# ============================================================


class CaseTruth(str, Enum):

    LEGITIMATE = "LEGIT"

    FRAUD = "FRAUD"


# ============================================================
# ECONOMIC CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class EconomicCosts:
    """
    Costs associated with Governor interventions.

    These are economic assumptions and are not model
    predictions.
    """

    evidence_request_cost: float = 25.0

    human_review_cost: float = 50.0

    false_intervention_cost: float = 0.0

    def __post_init__(self) -> None:

        values = (
            self.evidence_request_cost,
            self.human_review_cost,
            self.false_intervention_cost,
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "Economic costs must be finite."
            )

        if any(
            value < 0.0
            for value in values
        ):
            raise ValueError(
                "Economic costs cannot be negative."
            )


# ============================================================
# ECONOMIC CASE
# ============================================================


@dataclass(frozen=True)
class EconomicCase:
    """
    One economic evaluation case.

    IMPORTANT
    ---------
    `truth` is the evaluation label.

    The Governor must NOT use truth to determine its action.

    Instead, evidence_resolution and human_review_resolution
    describe what actually happens after an intervention.

    This allows the economic simulator to model imperfect
    interventions without giving the Governor access to
    ground truth.
    """

    case_id: str

    truth: CaseTruth

    requested_amount: float

    fraud_loss_exposure: float

    legitimate_refund_amount: float

    # --------------------------------------------------------
    # Outcome of requesting additional evidence.
    #
    # This is hidden from the Governor.
    # --------------------------------------------------------

    evidence_resolution: SupportDecision | None = None

    # --------------------------------------------------------
    # Outcome of human review.
    #
    # This is hidden from the Governor.
    # --------------------------------------------------------

    human_review_resolution: SupportDecision | None = None

    def __post_init__(self) -> None:

        if not self.case_id:
            raise ValueError(
                "case_id cannot be empty."
            )

        numeric_values = (
            self.requested_amount,
            self.fraud_loss_exposure,
            self.legitimate_refund_amount,
        )

        if not all(
            math.isfinite(value)
            for value in numeric_values
        ):
            raise ValueError(
                "Economic case values must be finite."
            )

        if self.requested_amount < 0.0:
            raise ValueError(
                "requested_amount cannot be negative."
            )

        if self.fraud_loss_exposure < 0.0:
            raise ValueError(
                "fraud_loss_exposure cannot be negative."
            )

        if self.legitimate_refund_amount < 0.0:
            raise ValueError(
                "legitimate_refund_amount cannot be negative."
            )

        # ----------------------------------------------------
        # Truth consistency.
        # ----------------------------------------------------

        if self.truth == CaseTruth.FRAUD:

            if self.legitimate_refund_amount != 0.0:

                raise ValueError(
                    "Fraud cases cannot have a legitimate "
                    "refund amount."
                )

        elif self.truth == CaseTruth.LEGITIMATE:

            if self.fraud_loss_exposure != 0.0:

                raise ValueError(
                    "Legitimate cases cannot have fraud "
                    "loss exposure."
                )

        else:

            raise ValueError(
                f"Unsupported case truth: {self.truth}"
            )

        # ----------------------------------------------------
        # Intervention resolutions must be terminal support
        # decisions.
        # ----------------------------------------------------

        terminal_decisions = {
            SupportDecision.APPROVE,
            SupportDecision.DENY,
            SupportDecision.ESCALATE,
        }

        if (
            self.evidence_resolution is not None
            and self.evidence_resolution
            not in terminal_decisions
        ):

            raise ValueError(
                "evidence_resolution must be APPROVE, "
                "DENY, or ESCALATE."
            )

        if (
            self.human_review_resolution is not None
            and self.human_review_resolution
            not in terminal_decisions
        ):

            raise ValueError(
                "human_review_resolution must be "
                "APPROVE, DENY, or ESCALATE."
            )


# ============================================================
# ECONOMIC OUTCOME
# ============================================================


@dataclass(frozen=True)
class EconomicOutcome:

    case_id: str

    governor_action: GovernorAction

    effective_decision: SupportDecision

    truth: CaseTruth

    fraud_loss_exposure: float

    loss_prevented: float

    remaining_fraud_loss: float

    legitimate_refund_preserved: float

    friction_cost: float

    human_review_cost: float

    false_intervention: bool

    net_benefit: float

    def __post_init__(self) -> None:

        numeric_values = (
            self.fraud_loss_exposure,
            self.loss_prevented,
            self.remaining_fraud_loss,
            self.legitimate_refund_preserved,
            self.friction_cost,
            self.human_review_cost,
            self.net_benefit,
        )

        if not all(
            math.isfinite(value)
            for value in numeric_values
        ):
            raise ValueError(
                "Economic outcome contains "
                "non-finite values."
            )

        if self.fraud_loss_exposure < 0.0:

            raise ValueError(
                "Fraud exposure cannot be negative."
            )

        if self.loss_prevented < 0.0:

            raise ValueError(
                "Loss prevented cannot be negative."
            )

        if (
            self.loss_prevented
            > self.fraud_loss_exposure + 1e-9
        ):

            raise ValueError(
                "Loss prevented cannot exceed "
                "fraud exposure."
            )

        if self.remaining_fraud_loss < 0.0:

            raise ValueError(
                "Remaining fraud loss cannot "
                "be negative."
            )

        if self.legitimate_refund_preserved < 0.0:

            raise ValueError(
                "Legitimate refund preserved cannot "
                "be negative."
            )

        if self.friction_cost < 0.0:

            raise ValueError(
                "Friction cost cannot be negative."
            )

        if self.human_review_cost < 0.0:

            raise ValueError(
                "Human review cost cannot be negative."
            )


# ============================================================
# NORMALIZATION
# ============================================================


def normalize_governor_action(
    action: str | GovernorAction,
) -> GovernorAction:

    if isinstance(
        action,
        GovernorAction,
    ):

        return action

    try:

        return GovernorAction(action)

    except ValueError as exc:

        raise ValueError(
            f"Unknown Governor action: {action}"
        ) from exc


def normalize_support_decision(
    decision: str | SupportDecision,
) -> SupportDecision:

    if isinstance(
        decision,
        SupportDecision,
    ):

        return decision

    try:

        return SupportDecision(decision)

    except ValueError as exc:

        raise ValueError(
            f"Unknown support decision: {decision}"
        ) from exc


# ============================================================
# GOVERNOR ACTION → EFFECTIVE DECISION
# ============================================================


def resolve_effective_decision(
    *,
    governor_action: str | GovernorAction,
    agent_decision: str | SupportDecision,
) -> SupportDecision:

    action = normalize_governor_action(
        governor_action
    )

    decision = normalize_support_decision(
        agent_decision
    )

    if (
        action
        == GovernorAction.ALLOW_AGENT_A_DECISION
    ):

        return decision

    if (
        action
        == GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
    ):

        return SupportDecision.REQUEST_EVIDENCE

    if (
        action
        == GovernorAction.ESCALATE_TO_HUMAN_REVIEW
    ):

        return SupportDecision.ESCALATE

    raise ValueError(
        f"Unsupported Governor action: {action}"
    )


# ============================================================
# TERMINAL ECONOMIC RESOLUTION
# ============================================================


def _resolve_intervention_outcome(
    *,
    case: EconomicCase,
    action: GovernorAction,
    effective_decision: SupportDecision,
) -> SupportDecision:

    # --------------------------------------------------------
    # Evidence request
    # --------------------------------------------------------

    if (
        action
        == GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
    ):

        if case.evidence_resolution is None:

            raise ValueError(
                "EconomicCase must define "
                "evidence_resolution when evaluating "
                "REQUEST_ADDITIONAL_EVIDENCE."
            )

        return case.evidence_resolution

    # --------------------------------------------------------
    # Human escalation
    # --------------------------------------------------------

    if (
        action
        == GovernorAction.ESCALATE_TO_HUMAN_REVIEW
    ):

        if case.human_review_resolution is None:

            raise ValueError(
                "EconomicCase must define "
                "human_review_resolution when evaluating "
                "ESCALATE_TO_HUMAN_REVIEW."
            )

        return case.human_review_resolution

    # --------------------------------------------------------
    # Allow preserves the SupportAgent decision.
    # --------------------------------------------------------

    return effective_decision


# ============================================================
# ECONOMIC SIMULATION
# ============================================================


def simulate_economic_outcome(
    *,
    case: EconomicCase,
    governor_action: str | GovernorAction,
    agent_decision: str | SupportDecision,
    costs: EconomicCosts | None = None,
) -> EconomicOutcome:
    """
    Simulate the economic consequence of the actual
    Governor + SupportAgent decision flow.

    Ground truth is NEVER used to determine the Governor
    action or the intervention resolution.

    Ground truth is used only after the final decision has
    been resolved, to calculate the economic consequence.
    """

    if costs is None:

        costs = EconomicCosts()

    action = normalize_governor_action(
        governor_action
    )

    effective_decision = resolve_effective_decision(
        governor_action=action,
        agent_decision=agent_decision,
    )

    # --------------------------------------------------------
    # Resolve intervention outcome.
    # --------------------------------------------------------

    terminal_decision = (
        _resolve_intervention_outcome(
            case=case,
            action=action,
            effective_decision=effective_decision,
        )
    )

    # --------------------------------------------------------
    # Costs caused by Governor intervention.
    # --------------------------------------------------------

    friction_cost = 0.0

    human_review_cost = 0.0

    if (
        action
        == GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
    ):

        friction_cost = (
            costs.evidence_request_cost
        )

    elif (
        action
        == GovernorAction.ESCALATE_TO_HUMAN_REVIEW
    ):

        human_review_cost = (
            costs.human_review_cost
        )

    # --------------------------------------------------------
    # Evaluate the terminal decision against truth.
    #
    # Ground truth enters ONLY here.
    # --------------------------------------------------------

    loss_prevented = 0.0

    remaining_fraud_loss = 0.0

    legitimate_refund_preserved = 0.0

    # ========================================================
    # APPROVE
    # ========================================================

    if (
        terminal_decision
        == SupportDecision.APPROVE
    ):

        if case.truth == CaseTruth.FRAUD:

            # Fraud was approved, therefore the entire fraud
            # exposure remains as economic loss.
            loss_prevented = 0.0

            remaining_fraud_loss = (
                case.fraud_loss_exposure
            )

        else:

            # Legitimate request was approved, therefore the
            # legitimate refund is preserved.
            legitimate_refund_preserved = (
                case.legitimate_refund_amount
            )

    # ========================================================
    # DENY
    # ========================================================

    elif (
        terminal_decision
        == SupportDecision.DENY
    ):

        if case.truth == CaseTruth.FRAUD:

            # Correctly denying fraud prevents the fraud loss.
            loss_prevented = (
                case.fraud_loss_exposure
            )

            remaining_fraud_loss = 0.0

        else:

            # Incorrectly denying a legitimate request means
            # the legitimate customer loses the refund.
            legitimate_refund_preserved = 0.0

    # ========================================================
    # ESCALATE
    # ========================================================

    elif (
        terminal_decision
        == SupportDecision.ESCALATE
    ):

        raise ValueError(
            "Economic simulation requires escalation "
            "to resolve to APPROVE or DENY."
        )

    # ========================================================
    # REQUEST_EVIDENCE
    # ========================================================

    elif (
        terminal_decision
        == SupportDecision.REQUEST_EVIDENCE
    ):

        raise ValueError(
            "Economic simulation requires evidence "
            "resolution to be APPROVE or DENY."
        )

    else:

        raise ValueError(
            "Unsupported terminal decision: "
            f"{terminal_decision}"
        )

    # ========================================================
    # FALSE INTERVENTION
    # ========================================================

    false_intervention = False

    if case.truth == CaseTruth.LEGITIMATE:

        # Any additional evidence or human-review intervention
        # on a legitimate case is counted as an intervention.
        if (
            action
            == GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
        ):

            false_intervention = True

        elif (
            action
            == GovernorAction.ESCALATE_TO_HUMAN_REVIEW
        ):

            false_intervention = True

        # A legitimate case that ends in DENY is also a false
        # intervention because the legitimate refund was lost.
        if (
            terminal_decision
            == SupportDecision.DENY
        ):

            false_intervention = True

    # ========================================================
    # ECONOMIC BENEFIT
    # ========================================================
    #
    # Correct economic accounting:
    #
    # Fraud:
    #   APPROVE -> -remaining fraud loss
    #   DENY    -> +loss prevented
    #
    # Legitimate:
    #   APPROVE -> +refund preserved
    #   DENY    -> -lost legitimate refund
    #
    # Therefore the value formula is based on the final
    # economic outcome, not simply on fraud exposure.
    # ========================================================

    if case.truth == CaseTruth.FRAUD:

        gross_economic_benefit = (
            loss_prevented
            - remaining_fraud_loss
        )

    else:

        gross_economic_benefit = (
            legitimate_refund_preserved
            - (
                case.legitimate_refund_amount
                - legitimate_refund_preserved
            )
        )

    net_benefit = (
        gross_economic_benefit
        - friction_cost
        - human_review_cost
        - (
            costs.false_intervention_cost
            if false_intervention
            else 0.0
        )
    )

    return EconomicOutcome(
        case_id=case.case_id,
        governor_action=action,
        effective_decision=terminal_decision,
        truth=case.truth,
        fraud_loss_exposure=(
            case.fraud_loss_exposure
        ),
        loss_prevented=loss_prevented,
        remaining_fraud_loss=(
            remaining_fraud_loss
        ),
        legitimate_refund_preserved=(
            legitimate_refund_preserved
        ),
        friction_cost=friction_cost,
        human_review_cost=human_review_cost,
        false_intervention=false_intervention,
        net_benefit=net_benefit,
    )


# ============================================================
# BATCH SUMMARY
# ============================================================


@dataclass(frozen=True)
class EconomicSummary:

    total_cases: int

    fraud_cases: int

    legitimate_cases: int

    fraud_exposure: float

    loss_prevented: float

    remaining_fraud_loss: float

    legitimate_refunds_preserved: float

    friction_cost: float

    human_review_cost: float

    false_interventions: int

    false_intervention_rate: float

    loss_prevention_rate: float

    net_economic_benefit: float

    average_benefit_per_case: float


def summarize_economic_outcomes(
    outcomes: Iterable[EconomicOutcome],
) -> EconomicSummary:

    outcomes = list(outcomes)

    if not outcomes:

        raise ValueError(
            "Cannot summarize an empty outcome set."
        )

    total_cases = len(outcomes)

    fraud_cases = sum(
        outcome.truth == CaseTruth.FRAUD
        for outcome in outcomes
    )

    legitimate_cases = sum(
        outcome.truth == CaseTruth.LEGITIMATE
        for outcome in outcomes
    )

    fraud_exposure = sum(
        outcome.fraud_loss_exposure
        for outcome in outcomes
    )

    loss_prevented = sum(
        outcome.loss_prevented
        for outcome in outcomes
    )

    remaining_fraud_loss = sum(
        outcome.remaining_fraud_loss
        for outcome in outcomes
    )

    legitimate_refunds_preserved = sum(
        outcome.legitimate_refund_preserved
        for outcome in outcomes
    )

    friction_cost = sum(
        outcome.friction_cost
        for outcome in outcomes
    )

    human_review_cost = sum(
        outcome.human_review_cost
        for outcome in outcomes
    )

    false_interventions = sum(
        outcome.false_intervention
        for outcome in outcomes
    )

    loss_prevention_rate = (
        loss_prevented / fraud_exposure
        if fraud_exposure > 0.0
        else 0.0
    )

    false_intervention_rate = (
        false_interventions / legitimate_cases
        if legitimate_cases > 0
        else 0.0
    )

    net_economic_benefit = sum(
        outcome.net_benefit
        for outcome in outcomes
    )

    average_benefit_per_case = (
        net_economic_benefit
        / total_cases
    )

    return EconomicSummary(
        total_cases=total_cases,
        fraud_cases=fraud_cases,
        legitimate_cases=legitimate_cases,
        fraud_exposure=fraud_exposure,
        loss_prevented=loss_prevented,
        remaining_fraud_loss=remaining_fraud_loss,
        legitimate_refunds_preserved=(
            legitimate_refunds_preserved
        ),
        friction_cost=friction_cost,
        human_review_cost=human_review_cost,
        false_interventions=(
            false_interventions
        ),
        false_intervention_rate=(
            false_intervention_rate
        ),
        loss_prevention_rate=(
            loss_prevention_rate
        ),
        net_economic_benefit=(
            net_economic_benefit
        ),
        average_benefit_per_case=(
            average_benefit_per_case
        ),
    )


# ============================================================
# FIXED POLICY EVALUATION
# ============================================================


def evaluate_policy(
    *,
    cases: Iterable[EconomicCase],
    governor_action: str | GovernorAction,
    agent_decision: str | SupportDecision,
    costs: EconomicCosts | None = None,
) -> EconomicSummary:
    """
    Evaluate one fixed Governor policy.

    Only the canonical Governor actions are accepted.

    The economic simulator never creates artificial
    Governor actions such as VERIFY_ALL or ESCALATE_ALL.
    """

    outcomes = [
        simulate_economic_outcome(
            case=case,
            governor_action=governor_action,
            agent_decision=agent_decision,
            costs=costs,
        )
        for case in cases
    ]

    return summarize_economic_outcomes(
        outcomes
    )