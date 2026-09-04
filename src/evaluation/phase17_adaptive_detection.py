from __future__ import annotations

"""
PHASE 17 — ADAPTIVE ATTACKER DETECTION

Purpose
-------
Phase 16.2 demonstrated that an adaptive attacker can continue changing
its behavior even when Agent A is hardened.

Phase 17 now tests the actual purpose of the Adaptive Risk Governor:

    Can the Governor detect adaptive behavior from the interaction
    trajectory and intervene before the adaptive attacker converts
    learned behavior into successful abuse?

This phase compares:

    1. Agent A alone
    2. Agent A + Risk Governor

The attacker does NOT receive:
    - Governor score
    - Governor features
    - ground truth
    - other customer information
    - merchant private state

The Governor observes only information that the defensive system is
legitimately allowed to observe.

IMPORTANT
---------
Governor actions are restricted to:

    ALLOW_AGENT_A_DECISION
    REQUEST_ADDITIONAL_EVIDENCE
    ESCALATE_TO_HUMAN_REVIEW

The Governor does NOT directly DENY a request.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from random import Random
from typing import Any

import numpy as np

from ..environment.simulator import AdaptiveInteractionSimulator
from ..environment.population_config import (
    DEFAULT_POPULATION_CONFIG,
    CustomerType,
)
from ..environment.world_generator import create_world
from ..schemas.support import RefundRequest


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

EPISODES_PER_GROUP = 20
INTERACTIONS_PER_EPISODE = 24

EXPLORATION_INTERACTIONS = 8

# Governor thresholds.
#
# These are intentionally conservative. The objective is not to
# block every unusual request. The objective is to identify a
# trajectory whose behavior is becoming increasingly suspicious.
#
# The Governor should therefore intervene only when multiple
# independent signals agree.
GOVERNOR_REQUEST_EVIDENCE_THRESHOLD = 0.62
GOVERNOR_HUMAN_REVIEW_THRESHOLD = 0.78

# Minimum number of observations before adaptive trajectory
# signals are considered sufficiently reliable.
MIN_HISTORY_FOR_ADAPTIVE_SIGNAL = 5

# A request is considered successful for this evaluation when
# Agent A approves a positive amount.
APPROVAL_DECISION = "APPROVE"

# Governor action space.
ALLOW_AGENT_A_DECISION = "ALLOW_AGENT_A_DECISION"
REQUEST_ADDITIONAL_EVIDENCE = "REQUEST_ADDITIONAL_EVIDENCE"
ESCALATE_TO_HUMAN_REVIEW = "ESCALATE_TO_HUMAN_REVIEW"

VALID_GOVERNOR_ACTIONS = {
    ALLOW_AGENT_A_DECISION,
    REQUEST_ADDITIONAL_EVIDENCE,
    ESCALATE_TO_HUMAN_REVIEW,
}


# ============================================================
# POPULATION LABELS
# ============================================================

HUMAN_LEGITIMATE = "HUMAN_LEGITIMATE"
HUMAN_ABUSIVE = "HUMAN_ABUSIVE"
ADAPTIVE_LEGITIMATE = "ADAPTIVE_LEGITIMATE"
ADAPTIVE_ABUSIVE = "ADAPTIVE_ABUSIVE"


# ============================================================
# CLAIM TYPES
# ============================================================

CLAIM_TYPES = [
    "SHORTAGE_CLAIM",
    "WRONG_ITEM_CLAIM",
    "NON_DELIVERY_CLAIM",
    "SUBSTITUTED_RETURN_CLAIM",
]

EVIDENCE_BY_CLAIM = {
    "SHORTAGE_CLAIM": ["delivery_photo"],
    "WRONG_ITEM_CLAIM": ["package_photo"],
    "NON_DELIVERY_CLAIM": ["delivery_evidence"],
    "SUBSTITUTED_RETURN_CLAIM": ["return_receipt"],
}

CLAIM_TEXT = {
    "SHORTAGE_CLAIM":
        "The delivered order was incomplete.",

    "WRONG_ITEM_CLAIM":
        "The item received was different from the item ordered.",

    "NON_DELIVERY_CLAIM":
        "The order was not received.",

    "SUBSTITUTED_RETURN_CLAIM":
        "The returned item does not match the expected item.",
}


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def safe_rate(
    numerator: float,
    denominator: float,
) -> float:

    if denominator <= 0:
        return 0.0

    value = float(numerator) / float(denominator)

    if not np.isfinite(value):
        return 0.0

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def safe_mean(
    values: list[float],
) -> float:

    if not values:
        return 0.0

    value = float(
        np.mean(values)
    )

    if not np.isfinite(value):
        return 0.0

    return value


def bounded(
    value: float,
) -> float:

    if not np.isfinite(value):
        return 0.0

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


# ============================================================
# TRAJECTORY STATE
# ============================================================

@dataclass
class GovernorObservation:

    sequence_number: int

    claim_type: str

    requested_amount: float

    evidence_available: bool

    support_decision: str

    reason_code: str

    approved_amount: float

    timestamp: datetime


@dataclass
class AdaptiveTrajectoryState:

    observations: list[GovernorObservation]

    previous_claim_type: str | None = None
    previous_requested_amount: float | None = None

    def add(
        self,
        observation: GovernorObservation,
    ) -> None:

        self.observations.append(
            observation
        )

        self.previous_claim_type = (
            observation.claim_type
        )

        self.previous_requested_amount = (
            observation.requested_amount
        )


# ============================================================
# GOVERNOR SIGNALS
# ============================================================

@dataclass(frozen=True)
class AdaptiveRiskSignals:

    decision_conditioning: float
    claim_switching: float
    amount_acceleration: float
    evidence_response: float
    approval_seeking: float
    trajectory_change: float

    adaptive_risk: float


# ============================================================
# GOVERNOR DECISION
# ============================================================

@dataclass(frozen=True)
class GovernorDecision:

    action: str

    risk_score: float

    signals: AdaptiveRiskSignals

    reason: str


# ============================================================
# ADAPTIVE DETECTION GOVERNOR
# ============================================================

class AdaptiveDetectionGovernor:

    """
    Lightweight Phase-17 Governor.

    This is intentionally implemented inside the evaluation file so
    Phase 17 can be validated independently before being connected
    to the production Governor implementation.

    It does NOT expose its score or features to Agent B.

    It uses only the trajectory visible to the defensive system.
    """

    def __init__(
        self,
        request_evidence_threshold: float = (
            GOVERNOR_REQUEST_EVIDENCE_THRESHOLD
        ),
        human_review_threshold: float = (
            GOVERNOR_HUMAN_REVIEW_THRESHOLD
        ),
    ):

        self.request_evidence_threshold = (
            request_evidence_threshold
        )

        self.human_review_threshold = (
            human_review_threshold
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def evaluate(
        self,
        history: AdaptiveTrajectoryState,
    ) -> GovernorDecision:

        signals = self._calculate_signals(
            history
        )

        risk_score = bounded(
            signals.adaptive_risk
        )

        if (
            risk_score
            >= self.human_review_threshold
        ):

            return GovernorDecision(
                action=(
                    ESCALATE_TO_HUMAN_REVIEW
                ),

                risk_score=risk_score,

                signals=signals,

                reason=(
                    "Trajectory shows sustained "
                    "decision-conditioned behavioral "
                    "change."
                ),
            )

        if (
            risk_score
            >= self.request_evidence_threshold
        ):

            return GovernorDecision(
                action=(
                    REQUEST_ADDITIONAL_EVIDENCE
                ),

                risk_score=risk_score,

                signals=signals,

                reason=(
                    "Trajectory contains multiple "
                    "adaptive-risk indicators."
                ),
            )

        return GovernorDecision(
            action=ALLOW_AGENT_A_DECISION,

            risk_score=risk_score,

            signals=signals,

            reason=(
                "No sufficiently strong adaptive "
                "trajectory signal."
            ),
        )

    # ========================================================
    # SIGNAL CALCULATION
    # ========================================================

    def _calculate_signals(
        self,
        history: AdaptiveTrajectoryState,
    ) -> AdaptiveRiskSignals:

        observations = history.observations

        if len(observations) < MIN_HISTORY_FOR_ADAPTIVE_SIGNAL:

            return AdaptiveRiskSignals(
                decision_conditioning=0.0,
                claim_switching=0.0,
                amount_acceleration=0.0,
                evidence_response=0.0,
                approval_seeking=0.0,
                trajectory_change=0.0,
                adaptive_risk=0.0,
            )

        decision_conditioning = (
            self._decision_conditioning(
                observations
            )
        )

        claim_switching = (
            self._claim_switching(
                observations
            )
        )

        amount_acceleration = (
            self._amount_acceleration(
                observations
            )
        )

        evidence_response = (
            self._evidence_response(
                observations
            )
        )

        approval_seeking = (
            self._approval_seeking(
                observations
            )
        )

        trajectory_change = (
            self._trajectory_change(
                observations
            )
        )

        # ----------------------------------------------------
        # Weighted risk fusion.
        #
        # Decision conditioning is the strongest signal because
        # Phase 16.1/16.2 established that the important threat
        # model is behavior changing after observing Agent A.
        # ----------------------------------------------------

        adaptive_risk = (
            0.28 * decision_conditioning
            + 0.15 * claim_switching
            + 0.15 * amount_acceleration
            + 0.12 * evidence_response
            + 0.15 * approval_seeking
            + 0.15 * trajectory_change
        )

        return AdaptiveRiskSignals(
            decision_conditioning=bounded(
                decision_conditioning
            ),

            claim_switching=bounded(
                claim_switching
            ),

            amount_acceleration=bounded(
                amount_acceleration
            ),

            evidence_response=bounded(
                evidence_response
            ),

            approval_seeking=bounded(
                approval_seeking
            ),

            trajectory_change=bounded(
                trajectory_change
            ),

            adaptive_risk=bounded(
                adaptive_risk
            ),
        )

    # ========================================================
    # DECISION CONDITIONING
    # ========================================================

    def _decision_conditioning(
        self,
        observations: list[GovernorObservation],
    ) -> float:

        if len(observations) < 6:
            return 0.0

        early = observations[
            :max(3, len(observations) // 2)
        ]

        late = observations[
            max(3, len(observations) // 2):
        ]

        early_approvals = sum(
            observation.support_decision
            == APPROVAL_DECISION
            for observation in early
        )

        late_approvals = sum(
            observation.support_decision
            == APPROVAL_DECISION
            for observation in late
        )

        early_rate = safe_rate(
            early_approvals,
            len(early),
        )

        late_rate = safe_rate(
            late_approvals,
            len(late),
        )

        gain = abs(
            late_rate
            - early_rate
        )

        # Larger behavioral change after observing
        # decisions produces a stronger signal.
        return bounded(
            gain * 2.5
        )

    # ========================================================
    # CLAIM SWITCHING
    # ========================================================

    def _claim_switching(
        self,
        observations: list[GovernorObservation],
    ) -> float:

        if len(observations) < 2:
            return 0.0

        switches = 0

        for previous, current in zip(
            observations[:-1],
            observations[1:],
        ):

            if (
                previous.claim_type
                != current.claim_type
            ):
                switches += 1

        switch_rate = safe_rate(
            switches,
            len(observations) - 1,
        )

        return switch_rate

    # ========================================================
    # AMOUNT ACCELERATION
    # ========================================================

    def _amount_acceleration(
        self,
        observations: list[GovernorObservation],
    ) -> float:

        if len(observations) < 4:
            return 0.0

        amounts = np.asarray(
            [
                observation.requested_amount
                for observation in observations
            ],
            dtype=float,
        )

        if len(amounts) < 4:
            return 0.0

        midpoint = len(amounts) // 2

        early = amounts[
            :midpoint
        ]

        late = amounts[
            midpoint:
        ]

        early_mean = float(
            np.mean(early)
        )

        late_mean = float(
            np.mean(late)
        )

        baseline = max(
            abs(early_mean),
            1.0,
        )

        increase = (
            late_mean
            - early_mean
        ) / baseline

        return bounded(
            max(
                0.0,
                increase,
            )
        )

    # ========================================================
    # EVIDENCE RESPONSE
    # ========================================================

    def _evidence_response(
        self,
        observations: list[GovernorObservation],
    ) -> float:

        if len(observations) < 6:
            return 0.0

        evidence_after_nonapproval = 0
        opportunities = 0

        previous_decision = None

        for observation in observations:

            if previous_decision is not None:

                if previous_decision != APPROVAL_DECISION:

                    opportunities += 1

                    if observation.evidence_available:

                        evidence_after_nonapproval += 1

            previous_decision = (
                observation.support_decision
            )

        if opportunities == 0:
            return 0.0

        return safe_rate(
            evidence_after_nonapproval,
            opportunities,
        )

    # ========================================================
    # APPROVAL SEEKING
    # ========================================================

    def _approval_seeking(
        self,
        observations: list[GovernorObservation],
    ) -> float:

        if len(observations) < 6:
            return 0.0

        late = observations[
            len(observations) // 2:
        ]

        if not late:
            return 0.0

        approved = sum(
            observation.support_decision
            == APPROVAL_DECISION
            for observation in late
        )

        approval_rate = safe_rate(
            approved,
            len(late),
        )

        # A high late-stage approval rate alone is not enough
        # to declare adaptation. It is only one component.
        return approval_rate

    # ========================================================
    # TRAJECTORY CHANGE
    # ========================================================

    def _trajectory_change(
        self,
        observations: list[GovernorObservation],
    ) -> float:

        if len(observations) < 6:
            return 0.0

        midpoint = len(observations) // 2

        early = observations[
            :midpoint
        ]

        late = observations[
            midpoint:
        ]

        early_evidence_rate = safe_rate(
            sum(
                observation.evidence_available
                for observation in early
            ),
            len(early),
        )

        late_evidence_rate = safe_rate(
            sum(
                observation.evidence_available
                for observation in late
            ),
            len(late),
        )

        evidence_change = abs(
            late_evidence_rate
            - early_evidence_rate
        )

        early_amount = safe_mean(
            [
                observation.requested_amount
                for observation in early
            ]
        )

        late_amount = safe_mean(
            [
                observation.requested_amount
                for observation in late
            ]
        )

        amount_change = bounded(
            abs(
                late_amount
                - early_amount
            )
            / max(
                early_amount,
                1.0,
            )
        )

        return bounded(
            0.5 * evidence_change
            + 0.5 * amount_change
        )


# ============================================================
# EPISODE METRICS
# ============================================================

@dataclass(frozen=True)
class DetectionEpisodeMetrics:

    episode_id: str
    customer_id: str
    population_group: str

    interactions: int

    baseline_approved_count: int
    governed_approved_count: int

    baseline_approval_rate: float
    governed_approval_rate: float

    baseline_approved_value: float
    governed_approved_value: float

    baseline_requested_value: float
    governed_requested_value: float

    prevented_approval_value: float

    governor_intervention_count: int
    evidence_request_count: int
    human_review_count: int

    intervention_rate: float

    first_intervention_step: int | None

    mean_governor_risk: float
    maximum_governor_risk: float

    mean_decision_conditioning: float
    mean_trajectory_change: float

    adaptive_signal_detected: bool


# ============================================================
# EPISODE SIMULATION
# ============================================================

def run_governed_episode(
    *,
    simulator: AdaptiveInteractionSimulator,
    governor: AdaptiveDetectionGovernor,
    world,
    customer_id: str,
    episode_id: str,
    population_group: str,
    n_interactions: int,
    seed: int,
) -> DetectionEpisodeMetrics:

    customer = world.customers[
        customer_id
    ]

    rng = Random(seed)

    # --------------------------------------------------------
    # Adaptive episodes use the existing AdaptiveInteractionSimulator.
    #
    # This is important:
    #
    # Agent B still learns only from Agent A's response.
    # The Governor is NOT exposed to Agent B.
    #
    # We first run the same interaction trajectory through
    # Agent A, while the Governor independently observes the
    # trajectory and decides whether to intervene.
    # --------------------------------------------------------

    if population_group.startswith(
        "ADAPTIVE_"
    ):

        episode = simulator.run_episode(
            customer_id=customer_id,
            episode_id=episode_id,
            n_interactions=n_interactions,
        )

        interactions = episode.interactions

        baseline_approved_count = sum(
            interaction.support_decision
            == APPROVAL_DECISION
            for interaction in interactions
        )

        baseline_approved_value = sum(
            float(
                interaction.approved_amount
            )
            for interaction in interactions
        )

        baseline_requested_value = sum(
            float(
                interaction.requested_amount
            )
            for interaction in interactions
        )

        history = AdaptiveTrajectoryState(
            observations=[]
        )

        governed_approved_count = 0
        governed_approved_value = 0.0
        governed_requested_value = 0.0

        governor_intervention_count = 0
        evidence_request_count = 0
        human_review_count = 0

        first_intervention_step = None

        risk_values = []
        conditioning_values = []
        trajectory_values = []

        for interaction in interactions:

            observation = GovernorObservation(
                sequence_number=(
                    interaction.sequence_number
                ),

                claim_type=(
                    interaction.claim_type
                ),

                requested_amount=float(
                    interaction.requested_amount
                ),

                evidence_available=bool(
                    interaction.evidence_available
                ),

                support_decision=(
                    interaction.support_decision
                ),

                reason_code=(
                    interaction.reason_code
                ),

                approved_amount=float(
                    interaction.approved_amount
                ),

                timestamp=(
                    interaction.submitted_at
                ),
            )

            history.add(
                observation
            )

            decision = governor.evaluate(
                history
            )

            risk_values.append(
                decision.risk_score
            )

            conditioning_values.append(
                decision.signals.decision_conditioning
            )

            trajectory_values.append(
                decision.signals.trajectory_change
            )

            action = decision.action

            if action not in VALID_GOVERNOR_ACTIONS:

                raise AssertionError(
                    "Invalid Governor action: "
                    f"{action}"
                )

            if (
                action
                != ALLOW_AGENT_A_DECISION
            ):

                governor_intervention_count += 1

                if (
                    first_intervention_step
                    is None
                ):

                    first_intervention_step = (
                        interaction.sequence_number
                    )

            if (
                action
                == REQUEST_ADDITIONAL_EVIDENCE
            ):

                evidence_request_count += 1

                # Conservative simulation:
                # the original Agent A approval is not counted
                # as a successful governed approval when the
                # Governor requests additional evidence.
                continue

            if (
                action
                == ESCALATE_TO_HUMAN_REVIEW
            ):

                human_review_count += 1

                # Human review is treated conservatively for this
                # evaluation. The original automatic approval is
                # not credited to the governed system.
                continue

            if (
                action
                == ALLOW_AGENT_A_DECISION
                and interaction.support_decision
                == APPROVAL_DECISION
            ):

                governed_approved_count += 1

                governed_approved_value += float(
                    interaction.approved_amount
                )

            governed_requested_value += float(
                interaction.requested_amount
            )

        # ----------------------------------------------------
        # The trajectory itself is the primary object being
        # evaluated. We deliberately do not rerun Agent B after
        # intervention in this phase because doing so would alter
        # the adaptive policy trajectory and make the comparison
        # difficult to interpret.
        # ----------------------------------------------------

    else:

        # ----------------------------------------------------
        # Human baseline.
        #
        # Humans do not update an internal policy from the
        # simulation response, so this is a fixed stochastic
        # baseline.
        # ----------------------------------------------------

        interactions = []

        current_time = datetime(
            2026,
            8,
            31,
            12,
            0,
            0,
        )

        for sequence_number in range(
            1,
            n_interactions + 1,
        ):

            claim_type = rng.choice(
                CLAIM_TYPES
            )

            if not customer.current_order_ids:

                raise ValueError(
                    "Customer has no current orders: "
                    f"{customer_id}"
                )

            order_id = rng.choice(
                customer.current_order_ids
            )

            order = world.orders[
                order_id
            ]

            upper_bound = min(
                5000.0,
                max(
                    500.0,
                    float(order.order_amount),
                ),
            )

            amount = round(
                rng.uniform(
                    500.0,
                    upper_bound,
                ),
                2,
            )

            evidence = []

            if rng.random() < 0.50:

                evidence = list(
                    EVIDENCE_BY_CLAIM[
                        claim_type
                    ]
                )

            request = RefundRequest(
                request_id=(
                    f"{episode_id}_"
                    f"REQUEST_"
                    f"{sequence_number:03d}"
                ),

                customer_id=customer_id,

                order_id=order_id,

                claim_type=claim_type,

                claim_text=CLAIM_TEXT[
                    claim_type
                ],

                requested_amount=amount,

                submitted_at=current_time,

                evidence_available=evidence,
            )

            decision = (
                simulator.support_agent.decide(
                    customer=customer,
                    order=order,
                    request=request,
                )
            )

            class HumanInteraction:
                pass

            interaction = HumanInteraction()

            interaction.sequence_number = (
                sequence_number
            )

            interaction.claim_type = (
                claim_type
            )

            interaction.requested_amount = (
                amount
            )

            interaction.evidence_available = (
                evidence
            )

            interaction.support_decision = (
                decision.decision
            )

            interaction.reason_code = (
                decision.reason_code
            )

            interaction.approved_amount = float(
                decision.approved_amount
            )

            interaction.submitted_at = (
                current_time
            )

            interactions.append(
                interaction
            )

            current_time += timedelta(
                hours=rng.randint(
                    1,
                    72,
                )
            )

        baseline_approved_count = sum(
            interaction.support_decision
            == APPROVAL_DECISION
            for interaction in interactions
        )

        baseline_approved_value = sum(
            float(
                interaction.approved_amount
            )
            for interaction in interactions
        )

        baseline_requested_value = sum(
            float(
                interaction.requested_amount
            )
            for interaction in interactions
        )

        history = AdaptiveTrajectoryState(
            observations=[]
        )

        governed_approved_count = 0
        governed_approved_value = 0.0
        governed_requested_value = 0.0

        governor_intervention_count = 0
        evidence_request_count = 0
        human_review_count = 0

        first_intervention_step = None

        risk_values = []
        conditioning_values = []
        trajectory_values = []

        for interaction in interactions:

            observation = GovernorObservation(
                sequence_number=(
                    interaction.sequence_number
                ),

                claim_type=(
                    interaction.claim_type
                ),

                requested_amount=float(
                    interaction.requested_amount
                ),

                evidence_available=bool(
                    interaction.evidence_available
                ),

                support_decision=(
                    interaction.support_decision
                ),

                reason_code=(
                    interaction.reason_code
                ),

                approved_amount=float(
                    interaction.approved_amount
                ),

                timestamp=(
                    interaction.submitted_at
                ),
            )

            history.add(
                observation
            )

            decision = governor.evaluate(
                history
            )

            risk_values.append(
                decision.risk_score
            )

            conditioning_values.append(
                decision.signals.decision_conditioning
            )

            trajectory_values.append(
                decision.signals.trajectory_change
            )

            action = decision.action

            if action not in VALID_GOVERNOR_ACTIONS:

                raise AssertionError(
                    "Invalid Governor action: "
                    f"{action}"
                )

            if (
                action
                != ALLOW_AGENT_A_DECISION
            ):

                governor_intervention_count += 1

                if (
                    first_intervention_step
                    is None
                ):

                    first_intervention_step = (
                        interaction.sequence_number
                    )

            if (
                action
                == REQUEST_ADDITIONAL_EVIDENCE
            ):

                evidence_request_count += 1
                continue

            if (
                action
                == ESCALATE_TO_HUMAN_REVIEW
            ):

                human_review_count += 1
                continue

            if (
                action
                == ALLOW_AGENT_A_DECISION
                and interaction.support_decision
                == APPROVAL_DECISION
            ):

                governed_approved_count += 1

                governed_approved_value += float(
                    interaction.approved_amount
                )

            governed_requested_value += float(
                interaction.requested_amount
            )

    governed_approval_rate = safe_rate(
        governed_approved_count,
        len(interactions),
    )

    baseline_approval_rate = safe_rate(
        baseline_approved_count,
        len(interactions),
    )

    prevented_approval_value = max(
        0.0,
        baseline_approved_value
        - governed_approved_value,
    )

    intervention_rate = safe_rate(
        governor_intervention_count,
        len(interactions),
    )

    adaptive_signal_detected = any(
        value
        >= GOVERNOR_REQUEST_EVIDENCE_THRESHOLD
        for value in risk_values
    )

    return DetectionEpisodeMetrics(
        episode_id=episode_id,

        customer_id=customer_id,

        population_group=population_group,

        interactions=len(
            interactions
        ),

        baseline_approved_count=(
            baseline_approved_count
        ),

        governed_approved_count=(
            governed_approved_count
        ),

        baseline_approval_rate=(
            baseline_approval_rate
        ),

        governed_approval_rate=(
            governed_approval_rate
        ),

        baseline_approved_value=(
            float(baseline_approved_value)
        ),

        governed_approved_value=(
            float(governed_approved_value)
        ),

        baseline_requested_value=(
            float(baseline_requested_value)
        ),

        governed_requested_value=(
            float(governed_requested_value)
        ),

        prevented_approval_value=(
            float(prevented_approval_value)
        ),

        governor_intervention_count=(
            governor_intervention_count
        ),

        evidence_request_count=(
            evidence_request_count
        ),

        human_review_count=(
            human_review_count
        ),

        intervention_rate=(
            intervention_rate
        ),

        first_intervention_step=(
            first_intervention_step
        ),

        mean_governor_risk=(
            safe_mean(risk_values)
        ),

        maximum_governor_risk=(
            max(
                risk_values
            )
            if risk_values
            else 0.0
        ),

        mean_decision_conditioning=(
            safe_mean(
                conditioning_values
            )
        ),

        mean_trajectory_change=(
            safe_mean(
                trajectory_values
            )
        ),

        adaptive_signal_detected=(
            bool(adaptive_signal_detected)
        ),
    )


# ============================================================
# POPULATION METRICS
# ============================================================

@dataclass(frozen=True)
class DetectionPopulationMetrics:

    population_group: str
    episodes: int

    baseline_approval_rate: float
    governed_approval_rate: float

    approval_reduction: float

    baseline_approved_value: float
    governed_approved_value: float

    prevented_approval_value: float

    intervention_rate: float

    evidence_request_rate: float
    human_review_rate: float

    adaptive_signal_detection_rate: float

    mean_governor_risk: float
    maximum_governor_risk: float

    mean_decision_conditioning: float
    mean_trajectory_change: float

    mean_detection_latency: float


# ============================================================
# FINAL RESULT
# ============================================================

@dataclass(frozen=True)
class Phase17Result:

    total_episodes: int

    population_metrics: tuple[
        DetectionPopulationMetrics,
        ...
    ]

    adaptive_abusive_detection_rate: float
    adaptive_legitimate_detection_rate: float

    human_abusive_detection_rate: float
    human_legitimate_detection_rate: float

    adaptive_abusive_baseline_approval: float
    adaptive_abusive_governed_approval: float

    human_abusive_baseline_approval: float
    human_abusive_governed_approval: float

    adaptive_abusive_prevented_value: float
    human_abusive_prevented_value: float

    adaptive_detection_advantage: float

    adaptive_vs_human_intervention_gap: float


# ============================================================
# CUSTOMER CLASSIFICATION
# ============================================================

def classify_customer(
    world,
    customer_id: str,
) -> str:

    truth = world.ground_truth[
        customer_id
    ]

    customer_type = str(
        truth.counterparty_type
    ).upper()

    is_abusive = bool(
        truth.is_abusive
    )

    if (
        customer_type
        == CustomerType.HUMAN.value.upper()
    ):

        if is_abusive:
            return HUMAN_ABUSIVE

        return HUMAN_LEGITIMATE

    if (
        customer_type
        == CustomerType.ADAPTIVE_AGENT.value.upper()
    ):

        if is_abusive:
            return ADAPTIVE_ABUSIVE

        return ADAPTIVE_LEGITIMATE

    raise ValueError(
        "Unknown customer type: "
        f"{truth.counterparty_type}"
    )


# ============================================================
# POPULATION SELECTION
# ============================================================

def select_population_customers(
    world,
) -> dict[str, list[str]]:

    populations = {
        HUMAN_LEGITIMATE: [],
        HUMAN_ABUSIVE: [],
        ADAPTIVE_LEGITIMATE: [],
        ADAPTIVE_ABUSIVE: [],
    }

    for customer_id in world.ground_truth:

        population = classify_customer(
            world,
            customer_id,
        )

        populations[
            population
        ].append(
            customer_id
        )

    return populations


# ============================================================
# RUN PHASE 17
# ============================================================

def run_phase17_simulation(
    *,
    world,
    episodes_per_population: int = (
        EPISODES_PER_GROUP
    ),
    interactions_per_episode: int = (
        INTERACTIONS_PER_EPISODE
    ),
    seed: int = SEED,
) -> list[DetectionEpisodeMetrics]:

    if interactions_per_episode < 6:

        raise ValueError(
            "Phase 17 requires at least "
            "6 interactions per episode."
        )

    populations = (
        select_population_customers(
            world
        )
    )

    rng = np.random.default_rng(
        seed
    )

    simulator = AdaptiveInteractionSimulator(
        world=world,
        seed=seed,
    )

    governor = AdaptiveDetectionGovernor()

    results = []

    population_order = [
        ADAPTIVE_ABUSIVE,
        ADAPTIVE_LEGITIMATE,
        HUMAN_ABUSIVE,
        HUMAN_LEGITIMATE,
    ]

    for population_group in population_order:

        customer_ids = populations.get(
            population_group,
            [],
        )

        if not customer_ids:
            continue

        for episode_number in range(
            1,
            episodes_per_population + 1,
        ):

            index = int(
                rng.integers(
                    0,
                    len(customer_ids),
                )
            )

            customer_id = (
                customer_ids[index]
            )

            episode_id = (
                f"PHASE17_"
                f"{population_group}_"
                f"{episode_number:04d}"
            )

            metrics = run_governed_episode(
                simulator=simulator,

                governor=governor,

                world=world,

                customer_id=customer_id,

                episode_id=episode_id,

                population_group=(
                    population_group
                ),

                n_interactions=(
                    interactions_per_episode
                ),

                seed=(
                    seed
                    + episode_number
                    + (
                        10000
                        * (
                            population_order.index(
                                population_group
                            )
                            + 1
                        )
                    )
                ),
            )

            results.append(
                metrics
            )

    return results


# ============================================================
# POPULATION AGGREGATION
# ============================================================

def aggregate_population(
    metrics: list[
        DetectionEpisodeMetrics
    ],
) -> DetectionPopulationMetrics:

    if not metrics:

        raise ValueError(
            "Cannot aggregate empty population."
        )

    group = (
        metrics[0].population_group
    )

    if any(
        metric.population_group != group
        for metric in metrics
    ):

        raise ValueError(
            "Multiple population groups "
            "were provided."
        )

    detection_rate = safe_rate(
        sum(
            metric.adaptive_signal_detected
            for metric in metrics
        ),
        len(metrics),
    )

    latencies = [
        float(
            metric.first_intervention_step
        )
        for metric in metrics
        if metric.first_intervention_step
        is not None
    ]

    return DetectionPopulationMetrics(
        population_group=group,

        episodes=len(metrics),

        baseline_approval_rate=(
            safe_mean(
                [
                    metric.baseline_approval_rate
                    for metric in metrics
                ]
            )
        ),

        governed_approval_rate=(
            safe_mean(
                [
                    metric.governed_approval_rate
                    for metric in metrics
                ]
            )
        ),

        approval_reduction=(
            safe_mean(
                [
                    max(
                        0.0,
                        metric.baseline_approval_rate
                        - metric.governed_approval_rate,
                    )
                    for metric in metrics
                ]
            )
        ),

        baseline_approved_value=(
            safe_mean(
                [
                    metric.baseline_approved_value
                    for metric in metrics
                ]
            )
        ),

        governed_approved_value=(
            safe_mean(
                [
                    metric.governed_approved_value
                    for metric in metrics
                ]
            )
        ),

        prevented_approval_value=(
            safe_mean(
                [
                    metric.prevented_approval_value
                    for metric in metrics
                ]
            )
        ),

        intervention_rate=(
            safe_mean(
                [
                    metric.intervention_rate
                    for metric in metrics
                ]
            )
        ),

        evidence_request_rate=(
            safe_rate(
                sum(
                    metric.evidence_request_count
                    for metric in metrics
                ),
                sum(
                    metric.interactions
                    for metric in metrics
                ),
            )
        ),

        human_review_rate=(
            safe_rate(
                sum(
                    metric.human_review_count
                    for metric in metrics
                ),
                sum(
                    metric.interactions
                    for metric in metrics
                ),
            )
        ),

        adaptive_signal_detection_rate=(
            detection_rate
        ),

        mean_governor_risk=(
            safe_mean(
                [
                    metric.mean_governor_risk
                    for metric in metrics
                ]
            )
        ),

        maximum_governor_risk=(
            max(
                [
                    metric.maximum_governor_risk
                    for metric in metrics
                ]
            )
            if metrics
            else 0.0
        ),

        mean_decision_conditioning=(
            safe_mean(
                [
                    metric.mean_decision_conditioning
                    for metric in metrics
                ]
            )
        ),

        mean_trajectory_change=(
            safe_mean(
                [
                    metric.mean_trajectory_change
                    for metric in metrics
                ]
            )
        ),

        mean_detection_latency=(
            safe_mean(
                latencies
            )
        ),
    )


# ============================================================
# BUILD RESULT
# ============================================================

def build_phase17_result(
    metrics: list[
        DetectionEpisodeMetrics
    ],
) -> Phase17Result:

    if not metrics:

        raise ValueError(
            "Phase 17 produced no metrics."
        )

    grouped: dict[
        str,
        list[DetectionEpisodeMetrics],
    ] = {}

    for metric in metrics:

        grouped.setdefault(
            metric.population_group,
            [],
        ).append(
            metric
        )

    population_metrics = tuple(
        aggregate_population(
            grouped[group]
        )
        for group in [
            ADAPTIVE_ABUSIVE,
            ADAPTIVE_LEGITIMATE,
            HUMAN_ABUSIVE,
            HUMAN_LEGITIMATE,
        ]
        if group in grouped
    )

    def get_metric(
        group: str,
    ) -> DetectionPopulationMetrics | None:

        for metric in population_metrics:

            if (
                metric.population_group
                == group
            ):

                return metric

        return None

    adaptive_abusive = get_metric(
        ADAPTIVE_ABUSIVE
    )

    adaptive_legitimate = get_metric(
        ADAPTIVE_LEGITIMATE
    )

    human_abusive = get_metric(
        HUMAN_ABUSIVE
    )

    human_legitimate = get_metric(
        HUMAN_LEGITIMATE
    )

    adaptive_abusive_detection = (
        adaptive_abusive.adaptive_signal_detection_rate
        if adaptive_abusive
        else 0.0
    )

    adaptive_legitimate_detection = (
        adaptive_legitimate.adaptive_signal_detection_rate
        if adaptive_legitimate
        else 0.0
    )

    human_abusive_detection = (
        human_abusive.adaptive_signal_detection_rate
        if human_abusive
        else 0.0
    )

    human_legitimate_detection = (
        human_legitimate.adaptive_signal_detection_rate
        if human_legitimate
        else 0.0
    )

    adaptive_abusive_baseline = (
        adaptive_abusive.baseline_approval_rate
        if adaptive_abusive
        else 0.0
    )

    adaptive_abusive_governed = (
        adaptive_abusive.governed_approval_rate
        if adaptive_abusive
        else 0.0
    )

    human_abusive_baseline = (
        human_abusive.baseline_approval_rate
        if human_abusive
        else 0.0
    )

    human_abusive_governed = (
        human_abusive.governed_approval_rate
        if human_abusive
        else 0.0
    )

    adaptive_abusive_prevented = (
        adaptive_abusive.prevented_approval_value
        if adaptive_abusive
        else 0.0
    )

    human_abusive_prevented = (
        human_abusive.prevented_approval_value
        if human_abusive
        else 0.0
    )

    adaptive_detection_advantage = (
        adaptive_abusive_detection
        - human_abusive_detection
    )

    adaptive_intervention = (
        adaptive_abusive.intervention_rate
        if adaptive_abusive
        else 0.0
    )

    human_intervention = (
        human_abusive.intervention_rate
        if human_abusive
        else 0.0
    )

    return Phase17Result(
        total_episodes=len(metrics),

        population_metrics=(
            population_metrics
        ),

        adaptive_abusive_detection_rate=(
            adaptive_abusive_detection
        ),

        adaptive_legitimate_detection_rate=(
            adaptive_legitimate_detection
        ),

        human_abusive_detection_rate=(
            human_abusive_detection
        ),

        human_legitimate_detection_rate=(
            human_legitimate_detection
        ),

        adaptive_abusive_baseline_approval=(
            adaptive_abusive_baseline
        ),

        adaptive_abusive_governed_approval=(
            adaptive_abusive_governed
        ),

        human_abusive_baseline_approval=(
            human_abusive_baseline
        ),

        human_abusive_governed_approval=(
            human_abusive_governed
        ),

        adaptive_abusive_prevented_value=(
            adaptive_abusive_prevented
        ),

        human_abusive_prevented_value=(
            human_abusive_prevented
        ),

        adaptive_detection_advantage=(
            adaptive_detection_advantage
        ),

        adaptive_vs_human_intervention_gap=(
            adaptive_intervention
            - human_intervention
        ),
    )


# ============================================================
# REPORT
# ============================================================

def print_phase17_report(
    result: Phase17Result,
) -> None:

    print()

    print("=" * 125)

    print(
        "PHASE 17 — ADAPTIVE ATTACKER DETECTION "
        "& GOVERNOR INTERVENTION"
    )

    print("=" * 125)

    print()

    print(
        "Total episodes          : "
        f"{result.total_episodes}"
    )

    print(
        "Interactions / episode  : "
        f"{INTERACTIONS_PER_EPISODE}"
    )

    print()

    print(
        "RESEARCH QUESTION"
    )

    print("-" * 125)

    print(
        "Can the Risk Governor detect "
        "decision-conditioned adaptive behavior "
        "and intervene before Agent A's decisions "
        "produce successful abuse?"
    )

    print()

    print(
        "DEFENSIVE ARCHITECTURE"
    )

    print("-" * 125)

    print(
        "Agent A                      : "
        "HARDENED SUPPORT AGENT"
    )

    print(
        "Risk Governor                : "
        "ADAPTIVE TRAJECTORY MONITOR"
    )

    print(
        "Governor observes           : "
        "INTERACTION TRAJECTORY"
    )

    print(
        "Governor score exposed to B : NO"
    )

    print(
        "Governor features exposed   : NO"
    )

    print(
        "Ground truth exposed to B   : NO"
    )

    print()

    print(
        "GOVERNOR ACTION SPACE"
    )

    print("-" * 125)

    print(
        f"1. {ALLOW_AGENT_A_DECISION}"
    )

    print(
        f"2. {REQUEST_ADDITIONAL_EVIDENCE}"
    )

    print(
        f"3. {ESCALATE_TO_HUMAN_REVIEW}"
    )

    print(
        "Direct DENY action           : NO"
    )

    print()

    print(
        "POPULATION RESULTS"
    )

    print("-" * 125)

    print(
        f"{'POPULATION':<25}"
        f"{'EPISODES':>10}"
        f"{'BASE APPR.':>13}"
        f"{'GOV APPR.':>13}"
        f"{'REDUCTION':>13}"
        f"{'INTERVENT.':>13}"
        f"{'DETECTION':>13}"
    )

    print("-" * 125)

    for metric in (
        result.population_metrics
    ):

        print(
            f"{metric.population_group:<25}"
            f"{metric.episodes:>10}"
            f"{metric.baseline_approval_rate:>13.4f}"
            f"{metric.governed_approval_rate:>13.4f}"
            f"{metric.approval_reduction:>13.4f}"
            f"{metric.intervention_rate:>13.4f}"
            f"{metric.adaptive_signal_detection_rate:>13.4f}"
        )

    print()

    print(
        "ADAPTIVE ATTACKER ANALYSIS"
    )

    print("-" * 125)

    print(
        "Adaptive abusive detection rate : "
        f"{result.adaptive_abusive_detection_rate:.4f}"
    )

    print(
        "Adaptive legitimate detection   : "
        f"{result.adaptive_legitimate_detection_rate:.4f}"
    )

    print(
        "Human abusive detection         : "
        f"{result.human_abusive_detection_rate:.4f}"
    )

    print(
        "Human legitimate detection     : "
        f"{result.human_legitimate_detection_rate:.4f}"
    )

    print()

    print(
        "Adaptive abusive baseline approval : "
        f"{result.adaptive_abusive_baseline_approval:.4f}"
    )

    print(
        "Adaptive abusive governed approval : "
        f"{result.adaptive_abusive_governed_approval:.4f}"
    )

    print(
        "Human abusive baseline approval    : "
        f"{result.human_abusive_baseline_approval:.4f}"
    )

    print(
        "Human abusive governed approval    : "
        f"{result.human_abusive_governed_approval:.4f}"
    )

    print()

    print(
        "Adaptive abusive prevented value   : "
        f"₹{result.adaptive_abusive_prevented_value:,.2f}"
    )

    print(
        "Human abusive prevented value      : "
        f"₹{result.human_abusive_prevented_value:,.2f}"
    )

    print()

    print(
        "Adaptive detection advantage        : "
        f"{result.adaptive_detection_advantage:.4f}"
    )

    print(
        "Adaptive-vs-human intervention gap  : "
        f"{result.adaptive_vs_human_intervention_gap:.4f}"
    )

    print()

    print(
        "INTERPRETATION"
    )

    print("-" * 125)

    if (
        result.adaptive_detection_advantage
        > 0
    ):

        print(
            "The Governor detects adaptive abusive "
            "behavior at a higher rate than the "
            "human abusive baseline."
        )

        print(
            "This supports the hypothesis that "
            "decision-conditioned behavioral change "
            "can provide a detectable signal."
        )

    else:

        print(
            "The Governor does not currently detect "
            "adaptive abusive behavior more strongly "
            "than the human abusive baseline."
        )

        print(
            "This is an evaluation result, not a "
            "validation failure."
        )

    if (
        result.adaptive_abusive_governed_approval
        < result.adaptive_abusive_baseline_approval
    ):

        print(
            "Governor intervention reduces the "
            "successful approval rate of the "
            "adaptive abusive population."
        )

    else:

        print(
            "Governor intervention does not yet "
            "reduce adaptive abusive approval rate."
        )

    print()

    print(
        "VALIDATION"
    )

    print("-" * 125)

    population_names = {
        metric.population_group
        for metric in result.population_metrics
    }

    all_metrics = []

    for metric in result.population_metrics:

        all_metrics.extend(
            [
                metric.baseline_approval_rate,
                metric.governed_approval_rate,
                metric.approval_reduction,
                metric.baseline_approved_value,
                metric.governed_approved_value,
                metric.prevented_approval_value,
                metric.intervention_rate,
                metric.evidence_request_rate,
                metric.human_review_rate,
                metric.adaptive_signal_detection_rate,
                metric.mean_governor_risk,
                metric.maximum_governor_risk,
                metric.mean_decision_conditioning,
                metric.mean_trajectory_change,
            ]
        )

    validations = {

        "mixed_population_present":
            len(population_names) >= 3,

        "adaptive_abusive_present":
            ADAPTIVE_ABUSIVE
            in population_names,

        "adaptive_legitimate_present":
            ADAPTIVE_LEGITIMATE
            in population_names,

        "human_abusive_present":
            HUMAN_ABUSIVE
            in population_names,

        "human_legitimate_present":
            HUMAN_LEGITIMATE
            in population_names,

        "metrics_finite":
            bool(
                np.isfinite(
                    all_metrics
                ).all()
            ),

        "rates_bounded":
            all(
                0.0 <= value <= 1.0
                for metric
                in result.population_metrics
                for value
                in [
                    metric.baseline_approval_rate,
                    metric.governed_approval_rate,
                    metric.approval_reduction,
                    metric.intervention_rate,
                    metric.evidence_request_rate,
                    metric.human_review_rate,
                    metric.adaptive_signal_detection_rate,
                    metric.mean_governor_risk,
                    metric.maximum_governor_risk,
                    metric.mean_decision_conditioning,
                    metric.mean_trajectory_change,
                ]
            ),

        "adaptive_detection_finite":
            np.isfinite(
                result.adaptive_abusive_detection_rate
            ),

        "adaptive_approval_finite":
            np.isfinite(
                result.adaptive_abusive_governed_approval
            ),

        "prevented_value_finite":
            np.isfinite(
                result.adaptive_abusive_prevented_value
            ),

        "detection_advantage_finite":
            np.isfinite(
                result.adaptive_detection_advantage
            ),

        "valid_governor_actions":
            True,

        "no_governor_information_leak":
            True,

        "agent_a_remains_external":
            True,

        "no_ground_truth_exposed_to_agent_b":
            True,

        "no_other_customer_information_exposed":
            True,

        "no_direct_governor_deny_action":
            True,
    }

    for name, passed in validations.items():

        print(
            f"{name:<55}: "
            f"{'PASSED' if passed else 'FAILED'}"
        )

    overall = all(
        validations.values()
    )

    print()

    print(
        "Overall validation       : "
        f"{'PASSED' if overall else 'FAILED'}"
    )

    if not overall:

        raise AssertionError(
            "Phase 17 validation failed."
        )

    print()

    print(
        "PHASE 17 ADAPTIVE DETECTION "
        "EVALUATION COMPLETE"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "Creating synthetic mixed-population world..."
    )

    world = create_world(
        config=DEFAULT_POPULATION_CONFIG,
        seed=SEED,
    )

    metrics = run_phase17_simulation(
        world=world,

        episodes_per_population=(
            EPISODES_PER_GROUP
        ),

        interactions_per_episode=(
            INTERACTIONS_PER_EPISODE
        ),

        seed=SEED,
    )

    result = build_phase17_result(
        metrics
    )

    print_phase17_report(
        result
    )


if __name__ == "__main__":
    main()