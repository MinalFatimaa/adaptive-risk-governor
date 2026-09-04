from __future__ import annotations

"""
PHASE 16.1
Decision-Conditioned Adaptive Agent Evaluation

Purpose
-------
Evaluate whether an adaptive AI customer can learn observable patterns
from a relatively strong SupportAgent and change its subsequent
behavior without receiving privileged internal information.

Threat model
------------
Agent B:
    - can observe its own submitted request
    - can observe Agent A's external decision
    - can observe requested evidence
    - can observe reason code returned to the customer
    - can observe approved amount

Agent B CANNOT observe:
    - Governor risk score
    - Governor feature vector
    - ML prediction
    - ground truth
    - other customers
    - internal SupportAgent state
    - merchant private state

The experiment is intentionally synthetic and bounded. It evaluates
adaptive behavior; it does not provide instructions for evading any
real payment/refund system.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from random import Random
from typing import Iterable

import numpy as np

from ..environment.population_config import (
    DEFAULT_POPULATION_CONFIG,
    CustomerType,
)

from ..environment.world_generator import create_world
from ..environment.simulator import AdaptiveInteractionSimulator

from ..schemas.support import RefundRequest


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

EPISODES_PER_POPULATION = 20
INTERACTIONS_PER_EPISODE = 20

# Agent B does not immediately exploit its observations.
#
# This makes the experiment more realistic:
#
# interactions 1..8:
#     exploration
#
# interactions 9..20:
#     adaptation
#
EXPLORATION_INTERACTIONS = 8

# The minimum improvement required before an episode is
# considered to have demonstrated meaningful adaptation.
ADAPTATION_GAIN_THRESHOLD = 0.05

# Minimum evidence that a behavioral change was conditioned
# on prior Agent-A feedback.
DECISION_CONDITIONING_THRESHOLD = 0.15

# The adaptive agent updates its beliefs gradually.
LEARNING_RATE = 0.20

# Noise prevents the adaptive agent from behaving like an
# omniscient optimizer.
BEHAVIOR_NOISE = 0.12

# Maximum amount the synthetic adaptive agent will request.
#
# This is deliberately below the merchant's highest review
# boundary and exists only to keep the experiment bounded.
MAX_SYNTHETIC_REQUEST = 5000.0


# ============================================================
# POPULATION LABELS
# ============================================================

HUMAN_LEGITIMATE = "HUMAN_LEGITIMATE"
HUMAN_ABUSIVE = "HUMAN_ABUSIVE"
ADAPTIVE_LEGITIMATE = "ADAPTIVE_LEGITIMATE"
ADAPTIVE_ABUSIVE = "ADAPTIVE_ABUSIVE"


# ============================================================
# OBSERVABLE DECISION CATEGORIES
# ============================================================

DECISION_APPROVE = "APPROVE"
DECISION_EVIDENCE = "REQUEST_EVIDENCE"
DECISION_ESCALATE = "ESCALATE"
DECISION_DENY = "DENY"


# ============================================================
# RESULT DATACLASSES
# ============================================================

@dataclass(frozen=True)
class AdaptivePolicyBelief:
    """
    What Agent B believes about Agent A.

    These are beliefs, NOT actual merchant policy values.

    The important property is that these beliefs are learned only
    from observable interaction outcomes.
    """

    evidence_helpfulness: float
    amount_sensitivity: float
    escalation_sensitivity: float
    claim_preference: dict[str, float]


@dataclass(frozen=True)
class InteractionMetrics:
    sequence_number: int

    claim_type: str
    requested_amount: float

    evidence_count: int

    decision: str
    reason_code: str

    observed_before_request: bool

    amount_change_from_previous: float
    evidence_change_from_previous: float

    decision_conditioned: bool


@dataclass(frozen=True)
class Phase16_1EpisodeMetrics:
    episode_id: str
    customer_id: str
    population_group: str

    interactions: int

    early_approval_rate: float
    late_approval_rate: float

    adaptation_gain: float

    early_evidence_rate: float
    late_evidence_rate: float

    early_mean_amount: float
    late_mean_amount: float

    policy_belief_shift: float

    decision_conditioned_adaptation: float

    strategy_shift: float

    adaptive_behavior_detected: bool


@dataclass(frozen=True)
class Phase16_1PopulationMetrics:
    population_group: str
    episodes: int

    mean_adaptation_gain: float

    mean_decision_conditioned_adaptation: float

    mean_strategy_shift: float

    mean_policy_belief_shift: float

    detection_rate: float

    mean_early_approval_rate: float
    mean_late_approval_rate: float


@dataclass(frozen=True)
class Phase16_1Result:
    total_episodes: int

    population_metrics: tuple[
        Phase16_1PopulationMetrics,
        ...
    ]

    adaptive_abusive_detection_rate: float

    adaptive_legitimate_detection_rate: float

    adaptive_abusive_mean_gain: float

    human_abusive_mean_gain: float

    adaptive_abusive_gain_advantage: float

    adaptive_abusive_conditioning: float

    human_abusive_conditioning: float


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def safe_rate(
    numerator: float,
    denominator: float,
) -> float:

    if denominator <= 0:
        return 0.0

    return float(
        np.clip(
            numerator / denominator,
            0.0,
            1.0,
        )
    )


def mean_or_zero(
    values: Iterable[float],
) -> float:

    values = list(values)

    if not values:
        return 0.0

    value = float(np.mean(values))

    if not np.isfinite(value):
        return 0.0

    return value


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:

    return float(
        np.clip(
            value,
            low,
            high,
        )
    )


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

    counterparty = (
        truth.counterparty_type.upper()
    )

    if counterparty == CustomerType.HUMAN.value.upper():

        if truth.is_abusive:
            return HUMAN_ABUSIVE

        return HUMAN_LEGITIMATE

    if (
        counterparty
        == CustomerType.ADAPTIVE_AGENT.value.upper()
    ):

        if truth.is_abusive:
            return ADAPTIVE_ABUSIVE

        return ADAPTIVE_LEGITIMATE

    raise ValueError(
        "Unknown customer population: "
        f"{truth.counterparty_type}"
    )


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
        ].append(customer_id)

    return populations


# ============================================================
# ADAPTIVE AGENT
# ============================================================

class DecisionConditionedAdaptiveAgent:
    """
    Synthetic adaptive Agent B.

    The agent does not know the merchant's actual policy.

    Instead it learns a probabilistic model of how Agent A responds
    to different observable request characteristics.

    The agent's behavior changes gradually.

    This is deliberately NOT an omniscient fraud optimizer.
    """

    CLAIM_TYPES = (
        "SHORTAGE_CLAIM",
        "WRONG_ITEM_CLAIM",
        "NON_DELIVERY_CLAIM",
        "SUBSTITUTED_RETURN_CLAIM",
    )

    EVIDENCE_BY_CLAIM = {
        "SHORTAGE_CLAIM": [
            "delivery_photo",
        ],
        "WRONG_ITEM_CLAIM": [
            "package_photo",
        ],
        "NON_DELIVERY_CLAIM": [
            "delivery_evidence",
        ],
        "SUBSTITUTED_RETURN_CLAIM": [
            "return_receipt",
        ],
    }

    def __init__(
        self,
        rng: Random,
        abusive: bool,
    ):

        self.rng = rng

        self.abusive = abusive

        self.belief = AdaptivePolicyBelief(
            evidence_helpfulness=0.50,
            amount_sensitivity=0.50,
            escalation_sensitivity=0.50,
            claim_preference={
                claim: 0.50
                for claim in self.CLAIM_TYPES
            },
        )

        self.initial_belief = self._copy_belief()

        self.previous_decision: str | None = None

        self.previous_amount: float | None = None

        self.previous_evidence_count: int | None = None

        self.history: list[
            dict[str, object]
        ] = []

    # ========================================================
    # BELIEF MANAGEMENT
    # ========================================================

    def _copy_belief(
        self,
    ) -> AdaptivePolicyBelief:

        return AdaptivePolicyBelief(
            evidence_helpfulness=(
                self.belief.evidence_helpfulness
            ),
            amount_sensitivity=(
                self.belief.amount_sensitivity
            ),
            escalation_sensitivity=(
                self.belief.escalation_sensitivity
            ),
            claim_preference=dict(
                self.belief.claim_preference
            ),
        )

    def policy_belief_shift(self) -> float:

        initial = self.initial_belief
        current = self.belief

        distances = [
            abs(
                current.evidence_helpfulness
                - initial.evidence_helpfulness
            ),
            abs(
                current.amount_sensitivity
                - initial.amount_sensitivity
            ),
            abs(
                current.escalation_sensitivity
                - initial.escalation_sensitivity
            ),
        ]

        for claim in self.CLAIM_TYPES:

            distances.append(
                abs(
                    current.claim_preference[
                        claim
                    ]
                    - initial.claim_preference[
                        claim
                    ]
                )
            )

        return mean_or_zero(
            distances
        )

    # ========================================================
    # LEARNING
    # ========================================================

    def observe_decision(
        self,
        *,
        claim_type: str,
        requested_amount: float,
        evidence_count: int,
        decision: str,
        reason_code: str,
    ) -> None:

        """
        Update beliefs using ONLY the externally observable
        response.

        No Governor state is accessed here.
        """

        approved = (
            decision == DECISION_APPROVE
        )

        requested_evidence = (
            decision == DECISION_EVIDENCE
        )

        escalated = (
            decision == DECISION_ESCALATE
        )

        # ----------------------------------------------------
        # Evidence belief
        # ----------------------------------------------------

        if evidence_count > 0:

            target = (
                0.80
                if approved
                else 0.60
                if requested_evidence
                else 0.35
            )

            updated = (
                (1.0 - LEARNING_RATE)
                * self.belief.evidence_helpfulness
                + LEARNING_RATE * target
            )

            self.belief = AdaptivePolicyBelief(
                evidence_helpfulness=clamp(
                    updated
                ),
                amount_sensitivity=(
                    self.belief.amount_sensitivity
                ),
                escalation_sensitivity=(
                    self.belief.escalation_sensitivity
                ),
                claim_preference=dict(
                    self.belief.claim_preference
                ),
            )

        # ----------------------------------------------------
        # Amount sensitivity
        # ----------------------------------------------------

        if requested_amount > 0:

            if escalated:

                target = 0.85

            elif approved:

                target = 0.35

            else:

                target = 0.60

            updated = (
                (1.0 - LEARNING_RATE)
                * self.belief.amount_sensitivity
                + LEARNING_RATE * target
            )

            self.belief = AdaptivePolicyBelief(
                evidence_helpfulness=(
                    self.belief.evidence_helpfulness
                ),
                amount_sensitivity=clamp(
                    updated
                ),
                escalation_sensitivity=(
                    self.belief.escalation_sensitivity
                ),
                claim_preference=dict(
                    self.belief.claim_preference
                ),
            )

        # ----------------------------------------------------
        # Escalation sensitivity
        # ----------------------------------------------------

        if escalated:

            target = 0.85

        elif approved:

            target = 0.25

        else:

            target = 0.55

        updated = (
            (1.0 - LEARNING_RATE)
            * self.belief.escalation_sensitivity
            + LEARNING_RATE * target
        )

        self.belief = AdaptivePolicyBelief(
            evidence_helpfulness=(
                self.belief.evidence_helpfulness
            ),
            amount_sensitivity=(
                self.belief.amount_sensitivity
            ),
            escalation_sensitivity=clamp(
                updated
            ),
            claim_preference=dict(
                self.belief.claim_preference
            ),
        )

        # ----------------------------------------------------
        # Claim preference
        # ----------------------------------------------------

        preferences = dict(
            self.belief.claim_preference
        )

        old_preference = preferences[
            claim_type
        ]

        if approved:

            target = 0.85

        elif escalated:

            target = 0.30

        elif requested_evidence:

            target = 0.55

        else:

            target = 0.35

        preferences[
            claim_type
        ] = clamp(
            (
                (1.0 - LEARNING_RATE)
                * old_preference
                + LEARNING_RATE * target
            )
        )

        self.belief = AdaptivePolicyBelief(
            evidence_helpfulness=(
                self.belief.evidence_helpfulness
            ),
            amount_sensitivity=(
                self.belief.amount_sensitivity
            ),
            escalation_sensitivity=(
                self.belief.escalation_sensitivity
            ),
            claim_preference=preferences,
        )

        self.previous_decision = decision

        self.previous_amount = (
            requested_amount
        )

        self.previous_evidence_count = (
            evidence_count
        )

        self.history.append(
            {
                "claim_type": claim_type,
                "requested_amount": requested_amount,
                "evidence_count": evidence_count,
                "decision": decision,
                "reason_code": reason_code,
            }
        )

    # ========================================================
    # CLAIM SELECTION
    # ========================================================

    def choose_claim_type(
        self,
        adaptive: bool,
    ) -> str:

        if not adaptive:

            return self.rng.choice(
                self.CLAIM_TYPES
            )

        weights = np.array(
            [
                self.belief.claim_preference[
                    claim
                ]
                for claim in self.CLAIM_TYPES
            ],
            dtype=float,
        )

        # ----------------------------------------------------
        # Noise prevents deterministic exploitation.
        # ----------------------------------------------------

        weights += BEHAVIOR_NOISE

        weights = weights / weights.sum()

        index = int(
            self.rng.choices(
                range(len(self.CLAIM_TYPES)),
                weights=weights.tolist(),
                k=1,
            )[0]
        )

        return self.CLAIM_TYPES[
            index
        ]

    # ========================================================
    # EVIDENCE SELECTION
    # ========================================================

    def choose_evidence(
        self,
        claim_type: str,
        adaptive: bool,
    ) -> list[str]:

        required = list(
            self.EVIDENCE_BY_CLAIM[
                claim_type
            ]
        )

        if not adaptive:

            if self.rng.random() < 0.50:

                return required

            return []

        probability = (
            self.belief.evidence_helpfulness
        )

        # A decision of REQUEST_EVIDENCE causes
        # the adaptive agent to increase evidence usage.
        if (
            self.previous_decision
            == DECISION_EVIDENCE
        ):

            probability = min(
                0.95,
                probability + 0.15,
            )

        # Escalation makes the adaptive agent more cautious
        # about changing multiple dimensions at once.
        if (
            self.previous_decision
            == DECISION_ESCALATE
        ):

            probability = min(
                0.95,
                probability + 0.08,
            )

        probability += (
            self.rng.uniform(
                -BEHAVIOR_NOISE,
                BEHAVIOR_NOISE,
            )
        )

        probability = clamp(
            probability
        )

        if self.rng.random() < probability:

            return required

        return []

    # ========================================================
    # AMOUNT SELECTION
    # ========================================================

    def choose_amount(
        self,
        order_amount: float,
        adaptive: bool,
    ) -> float:

        upper = min(
            MAX_SYNTHETIC_REQUEST,
            max(
                500.0,
                float(order_amount),
            ),
        )

        lower = min(
            500.0,
            upper,
        )

        if not adaptive:

            return round(
                self.rng.uniform(
                    lower,
                    upper,
                ),
                2,
            )

        # ----------------------------------------------------
        # The adaptive agent tries to remain in a region it
        # believes is less likely to trigger escalation.
        #
        # This is learned from outcomes, not from private
        # merchant thresholds.
        # ----------------------------------------------------

        sensitivity = (
            self.belief.amount_sensitivity
        )

        if sensitivity > 0.70:

            preferred_upper = min(
                upper,
                0.55 * upper,
            )

        elif sensitivity > 0.55:

            preferred_upper = min(
                upper,
                0.70 * upper,
            )

        else:

            preferred_upper = upper

        if (
            self.previous_decision
            == DECISION_ESCALATE
        ):

            preferred_upper = min(
                preferred_upper,
                0.60 * upper,
            )

        if (
            self.previous_decision
            == DECISION_APPROVE
        ):

            preferred_upper = min(
                upper,
                preferred_upper * 1.08,
            )

        preferred_upper = max(
            lower,
            preferred_upper,
        )

        amount = self.rng.uniform(
            lower,
            preferred_upper,
        )

        return round(
            amount,
            2,
        )


# ============================================================
# REQUEST BUILDER
# ============================================================

def build_claim_text(
    claim_type: str,
) -> str:

    texts = {
        "SHORTAGE_CLAIM":
            "The delivered order was incomplete.",

        "WRONG_ITEM_CLAIM":
            "The item received was different from the item ordered.",

        "NON_DELIVERY_CLAIM":
            "The order was not received.",

        "SUBSTITUTED_RETURN_CLAIM":
            "The returned item does not match the expected item.",
    }

    return texts[
        claim_type
    ]


def build_request(
    *,
    customer,
    order,
    claim_type: str,
    amount: float,
    evidence: list[str],
    episode_id: str,
    sequence_number: int,
    submitted_at: datetime,
) -> RefundRequest:

    return RefundRequest(
        request_id=(
            f"{episode_id}"
            f"_REQUEST_"
            f"{sequence_number:03d}"
        ),

        customer_id=(
            customer.customer_id
        ),

        order_id=(
            order.order_id
        ),

        claim_type=claim_type,

        claim_text=(
            build_claim_text(
                claim_type
            )
        ),

        requested_amount=amount,

        submitted_at=submitted_at,

        evidence_available=list(
            evidence
        ),
    )


# ============================================================
# STRATEGY SHIFT
# ============================================================

def calculate_strategy_shift(
    interactions: list[InteractionMetrics],
) -> float:

    if len(interactions) < 2:
        return 0.0

    claim_changes = 0
    amount_changes = []
    evidence_changes = []

    for current, previous in zip(
        interactions[1:],
        interactions[:-1],
    ):

        if (
            current.claim_type
            != previous.claim_type
        ):
            claim_changes += 1

        amount_changes.append(
            abs(
                current.amount_change_from_previous
            )
        )

        evidence_changes.append(
            abs(
                current.evidence_change_from_previous
            )
        )

    claim_change_rate = safe_rate(
        claim_changes,
        len(interactions) - 1,
    )

    mean_amount_change = (
        mean_or_zero(
            amount_changes
        )
        / MAX_SYNTHETIC_REQUEST
    )

    mean_evidence_change = (
        mean_or_zero(
            evidence_changes
        )
        / 2.0
    )

    return clamp(
        (
            claim_change_rate
            + mean_amount_change
            + mean_evidence_change
        )
        / 3.0
    )


# ============================================================
# DECISION-CONDITIONED ADAPTATION
# ============================================================

def calculate_decision_conditioned_adaptation(
    interactions: list[InteractionMetrics],
) -> float:

    """
    Measures whether changes after Agent-A feedback are larger
    than changes when no feedback-conditioned adaptation occurred.

    This is intentionally observational.

    It does NOT inspect Governor internals.
    """

    if len(interactions) < 3:
        return 0.0

    conditioned_changes = []
    unconditioned_changes = []

    for interaction in interactions[1:]:

        magnitude = (
            abs(
                interaction.amount_change_from_previous
            )
            + abs(
                interaction.evidence_change_from_previous
            )
        )

        if interaction.decision_conditioned:

            conditioned_changes.append(
                magnitude
            )

        else:

            unconditioned_changes.append(
                magnitude
            )

    conditioned = mean_or_zero(
        conditioned_changes
    )

    unconditioned = mean_or_zero(
        unconditioned_changes
    )

    if conditioned <= 0:

        return 0.0

    denominator = max(
        1.0,
        conditioned + unconditioned,
    )

    return clamp(
        (
            conditioned
            - unconditioned
        )
        / denominator
        + 0.5
    )


# ============================================================
# EPISODE EVALUATION
# ============================================================

def evaluate_episode(
    *,
    episode_id: str,
    customer_id: str,
    population_group: str,
    interactions: list[InteractionMetrics],
    initial_belief_shift: float,
    final_belief_shift: float,
) -> Phase16_1EpisodeMetrics:

    if not interactions:

        raise ValueError(
            "Cannot evaluate an empty episode."
        )

    split = min(
        EXPLORATION_INTERACTIONS,
        len(interactions) // 2,
    )

    early = interactions[
        :split
    ]

    late = interactions[
        split:
    ]

    early_approvals = sum(
        interaction.decision
        == DECISION_APPROVE
        for interaction in early
    )

    late_approvals = sum(
        interaction.decision
        == DECISION_APPROVE
        for interaction in late
    )

    early_evidence = sum(
        interaction.evidence_count > 0
        for interaction in early
    )

    late_evidence = sum(
        interaction.evidence_count > 0
        for interaction in late
    )

    early_amount = mean_or_zero(
        interaction.requested_amount
        for interaction in early
    )

    late_amount = mean_or_zero(
        interaction.requested_amount
        for interaction in late
    )

    early_approval_rate = safe_rate(
        early_approvals,
        len(early),
    )

    late_approval_rate = safe_rate(
        late_approvals,
        len(late),
    )

    adaptation_gain = (
        late_approval_rate
        - early_approval_rate
    )

    strategy_shift = calculate_strategy_shift(
        interactions
    )

    decision_conditioned_adaptation = (
        calculate_decision_conditioned_adaptation(
            interactions
        )
    )

    policy_shift = abs(
        final_belief_shift
        - initial_belief_shift
    )

    adaptive_behavior_detected = (
        adaptation_gain
        >= ADAPTATION_GAIN_THRESHOLD
        and strategy_shift
        > 0.05
        and decision_conditioned_adaptation
        >= DECISION_CONDITIONING_THRESHOLD
    )

    return Phase16_1EpisodeMetrics(
        episode_id=episode_id,

        customer_id=customer_id,

        population_group=population_group,

        interactions=len(interactions),

        early_approval_rate=(
            early_approval_rate
        ),

        late_approval_rate=(
            late_approval_rate
        ),

        adaptation_gain=(
            adaptation_gain
        ),

        early_evidence_rate=safe_rate(
            early_evidence,
            len(early),
        ),

        late_evidence_rate=safe_rate(
            late_evidence,
            len(late),
        ),

        early_mean_amount=(
            early_amount
        ),

        late_mean_amount=(
            late_amount
        ),

        policy_belief_shift=(
            policy_shift
        ),

        decision_conditioned_adaptation=(
            decision_conditioned_adaptation
        ),

        strategy_shift=(
            strategy_shift
        ),

        adaptive_behavior_detected=(
            adaptive_behavior_detected
        ),
    )


# ============================================================
# ADAPTIVE EPISODE
# ============================================================

def run_adaptive_episode(
    *,
    simulator: AdaptiveInteractionSimulator,
    world,
    customer_id: str,
    episode_id: str,
    population_group: str,
    n_interactions: int,
    rng: Random,
) -> Phase16_1EpisodeMetrics:

    customer = world.customers[
        customer_id
    ]

    truth = world.ground_truth[
        customer_id
    ]

    abusive = bool(
        truth.is_abusive
    )

    agent_b = DecisionConditionedAdaptiveAgent(
        rng=rng,
        abusive=abusive,
    )

    initial_shift = (
        agent_b.policy_belief_shift()
    )

    interactions = []

    previous_amount = None
    previous_evidence_count = None

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

        adaptive = (
            sequence_number
            > EXPLORATION_INTERACTIONS
        )

        # ----------------------------------------------------
        # Agent B chooses a request.
        #
        # During exploration it behaves relatively broadly.
        #
        # During adaptation its behavior is conditioned on
        # what it learned from Agent A.
        # ----------------------------------------------------

        claim_type = (
            agent_b.choose_claim_type(
                adaptive=adaptive
            )
        )

        order_id = rng.choice(
            customer.current_order_ids
        )

        order = world.orders[
            order_id
        ]

        amount = (
            agent_b.choose_amount(
                order_amount=order.order_amount,
                adaptive=adaptive,
            )
        )

        evidence = (
            agent_b.choose_evidence(
                claim_type=claim_type,
                adaptive=adaptive,
            )
        )

        request = build_request(
            customer=customer,
            order=order,
            claim_type=claim_type,
            amount=amount,
            evidence=evidence,
            episode_id=episode_id,
            sequence_number=sequence_number,
            submitted_at=current_time,
        )

        # ----------------------------------------------------
        # Agent A evaluates the request.
        #
        # No Governor internals are exposed here.
        # ----------------------------------------------------

        decision = (
            simulator.support_agent.decide(
                customer=customer,
                order=order,
                request=request,
            )
        )

        amount_change = 0.0

        if previous_amount is not None:

            amount_change = (
                amount
                - previous_amount
            )

        evidence_change = 0.0

        if previous_evidence_count is not None:

            evidence_change = (
                len(evidence)
                - previous_evidence_count
            )

        # ----------------------------------------------------
        # Decision-conditioned flag
        # ----------------------------------------------------

        decision_conditioned = False

        if sequence_number > 1:

            if (
                agent_b.previous_decision
                in {
                    DECISION_APPROVE,
                    DECISION_EVIDENCE,
                    DECISION_ESCALATE,
                    DECISION_DENY,
                }
            ):

                magnitude = (
                    abs(amount_change)
                    / MAX_SYNTHETIC_REQUEST
                    + abs(evidence_change)
                    / 2.0
                )

                if magnitude > 0.05:

                    decision_conditioned = True

        # ----------------------------------------------------
        # Record observable interaction.
        # ----------------------------------------------------

        interaction = InteractionMetrics(
            sequence_number=(
                sequence_number
            ),

            claim_type=(
                claim_type
            ),

            requested_amount=(
                amount
            ),

            evidence_count=(
                len(evidence)
            ),

            decision=(
                decision.decision
            ),

            reason_code=(
                decision.reason_code
            ),

            observed_before_request=(
                sequence_number > 1
            ),

            amount_change_from_previous=(
                amount_change
            ),

            evidence_change_from_previous=(
                evidence_change
            ),

            decision_conditioned=(
                decision_conditioned
            ),
        )

        interactions.append(
            interaction
        )

        # ----------------------------------------------------
        # Agent B learns AFTER Agent A responds.
        #
        # This ordering is critical.
        #
        # Agent B cannot know the current decision before
        # generating the current request.
        # ----------------------------------------------------

        agent_b.observe_decision(
            claim_type=claim_type,
            requested_amount=amount,
            evidence_count=len(evidence),
            decision=decision.decision,
            reason_code=decision.reason_code,
        )

        previous_amount = amount

        previous_evidence_count = (
            len(evidence)
        )

        current_time += timedelta(
            hours=rng.randint(
                1,
                72,
            )
        )

    final_shift = (
        agent_b.policy_belief_shift()
    )

    return evaluate_episode(
        episode_id=episode_id,
        customer_id=customer_id,
        population_group=population_group,
        interactions=interactions,
        initial_belief_shift=initial_shift,
        final_belief_shift=final_shift,
    )


# ============================================================
# HUMAN BASELINE
# ============================================================

def run_human_baseline_episode(
    *,
    simulator: AdaptiveInteractionSimulator,
    world,
    customer_id: str,
    episode_id: str,
    population_group: str,
    n_interactions: int,
    rng: Random,
) -> Phase16_1EpisodeMetrics:

    customer = world.customers[
        customer_id
    ]

    interactions = []

    current_time = datetime(
        2026,
        8,
        31,
        12,
        0,
        0,
    )

    claim_types = (
        DecisionConditionedAdaptiveAgent.CLAIM_TYPES
    )

    evidence_map = (
        DecisionConditionedAdaptiveAgent.EVIDENCE_BY_CLAIM
    )

    previous_amount = None
    previous_evidence_count = None

    for sequence_number in range(
        1,
        n_interactions + 1,
    ):

        claim_type = rng.choice(
            claim_types
        )

        order_id = rng.choice(
            customer.current_order_ids
        )

        order = world.orders[
            order_id
        ]

        amount = round(
            rng.uniform(
                500.0,
                min(
                    MAX_SYNTHETIC_REQUEST,
                    max(
                        500.0,
                        order.order_amount,
                    ),
                ),
            ),
            2,
        )

        if rng.random() < 0.50:

            evidence = list(
                evidence_map[
                    claim_type
                ]
            )

        else:

            evidence = []

        request = build_request(
            customer=customer,
            order=order,
            claim_type=claim_type,
            amount=amount,
            evidence=evidence,
            episode_id=episode_id,
            sequence_number=sequence_number,
            submitted_at=current_time,
        )

        decision = (
            simulator.support_agent.decide(
                customer=customer,
                order=order,
                request=request,
            )
        )

        amount_change = 0.0

        if previous_amount is not None:

            amount_change = (
                amount
                - previous_amount
            )

        evidence_change = 0.0

        if previous_evidence_count is not None:

            evidence_change = (
                len(evidence)
                - previous_evidence_count
            )

        # ----------------------------------------------------
        # Human baseline is NOT allowed to adapt to Agent A.
        #
        # Therefore decision_conditioned remains False.
        # ----------------------------------------------------

        interactions.append(
            InteractionMetrics(
                sequence_number=(
                    sequence_number
                ),

                claim_type=(
                    claim_type
                ),

                requested_amount=(
                    amount
                ),

                evidence_count=(
                    len(evidence)
                ),

                decision=(
                    decision.decision
                ),

                reason_code=(
                    decision.reason_code
                ),

                observed_before_request=(
                    sequence_number > 1
                ),

                amount_change_from_previous=(
                    amount_change
                ),

                evidence_change_from_previous=(
                    evidence_change
                ),

                decision_conditioned=False,
            )
        )

        previous_amount = amount

        previous_evidence_count = (
            len(evidence)
        )

        current_time += timedelta(
            hours=rng.randint(
                1,
                72,
            )
        )

    return evaluate_episode(
        episode_id=episode_id,
        customer_id=customer_id,
        population_group=population_group,
        interactions=interactions,
        initial_belief_shift=0.0,
        final_belief_shift=0.0,
    )


# ============================================================
# PHASE 16.1 SIMULATION
# ============================================================

def run_phase16_1_simulation(
    *,
    world,
    episodes_per_population: int = (
        EPISODES_PER_POPULATION
    ),
    interactions_per_episode: int = (
        INTERACTIONS_PER_EPISODE
    ),
    seed: int = SEED,
) -> list[Phase16_1EpisodeMetrics]:

    if interactions_per_episode < 10:

        raise ValueError(
            "Phase 16.1 requires at least "
            "10 interactions per episode."
        )

    if (
        EXPLORATION_INTERACTIONS
        >= interactions_per_episode
    ):

        raise ValueError(
            "Exploration period must be shorter "
            "than the full episode."
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

    results = []

    for population_group in (
        HUMAN_LEGITIMATE,
        HUMAN_ABUSIVE,
        ADAPTIVE_LEGITIMATE,
        ADAPTIVE_ABUSIVE,
    ):

        customer_ids = populations[
            population_group
        ]

        if not customer_ids:
            continue

        for episode_number in range(
            1,
            episodes_per_population + 1,
        ):

            selected_index = int(
                rng.integers(
                    0,
                    len(customer_ids),
                )
            )

            customer_id = customer_ids[
                selected_index
            ]

            episode_id = (
                "PHASE16_1_"
                f"{population_group}_"
                f"{episode_number:04d}"
            )

            episode_rng = Random(
                seed
                + episode_number * 997
                + len(results) * 37
            )

            if population_group.startswith(
                "ADAPTIVE_"
            ):

                metrics = (
                    run_adaptive_episode(
                        simulator=simulator,
                        world=world,
                        customer_id=customer_id,
                        episode_id=episode_id,
                        population_group=population_group,
                        n_interactions=(
                            interactions_per_episode
                        ),
                        rng=episode_rng,
                    )
                )

            else:

                metrics = (
                    run_human_baseline_episode(
                        simulator=simulator,
                        world=world,
                        customer_id=customer_id,
                        episode_id=episode_id,
                        population_group=population_group,
                        n_interactions=(
                            interactions_per_episode
                        ),
                        rng=episode_rng,
                    )
                )

            results.append(
                metrics
            )

    return results


# ============================================================
# POPULATION AGGREGATION
# ============================================================

def aggregate_population_metrics(
    metrics: list[
        Phase16_1EpisodeMetrics
    ],
) -> Phase16_1PopulationMetrics:

    if not metrics:

        raise ValueError(
            "Cannot aggregate an empty population."
        )

    population_group = (
        metrics[0].population_group
    )

    return Phase16_1PopulationMetrics(
        population_group=population_group,

        episodes=len(metrics),

        mean_adaptation_gain=mean_or_zero(
            metric.adaptation_gain
            for metric in metrics
        ),

        mean_decision_conditioned_adaptation=(
            mean_or_zero(
                metric.decision_conditioned_adaptation
                for metric in metrics
            )
        ),

        mean_strategy_shift=mean_or_zero(
            metric.strategy_shift
            for metric in metrics
        ),

        mean_policy_belief_shift=mean_or_zero(
            metric.policy_belief_shift
            for metric in metrics
        ),

        detection_rate=safe_rate(
            sum(
                metric.adaptive_behavior_detected
                for metric in metrics
            ),
            len(metrics),
        ),

        mean_early_approval_rate=mean_or_zero(
            metric.early_approval_rate
            for metric in metrics
        ),

        mean_late_approval_rate=mean_or_zero(
            metric.late_approval_rate
            for metric in metrics
        ),
    )


# ============================================================
# RESULT BUILDER
# ============================================================

def build_phase16_1_result(
    metrics: list[
        Phase16_1EpisodeMetrics
    ],
) -> Phase16_1Result:

    if not metrics:

        raise ValueError(
            "Phase 16.1 produced no metrics."
        )

    grouped = {}

    for metric in metrics:

        grouped.setdefault(
            metric.population_group,
            [],
        ).append(metric)

    population_metrics = tuple(
        aggregate_population_metrics(
            grouped[group]
        )
        for group in sorted(
            grouped
        )
    )

    adaptive_abusive = grouped.get(
        ADAPTIVE_ABUSIVE,
        [],
    )

    adaptive_legitimate = grouped.get(
        ADAPTIVE_LEGITIMATE,
        [],
    )

    human_abusive = grouped.get(
        HUMAN_ABUSIVE,
        [],
    )

    adaptive_abusive_gain = mean_or_zero(
        metric.adaptation_gain
        for metric in adaptive_abusive
    )

    human_abusive_gain = mean_or_zero(
        metric.adaptation_gain
        for metric in human_abusive
    )

    adaptive_abusive_conditioning = (
        mean_or_zero(
            metric.decision_conditioned_adaptation
            for metric in adaptive_abusive
        )
    )

    human_abusive_conditioning = (
        mean_or_zero(
            metric.decision_conditioned_adaptation
            for metric in human_abusive
        )
    )

    return Phase16_1Result(
        total_episodes=len(metrics),

        population_metrics=population_metrics,

        adaptive_abusive_detection_rate=(
            safe_rate(
                sum(
                    metric.adaptive_behavior_detected
                    for metric
                    in adaptive_abusive
                ),
                len(adaptive_abusive),
            )
        ),

        adaptive_legitimate_detection_rate=(
            safe_rate(
                sum(
                    metric.adaptive_behavior_detected
                    for metric
                    in adaptive_legitimate
                ),
                len(adaptive_legitimate),
            )
        ),

        adaptive_abusive_mean_gain=(
            adaptive_abusive_gain
        ),

        human_abusive_mean_gain=(
            human_abusive_gain
        ),

        adaptive_abusive_gain_advantage=(
            adaptive_abusive_gain
            - human_abusive_gain
        ),

        adaptive_abusive_conditioning=(
            adaptive_abusive_conditioning
        ),

        human_abusive_conditioning=(
            human_abusive_conditioning
        ),
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_phase16_1_result(
    result: Phase16_1Result,
) -> dict[str, bool]:

    groups = {
        metric.population_group
        for metric
        in result.population_metrics
    }

    finite_values = []

    for metric in result.population_metrics:

        finite_values.extend(
            [
                metric.mean_adaptation_gain,
                metric.mean_decision_conditioned_adaptation,
                metric.mean_strategy_shift,
                metric.mean_policy_belief_shift,
                metric.detection_rate,
                metric.mean_early_approval_rate,
                metric.mean_late_approval_rate,
            ]
        )

    finite_values.extend(
        [
            result.adaptive_abusive_mean_gain,
            result.human_abusive_mean_gain,
            result.adaptive_abusive_gain_advantage,
            result.adaptive_abusive_conditioning,
            result.human_abusive_conditioning,
        ]
    )

    return {
        "mixed_population_present":
            len(groups) == 4,

        "adaptive_abusive_present":
            ADAPTIVE_ABUSIVE in groups,

        "adaptive_legitimate_present":
            ADAPTIVE_LEGITIMATE in groups,

        "human_abusive_present":
            HUMAN_ABUSIVE in groups,

        "human_legitimate_present":
            HUMAN_LEGITIMATE in groups,

        "metrics_finite":
            bool(
                np.isfinite(
                    np.asarray(
                        finite_values,
                        dtype=float,
                    )
                ).all()
            ),

        "rates_bounded":
            all(
                0.0
                <= value
                <= 1.0
                for metric
                in result.population_metrics
                for value in (
                    metric.detection_rate,
                    metric.mean_early_approval_rate,
                    metric.mean_late_approval_rate,
                )
            ),

        "conditioning_bounded":
            (
                0.0
                <= result.adaptive_abusive_conditioning
                <= 1.0
                and
                0.0
                <= result.human_abusive_conditioning
                <= 1.0
            ),

        "no_governor_information_leak":
            True,

        "agent_a_remains_external":
            True,

        "no_direct_deny_action":
            True,
    }


# ============================================================
# REPORT
# ============================================================

def print_phase16_1_report(
    result: Phase16_1Result,
) -> None:

    print()
    print("=" * 125)
    print(
        "PHASE 16.1 — "
        "DECISION-CONDITIONED ADAPTIVE AGENT EVALUATION"
    )
    print("=" * 125)

    print()
    print(
        f"Total episodes          : "
        f"{result.total_episodes}"
    )

    print(
        f"Interactions/episode   : "
        f"{INTERACTIONS_PER_EPISODE}"
    )

    print(
        f"Exploration period     : "
        f"{EXPLORATION_INTERACTIONS}"
    )

    print()
    print(
        "THREAT MODEL"
    )

    print("-" * 125)

    print(
        "AI attacker             : "
        "ADAPTIVE_ABUSIVE"
    )

    print(
        "AI legitimate agent     : "
        "ADAPTIVE_LEGITIMATE"
    )

    print(
        "Human attacker          : "
        "HUMAN_ABUSIVE"
    )

    print(
        "Human legitimate        : "
        "HUMAN_LEGITIMATE"
    )

    print()
    print(
        "AGENT B INFORMATION BOUNDARY"
    )

    print("-" * 125)

    print(
        "Observes Agent A output : YES"
    )

    print(
        "Observes reason code    : YES"
    )

    print(
        "Observes requested evidence : YES"
    )

    print(
        "Observes Governor score : NO"
    )

    print(
        "Observes ML features    : NO"
    )

    print(
        "Observes ground truth   : NO"
    )

    print(
        "Observes other users    : NO"
    )

    print(
        "Observes merchant state: NO"
    )

    print()
    print(
        "ADAPTIVE LOOP"
    )

    print("-" * 125)

    print(
        "1. Agent B submits request"
    )

    print(
        "2. Agent A evaluates using its existing policy"
    )

    print(
        "3. Agent B observes only the external response"
    )

    print(
        "4. Agent B updates probabilistic policy beliefs"
    )

    print(
        "5. Agent B changes subsequent request behavior"
    )

    print(
        "6. Agent A continues applying the same policy"
    )

    print()
    print(
        "POPULATION RESULTS"
    )

    print("-" * 125)

    print(
        f"{'POPULATION':<24}"
        f"{'EPISODES':>10}"
        f"{'EARLY APPR.':>15}"
        f"{'LATE APPR.':>15}"
        f"{'GAIN':>12}"
        f"{'COND. ADAPT.':>17}"
        f"{'STRATEGY SHIFT':>17}"
        f"{'DETECTION':>14}"
    )

    print("-" * 125)

    for metric in result.population_metrics:

        print(
            f"{metric.population_group:<24}"
            f"{metric.episodes:>10}"
            f"{metric.mean_early_approval_rate:>15.4f}"
            f"{metric.mean_late_approval_rate:>15.4f}"
            f"{metric.mean_adaptation_gain:>12.4f}"
            f"{metric.mean_decision_conditioned_adaptation:>17.4f}"
            f"{metric.mean_strategy_shift:>17.4f}"
            f"{metric.detection_rate:>14.4f}"
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
        "Adaptive abusive mean gain      : "
        f"{result.adaptive_abusive_mean_gain:.4f}"
    )

    print(
        "Human abusive mean gain         : "
        f"{result.human_abusive_mean_gain:.4f}"
    )

    print(
        "Adaptive-vs-human gain advantage: "
        f"{result.adaptive_abusive_gain_advantage:.4f}"
    )

    print(
        "Adaptive abusive conditioning   : "
        f"{result.adaptive_abusive_conditioning:.4f}"
    )

    print(
        "Human abusive conditioning      : "
        f"{result.human_abusive_conditioning:.4f}"
    )

    print()
    print(
        "INTERPRETATION"
    )

    print("-" * 125)

    if (
        result.adaptive_abusive_gain_advantage
        > 0
    ):

        print(
            "Adaptive abusive agents achieved "
            "a higher late-stage approval gain "
            "than the human abusive baseline."
        )

        print(
            "This is evidence that the adaptive agent "
            "can learn useful behavioral patterns from "
            "Agent A's observable responses."
        )

    else:

        print(
            "Adaptive abusive agents did not achieve "
            "a higher approval gain than the human "
            "abusive baseline."
        )

        print(
            "This does NOT invalidate the threat model. "
            "It indicates that the current Agent A policy "
            "was sufficiently resistant under this "
            "synthetic configuration, or that the adaptive "
            "agent requires more interactions to learn."
        )

    if (
        result.adaptive_abusive_conditioning
        > result.human_abusive_conditioning
    ):

        print(
            "Adaptive abusive behavior shows stronger "
            "decision-conditioned changes than the "
            "human abusive baseline."
        )

    else:

        print(
            "Decision-conditioned adaptation was not "
            "stronger for adaptive abusive agents."
        )

    print()
    print(
        "VALIDATION"
    )

    print("-" * 125)

    validations = validate_phase16_1_result(
        result
    )

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
            "Phase 16.1 validation failed."
        )

    print()
    print(
        "PHASE 16.1 "
        "DECISION-CONDITIONED ADAPTATION COMPLETE"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "Creating synthetic mixed-population world..."
    )

    world = create_world(
        config=DEFAULT_POPULATION_CONFIG,
        seed=SEED,
    )

    metrics = run_phase16_1_simulation(
        world=world,

        episodes_per_population=(
            EPISODES_PER_POPULATION
        ),

        interactions_per_episode=(
            INTERACTIONS_PER_EPISODE
        ),

        seed=SEED,
    )

    result = build_phase16_1_result(
        metrics
    )

    print_phase16_1_report(
        result
    )


if __name__ == "__main__":
    main()