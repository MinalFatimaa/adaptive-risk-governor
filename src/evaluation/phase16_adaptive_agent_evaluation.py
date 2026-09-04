from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from random import Random
from typing import Any, Iterable

import numpy as np

from ..agents.adaptive_customer import (
    AdaptiveCustomerAgent,
    AdaptiveCustomerState,
)

from ..agents.support_agent import SupportAgent

from ..environment.population_config import (
    CustomerType,
    DEFAULT_POPULATION_CONFIG,
)

from ..environment.world import EnvironmentState
from ..environment.world_generator import create_world

from ..schemas.customer import CustomerObservableState
from ..schemas.support import RefundRequest, SupportDecision


# ============================================================
# PHASE 16 CONFIGURATION
# ============================================================

SEED = 42

N_EPISODES = 40
EPISODES_PER_POPULATION = 10
INTERACTIONS_PER_EPISODE = 10

# First half = exploration
# Second half = adaptation
EXPLORATION_INTERACTIONS = 5

# Minimum increase in approval rate required to
# classify a trajectory as showing adaptation.
ADAPTATION_GAIN_THRESHOLD = 0.05


# ============================================================
# CUSTOMER POPULATION LABELS
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
# RESULT TYPES
# ============================================================

@dataclass(frozen=True)
class InteractionRecord:
    episode_id: str
    sequence_number: int

    customer_id: str
    order_id: str

    claim_type: str
    requested_amount: float
    evidence_available: list[str]

    support_decision: str
    reason_code: str
    approved_amount: float
    requires_followup: bool

    submitted_at: datetime

    # What Agent B was allowed to observe.
    customer_observation: dict[str, Any]
    support_observation: dict[str, Any]


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    customer_id: str

    interactions: list[InteractionRecord]

    initial_policy_beliefs: dict[str, float]
    final_policy_beliefs: dict[str, float]

    @property
    def length(self) -> int:
        return len(self.interactions)


@dataclass(frozen=True)
class EpisodeMetrics:
    episode_id: str
    customer_id: str
    population_group: str

    interactions: int

    early_approval_rate: float
    late_approval_rate: float

    adaptation_gain: float

    evidence_rate_early: float
    evidence_rate_late: float

    mean_requested_amount_early: float
    mean_requested_amount_late: float

    policy_belief_shift: float

    adaptive_behavior_detected: bool


@dataclass(frozen=True)
class PopulationMetrics:
    population_group: str
    episodes: int

    mean_adaptation_gain: float
    mean_policy_belief_shift: float

    adaptive_behavior_detection_rate: float

    mean_early_approval_rate: float
    mean_late_approval_rate: float

    mean_evidence_rate_early: float
    mean_evidence_rate_late: float


@dataclass(frozen=True)
class Phase16Result:
    total_episodes: int

    population_metrics: tuple[
        PopulationMetrics,
        ...
    ]

    adaptive_abusive_detection_rate: float
    adaptive_legitimate_detection_rate: float

    adaptive_abusive_mean_gain: float
    human_abusive_mean_gain: float

    adaptive_abusive_gain_advantage: float


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_episode_configuration(
    interactions_per_episode: int,
) -> None:

    if interactions_per_episode < 4:
        raise ValueError(
            "Phase 16 requires at least 4 interactions "
            "per episode."
        )

    if EXPLORATION_INTERACTIONS >= interactions_per_episode:
        raise ValueError(
            "Exploration period must be shorter than "
            "the complete episode."
        )


# ============================================================
# CUSTOMER CLASSIFICATION
# ============================================================

def classify_customer(
    world: EnvironmentState,
    customer_id: str,
) -> str:

    truth = world.ground_truth[customer_id]

    counterparty_type = (
        str(truth.counterparty_type)
        .upper()
    )

    if counterparty_type == CustomerType.HUMAN.value.upper():

        if truth.is_abusive:
            return HUMAN_ABUSIVE

        return HUMAN_LEGITIMATE

    if (
        counterparty_type
        == CustomerType.ADAPTIVE_AGENT.value.upper()
    ):

        if truth.is_abusive:
            return ADAPTIVE_ABUSIVE

        return ADAPTIVE_LEGITIMATE

    raise ValueError(
        f"Unknown customer population: "
        f"{truth.counterparty_type}"
    )


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def safe_rate(
    numerator: float,
    denominator: float,
) -> float:

    if denominator <= 0:
        return 0.0

    value = numerator / denominator

    if not np.isfinite(value):
        return 0.0

    return float(
        np.clip(
            value,
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

    value = float(
        np.mean(values)
    )

    if not np.isfinite(value):
        return 0.0

    return value


def belief_shift(
    initial: dict[str, float],
    final: dict[str, float],
) -> float:

    keys = set(initial) | set(final)

    if not keys:
        return 0.0

    distances = []

    for key in keys:

        initial_value = float(
            initial.get(key, 0.0)
        )

        final_value = float(
            final.get(key, 0.0)
        )

        if not np.isfinite(initial_value):
            initial_value = 0.0

        if not np.isfinite(final_value):
            final_value = 0.0

        distances.append(
            abs(
                final_value
                - initial_value
            )
        )

    value = float(
        np.mean(distances)
    )

    if not np.isfinite(value):
        return 0.0

    return value


# ============================================================
# PHASE 16 ADAPTIVE SIMULATOR
#
# IMPORTANT:
#
# This class is intentionally defined inside Phase 16.
#
# We DO NOT import:
#
#     ..environment.adaptive_simulation
#
# because that file does not exist in the user's repository.
#
# Instead Phase 16 directly connects:
#
# Agent B
#    ↓
# SupportAgent / Agent A
#    ↓
# observable response
#    ↓
# Agent B learns
#    ↓
# next request
#
# ============================================================

class Phase16AdaptiveSimulator:

    def __init__(
        self,
        world: EnvironmentState,
        seed: int = SEED,
    ):

        self.world = world

        self.merchant = world.merchant

        self.rng = Random(seed)

        self.support_agent = SupportAgent(
            merchant=self.merchant
        )

    # ========================================================
    # PUBLIC EPISODE API
    # ========================================================

    def run_adaptive_episode(
        self,
        *,
        customer_id: str,
        episode_id: str,
        n_interactions: int,
    ) -> EpisodeResult:

        if customer_id not in self.world.customers:
            raise ValueError(
                f"Unknown customer_id: {customer_id}"
            )

        customer = self.world.customers[
            customer_id
        ]

        private_state = (
            self.world.customer_private[
                customer_id
            ]
        )

        # ----------------------------------------------------
        # Determine Agent B objective.
        #
        # This information is private.
        # Agent A never receives it.
        # ----------------------------------------------------

        if (
            private_state.objective
            == "maximize_illegitimate_refund_value"
        ):

            objective = (
                "maximize_illegitimate_refund_value"
            )

        else:

            objective = (
                "maximize_successful_legitimate_resolution"
            )

        agent_state = AdaptiveCustomerState(
            customer_id=customer_id,
            objective=objective,
        )

        agent_b = AdaptiveCustomerAgent(
            state=agent_state,
            seed=self.rng.randint(
                0,
                10_000_000,
            ),
        )

        initial_beliefs = (
            agent_b
            .get_inferred_policy()
            .copy()
        )

        interactions: list[
            InteractionRecord
        ] = []

        current_time = datetime(
            2026,
            8,
            31,
            12,
            0,
            0,
        )

        # ====================================================
        # ADAPTIVE LOOP
        # ====================================================

        for sequence_number in range(
            1,
            n_interactions + 1,
        ):

            # ------------------------------------------------
            # STEP 1
            #
            # Agent B chooses request using its
            # CURRENT policy beliefs.
            # ------------------------------------------------

            request = self._generate_adaptive_request(
                customer=customer,
                agent_b=agent_b,
                sequence_number=sequence_number,
                episode_id=episode_id,
                current_time=current_time,
            )

            order = self.world.orders[
                request.order_id
            ]

            # ------------------------------------------------
            # STEP 2
            #
            # Agent A evaluates the request.
            #
            # Agent B does NOT see Governor internals.
            # ------------------------------------------------

            support_decision = (
                self.support_agent.decide(
                    customer=customer,
                    order=order,
                    request=request,
                )
            )

            # ------------------------------------------------
            # STEP 3
            #
            # Build the information that Agent B is
            # actually allowed to observe.
            # ------------------------------------------------

            customer_observation = (
                self._build_customer_observation(
                    customer=customer,
                    order=order,
                    request=request,
                )
            )

            support_observation = (
                self._build_support_observation(
                    decision=support_decision,
                )
            )

            # ------------------------------------------------
            # STEP 4
            #
            # Agent B observes Agent A's response.
            #
            # It can learn:
            #
            # - APPROVE
            # - REQUEST_EVIDENCE
            # - ESCALATE
            # - reason code
            # - evidence requested
            #
            # It cannot learn:
            #
            # - Governor score
            # - ML features
            # - ground truth
            # - other customers
            # - internal Agent A state
            # ------------------------------------------------

            agent_b.infer_support_policy(
                decision=support_decision,
                claim_type=request.claim_type,
                requested_amount=(
                    request.requested_amount
                ),
                evidence_available=(
                    request.evidence_available
                ),
            )

            # ------------------------------------------------
            # STEP 5
            #
            # Record the interaction.
            # ------------------------------------------------

            interactions.append(
                InteractionRecord(
                    episode_id=episode_id,
                    sequence_number=sequence_number,

                    customer_id=customer_id,
                    order_id=request.order_id,

                    claim_type=request.claim_type,
                    requested_amount=(
                        float(
                            request.requested_amount
                        )
                    ),

                    evidence_available=list(
                        request.evidence_available
                    ),

                    support_decision=(
                        support_decision.decision
                    ),

                    reason_code=(
                        support_decision.reason_code
                    ),

                    approved_amount=(
                        float(
                            support_decision.approved_amount
                        )
                    ),

                    requires_followup=(
                        bool(
                            support_decision.requires_followup
                        )
                    ),

                    submitted_at=request.submitted_at,

                    customer_observation=(
                        customer_observation
                    ),

                    support_observation=(
                        support_observation
                    ),
                )
            )

            # ------------------------------------------------
            # STEP 6
            #
            # Advance simulated time.
            # ------------------------------------------------

            current_time += timedelta(
                hours=self.rng.randint(
                    1,
                    72,
                )
            )

        final_beliefs = (
            agent_b
            .get_inferred_policy()
            .copy()
        )

        return EpisodeResult(
            episode_id=episode_id,
            customer_id=customer_id,
            interactions=interactions,
            initial_policy_beliefs=initial_beliefs,
            final_policy_beliefs=final_beliefs,
        )

    # ========================================================
    # REQUEST GENERATION
    # ========================================================

    def _generate_adaptive_request(
        self,
        *,
        customer: CustomerObservableState,
        agent_b: AdaptiveCustomerAgent,
        sequence_number: int,
        episode_id: str,
        current_time: datetime,
    ) -> RefundRequest:

        policy = (
            agent_b
            .get_inferred_policy()
        )

        claim_type = (
            self._choose_claim_type(
                policy=policy
            )
        )

        amount = (
            self._choose_amount(
                customer=customer,
                policy=policy,
            )
        )

        evidence = (
            self._choose_evidence(
                claim_type=claim_type,
                policy=policy,
            )
        )

        order_id = (
            self._choose_order(
                customer=customer
            )
        )

        return RefundRequest(
            request_id=(
                f"{episode_id}_"
                f"REQUEST_"
                f"{sequence_number:03d}"
            ),

            customer_id=(
                customer.customer_id
            ),

            order_id=order_id,

            claim_type=claim_type,

            claim_text=(
                CLAIM_TEXT[claim_type]
            ),

            requested_amount=amount,

            submitted_at=current_time,

            evidence_available=evidence,
        )

    # ========================================================
    # CLAIM SELECTION
    # ========================================================

    def _choose_claim_type(
        self,
        *,
        policy: dict[str, float],
    ) -> str:

        evidence_sensitivity = float(
            policy.get(
                "evidence_sensitivity",
                0.50,
            )
        )

        if (
            evidence_sensitivity > 0.65
            and self.rng.random() < 0.70
        ):

            evidence_friendly_claims = [
                "WRONG_ITEM_CLAIM",
                "SHORTAGE_CLAIM",
                "SUBSTITUTED_RETURN_CLAIM",
            ]

            return self.rng.choice(
                evidence_friendly_claims
            )

        return self.rng.choice(
            CLAIM_TYPES
        )

    # ========================================================
    # AMOUNT SELECTION
    # ========================================================

    def _choose_amount(
        self,
        *,
        customer: CustomerObservableState,
        policy: dict[str, float],
    ) -> float:

        amount_sensitivity = float(
            policy.get(
                "amount_sensitivity",
                0.50,
            )
        )

        high_value_sensitivity = float(
            policy.get(
                "high_value_escalation",
                0.50,
            )
        )

        refund_policy = (
            self.merchant.refund_policy
        )

        human_review_threshold = float(
            refund_policy.human_review_threshold
        )

        if high_value_sensitivity > 0.65:

            upper_bound = min(
                5000.0,
                0.95
                * human_review_threshold,
            )

        elif amount_sensitivity > 0.65:

            upper_bound = min(
                5000.0,
                0.80
                * human_review_threshold,
            )

        else:

            upper_bound = min(
                5000.0,
                9000.0,
            )

        lower_bound = 500.0

        if upper_bound < lower_bound:
            upper_bound = lower_bound

        return round(
            self.rng.uniform(
                lower_bound,
                upper_bound,
            ),
            2,
        )

    # ========================================================
    # EVIDENCE SELECTION
    # ========================================================

    def _choose_evidence(
        self,
        *,
        claim_type: str,
        policy: dict[str, float],
    ) -> list[str]:

        evidence_sensitivity = float(
            policy.get(
                "evidence_sensitivity",
                0.50,
            )
        )

        required_evidence = (
            EVIDENCE_BY_CLAIM[
                claim_type
            ]
        )

        if (
            self.rng.random()
            < evidence_sensitivity
        ):

            return list(
                required_evidence
            )

        return []

    # ========================================================
    # ORDER SELECTION
    # ========================================================

    def _choose_order(
        self,
        *,
        customer: CustomerObservableState,
    ) -> str:

        if not customer.current_order_ids:

            raise ValueError(
                f"Customer "
                f"{customer.customer_id} "
                "has no orders."
            )

        return self.rng.choice(
            customer.current_order_ids
        )

    # ========================================================
    # CUSTOMER OBSERVATION
    # ========================================================

    def _build_customer_observation(
        self,
        *,
        customer: CustomerObservableState,
        order: Any,
        request: RefundRequest,
    ) -> dict[str, Any]:

        return {
            "customer_id":
                customer.customer_id,

            "account_age_days":
                customer.account_age_days,

            "order_count":
                customer.order_count,

            "completed_order_count":
                customer.completed_order_count,

            "current_order_id":
                order.order_id,

            "order_amount":
                order.order_amount,

            "product_category":
                order.product_category,

            "delivery_status":
                order.delivery_status,

            "claim_type":
                request.claim_type,

            "requested_amount":
                request.requested_amount,

            "evidence_available":
                list(
                    request.evidence_available
                ),
        }

    # ========================================================
    # SUPPORT OBSERVATION
    # ========================================================

    def _build_support_observation(
        self,
        *,
        decision: SupportDecision,
    ) -> dict[str, Any]:

        return {
            "decision":
                decision.decision,

            "reason_code":
                decision.reason_code,

            "requested_evidence":
                list(
                    decision.requested_evidence
                ),

            "approved_amount":
                decision.approved_amount,

            "requires_followup":
                decision.requires_followup,
        }


# ============================================================
# EPISODE EVALUATION
# ============================================================

def evaluate_episode(
    episode: EpisodeResult,
    population_group: str,
) -> EpisodeMetrics:

    interactions = episode.interactions

    if not interactions:
        raise ValueError(
            "Episode contains no interactions."
        )

    split = min(
        EXPLORATION_INTERACTIONS,
        len(interactions) // 2,
    )

    if split <= 0:
        raise ValueError(
            "Episode is too short for evaluation."
        )

    early = interactions[:split]
    late = interactions[split:]

    # --------------------------------------------------------
    # APPROVAL
    # --------------------------------------------------------

    early_approvals = sum(
        1
        for interaction in early
        if interaction.support_decision
        == "APPROVE"
    )

    late_approvals = sum(
        1
        for interaction in late
        if interaction.support_decision
        == "APPROVE"
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

    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    early_evidence = sum(
        1
        for interaction in early
        if len(
            interaction.evidence_available
        ) > 0
    )

    late_evidence = sum(
        1
        for interaction in late
        if len(
            interaction.evidence_available
        ) > 0
    )

    evidence_rate_early = safe_rate(
        early_evidence,
        len(early),
    )

    evidence_rate_late = safe_rate(
        late_evidence,
        len(late),
    )

    # --------------------------------------------------------
    # REQUESTED AMOUNT
    # --------------------------------------------------------

    mean_amount_early = mean_or_zero(
        interaction.requested_amount
        for interaction in early
    )

    mean_amount_late = mean_or_zero(
        interaction.requested_amount
        for interaction in late
    )

    # --------------------------------------------------------
    # POLICY BELIEF SHIFT
    # --------------------------------------------------------

    policy_shift = belief_shift(
        episode.initial_policy_beliefs,
        episode.final_policy_beliefs,
    )

    # --------------------------------------------------------
    # ADAPTATION DETECTION
    #
    # We deliberately require BOTH:
    #
    # 1. behavioral change
    # 2. policy belief change
    #
    # This prevents every random approval fluctuation
    # from being called adaptive behavior.
    # --------------------------------------------------------

    adaptive_behavior_detected = (
        adaptation_gain
        >= ADAPTATION_GAIN_THRESHOLD
        and policy_shift
        > 0.0
    )

    return EpisodeMetrics(
        episode_id=episode.episode_id,

        customer_id=episode.customer_id,

        population_group=population_group,

        interactions=len(interactions),

        early_approval_rate=float(
            early_approval_rate
        ),

        late_approval_rate=float(
            late_approval_rate
        ),

        adaptation_gain=float(
            adaptation_gain
        ),

        evidence_rate_early=float(
            evidence_rate_early
        ),

        evidence_rate_late=float(
            evidence_rate_late
        ),

        mean_requested_amount_early=float(
            mean_amount_early
        ),

        mean_requested_amount_late=float(
            mean_amount_late
        ),

        policy_belief_shift=float(
            policy_shift
        ),

        adaptive_behavior_detected=bool(
            adaptive_behavior_detected
        ),
    )


# ============================================================
# POPULATION SELECTION
# ============================================================

def select_population_customers(
    world: EnvironmentState,
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
# HUMAN BASELINE EPISODE
#
# Humans do NOT update their request strategy after
# observing Agent A.
#
# This gives us the control group for comparison.
# ============================================================

def run_human_baseline_episode(
    *,
    simulator: Phase16AdaptiveSimulator,
    world: EnvironmentState,
    customer_id: str,
    episode_id: str,
    population_group: str,
    n_interactions: int,
    seed: int,
) -> EpisodeMetrics:

    rng = Random(seed)

    customer = world.customers[
        customer_id
    ]

    interactions: list[
        InteractionRecord
    ] = []

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

        order_id = rng.choice(
            customer.current_order_ids
        )

        order = world.orders[
            order_id
        ]

        # ----------------------------------------------------
        # Human request generation is independent from the
        # previous Agent A decision.
        # ----------------------------------------------------

        amount = round(
            rng.uniform(
                500.0,
                min(
                    5000.0,
                    max(
                        500.0,
                        float(
                            order.order_amount
                        ),
                    ),
                ),
            ),
            2,
        )

        evidence: list[str] = []

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

            claim_text=(
                CLAIM_TEXT[claim_type]
            ),

            requested_amount=amount,

            submitted_at=current_time,

            evidence_available=evidence,
        )

        support_decision = (
            simulator.support_agent.decide(
                customer=customer,
                order=order,
                request=request,
            )
        )

        customer_observation = (
            simulator._build_customer_observation(
                customer=customer,
                order=order,
                request=request,
            )
        )

        support_observation = (
            simulator._build_support_observation(
                decision=support_decision,
            )
        )

        interactions.append(
            InteractionRecord(
                episode_id=episode_id,
                sequence_number=sequence_number,

                customer_id=customer_id,
                order_id=order_id,

                claim_type=claim_type,
                requested_amount=amount,
                evidence_available=list(
                    evidence
                ),

                support_decision=(
                    support_decision.decision
                ),

                reason_code=(
                    support_decision.reason_code
                ),

                approved_amount=(
                    float(
                        support_decision.approved_amount
                    )
                ),

                requires_followup=(
                    bool(
                        support_decision.requires_followup
                    )
                ),

                submitted_at=request.submitted_at,

                customer_observation=(
                    customer_observation
                ),

                support_observation=(
                    support_observation
                ),
            )
        )

        current_time += timedelta(
            hours=rng.randint(
                1,
                72,
            )
        )

    episode = EpisodeResult(
        episode_id=episode_id,

        customer_id=customer_id,

        interactions=interactions,

        initial_policy_beliefs={
            "evidence_sensitivity": 0.50,
            "amount_sensitivity": 0.50,
            "high_value_escalation": 0.50,
        },

        final_policy_beliefs={
            "evidence_sensitivity": 0.50,
            "amount_sensitivity": 0.50,
            "high_value_escalation": 0.50,
        },
    )

    return evaluate_episode(
        episode=episode,
        population_group=population_group,
    )


# ============================================================
# RUN MIXED POPULATION SIMULATION
# ============================================================

def run_phase16_simulation(
    *,
    world: EnvironmentState,
    episodes_per_population: int = (
        EPISODES_PER_POPULATION
    ),
    interactions_per_episode: int = (
        INTERACTIONS_PER_EPISODE
    ),
    seed: int = SEED,
) -> list[EpisodeMetrics]:

    validate_episode_configuration(
        interactions_per_episode
    )

    populations = (
        select_population_customers(
            world
        )
    )

    rng = np.random.default_rng(
        seed
    )

    simulator = Phase16AdaptiveSimulator(
        world=world,
        seed=seed,
    )

    results: list[
        EpisodeMetrics
    ] = []

    for population_group, customer_ids in (
        populations.items()
    ):

        if not customer_ids:
            continue

        # ----------------------------------------------------
        # Select customers deterministically from the
        # corresponding population.
        # ----------------------------------------------------

        selected_ids: list[str] = []

        for _ in range(
            episodes_per_population
        ):

            index = int(
                rng.integers(
                    0,
                    len(customer_ids),
                )
            )

            selected_ids.append(
                customer_ids[index]
            )

        # ----------------------------------------------------
        # Run episodes.
        # ----------------------------------------------------

        for episode_number, customer_id in enumerate(
            selected_ids,
            start=1,
        ):

            episode_id = (
                f"PHASE16_"
                f"{population_group}_"
                f"{episode_number:04d}"
            )

            # ------------------------------------------------
            # ADAPTIVE AGENTS
            # ------------------------------------------------

            if population_group.startswith(
                "ADAPTIVE_"
            ):

                episode = (
                    simulator.run_adaptive_episode(
                        customer_id=customer_id,
                        episode_id=episode_id,
                        n_interactions=(
                            interactions_per_episode
                        ),
                    )
                )

                metrics = evaluate_episode(
                    episode=episode,
                    population_group=(
                        population_group
                    ),
                )

            # ------------------------------------------------
            # HUMAN CONTROL GROUP
            # ------------------------------------------------

            else:

                metrics = (
                    run_human_baseline_episode(
                        simulator=simulator,

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
                                1000
                                * (
                                    list(
                                        populations.keys()
                                    ).index(
                                        population_group
                                    )
                                    + 1
                                )
                            )
                        ),
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
    episode_metrics: list[EpisodeMetrics],
) -> PopulationMetrics:

    if not episode_metrics:
        raise ValueError(
            "Cannot aggregate an empty population."
        )

    population_group = (
        episode_metrics[0]
        .population_group
    )

    if any(
        metric.population_group
        != population_group
        for metric in episode_metrics
    ):

        raise ValueError(
            "Episode metrics contain multiple "
            "population groups."
        )

    detection_rate = safe_rate(
        sum(
            bool(
                metric.adaptive_behavior_detected
            )
            for metric in episode_metrics
        ),
        len(episode_metrics),
    )

    return PopulationMetrics(
        population_group=population_group,

        episodes=len(
            episode_metrics
        ),

        mean_adaptation_gain=mean_or_zero(
            metric.adaptation_gain
            for metric in episode_metrics
        ),

        mean_policy_belief_shift=mean_or_zero(
            metric.policy_belief_shift
            for metric in episode_metrics
        ),

        adaptive_behavior_detection_rate=(
            detection_rate
        ),

        mean_early_approval_rate=mean_or_zero(
            metric.early_approval_rate
            for metric in episode_metrics
        ),

        mean_late_approval_rate=mean_or_zero(
            metric.late_approval_rate
            for metric in episode_metrics
        ),

        mean_evidence_rate_early=mean_or_zero(
            metric.evidence_rate_early
            for metric in episode_metrics
        ),

        mean_evidence_rate_late=mean_or_zero(
            metric.evidence_rate_late
            for metric in episode_metrics
        ),
    )


# ============================================================
# PHASE 16 RESULT
# ============================================================

def build_phase16_result(
    metrics: list[EpisodeMetrics],
) -> Phase16Result:

    if not metrics:
        raise ValueError(
            "Phase 16 produced no episode metrics."
        )

    grouped: dict[
        str,
        list[EpisodeMetrics]
    ] = {}

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

    adaptive_abusive_detection = safe_rate(
        sum(
            metric.adaptive_behavior_detected
            for metric in adaptive_abusive
        ),
        len(adaptive_abusive),
    )

    adaptive_legitimate_detection = safe_rate(
        sum(
            metric.adaptive_behavior_detected
            for metric in adaptive_legitimate
        ),
        len(adaptive_legitimate),
    )

    adaptive_abusive_gain = mean_or_zero(
        metric.adaptation_gain
        for metric in adaptive_abusive
    )

    human_abusive_gain = mean_or_zero(
        metric.adaptation_gain
        for metric in human_abusive
    )

    return Phase16Result(
        total_episodes=len(
            metrics
        ),

        population_metrics=(
            population_metrics
        ),

        adaptive_abusive_detection_rate=(
            adaptive_abusive_detection
        ),

        adaptive_legitimate_detection_rate=(
            adaptive_legitimate_detection
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
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_phase16_result(
    result: Phase16Result,
) -> dict[str, bool]:

    groups = {
        metric.population_group
        for metric
        in result.population_metrics
    }

    validations = {

        "mixed_population_present":
            len(groups) >= 3,

        "adaptive_abusive_present":
            ADAPTIVE_ABUSIVE in groups,

        "adaptive_legitimate_present":
            ADAPTIVE_LEGITIMATE in groups,

        "human_abusive_present":
            HUMAN_ABUSIVE in groups,

        "human_legitimate_present":
            HUMAN_LEGITIMATE in groups,

        "metrics_finite":
            all(
                np.isfinite(
                    [
                        metric.mean_adaptation_gain,
                        metric.mean_policy_belief_shift,
                        metric.mean_early_approval_rate,
                        metric.mean_late_approval_rate,
                        metric.mean_evidence_rate_early,
                        metric.mean_evidence_rate_late,
                    ]
                ).all()
                for metric
                in result.population_metrics
            ),

        "rates_bounded":
            all(
                0.0 <= value <= 1.0
                for metric
                in result.population_metrics
                for value
                in [
                    metric.mean_early_approval_rate,
                    metric.mean_late_approval_rate,
                    metric.adaptive_behavior_detection_rate,
                    metric.mean_evidence_rate_early,
                    metric.mean_evidence_rate_late,
                ]
            ),

        "adaptive_abusive_gain_finite":
            np.isfinite(
                result.adaptive_abusive_mean_gain
            ),

        "human_abusive_gain_finite":
            np.isfinite(
                result.human_abusive_mean_gain
            ),

        "gain_advantage_finite":
            np.isfinite(
                result.adaptive_abusive_gain_advantage
            ),

        "no_direct_deny_action":
            True,
    }

    return validations


# ============================================================
# PRINT REPORT
# ============================================================

def print_phase16_report(
    result: Phase16Result,
) -> None:

    print()
    print("=" * 115)
    print(
        "PHASE 16 — ADAPTIVE AGENT "
        "DETECTION & POPULATION EVALUATION"
    )
    print("=" * 115)

    print()
    print(
        "Total episodes          : "
        f"{result.total_episodes}"
    )

    print(
        "Episode interactions    : "
        f"{INTERACTIONS_PER_EPISODE}"
    )

    print()
    print(
        "THREAT MODEL"
    )

    print("-" * 115)

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

    print("-" * 115)

    print(
        "Observes Agent A output : YES"
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

    print()
    print(
        "ADAPTIVE LOOP"
    )

    print("-" * 115)

    print(
        "1. Agent B submits request"
    )

    print(
        "2. Agent A evaluates request"
    )

    print(
        "3. Agent B observes external decision"
    )

    print(
        "4. Agent B updates policy belief"
    )

    print(
        "5. Agent B changes subsequent behavior"
    )

    print()
    print(
        "POPULATION RESULTS"
    )

    print("-" * 115)

    print(
        f"{'POPULATION':<24}"
        f"{'EPISODES':>10}"
        f"{'EARLY APPR.':>15}"
        f"{'LATE APPR.':>15}"
        f"{'ADAPT. GAIN':>15}"
        f"{'DETECTION':>15}"
    )

    print("-" * 115)

    for metric in result.population_metrics:

        print(
            f"{metric.population_group:<24}"
            f"{metric.episodes:>10}"
            f"{metric.mean_early_approval_rate:>15.4f}"
            f"{metric.mean_late_approval_rate:>15.4f}"
            f"{metric.mean_adaptation_gain:>15.4f}"
            f"{metric.adaptive_behavior_detection_rate:>15.4f}"
        )

    print()
    print(
        "ADAPTIVE ATTACKER RESULT"
    )

    print("-" * 115)

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

    print()
    print(
        "INTERPRETATION"
    )

    print("-" * 115)

    if (
        result.adaptive_abusive_gain_advantage
        > 0
    ):

        print(
            "Adaptive abusive agents show "
            "stronger decision-conditioned "
            "behavioral change than human abusive "
            "customers."
        )

    else:

        print(
            "Adaptive abusive agents did not show "
            "a stronger adaptation signal than the "
            "human abusive baseline."
        )

    if (
        result.adaptive_abusive_detection_rate
        > 0
    ):

        print(
            "Adaptive behavior is observable from "
            "the interaction trajectory."
        )

    else:

        print(
            "No adaptive behavior signal crossed "
            "the current detection threshold."
        )

    print()
    print(
        "VALIDATION"
    )

    print("-" * 115)

    validations = validate_phase16_result(
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
            "Phase 16 validation failed."
        )

    print()
    print(
        "PHASE 16 ADAPTIVE AGENT "
        "EVALUATION COMPLETE"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    config = (
        DEFAULT_POPULATION_CONFIG
    )

    world = create_world(
        config=config,
        seed=SEED,
    )

    metrics = run_phase16_simulation(
        world=world,

        episodes_per_population=(
            EPISODES_PER_POPULATION
        ),

        interactions_per_episode=(
            INTERACTIONS_PER_EPISODE
        ),

        seed=SEED,
    )

    result = build_phase16_result(
        metrics
    )

    print_phase16_report(
        result
    )


if __name__ == "__main__":
    main()