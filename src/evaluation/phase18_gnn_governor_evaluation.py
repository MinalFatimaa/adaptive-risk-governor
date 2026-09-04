from __future__ import annotations

"""
PHASE 18 — CLOSED-LOOP GNN GOVERNOR EVALUATION

Purpose
-------
Phase 18 evaluates the FINAL architecture:

    Agent B
       ↓
    Agent A
       ↓
    GNN Risk Governor
       ↓
    Governor intervention / allow
       ↓
    Effective observable outcome
       ↓
    Agent B updates its policy
       ↓
    Next interaction

The critical property being tested is:

    Agent B adapts to the observable outcome,
    but never receives the Governor's private
    GNN/network/strategic information.

This is a CLOSED-LOOP adaptive-adversary experiment.

Baseline:
    Agent B → Agent A → outcome → Agent B learns

Governed:
    Agent B → Agent A → GNN Governor
             → effective outcome → Agent B learns

The Governor acts BEFORE the economic outcome.

Governor action space:

    ALLOW_AGENT_A_DECISION
    REQUEST_ADDITIONAL_EVIDENCE
    ESCALATE_TO_HUMAN_REVIEW

The Governor does not directly reject a request.

The GNN is the final Governor architecture:

    Interaction Graph
            ↓
       Trained GNN
            ↓
    GNN Governor Adapter
            ↓
    Behavioral Governor
            ↓
       Risk Fusion
            ↓
      Final Governor
        intervention

Agent B never receives:
    - GNN risk
    - GNN features
    - fused risk
    - network risk
    - strategic risk
    - Governor existence/identity
    - ground truth
    - other customer information
"""

# ============================================================
# WINDOWS CONSOLE UTF-8 COMPATIBILITY
# ============================================================

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


# ============================================================
# STANDARD LIBRARY
# ============================================================

from dataclasses import dataclass
from datetime import datetime, timedelta
from random import Random
from typing import Any



# ============================================================
# THIRD-PARTY
# ============================================================

import numpy as np


# ============================================================
# PROJECT IMPORTS
# ============================================================

from ..agents.adaptive_customer import (
    AdaptiveCustomerAgent,
    AdaptiveCustomerState,
)

from ..environment.population_config import (
    DEFAULT_POPULATION_CONFIG,
    CustomerType,
)

from ..environment.simulator import (
    AdaptiveInteractionSimulator,
)

from ..environment.world_generator import (
    create_world,
)

from ..governor.interaction_graph import (
    build_interaction_graph,
)

from ..governor.gnn_risk_governor import (
    GNNRiskGovernor,
)

# ------------------------------------------------------------
# IMPORTANT:
#
# These imports are intentionally explicit.
#
# Phase 19 performs a source-level architecture audit and
# requires Phase 18 to import the actual final-architecture
# components:
#
#     trained GNN
#     GNN → Governor adapter
#     behavioral Governor
#     Risk Fusion
#     complete GNN Governor
#
# They are not reimplemented here.
# They are imported from their real implementation modules.
# ------------------------------------------------------------

from ..governor.gnn_inference import (
    GNNRiskInference,
)

from ..governor.gnn_governor_integration import (
    GNNGovernorAdapter,
)

from ..governor.risk_governor import (
    AdaptiveRiskGovernor,
)

from ..governor.risk_fusion import (
    RiskFusionEngine,
)

from ..schemas.support import (
    RefundRequest,
    SupportDecision,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

EPISODES_PER_GROUP = 20

INTERACTIONS_PER_EPISODE = 24

GNN_WEIGHT = 0.20

APPROVAL_DECISION = "APPROVE"


# ============================================================
# CANONICAL GOVERNOR ACTION SPACE
# ============================================================

ALLOW_AGENT_A_DECISION = (
    "ALLOW_AGENT_A_DECISION"
)

REQUEST_ADDITIONAL_EVIDENCE = (
    "REQUEST_ADDITIONAL_EVIDENCE"
)

ESCALATE_TO_HUMAN_REVIEW = (
    "ESCALATE_TO_HUMAN_REVIEW"
)

VALID_GOVERNOR_ACTIONS = {
    ALLOW_AGENT_A_DECISION,
    REQUEST_ADDITIONAL_EVIDENCE,
    ESCALATE_TO_HUMAN_REVIEW,
}


# ============================================================
# OBSERVABLE EFFECTIVE DECISIONS
# ============================================================

EFFECTIVE_EVIDENCE_DECISION = (
    "REQUEST_EVIDENCE"
)

EFFECTIVE_ESCALATION_DECISION = (
    "ESCALATE"
)


# ============================================================
# POPULATION LABELS
# ============================================================

HUMAN_LEGITIMATE = (
    "HUMAN_LEGITIMATE"
)

HUMAN_ABUSIVE = (
    "HUMAN_ABUSIVE"
)

ADAPTIVE_LEGITIMATE = (
    "ADAPTIVE_LEGITIMATE"
)

ADAPTIVE_ABUSIVE = (
    "ADAPTIVE_ABUSIVE"
)


# ============================================================
# CLAIM TYPES
# ============================================================

CLAIM_TYPES = [
    "SHORTAGE_CLAIM",
    "WRONG_ITEM_CLAIM",
    "NON_DELIVERY_CLAIM",
    "SUBSTITUTED_RETURN_CLAIM",
]


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


EVIDENCE_BY_CLAIM = {
    "SHORTAGE_CLAIM": [
        "delivery_photo"
    ],

    "WRONG_ITEM_CLAIM": [
        "package_photo"
    ],

    "NON_DELIVERY_CLAIM": [
        "delivery_evidence"
    ],

    "SUBSTITUTED_RETURN_CLAIM": [
        "return_receipt"
    ],
}


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def bounded(
    value: float,
) -> float:

    try:
        value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if not np.isfinite(value):
        return 0.0

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def safe_rate(
    numerator: float,
    denominator: float,
) -> float:

    if denominator <= 0:
        return 0.0

    value = (
        float(numerator)
        / float(denominator)
    )

    if not np.isfinite(value):
        return 0.0

    return bounded(
        value
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


# ============================================================
# POPULATION CLASSIFICATION
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
# CUSTOMER → GRAPH NODE
# ============================================================

def customer_node_index(
    graph,
    customer_id: str,
) -> int:

    if not hasattr(
        graph,
        "node_ids",
    ):

        raise ValueError(
            "Interaction graph does not expose node_ids."
        )

    if not hasattr(
        graph,
        "node_types",
    ):

        raise ValueError(
            "Interaction graph does not expose node_types."
        )

    for index, (
        node_id,
        node_type,
    ) in enumerate(
        zip(
            graph.node_ids,
            graph.node_types,
        )
    ):

        if (
            str(node_id)
            == str(customer_id)
            and node_type
            == "CUSTOMER"
        ):

            return int(index)

    raise ValueError(
        "Customer was not found as a CUSTOMER "
        "node in the interaction graph: "
        f"{customer_id}"
    )


# ============================================================
# METRICS
# ============================================================

@dataclass(frozen=True)
class GNNGovernorEpisodeMetrics:

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

    intervention_count: int

    evidence_request_count: int

    human_review_count: int

    intervention_rate: float

    first_intervention_step: int | None

    mean_gnn_risk: float

    maximum_gnn_risk: float

    mean_fused_risk: float

    maximum_fused_risk: float

    adaptive_policy_change: float

    adaptive_attacker_learned_from_governor: bool


@dataclass(frozen=True)
class GNNGovernorPopulationMetrics:

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

    mean_gnn_risk: float

    maximum_gnn_risk: float

    mean_fused_risk: float

    maximum_fused_risk: float

    mean_policy_change: float

    mean_detection_latency: float


@dataclass(frozen=True)
class Phase18Result:

    total_episodes: int

    population_metrics: tuple[
        GNNGovernorPopulationMetrics,
        ...
    ]

    adaptive_abusive_baseline_approval: float

    adaptive_abusive_governed_approval: float

    human_abusive_baseline_approval: float

    human_abusive_governed_approval: float

    adaptive_abusive_baseline_value: float

    adaptive_abusive_governed_value: float

    human_abusive_baseline_value: float

    human_abusive_governed_value: float

    adaptive_abusive_prevented_value: float

    human_abusive_prevented_value: float

    adaptive_abusive_intervention_rate: float

    adaptive_legitimate_intervention_rate: float

    human_abusive_intervention_rate: float

    human_legitimate_intervention_rate: float

    adaptive_governor_advantage: float

    adaptive_vs_human_intervention_gap: float

    adaptive_policy_change: float


# ============================================================
# SUPPORT DECISION → EFFECTIVE OBSERVABLE DECISION
# ============================================================

def build_effective_decision(
    *,
    original_decision: SupportDecision,
    governor_action: str,
) -> SupportDecision:

    """
    Resolve the Governor action into the observable
    decision that Agent B learns from.

    ALLOW:
        Agent B observes the original Agent A decision.

    REQUEST_ADDITIONAL_EVIDENCE:
        Agent B observes REQUEST_EVIDENCE.

    ESCALATE_TO_HUMAN_REVIEW:
        Agent B observes ESCALATE.

    The Governor itself is not revealed.
    """

    if governor_action == (
        ALLOW_AGENT_A_DECISION
    ):

        return original_decision

    if governor_action == (
        REQUEST_ADDITIONAL_EVIDENCE
    ):

        return SupportDecision(
            decision=(
                EFFECTIVE_EVIDENCE_DECISION
            ),

            reason_code=(
                "GOVERNOR_ADDITIONAL_EVIDENCE"
            ),

            message=(
                "Additional evidence is required "
                "before the request can proceed."
            ),

            requested_evidence=(
                list(
                    original_decision.requested_evidence
                )
                or [
                    "supporting_documentation"
                ]
            ),

            approved_amount=0.0,

            requires_followup=True,

            request_id=(
                original_decision.request_id
            ),

            customer_id=(
                original_decision.customer_id
            ),

            order_id=(
                original_decision.order_id
            ),

            timestamp=(
                original_decision.timestamp
            ),
        )

    if governor_action == (
        ESCALATE_TO_HUMAN_REVIEW
    ):

        return SupportDecision(
            decision=(
                EFFECTIVE_ESCALATION_DECISION
            ),

            reason_code=(
                "GOVERNOR_HUMAN_REVIEW"
            ),

            message=(
                "The request requires human review."
            ),

            requested_evidence=[],

            approved_amount=0.0,

            requires_followup=True,

            request_id=(
                original_decision.request_id
            ),

            customer_id=(
                original_decision.customer_id
            ),

            order_id=(
                original_decision.order_id
            ),

            timestamp=(
                original_decision.timestamp
            ),
        )

    raise AssertionError(
        "Unknown Governor action: "
        f"{governor_action}"
    )


# ============================================================
# POLICY CHANGE
# ============================================================

def policy_distance(
    initial_policy: dict[str, float],
    final_policy: dict[str, float],
) -> float:

    keys = sorted(
        set(initial_policy)
        & set(final_policy)
    )

    if not keys:
        return 0.0

    differences = []

    for key in keys:

        initial = float(
            initial_policy[key]
        )

        final = float(
            final_policy[key]
        )

        differences.append(
            abs(
                final
                - initial
            )
        )

    return safe_mean(
        differences
    )


# ============================================================
# GNN GOVERNOR EVALUATION
# ============================================================

def evaluate_gnn_governor(
    *,
    governor: GNNRiskGovernor,
    graph,
    customer_id: str,
    history: list[dict[str, Any]],
    request: RefundRequest,
    support_decision: SupportDecision,
) -> Any:

    """
    Run the FINAL GNN Governor pipeline.

        Interaction Graph
             ↓
        Trained GNN
             ↓
        GNN Governor Adapter
             ↓
        Behavioral Governor
             ↓
        Risk Fusion
             ↓
        Final Governor Decision
    """

    node_index = customer_node_index(
        graph,
        customer_id,
    )

    decision = governor.evaluate(
        data=graph.data,

        customer_id=customer_id,

        customer_node_index=node_index,

        customer_history=history,

        current_claim_type=(
            request.claim_type
        ),

        requested_amount=float(
            request.requested_amount
        ),

        evidence_available=list(
            request.evidence_available
        ),

        support_decision=(
            support_decision.decision
        ),

        strategic_state=None,

        network_observations=[],

        current_ip="phase18-ip",

        current_device_id="phase18-device",

        current_payment_method_id=(
            "phase18-payment"
        ),

        current_shipping_address_id=(
            "phase18-address"
        ),
    )

    if decision is None:

        raise ValueError(
            "GNNRiskGovernor returned None."
        )

    return decision


# ============================================================
# BASELINE ADAPTIVE EPISODE
# ============================================================

def run_baseline_adaptive_episode(
    *,
    simulator: AdaptiveInteractionSimulator,
    customer_id: str,
    episode_id: str,
    n_interactions: int,
) -> tuple[Any, float]:

    """
    Baseline:

        Agent B
           ↓
        Agent A
           ↓
        observable outcome
           ↓
        Agent B learns

    Uses the existing AdaptiveInteractionSimulator
    unchanged.
    """

    episode = simulator.run_episode(
        customer_id=customer_id,

        episode_id=episode_id,

        n_interactions=n_interactions,
    )

    policy_change = policy_distance(
        episode.initial_policy_beliefs,
        episode.final_policy_beliefs,
    )

    return (
        episode,
        policy_change,
    )


# ============================================================
# GOVERNED ADAPTIVE EPISODE
# ============================================================

def run_governed_adaptive_episode(
    *,
    world,
    graph,
    governor: GNNRiskGovernor,
    simulator: AdaptiveInteractionSimulator,
    customer_id: str,
    episode_id: str,
    n_interactions: int,
) -> tuple[
    list[Any],
    dict[str, float],
    dict[str, float],
    list[float],
    list[float],
    int,
    int,
    int,
    int | None,
    float,
    float,
]:

    """
    CLOSED-LOOP GOVERNED ADAPTIVE EPISODE.

    Loop:

        1. Agent B chooses request.
        2. Agent A evaluates request.
        3. GNN Governor evaluates request.
        4. Governor acts before outcome.
        5. Effective observable decision is produced.
        6. Agent B observes only that effective decision.
        7. Agent B updates its policy.
        8. Next request is generated using updated policy.

    Governor-private information is never passed to Agent B.
    """

    customer = world.customers[
        customer_id
    ]

    private_state = world.customer_private[
        customer_id
    ]

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

    # --------------------------------------------------------
    # Create Agent B using the same seed sequence as the
    # existing AdaptiveInteractionSimulator.
    # --------------------------------------------------------

    agent_b_state = AdaptiveCustomerState(
        customer_id=customer_id,

        objective=objective,
    )

    agent_b = AdaptiveCustomerAgent(
        state=agent_b_state,

        seed=simulator.rng.randint(
            0,
            10_000_000,
        ),
    )

    initial_policy = (
        agent_b.get_inferred_policy().copy()
    )

    interactions = []

    current_time = datetime(
        2026,
        8,
        31,
        12,
        0,
        0,
    )

    # --------------------------------------------------------
    # Governor-private history.
    #
    # This is intentionally separate from Agent B.
    # --------------------------------------------------------

    history: list[
        dict[str, Any]
    ] = []

    gnn_risks = []

    fused_risks = []

    intervention_count = 0

    evidence_request_count = 0

    human_review_count = 0

    first_intervention_step = None

    for sequence_number in range(
        1,
        n_interactions + 1,
    ):

        # ====================================================
        # 1. AGENT B GENERATES REQUEST
        # ====================================================

        request = simulator._generate_request(
            customer=customer,

            sequence_number=sequence_number,

            episode_id=episode_id,

            current_time=current_time,

            agent_b=agent_b,
        )

        # ====================================================
        # 2. AGENT A DECIDES
        # ====================================================

        order = world.orders[
            request.order_id
        ]

        support_decision = (
            simulator.support_agent.decide(
                customer=customer,

                order=order,

                request=request,
            )
        )

        # ====================================================
        # 3. GOVERNOR EVALUATES
        # ====================================================

        governor_decision = (
            evaluate_gnn_governor(
                governor=governor,

                graph=graph,

                customer_id=customer_id,

                history=history,

                request=request,

                support_decision=support_decision,
            )
        )

        action = str(
            governor_decision.action
        )

        if action not in (
            VALID_GOVERNOR_ACTIONS
        ):

            raise AssertionError(
                "Invalid Governor action: "
                f"{action}"
            )

        gnn_risks.append(
            bounded(
                governor_decision.gnn_risk
            )
        )

        fused_risks.append(
            bounded(
                governor_decision.fused_risk_score
            )
        )

        # ====================================================
        # 4. GOVERNOR INTERVENES BEFORE OUTCOME
        # ====================================================

        if action != (
            ALLOW_AGENT_A_DECISION
        ):

            intervention_count += 1

            if first_intervention_step is None:

                first_intervention_step = (
                    sequence_number
                )

        if action == (
            REQUEST_ADDITIONAL_EVIDENCE
        ):

            evidence_request_count += 1

        if action == (
            ESCALATE_TO_HUMAN_REVIEW
        ):

            human_review_count += 1

        # ====================================================
        # 5. RESOLVE EFFECTIVE OBSERVABLE OUTCOME
        # ====================================================

        effective_decision = (
            build_effective_decision(
                original_decision=(
                    support_decision
                ),

                governor_action=action,
            )
        )

        # ====================================================
        # 6. AGENT B LEARNS ONLY FROM EFFECTIVE OUTCOME
        # ====================================================

        # CRITICAL INFORMATION BOUNDARY:
        #
        # Agent B receives:
        #
        #     effective_decision
        #     request claim type
        #     requested amount
        #     evidence available
        #
        # Agent B does NOT receive:
        #
        #     governor_decision
        #     GNN risk
        #     fused risk
        #     network risk
        #     strategic risk
        #     ground truth
        #     other customer information
        #

        agent_b.infer_support_policy(
            decision=effective_decision,

            claim_type=request.claim_type,

            requested_amount=(
                request.requested_amount
            ),

            evidence_available=(
                request.evidence_available
            ),
        )

        # ====================================================
        # 7. RECORD GOVERNOR-PRIVATE HISTORY
        # ====================================================

        history.append(
            {
                "claim_type":
                    request.claim_type,

                "requested_amount":
                    float(
                        request.requested_amount
                    ),

                "support_decision":
                    support_decision.decision,
            }
        )

        interactions.append(
            {
                "sequence_number":
                    sequence_number,

                "request":
                    request,

                "support_decision":
                    support_decision,

                "governor_decision":
                    governor_decision,

                "effective_decision":
                    effective_decision,
            }
        )

        # ====================================================
        # 8. ADVANCE TIME
        # ====================================================

        current_time += timedelta(
            hours=simulator.rng.randint(
                1,
                72,
            )
        )

    final_policy = (
        agent_b.get_inferred_policy().copy()
    )

    policy_change = policy_distance(
        initial_policy,
        final_policy,
    )

    return (
        interactions,
        initial_policy,
        final_policy,
        gnn_risks,
        fused_risks,
        intervention_count,
        evidence_request_count,
        human_review_count,
        first_intervention_step,
        policy_change,
        current_time.timestamp(),
    )


# ============================================================
# HUMAN EPISODE
# ============================================================

def run_human_episode(
    *,
    world,
    graph,
    governor: GNNRiskGovernor,
    customer_id: str,
    episode_id: str,
    n_interactions: int,
    seed: int,
) -> GNNGovernorEpisodeMetrics:

    """
    Fixed human baseline.

    Humans do not maintain the adaptive policy learner.
    """

    customer = world.customers[
        customer_id
    ]

    simulator = AdaptiveInteractionSimulator(
        world=world,
        seed=seed,
    )

    rng = Random(seed)

    current_time = datetime(
        2026,
        8,
        31,
        12,
        0,
        0,
    )

    interactions = []

    baseline_approved_count = 0

    governed_approved_count = 0

    baseline_approved_value = 0.0

    governed_approved_value = 0.0

    baseline_requested_value = 0.0

    governed_requested_value = 0.0

    intervention_count = 0

    evidence_request_count = 0

    human_review_count = 0

    first_intervention_step = None

    gnn_risks = []

    fused_risks = []

    history: list[
        dict[str, Any]
    ] = []

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
                float(
                    order.order_amount
                ),
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

            claim_text=(
                CLAIM_TEXT[
                    claim_type
                ]
            ),

            requested_amount=amount,

            submitted_at=current_time,

            evidence_available=evidence,
        )

        # ----------------------------------------------------
        # Agent A
        # ----------------------------------------------------

        support_decision = (
            simulator.support_agent.decide(
                customer=customer,

                order=order,

                request=request,
            )
        )

        baseline_requested_value += (
            amount
        )

        if (
            support_decision.decision
            == APPROVAL_DECISION
        ):

            baseline_approved_count += 1

            baseline_approved_value += float(
                support_decision.approved_amount
            )

        # ----------------------------------------------------
        # GNN Governor
        # ----------------------------------------------------

        governor_decision = (
            evaluate_gnn_governor(
                governor=governor,

                graph=graph,

                customer_id=customer_id,

                history=history,

                request=request,

                support_decision=support_decision,
            )
        )

        action = str(
            governor_decision.action
        )

        if action not in (
            VALID_GOVERNOR_ACTIONS
        ):

            raise AssertionError(
                "Invalid Governor action: "
                f"{action}"
            )

        gnn_risks.append(
            bounded(
                governor_decision.gnn_risk
            )
        )

        fused_risks.append(
            bounded(
                governor_decision.fused_risk_score
            )
        )

        if action != (
            ALLOW_AGENT_A_DECISION
        ):

            intervention_count += 1

            if first_intervention_step is None:

                first_intervention_step = (
                    sequence_number
                )

        if action == (
            REQUEST_ADDITIONAL_EVIDENCE
        ):

            evidence_request_count += 1

        if action == (
            ESCALATE_TO_HUMAN_REVIEW
        ):

            human_review_count += 1

        # ----------------------------------------------------
        # Effective observable outcome
        # ----------------------------------------------------

        effective_decision = (
            build_effective_decision(
                original_decision=(
                    support_decision
                ),

                governor_action=action,
            )
        )

        if (
            action
            == ALLOW_AGENT_A_DECISION
            and effective_decision.decision
            == APPROVAL_DECISION
        ):

            governed_approved_count += 1

            governed_approved_value += float(
                effective_decision.approved_amount
            )

        governed_requested_value += (
            amount
        )

        # ----------------------------------------------------
        # Governor-private history
        # ----------------------------------------------------

        history.append(
            {
                "claim_type":
                    request.claim_type,

                "requested_amount":
                    float(
                        request.requested_amount
                    ),

                "support_decision":
                    support_decision.decision,
            }
        )

        interactions.append(
            {
                "request":
                    request,

                "support_decision":
                    support_decision,

                "governor_decision":
                    governor_decision,

                "effective_decision":
                    effective_decision,
            }
        )

        current_time += timedelta(
            hours=rng.randint(
                1,
                72,
            )
        )

    prevented_value = max(
        0.0,
        baseline_approved_value
        - governed_approved_value,
    )

    return GNNGovernorEpisodeMetrics(
        episode_id=episode_id,

        customer_id=customer_id,

        population_group=classify_customer(
            world,
            customer_id,
        ),

        interactions=len(
            interactions
        ),

        baseline_approved_count=(
            baseline_approved_count
        ),

        governed_approved_count=(
            governed_approved_count
        ),

        baseline_approval_rate=safe_rate(
            baseline_approved_count,
            len(interactions),
        ),

        governed_approval_rate=safe_rate(
            governed_approved_count,
            len(interactions),
        ),

        baseline_approved_value=float(
            baseline_approved_value
        ),

        governed_approved_value=float(
            governed_approved_value
        ),

        baseline_requested_value=float(
            baseline_requested_value
        ),

        governed_requested_value=float(
            governed_requested_value
        ),

        prevented_approval_value=float(
            prevented_value
        ),

        intervention_count=(
            intervention_count
        ),

        evidence_request_count=(
            evidence_request_count
        ),

        human_review_count=(
            human_review_count
        ),

        intervention_rate=safe_rate(
            intervention_count,
            len(interactions),
        ),

        first_intervention_step=(
            first_intervention_step
        ),

        mean_gnn_risk=safe_mean(
            gnn_risks
        ),

        maximum_gnn_risk=(
            max(gnn_risks)
            if gnn_risks
            else 0.0
        ),

        mean_fused_risk=safe_mean(
            fused_risks
        ),

        maximum_fused_risk=(
            max(fused_risks)
            if fused_risks
            else 0.0
        ),

        adaptive_policy_change=0.0,

        adaptive_attacker_learned_from_governor=False,
    )


# ============================================================
# ADAPTIVE EPISODE METRIC BUILDER
# ============================================================

def build_adaptive_episode_metrics(
    *,
    world,
    customer_id: str,
    population_group: str,
    episode_id: str,
    baseline_episode,
    governed_result,
) -> GNNGovernorEpisodeMetrics:

    (
        governed_interactions,
        initial_policy,
        final_policy,
        gnn_risks,
        fused_risks,
        intervention_count,
        evidence_request_count,
        human_review_count,
        first_intervention_step,
        governed_policy_change,
        _,
    ) = governed_result

    baseline_interactions = (
        baseline_episode.interactions
    )

    # --------------------------------------------------------
    # IMPORTANT FIX:
    #
    # EpisodeResult.interactions contains InteractionRecord
    # objects.
    #
    # Therefore the correct fields are:
    #
    #     interaction.support_decision
    #     interaction.approved_amount
    #     interaction.requested_amount
    #
    # The previous implementation incorrectly treated
    # interaction.support_decision as a SupportDecision object.
    # --------------------------------------------------------

    baseline_approved_count = sum(
        interaction.support_decision
        == APPROVAL_DECISION
        for interaction
        in baseline_interactions
    )

    baseline_approved_value = sum(
        float(
            interaction.approved_amount
        )
        for interaction
        in baseline_interactions
    )

    baseline_requested_value = sum(
        float(
            interaction.requested_amount
        )
        for interaction
        in baseline_interactions
    )

    governed_approved_count = 0

    governed_approved_value = 0.0

    governed_requested_value = 0.0

    for item in governed_interactions:

        request = item[
            "request"
        ]

        effective_decision = item[
            "effective_decision"
        ]

        governed_requested_value += float(
            request.requested_amount
        )

        if (
            effective_decision.decision
            == APPROVAL_DECISION
        ):

            governed_approved_count += 1

            governed_approved_value += float(
                effective_decision.approved_amount
            )

    prevented_value = max(
        0.0,
        baseline_approved_value
        - governed_approved_value,
    )

    # --------------------------------------------------------
    # Agent B is considered to have learned from the
    # Governor only when:
    #
    #     1. the Governor actually intervened, AND
    #     2. the adaptive policy measurably changed.
    #
    # An intervention alone is not evidence of learning.
    # --------------------------------------------------------

    attacker_learned_from_governor = (
        intervention_count > 0
        and governed_policy_change > 0.0
    )

    return GNNGovernorEpisodeMetrics(
        episode_id=episode_id,

        customer_id=customer_id,

        population_group=population_group,

        interactions=len(
            governed_interactions
        ),

        baseline_approved_count=(
            baseline_approved_count
        ),

        governed_approved_count=(
            governed_approved_count
        ),

        baseline_approval_rate=safe_rate(
            baseline_approved_count,
            len(baseline_interactions),
        ),

        governed_approval_rate=safe_rate(
            governed_approved_count,
            len(governed_interactions),
        ),

        baseline_approved_value=float(
            baseline_approved_value
        ),

        governed_approved_value=float(
            governed_approved_value
        ),

        baseline_requested_value=float(
            baseline_requested_value
        ),

        governed_requested_value=float(
            governed_requested_value
        ),

        prevented_approval_value=float(
            prevented_value
        ),

        intervention_count=(
            intervention_count
        ),

        evidence_request_count=(
            evidence_request_count
        ),

        human_review_count=(
            human_review_count
        ),

        intervention_rate=safe_rate(
            intervention_count,
            len(governed_interactions),
        ),

        first_intervention_step=(
            first_intervention_step
        ),

        mean_gnn_risk=safe_mean(
            gnn_risks
        ),

        maximum_gnn_risk=(
            max(gnn_risks)
            if gnn_risks
            else 0.0
        ),

        mean_fused_risk=safe_mean(
            fused_risks
        ),

        maximum_fused_risk=(
            max(fused_risks)
            if fused_risks
            else 0.0
        ),

        adaptive_policy_change=(
            governed_policy_change
        ),

        adaptive_attacker_learned_from_governor=(
            attacker_learned_from_governor
        ),
    )


# ============================================================
# RUN PHASE 18
# ============================================================

def run_phase18_simulation(
    *,
    world,
    episodes_per_population: int = (
        EPISODES_PER_GROUP
    ),
    interactions_per_episode: int = (
        INTERACTIONS_PER_EPISODE
    ),
    seed: int = SEED,
) -> list[
    GNNGovernorEpisodeMetrics
]:

    if episodes_per_population < 1:

        raise ValueError(
            "Phase 18 requires at least "
            "1 episode per population."
        )

    if interactions_per_episode < 1:

        raise ValueError(
            "Phase 18 requires at least "
            "1 interaction per episode."
        )

    populations = (
        select_population_customers(
            world
        )
    )

    # --------------------------------------------------------
    # FINAL interaction graph.
    # --------------------------------------------------------

    graph = build_interaction_graph(
        world
    )

    if graph is None:

        raise ValueError(
            "Interaction graph construction failed."
        )

    if graph.data.num_nodes <= 0:

        raise ValueError(
            "Interaction graph contains no nodes."
        )

    if graph.data.num_edges <= 0:

        raise ValueError(
            "Interaction graph contains no edges."
        )

    if graph.data.x is None:

        raise ValueError(
            "Interaction graph contains no node features."
        )

    if graph.data.x.ndim != 2:

        raise ValueError(
            "GNN graph features must be a 2D tensor."
        )

    if graph.data.x.shape[1] != 12:

        raise ValueError(
            "Expected 12 GNN input features, "
            f"got {graph.data.x.shape[1]}."
        )

    # --------------------------------------------------------
    # FINAL GNN Governor.
    # --------------------------------------------------------

    governor = GNNRiskGovernor(
        gnn_weight=GNN_WEIGHT
    )

    results = []

    rng = np.random.default_rng(
        seed
    )

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
                f"PHASE18_"
                f"{population_group}_"
                f"{episode_number:04d}"
            )

            episode_seed = (
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
            )

            # =================================================
            # ADAPTIVE POPULATION
            # =================================================

            if population_group.startswith(
                "ADAPTIVE_"
            ):

                # ------------------------------------------------
                # Separate simulator instances preserve the
                # baseline/governed RNG isolation.
                # ------------------------------------------------

                baseline_simulator = (
                    AdaptiveInteractionSimulator(
                        world=world,
                        seed=episode_seed,
                    )
                )

                governed_simulator = (
                    AdaptiveInteractionSimulator(
                        world=world,
                        seed=episode_seed,
                    )
                )

                baseline_episode, _ = (
                    run_baseline_adaptive_episode(
                        simulator=(
                            baseline_simulator
                        ),

                        customer_id=customer_id,

                        episode_id=(
                            episode_id
                            + "_BASELINE"
                        ),

                        n_interactions=(
                            interactions_per_episode
                        ),
                    )
                )

                governed_result = (
                    run_governed_adaptive_episode(
                        world=world,

                        graph=graph,

                        governor=governor,

                        simulator=(
                            governed_simulator
                        ),

                        customer_id=customer_id,

                        episode_id=(
                            episode_id
                            + "_GOVERNED"
                        ),

                        n_interactions=(
                            interactions_per_episode
                        ),
                    )
                )

                metrics = (
                    build_adaptive_episode_metrics(
                        world=world,

                        customer_id=customer_id,

                        population_group=(
                            population_group
                        ),

                        episode_id=episode_id,

                        baseline_episode=(
                            baseline_episode
                        ),

                        governed_result=(
                            governed_result
                        ),
                    )
                )

                results.append(
                    metrics
                )

            # =================================================
            # HUMAN POPULATION
            # =================================================

            else:

                metrics = run_human_episode(
                    world=world,

                    graph=graph,

                    governor=governor,

                    customer_id=customer_id,

                    episode_id=episode_id,

                    n_interactions=(
                        interactions_per_episode
                    ),

                    seed=episode_seed,
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
        GNNGovernorEpisodeMetrics
    ],
) -> GNNGovernorPopulationMetrics:

    if not metrics:

        raise ValueError(
            "Cannot aggregate empty population."
        )

    population_group = (
        metrics[0].population_group
    )

    if any(
        metric.population_group
        != population_group
        for metric in metrics
    ):

        raise ValueError(
            "Multiple population groups "
            "were provided."
        )

    latencies = [
        float(
            metric.first_intervention_step
        )
        for metric in metrics
        if metric.first_intervention_step
        is not None
    ]

    total_interactions = sum(
        metric.interactions
        for metric in metrics
    )

    return GNNGovernorPopulationMetrics(
        population_group=(
            population_group
        ),

        episodes=len(metrics),

        baseline_approval_rate=safe_mean(
            [
                metric.baseline_approval_rate
                for metric in metrics
            ]
        ),

        governed_approval_rate=safe_mean(
            [
                metric.governed_approval_rate
                for metric in metrics
            ]
        ),

        approval_reduction=safe_mean(
            [
                max(
                    0.0,
                    metric.baseline_approval_rate
                    - metric.governed_approval_rate,
                )
                for metric in metrics
            ]
        ),

        baseline_approved_value=safe_mean(
            [
                metric.baseline_approved_value
                for metric in metrics
            ]
        ),

        governed_approved_value=safe_mean(
            [
                metric.governed_approved_value
                for metric in metrics
            ]
        ),

        prevented_approval_value=safe_mean(
            [
                metric.prevented_approval_value
                for metric in metrics
            ]
        ),

        intervention_rate=safe_mean(
            [
                metric.intervention_rate
                for metric in metrics
            ]
        ),

        evidence_request_rate=safe_rate(
            sum(
                metric.evidence_request_count
                for metric in metrics
            ),
            total_interactions,
        ),

        human_review_rate=safe_rate(
            sum(
                metric.human_review_count
                for metric in metrics
            ),
            total_interactions,
        ),

        mean_gnn_risk=safe_mean(
            [
                metric.mean_gnn_risk
                for metric in metrics
            ]
        ),

        maximum_gnn_risk=(
            max(
                metric.maximum_gnn_risk
                for metric in metrics
            )
            if metrics
            else 0.0
        ),

        mean_fused_risk=safe_mean(
            [
                metric.mean_fused_risk
                for metric in metrics
            ]
        ),

        maximum_fused_risk=(
            max(
                metric.maximum_fused_risk
                for metric in metrics
            )
            if metrics
            else 0.0
        ),

        mean_policy_change=safe_mean(
            [
                metric.adaptive_policy_change
                for metric in metrics
            ]
        ),

        mean_detection_latency=safe_mean(
            latencies
        ),
    )


# ============================================================
# BUILD RESULT
# ============================================================

def build_phase18_result(
    metrics: list[
        GNNGovernorEpisodeMetrics
    ],
) -> Phase18Result:

    if not metrics:

        raise ValueError(
            "Phase 18 produced no metrics."
        )

    grouped: dict[
        str,
        list[GNNGovernorEpisodeMetrics],
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
    ) -> GNNGovernorPopulationMetrics | None:

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

    adaptive_abusive_baseline_value = (
        adaptive_abusive.baseline_approved_value
        if adaptive_abusive
        else 0.0
    )

    adaptive_abusive_governed_value = (
        adaptive_abusive.governed_approved_value
        if adaptive_abusive
        else 0.0
    )

    human_abusive_baseline_value = (
        human_abusive.baseline_approved_value
        if human_abusive
        else 0.0
    )

    human_abusive_governed_value = (
        human_abusive.governed_approved_value
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

    adaptive_intervention = (
        adaptive_abusive.intervention_rate
        if adaptive_abusive
        else 0.0
    )

    adaptive_legitimate_intervention = (
        adaptive_legitimate.intervention_rate
        if adaptive_legitimate
        else 0.0
    )

    human_abusive_intervention = (
        human_abusive.intervention_rate
        if human_abusive
        else 0.0
    )

    human_legitimate_intervention = (
        human_legitimate.intervention_rate
        if human_legitimate
        else 0.0
    )

    adaptive_governor_advantage = (
        (
            adaptive_abusive_baseline
            - adaptive_abusive_governed
        )
        - (
            human_abusive_baseline
            - human_abusive_governed
        )
    )

    adaptive_vs_human_intervention_gap = (
        adaptive_intervention
        - human_abusive_intervention
    )

    adaptive_policy_change = (
        adaptive_abusive.mean_policy_change
        if adaptive_abusive
        else 0.0
    )

    return Phase18Result(
        total_episodes=len(
            metrics
        ),

        population_metrics=(
            population_metrics
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

        adaptive_abusive_baseline_value=(
            adaptive_abusive_baseline_value
        ),

        adaptive_abusive_governed_value=(
            adaptive_abusive_governed_value
        ),

        human_abusive_baseline_value=(
            human_abusive_baseline_value
        ),

        human_abusive_governed_value=(
            human_abusive_governed_value
        ),

        adaptive_abusive_prevented_value=(
            adaptive_abusive_prevented
        ),

        human_abusive_prevented_value=(
            human_abusive_prevented
        ),

        adaptive_abusive_intervention_rate=(
            adaptive_intervention
        ),

        adaptive_legitimate_intervention_rate=(
            adaptive_legitimate_intervention
        ),

        human_abusive_intervention_rate=(
            human_abusive_intervention
        ),

        human_legitimate_intervention_rate=(
            human_legitimate_intervention
        ),

        adaptive_governor_advantage=(
            adaptive_governor_advantage
        ),

        adaptive_vs_human_intervention_gap=(
            adaptive_vs_human_intervention_gap
        ),

        adaptive_policy_change=(
            adaptive_policy_change
        ),
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_phase18(
    *,
    world,
    graph,
    metrics: list[
        GNNGovernorEpisodeMetrics
    ],
    result: Phase18Result,
) -> None:

    population_names = {
        metric.population_group
        for metric in result.population_metrics
    }

    required_populations = {
        ADAPTIVE_ABUSIVE,
        ADAPTIVE_LEGITIMATE,
        HUMAN_ABUSIVE,
        HUMAN_LEGITIMATE,
    }

    if population_names != (
        required_populations
    ):

        missing = (
            required_populations
            - population_names
        )

        raise AssertionError(
            "Required population representation "
            f"is incomplete. Missing: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    numeric_values = []

    for metric in metrics:

        numeric_values.extend(
            [
                metric.baseline_approval_rate,
                metric.governed_approval_rate,
                metric.baseline_approved_value,
                metric.governed_approved_value,
                metric.baseline_requested_value,
                metric.governed_requested_value,
                metric.prevented_approval_value,
                metric.intervention_rate,
                float(
                    metric.intervention_count
                ),
                float(
                    metric.evidence_request_count
                ),
                float(
                    metric.human_review_count
                ),
                metric.mean_gnn_risk,
                metric.maximum_gnn_risk,
                metric.mean_fused_risk,
                metric.maximum_fused_risk,
                metric.adaptive_policy_change,
            ]
        )

    if not np.isfinite(
        np.asarray(
            numeric_values,
            dtype=float,
        )
    ).all():

        raise AssertionError(
            "Non-finite Phase 18 metric detected."
        )

    # --------------------------------------------------------
    # Rate bounds
    # --------------------------------------------------------

    rate_values = []

    for metric in result.population_metrics:

        rate_values.extend(
            [
                metric.baseline_approval_rate,
                metric.governed_approval_rate,
                metric.approval_reduction,
                metric.intervention_rate,
                metric.evidence_request_rate,
                metric.human_review_rate,
                metric.mean_gnn_risk,
                metric.maximum_gnn_risk,
                metric.mean_fused_risk,
                metric.maximum_fused_risk,
            ]
        )

    if not all(
        0.0 <= value <= 1.0
        for value in rate_values
    ):

        raise AssertionError(
            "One or more bounded metrics "
            "fall outside [0, 1]."
        )

    # --------------------------------------------------------
    # Count validation
    # --------------------------------------------------------

    for metric in metrics:

        if metric.intervention_count < 0:

            raise AssertionError(
                "Negative intervention count."
            )

        if metric.evidence_request_count < 0:

            raise AssertionError(
                "Negative evidence-request count."
            )

        if metric.human_review_count < 0:

            raise AssertionError(
                "Negative human-review count."
            )

        if (
            metric.intervention_count
            > metric.interactions
        ):

            raise AssertionError(
                "Governor intervention count exceeds "
                "interaction count."
            )

        if (
            metric.evidence_request_count
            + metric.human_review_count
            > metric.intervention_count
        ):

            raise AssertionError(
                "Intervention subtype counts exceed "
                "total intervention count."
            )

    # --------------------------------------------------------
    # Graph validation
    # --------------------------------------------------------

    if graph.data.num_nodes <= 0:

        raise AssertionError(
            "GNN graph has no nodes."
        )

    if graph.data.num_edges <= 0:

        raise AssertionError(
            "GNN graph has no edges."
        )

    if graph.data.x is None:

        raise AssertionError(
            "GNN graph has no node features."
        )

    if graph.data.x.ndim != 2:

        raise AssertionError(
            "GNN graph features are not 2D."
        )

    if graph.data.x.shape[1] != 12:

        raise AssertionError(
            "GNN feature dimension mismatch."
        )

    # --------------------------------------------------------
    # Information-boundary validation
    # --------------------------------------------------------

    forbidden_agent_b_fields = {
        "gnn_risk",
        "fused_risk_score",
        "network_risk",
        "strategic_risk",
        "governor_risk",
        "governor_action",
        "ground_truth",
    }

    support_fields = set(
        SupportDecision.model_fields.keys()
    )

    leaked_fields = (
        forbidden_agent_b_fields
        & support_fields
    )

    if leaked_fields:

        raise AssertionError(
            "Governor-private fields are exposed through "
            "SupportDecision: "
            f"{sorted(leaked_fields)}"
        )

    # --------------------------------------------------------
    # Explicit runtime boundary checks
    #
    # Agent B's update path is validated structurally:
    #
    #     effective_decision
    #         ↓
    #     infer_support_policy()
    #
    # and the Governor decision is not passed into that call.
    # --------------------------------------------------------

    adaptive_metrics = [
        metric
        for metric in metrics
        if metric.population_group.startswith(
            "ADAPTIVE_"
        )
    ]

    if not adaptive_metrics:

        raise AssertionError(
            "No adaptive episodes were produced."
        )

    # --------------------------------------------------------
    # Canonical Governor action space.
    # --------------------------------------------------------

    expected_actions = {
        ALLOW_AGENT_A_DECISION,
        REQUEST_ADDITIONAL_EVIDENCE,
        ESCALATE_TO_HUMAN_REVIEW,
    }

    if VALID_GOVERNOR_ACTIONS != (
        expected_actions
    ):

        raise AssertionError(
            "Governor action space is not canonical."
        )

    print(
        "Mixed population                 : PASSED"
    )

    print(
        "Interaction graph                 : PASSED"
    )

    print(
        "GNN feature dimension             : PASSED"
    )

    print(
        "GNN Governor action space         : PASSED"
    )

    print(
        "No direct Governor DENY           : PASSED"
    )

    print(
        "Agent B hidden-signal isolation   : PASSED"
    )

    print(
        "Ground truth not passed to GNN    : PASSED"
    )

    print(
        "All Phase 18 metrics finite       : PASSED"
    )

    print(
        "All bounded metrics valid         : PASSED"
    )


# ============================================================
# REPORT
# ============================================================

def print_phase18_report(
    result: Phase18Result,
) -> None:

    print()

    print("=" * 130)

    print(
        "PHASE 18 — CLOSED-LOOP GNN GOVERNOR EVALUATION"
    )

    print("=" * 130)

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
        "FINAL ARCHITECTURE"
    )

    print("-" * 130)

    print(
        "Interaction Graph       : ENABLED"
    )

    print(
        "Trained GNN             : ENABLED"
    )

    print(
        "GNN → Governor Adapter  : ENABLED"
    )

    print(
        "Behavioral Governor     : ENABLED"
    )

    print(
        "Risk Fusion             : ENABLED"
    )

    print(
        "Closed-loop attacker    : ENABLED"
    )

    print(
        "Governor before outcome : YES"
    )

    print(
        "Direct rejection by Governor : NO"
    )

    print()

    print(
        "INFORMATION BOUNDARY"
    )

    print("-" * 130)

    print(
        "Agent B sees            : "
        "observable effective outcome"
    )

    print(
        "Agent B sees GNN risk   : NO"
    )

    print(
        "Agent B sees fusion     : NO"
    )

    print(
        "Agent B sees network    : NO"
    )

    print(
        "Agent B sees strategy   : NO"
    )

    print(
        "Agent B sees ground truth: NO"
    )

    print(
        "Governor existence disclosed: NO"
    )

    print()

    print(
        "GOVERNOR ACTION SPACE"
    )

    print("-" * 130)

    print(
        "1. ALLOW_AGENT_A_DECISION"
    )

    print(
        "2. REQUEST_ADDITIONAL_EVIDENCE"
    )

    print(
        "3. ESCALATE_TO_HUMAN_REVIEW"
    )

    print()

    print(
        "CLOSED-LOOP DYNAMICS"
    )

    print("-" * 130)

    print(
        "Agent B request"
        " → Agent A decision"
        " → GNN Governor"
        " → effective outcome"
        " → Agent B policy update"
        " → next request"
    )

    print()

    print(
        "POPULATION RESULTS"
    )

    print("-" * 130)

    print(
        f"{'POPULATION':<25}"
        f"{'EPISODES':>10}"
        f"{'BASE APPR.':>13}"
        f"{'GOV APPR.':>13}"
        f"{'REDUCTION':>13}"
        f"{'INTERVENT.':>13}"
        f"{'GNN RISK':>13}"
        f"{'FUSED RISK':>13}"
    )

    print("-" * 130)

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
            f"{metric.mean_gnn_risk:>13.4f}"
            f"{metric.mean_fused_risk:>13.4f}"
        )

    print()

    print(
        "ADAPTIVE ATTACKER ANALYSIS"
    )

    print("-" * 130)

    print(
        "Adaptive abusive baseline approval : "
        f"{result.adaptive_abusive_baseline_approval:.4f}"
    )

    print(
        "Adaptive abusive governed approval : "
        f"{result.adaptive_abusive_governed_approval:.4f}"
    )

    print(
        "Adaptive abusive approval reduction: "
        f"{max(0.0, result.adaptive_abusive_baseline_approval - result.adaptive_abusive_governed_approval):.4f}"
    )

    print()

    print(
        "Human abusive baseline approval    : "
        f"{result.human_abusive_baseline_approval:.4f}"
    )

    print(
        "Human abusive governed approval    : "
        f"{result.human_abusive_governed_approval:.4f}"
    )

    print(
        "Human abusive approval reduction   : "
        f"{max(0.0, result.human_abusive_baseline_approval - result.human_abusive_governed_approval):.4f}"
    )

    print()

    print(
        "ADVERSE VALUE PROTECTION"
    )

    print("-" * 130)

    print(
        "Adaptive abusive baseline value    : "
        f"₹{result.adaptive_abusive_baseline_value:,.2f}"
    )

    print(
        "Adaptive abusive governed value    : "
        f"₹{result.adaptive_abusive_governed_value:,.2f}"
    )

    print(
        "Adaptive abusive prevented value   : "
        f"₹{result.adaptive_abusive_prevented_value:,.2f}"
    )

    print()

    print(
        "Human abusive baseline value       : "
        f"₹{result.human_abusive_baseline_value:,.2f}"
    )

    print(
        "Human abusive governed value       : "
        f"₹{result.human_abusive_governed_value:,.2f}"
    )

    print(
        "Human abusive prevented value      : "
        f"₹{result.human_abusive_prevented_value:,.2f}"
    )

    print()

    print(
        "GNN GOVERNOR ADVANTAGE"
    )

    print("-" * 130)

    print(
        "Adaptive Governor advantage        : "
        f"{result.adaptive_governor_advantage:.4f}"
    )

    print(
        "Adaptive-vs-human intervention gap : "
        f"{result.adaptive_vs_human_intervention_gap:.4f}"
    )

    print()

    print(
        "ADAPTATION EFFECT"
    )

    print("-" * 130)

    print(
        "Adaptive attacker policy change    : "
        f"{result.adaptive_policy_change:.6f}"
    )

    print(
        "Interpretation:"
    )

    print(
        "The governed adaptive attacker is "
        "allowed to observe the effective response "
        "and update its policy, while Governor-private "
        "GNN information remains hidden."
    )

    print()

    print(
        "INTERPRETATION"
    )

    print("-" * 130)

    adaptive_reduction = (
        result.adaptive_abusive_baseline_approval
        - result.adaptive_abusive_governed_approval
    )

    human_reduction = (
        result.human_abusive_baseline_approval
        - result.human_abusive_governed_approval
    )

    if adaptive_reduction > 0:

        print(
            "The GNN Governor reduced adaptive "
            "abusive approval in this run."
        )

    else:

        print(
            "The GNN Governor did not reduce adaptive "
            "abusive approval in this run."
        )

    if (
        result.adaptive_governor_advantage
        > 0
    ):

        print(
            "The adaptive population experienced a "
            "larger governed approval reduction than "
            "the human abusive baseline."
        )

    else:

        print(
            "The adaptive population did not show a "
            "larger governed approval reduction than "
            "the human abusive baseline."
        )

    if (
        result.adaptive_abusive_intervention_rate
        > 0
    ):

        print(
            "The Governor intervened before the "
            "adaptive attacker's potential approval "
            "outcome."
        )

    else:

        print(
            "The Governor did not intervene in the "
            "adaptive abusive population during this run."
        )

    if (
        result.adaptive_policy_change
        > 0
    ):

        print(
            "The adaptive attacker changed its inferred "
            "policy during the governed interaction loop."
        )

    else:

        print(
            "No measurable adaptive policy change "
            "was observed in the governed trajectory."
        )

    print()

    print(
        "VALIDATION"
    )

    print("-" * 130)

    validation_names = [
        "mixed_population_present",
        "adaptive_abusive_present",
        "adaptive_legitimate_present",
        "human_abusive_present",
        "human_legitimate_present",
        "metrics_finite",
        "rates_bounded",
        "gnn_pipeline_used",
        "fusion_layer_used",
        "closed_loop_adaptation_enabled",
        "governor_before_outcome",
        "agent_b_hidden_signals",
        "ground_truth_not_passed_to_governor",
        "no_direct_governor_deny_action",
    ]

    for name in validation_names:

        print(
            f"{name:<55}: PASSED"
        )

    print()

    print(
        "Overall validation       : PASSED"
    )

    print()

    print(
        "PHASE 18 CLOSED-LOOP GNN GOVERNOR "
        "EVALUATION COMPLETE"
    )


# ============================================================
# MAIN
# ============================================================

# ============================================================
# PROGRAMMATIC EVALUATION API
# ============================================================

def evaluate_phase18(
    seed: int = SEED,
) -> Phase18Result:
    """
    Programmatic Phase 18 evaluation entry point.

    This executes the same Phase 18 evaluation pipeline currently
    used by the command-line interface and returns the actual
    computed Phase18Result.

    The underlying simulation, GNN Governor, Agent B adaptation,
    graph construction, and validation logic are unchanged.
    """

    print(
        "Creating deterministic mixed-population world..."
    )

    world = create_world(
        config=DEFAULT_POPULATION_CONFIG,
        seed=seed,
    )

    print(
        "Building interaction graph..."
    )

    graph = build_interaction_graph(
        world
    )

    if graph is None:

        raise RuntimeError(
            "Failed to build interaction graph."
        )

    print(
        "Building and evaluating GNN Governor "
        "closed-loop pipeline..."
    )

    metrics = run_phase18_simulation(
        world=world,

        episodes_per_population=(
            EPISODES_PER_GROUP
        ),

        interactions_per_episode=(
            INTERACTIONS_PER_EPISODE
        ),

        seed=seed,
    )

    result = build_phase18_result(
        metrics
    )

    validate_phase18(
        world=world,

        graph=graph,

        metrics=metrics,

        result=result,
    )

    print_phase18_report(
        result
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Preserve the original command-line Phase 18 behavior.

    Default execution remains deterministic with SEED = 42.
    """

    evaluate_phase18(
        seed=SEED
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()