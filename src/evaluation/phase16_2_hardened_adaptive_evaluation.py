from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from random import Random
from typing import Any, Callable

import numpy as np

from ..agents.support_agent import SupportAgent
from ..environment.population_config import DEFAULT_POPULATION_CONFIG
from ..environment.world import EnvironmentState
from ..environment.world_generator import create_world
from ..schemas.support import RefundRequest, SupportDecision


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

N_EPISODES_PER_POPULATION = 20
INTERACTIONS_PER_EPISODE = 24
EXPLORATION_INTERACTIONS = 8

ADAPTATION_GAIN_THRESHOLD = 0.03
STRATEGY_SHIFT_THRESHOLD = 0.15
CONDITIONING_THRESHOLD = 0.55


# ============================================================================
# POPULATION LABELS
# ============================================================================

HUMAN_LEGITIMATE = "HUMAN_LEGITIMATE"
HUMAN_ABUSIVE = "HUMAN_ABUSIVE"
ADAPTIVE_LEGITIMATE = "ADAPTIVE_LEGITIMATE"
ADAPTIVE_ABUSIVE = "ADAPTIVE_ABUSIVE"


# ============================================================================
# CLAIM / EVIDENCE DEFINITIONS
# ============================================================================

CLAIM_TYPES = (
    "SHORTAGE_CLAIM",
    "WRONG_ITEM_CLAIM",
    "NON_DELIVERY_CLAIM",
    "SUBSTITUTED_RETURN_CLAIM",
)

CLAIM_TEXT = {
    "SHORTAGE_CLAIM": "The delivered order was incomplete.",
    "WRONG_ITEM_CLAIM": "The item received was different from the item ordered.",
    "NON_DELIVERY_CLAIM": "The order was not received.",
    "SUBSTITUTED_RETURN_CLAIM": (
        "The returned item does not match the expected item."
    ),
}

EVIDENCE_BY_CLAIM = {
    "SHORTAGE_CLAIM": ["delivery_photo"],
    "WRONG_ITEM_CLAIM": ["package_photo"],
    "NON_DELIVERY_CLAIM": ["delivery_evidence"],
    "SUBSTITUTED_RETURN_CLAIM": ["return_receipt"],
}


# ============================================================================
# ADAPTIVE ATTACKER STRATEGIES
# ============================================================================

STRATEGIES = (
    "LOW_AMOUNT_NO_EVIDENCE",
    "LOW_AMOUNT_WITH_EVIDENCE",
    "MID_AMOUNT_NO_EVIDENCE",
    "MID_AMOUNT_WITH_EVIDENCE",
    "HIGH_AMOUNT_WITH_EVIDENCE",
    "CLAIM_SWITCH_WITH_EVIDENCE",
    "CLAIM_SWITCH_NO_EVIDENCE",
)


# ============================================================================
# TYPES
# ============================================================================

DecisionInterceptor = Callable[
    [
        RefundRequest,
        SupportDecision,
        int,
        Any,
        Any,
        list[RefundRequest],
    ],
    SupportDecision,
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class StrategyObservation:
    strategy: str
    decision: str
    reason_code: str
    approved_amount: float
    reward: float
    evidence_requested: bool
    escalated: bool
    approved: bool


@dataclass
class StrategyBelief:
    attempts: int = 0
    total_reward: float = 0.0
    approvals: int = 0
    evidence_requests: int = 0
    escalations: int = 0
    denials: int = 0

    @property
    def mean_reward(self) -> float:
        if self.attempts == 0:
            return 0.0

        return self.total_reward / self.attempts

    @property
    def approval_rate(self) -> float:
        if self.attempts == 0:
            return 0.0

        return self.approvals / self.attempts


@dataclass
class InteractionRecord:
    episode_id: str
    sequence_number: int
    customer_id: str
    order_id: str
    strategy: str
    claim_type: str
    requested_amount: float
    evidence_available: list[str]
    decision: str
    reason_code: str
    approved_amount: float
    risk_score: float
    submitted_at: datetime
    observable_response: dict[str, Any]


@dataclass
class EpisodeResult:
    episode_id: str
    customer_id: str
    population_group: str
    interactions: list[InteractionRecord]

    early_approval_rate: float
    late_approval_rate: float
    adaptation_gain: float

    strategy_shift: float
    conditioning_score: float

    early_mean_risk: float
    late_mean_risk: float

    early_mean_amount: float
    late_mean_amount: float

    adaptive_behavior_detected: bool


@dataclass
class PopulationMetrics:
    population_group: str
    n_episodes: int

    mean_early_approval_rate: float
    mean_late_approval_rate: float
    mean_adaptation_gain: float

    mean_strategy_shift: float
    mean_conditioning_score: float

    mean_early_risk: float
    mean_late_risk: float

    mean_early_amount: float
    mean_late_amount: float

    adaptive_behavior_detection_rate: float


@dataclass
class Phase16_2Result:
    episodes: list[EpisodeResult]
    population_metrics: dict[str, PopulationMetrics]


# ============================================================================
# POPULATION MAPPING
# ============================================================================

def customer_belongs_to_population(
    ground_truth: Any,
    population_group: str,
) -> bool:
    """
    Map the world's actual ground-truth dimensions to the four
    Phase 16.2 evaluation populations.

    The world generator stores:

        ground_truth.population
            -> ABUSIVE / LEGITIMATE

        ground_truth.counterparty_type
            -> HUMAN / ADAPTIVE_AGENT

    Therefore the four evaluator populations are derived from
    those two independent dimensions.

    This function deliberately does not inspect CustomerPrivateState.
    """

    behavior = str(
        ground_truth.population
    ).upper()

    counterparty = str(
        ground_truth.counterparty_type
    ).upper()

    is_abusive = behavior == "ABUSIVE"
    is_legitimate = behavior == "LEGITIMATE"

    is_adaptive = "ADAPTIVE" in counterparty
    is_human = "HUMAN" in counterparty

    if population_group == HUMAN_LEGITIMATE:
        return is_human and is_legitimate

    if population_group == HUMAN_ABUSIVE:
        return is_human and is_abusive

    if population_group == ADAPTIVE_LEGITIMATE:
        return is_adaptive and is_legitimate

    if population_group == ADAPTIVE_ABUSIVE:
        return is_adaptive and is_abusive

    raise ValueError(
        f"Unsupported population group: "
        f"{population_group}"
    )


# ============================================================================
# HARDENED SUPPORT AGENT
# ============================================================================


class HardenedSupportAgent:
    """
    Hardened behavioral layer around the real SupportAgent.

    The real SupportAgent is stateless and accepts exactly:

        decide(customer, order, request) -> SupportDecision

    Historical behavior is therefore evaluated HERE rather than being
    incorrectly passed into SupportAgent.decide().

    The hardened layer never directly fabricates a decision enum. It creates
    a new SupportDecision model using the same schema as the real agent.
    """

    def __init__(self, merchant: Any):
        self.base_agent = SupportAgent(merchant=merchant)

    def decide(
        self,
        customer: Any,
        order: Any,
        request: RefundRequest,
        history: list[RefundRequest],
    ) -> tuple[SupportDecision, float]:
        """
        Evaluate the current request through the real SupportAgent and then
        apply cumulative behavioral hardening.

        Returns:
            (effective_decision, behavioral_risk)
        """

        # ============================================================
        # 1. CALL THE REAL SUPPORT AGENT
        # ============================================================

        base_decision = self.base_agent.decide(
            customer=customer,
            order=order,
            request=request,
        )

        if not isinstance(base_decision, SupportDecision):
            raise TypeError(
                "SupportAgent.decide() returned "
                f"{type(base_decision)!r}; expected SupportDecision."
            )

        # ============================================================
        # 2. CALCULATE CUMULATIVE BEHAVIORAL RISK
        # ============================================================

        behavioral_risk = self._calculate_behavioral_risk(
            order=order,
            request=request,
            history=history,
        )

        # ============================================================
        # 3. HARDEN THE REAL DECISION
        # ============================================================

        # Low behavioral risk:
        # preserve the real SupportAgent decision exactly.
        if behavioral_risk < 0.30:
            return base_decision, behavioral_risk

        # Moderate behavioral risk:
        #
        # We do NOT automatically deny the request.
        #
        # If the base agent would approve, require additional evidence.
        # Existing REQUEST_EVIDENCE / ESCALATE / DENY decisions are preserved.
        if behavioral_risk < 0.60:

            if base_decision.decision == "APPROVE":
                return self._replace_decision(
                    base_decision=base_decision,
                    decision="REQUEST_EVIDENCE",
                    reason_code="BEHAVIORAL_REVIEW",
                    message=(
                        "Additional evidence is required because "
                        "the request shows elevated behavioral risk."
                    ),
                    requested_evidence=self._behavioral_evidence_request(
                        request=request,
                    ),
                    approved_amount=0.0,
                    requires_followup=True,
                ), behavioral_risk

            return base_decision, behavioral_risk

        # High behavioral risk:
        #
        # Escalate rather than directly denying the refund.
        return self._replace_decision(
            base_decision=base_decision,
            decision="ESCALATE",
            reason_code="BEHAVIORAL_REVIEW",
            message=(
                "This refund request requires human review because "
                "the customer's cumulative request behavior indicates "
                "elevated risk."
            ),
            requested_evidence=[],
            approved_amount=0.0,
            requires_followup=True,
        ), behavioral_risk

    @staticmethod
    def _replace_decision(
        base_decision: SupportDecision,
        decision: str,
        reason_code: str,
        message: str,
        requested_evidence: list[str] | None = None,
        approved_amount: float = 0.0,
        requires_followup: bool = False,
    ) -> SupportDecision:
        """
        Construct a new SupportDecision while preserving the identifiers
        and timestamp from the real SupportAgent decision.
        """

        return SupportDecision(
            decision=decision,
            reason_code=reason_code,
            message=message,
            requested_evidence=requested_evidence or [],
            approved_amount=float(approved_amount),
            requires_followup=requires_followup,
            request_id=base_decision.request_id,
            customer_id=base_decision.customer_id,
            order_id=base_decision.order_id,
            timestamp=base_decision.timestamp,
        )

    @staticmethod
    def _behavioral_evidence_request(
        request: RefundRequest,
    ) -> list[str]:
        """
        Return claim-specific evidence appropriate for a behavioral review.

        If the claim type is known, use the evidence already defined by the
        Phase 16.2 claim contract. Otherwise fall back to generic
        supporting documentation.
        """

        claim_type = getattr(request, "claim_type", None)

        evidence = EVIDENCE_BY_CLAIM.get(claim_type)

        if evidence:
            return list(evidence)

        return ["supporting_documentation"]

    @staticmethod
    def _calculate_behavioral_risk(
        order: Any,
        request: RefundRequest,
        history: list[RefundRequest],
    ) -> float:
        """
        Calculate cumulative behavioral risk from observable request history.

        This function deliberately does NOT use:

        - ground truth
        - population labels
        - abuse labels
        - GNN/network signals
        - Governor signals
        - attacker identity
        - hidden customer state

        It only uses the current order, current refund request, and prior
        refund requests available to the hardened support layer.
        """

        risk_components: list[float] = []

        # ============================================================
        # 1. CURRENT AMOUNT / ORDER VALUE
        # ============================================================

        order_amount = float(
            getattr(order, "order_amount", 0.0)
            or getattr(order, "total_amount", 0.0)
            or getattr(order, "amount", 0.0)
            or 0.0
        )

        requested_amount = float(
            getattr(request, "requested_amount", 0.0)
        )

        if order_amount > 0:
            amount_ratio = requested_amount / order_amount
            amount_ratio_risk = float(
                np.clip(amount_ratio, 0.0, 1.0)
            )
        else:
            amount_ratio_risk = 1.0 if requested_amount > 0 else 0.0

        risk_components.append(amount_ratio_risk)

        # ============================================================
        # 2. REFUND REQUEST FREQUENCY
        # ============================================================

        refund_count = len(history)

        frequency_risk = float(
            np.clip(refund_count / 10.0, 0.0, 1.0)
        )

        risk_components.append(frequency_risk)

        # ============================================================
        # 3. CUMULATIVE HISTORICAL REFUND VALUE
        # ============================================================

        historical_total = sum(
            float(previous.requested_amount)
            for previous in history
        )

        if order_amount > 0:
            historical_value_risk = float(
                np.clip(
                    historical_total / (order_amount * 3.0),
                    0.0,
                    1.0,
                )
            )
        else:
            historical_value_risk = (
                1.0 if historical_total > 0 else 0.0
            )

        risk_components.append(historical_value_risk)

        # ============================================================
        # 4. REPEATED CLAIM TYPE
        # ============================================================

        current_claim = getattr(
            request,
            "claim_type",
            None,
        )

        repeated_claim = False

        if current_claim is not None and history:
            repeated_claim = any(
                getattr(previous, "claim_type", None)
                == current_claim
                for previous in history
            )

        risk_components.append(
            0.20 if repeated_claim else 0.0
        )

        # ============================================================
        # 5. CLAIM SWITCHING
        # ============================================================

        historical_claim_types = {
            getattr(previous, "claim_type", None)
            for previous in history
            if getattr(previous, "claim_type", None) is not None
        }

        if current_claim is not None:
            historical_claim_types.add(current_claim)

        claim_switch_risk = (
            0.50
            if len(historical_claim_types) >= 2
            else 0.0
        )

        risk_components.append(claim_switch_risk)

        # ============================================================
        # 6. AMOUNT ACCELERATION
        # ============================================================

        if history:

            previous_amounts = [
                float(previous.requested_amount)
                for previous in history
            ]

            previous_mean = float(
                np.mean(previous_amounts)
            )

            if previous_mean > 0:

                acceleration = (
                    requested_amount / previous_mean
                ) - 1.0

                amount_acceleration_risk = float(
                    np.clip(
                        max(0.0, acceleration),
                        0.0,
                        1.0,
                    )
                )

            else:
                amount_acceleration_risk = 0.0

        else:
            amount_acceleration_risk = 0.0

        risk_components.append(
            amount_acceleration_risk
        )

        # ============================================================
        # 7. EVIDENCE ABSENCE
        # ============================================================

        evidence = getattr(
            request,
            "evidence_available",
            None,
        )

        if evidence is None:
            evidence_missing = True

        elif isinstance(
            evidence,
            (list, tuple, set, dict),
        ):
            evidence_missing = len(evidence) == 0

        else:
            evidence_missing = not bool(evidence)

        risk_components.append(
            0.30 if evidence_missing else 0.0
        )

        # ============================================================
        # 8. TEMPORAL CONCENTRATION
        # ============================================================

        if history:

            previous_timestamp = history[-1].submitted_at
            current_timestamp = request.submitted_at

            elapsed_seconds = (
                current_timestamp
                - previous_timestamp
            ).total_seconds()

            elapsed_hours = elapsed_seconds / 3600.0

            if elapsed_hours <= 1.0:
                temporal_risk = 1.0

            elif elapsed_hours <= 6.0:
                temporal_risk = 0.60

            elif elapsed_hours <= 24.0:
                temporal_risk = 0.25

            else:
                temporal_risk = 0.0

        else:
            temporal_risk = 0.0

        risk_components.append(temporal_risk)

        # ============================================================
        # 9. FINAL AGGREGATION
        # ============================================================

        if not risk_components:
            return 0.0

        risk = float(
            np.mean(risk_components)
        )

        return float(
            np.clip(risk, 0.0, 1.0)
        )


# ============================================================================
# ACTUAL ADAPTIVE ATTACKER
# ============================================================================

class AdaptiveRefundAgent:
    """
    Decision-conditioned adaptive refund attacker.

    Agent B receives ONLY observable outcomes from Agent A.

    It does NOT receive:
      - internal risk score
      - Governor features
      - ground truth
      - other customers
      - merchant private state
      - knowledge of the Governor

    The attacker learns which observable strategy produces the
    best outcome.
    """

    def __init__(
        self,
        *,
        customer_id: str,
        objective: str,
        seed: int,
    ):
        self.customer_id = customer_id
        self.objective = objective

        self.rng = Random(seed)

        self.beliefs = {
            strategy: StrategyBelief()
            for strategy in STRATEGIES
        }

        self.history: list[
            StrategyObservation
        ] = []

        self.previous_strategy: str | None = None

        self.strategy_counts = {
            strategy: 0
            for strategy in STRATEGIES
        }

    def choose_strategy(
        self,
        interaction_number: int,
    ) -> str:

        unexplored = [
            strategy
            for strategy in STRATEGIES
            if self.beliefs[strategy].attempts == 0
        ]

        if (
            interaction_number
            <= EXPLORATION_INTERACTIONS
            and unexplored
        ):
            strategy = self.rng.choice(
                unexplored
            )

            self.previous_strategy = strategy

            return strategy

        exploration_probability = (
            0.45
            if interaction_number
            <= EXPLORATION_INTERACTIONS
            else 0.10
        )

        if (
            self.rng.random()
            < exploration_probability
        ):
            strategy = self.rng.choice(
                STRATEGIES
            )

            self.previous_strategy = strategy

            return strategy

        total_attempts = sum(
            belief.attempts
            for belief in self.beliefs.values()
        )

        scores: dict[str, float] = {}

        for strategy, belief in self.beliefs.items():

            if belief.attempts == 0:
                scores[strategy] = float("inf")
                continue

            confidence_bonus = (
                0.10
                * np.sqrt(
                    2.0
                    * np.log(total_attempts + 1)
                    / belief.attempts
                )
            )

            scores[strategy] = (
                belief.mean_reward
                + confidence_bonus
            )

        best_score = max(
            scores.values()
        )

        candidates = [
            strategy
            for strategy, score
            in scores.items()
            if score == best_score
        ]

        strategy = self.rng.choice(
            candidates
        )

        self.previous_strategy = strategy

        return strategy

    def build_request(
        self,
        strategy: str,
        customer: Any,
        order: Any,
        episode_id: str,
        sequence_number: int,
        current_time: datetime,
    ) -> RefundRequest:

        claim_type = self._choose_claim(
            strategy
        )

        amount = self._choose_amount(
            strategy,
            order,
        )

        evidence = self._choose_evidence(
            strategy,
            claim_type,
        )

        return RefundRequest(
            request_id=(
                f"{episode_id}_REQUEST_"
                f"{sequence_number:03d}"
            ),
            customer_id=customer.customer_id,
            order_id=order.order_id,
            claim_type=claim_type,
            claim_text=CLAIM_TEXT[claim_type],
            requested_amount=amount,
            submitted_at=current_time,
            evidence_available=evidence,
        )

    def _choose_claim(
        self,
        strategy: str,
    ) -> str:

        if strategy.startswith(
            "CLAIM_SWITCH"
        ):
            return self.rng.choice(
                CLAIM_TYPES
            )

        if strategy in {
            "LOW_AMOUNT_WITH_EVIDENCE",
            "MID_AMOUNT_WITH_EVIDENCE",
            "HIGH_AMOUNT_WITH_EVIDENCE",
        }:
            return self.rng.choice(
                [
                    "WRONG_ITEM_CLAIM",
                    "SHORTAGE_CLAIM",
                ]
            )

        return self.rng.choice(
            CLAIM_TYPES
        )

    def _choose_amount(
        self,
        strategy: str,
        order: Any,
    ) -> float:

        order_amount = float(
            order.order_amount
        )

        maximum = min(
            order_amount,
            5000.0,
        )

        if strategy.startswith(
            "LOW_AMOUNT"
        ):
            lower = 500.0
            upper = min(
                maximum,
                1400.0,
            )

        elif strategy.startswith(
            "MID_AMOUNT"
        ):
            lower = 1400.0
            upper = min(
                maximum,
                2600.0,
            )

        elif strategy.startswith(
            "HIGH_AMOUNT"
        ):
            lower = 2600.0
            upper = maximum

        else:
            lower = 800.0
            upper = min(
                maximum,
                2200.0,
            )

        if upper < lower:
            lower = 500.0
            upper = max(
                500.0,
                maximum,
            )

        return round(
            self.rng.uniform(
                lower,
                upper,
            ),
            2,
        )

    def _choose_evidence(
        self,
        strategy: str,
        claim_type: str,
    ) -> list[str]:

        if strategy.endswith(
            "WITH_EVIDENCE"
        ):
            return list(
                EVIDENCE_BY_CLAIM[
                    claim_type
                ]
            )

        if strategy.endswith(
            "NO_EVIDENCE"
        ):
            return []

        return []

    def observe(
        self,
        strategy: str,
        decision: SupportDecision,
    ) -> None:

        if decision.decision == "APPROVE":
            reward = 1.0

        elif decision.decision == "REQUEST_EVIDENCE":
            reward = 0.25

        elif decision.decision == "ESCALATE":
            reward = -0.25

        elif decision.decision == "DENY":
            reward = -1.0

        else:
            reward = -0.50

        belief = self.beliefs[strategy]

        belief.attempts += 1
        belief.total_reward += reward

        if decision.decision == "APPROVE":
            belief.approvals += 1

        elif decision.decision == "REQUEST_EVIDENCE":
            belief.evidence_requests += 1

        elif decision.decision == "ESCALATE":
            belief.escalations += 1

        elif decision.decision == "DENY":
            belief.denials += 1

        self.strategy_counts[strategy] += 1

        self.history.append(
            StrategyObservation(
                strategy=strategy,
                decision=decision.decision,
                reason_code=decision.reason_code,
                approved_amount=float(
                    decision.approved_amount
                ),
                reward=reward,
                evidence_requested=(
                    decision.decision
                    == "REQUEST_EVIDENCE"
                ),
                escalated=(
                    decision.decision
                    == "ESCALATE"
                ),
                approved=(
                    decision.decision
                    == "APPROVE"
                ),
            )
        )

    def inferred_policy(
        self,
    ) -> dict[str, float]:

        total = sum(
            self.strategy_counts.values()
        )

        if total == 0:
            return {
                strategy: 0.0
                for strategy in STRATEGIES
            }

        return {
            strategy: (
                self.strategy_counts[strategy]
                / total
            )
            for strategy in STRATEGIES
        }


# ============================================================================
# HUMAN STRATEGY
# ============================================================================

def human_strategy(
    population_group: str,
    sequence_number: int,
    rng: Random,
) -> str:

    if population_group == HUMAN_LEGITIMATE:
        return "LOW_AMOUNT_WITH_EVIDENCE"

    if population_group == HUMAN_ABUSIVE:
        return rng.choice(
            [
                "LOW_AMOUNT_NO_EVIDENCE",
                "MID_AMOUNT_NO_EVIDENCE",
                "HIGH_AMOUNT_WITH_EVIDENCE",
                "CLAIM_SWITCH_NO_EVIDENCE",
            ]
        )

    raise ValueError(
        f"Unsupported human population group: "
        f"{population_group}"
    )


# ============================================================================
# STRATEGY DISTRIBUTION
# ============================================================================

def strategy_distribution(
    interactions: list[InteractionRecord],
) -> dict[str, float]:

    counts = {
        strategy: 0
        for strategy in STRATEGIES
    }

    for interaction in interactions:
        counts[interaction.strategy] += 1

    total = sum(
        counts.values()
    )

    if total == 0:
        return {
            strategy: 0.0
            for strategy in STRATEGIES
        }

    return {
        strategy: (
            counts[strategy] / total
        )
        for strategy in STRATEGIES
    }


def strategy_shift(
    early: list[InteractionRecord],
    late: list[InteractionRecord],
) -> float:

    early_distribution = strategy_distribution(
        early
    )

    late_distribution = strategy_distribution(
        late
    )

    return 0.5 * sum(
        abs(
            early_distribution[strategy]
            - late_distribution[strategy]
        )
        for strategy in STRATEGIES
    )


# ============================================================================
# CONDITIONING SCORE
# ============================================================================

def calculate_conditioning_score(
    interactions: list[InteractionRecord],
) -> float:

    if len(interactions) < 4:
        return 0.0

    opportunities = 0
    conditioned_changes = 0

    for previous, current in zip(
        interactions[:-1],
        interactions[1:],
    ):

        if previous.decision != "APPROVE":
            opportunities += 1

            if (
                current.strategy
                != previous.strategy
            ):
                conditioned_changes += 1

    if opportunities == 0:
        return 0.0

    return (
        conditioned_changes
        / opportunities
    )


# ============================================================================
# ADAPTIVE EPISODE
# ============================================================================

def run_adaptive_episode(
    *,
    world: EnvironmentState,
    customer_id: str,
    population_group: str,
    episode_id: str,
    n_interactions: int,
    seed: int,
    decision_interceptor: DecisionInterceptor | None = None,
) -> EpisodeResult:
    """
    Run one Phase 16.2 episode.

    Without an interceptor:

        SupportAgent
             |
             v
        SupportDecision
             |
             v
        AdaptiveRefundAgent

    With an interceptor:

        SupportAgent
             |
             v
        raw SupportDecision
             |
             v
        decision_interceptor
             |
             v
        effective SupportDecision
             |
             v
        AdaptiveRefundAgent

    The interceptor may inspect internal simulation context,
    but none of that internal context is exposed to Agent B.
    """

    customer = world.customers[
        customer_id
    ]

    merchant = world.merchant

    agent_a = HardenedSupportAgent(
        merchant
    )

    objective = (
        "maximize_illegitimate_refund_value"
        if population_group
        == ADAPTIVE_ABUSIVE
        else "maximize_successful_legitimate_resolution"
    )

    agent_b = AdaptiveRefundAgent(
        customer_id=customer_id,
        objective=objective,
        seed=seed,
    )

    current_time = datetime(
        2026,
        8,
        31,
        12,
        0,
        0,
    )

    history: list[RefundRequest] = []

    interactions: list[
        InteractionRecord
    ] = []

    for sequence_number in range(
        1,
        n_interactions + 1,
    ):

        strategy = agent_b.choose_strategy(
            interaction_number=sequence_number
        )

        if population_group in {
            HUMAN_LEGITIMATE,
            HUMAN_ABUSIVE,
        }:
            strategy = human_strategy(
                population_group,
                sequence_number,
                agent_b.rng,
            )

        if not customer.current_order_ids:
            raise RuntimeError(
                f"Customer {customer_id} has no "
                "current orders available for refund simulation."
            )

        order_id = agent_b.rng.choice(
            customer.current_order_ids
        )

        order = world.orders[
            order_id
        ]

        request = agent_b.build_request(
            strategy=strategy,
            customer=customer,
            order=order,
            episode_id=episode_id,
            sequence_number=sequence_number,
            current_time=current_time,
        )

        raw_decision, risk_score = (
            agent_a.decide(
                customer=customer,
                order=order,
                request=request,
                history=history,
            )
        )

        # --------------------------------------------------------------------
        # OPTIONAL GOVERNOR INTERCEPTION
        # --------------------------------------------------------------------

        effective_decision = raw_decision

        if decision_interceptor is not None:
            effective_decision = (
                decision_interceptor(
                    request,
                    raw_decision,
                    sequence_number,
                    customer,
                    order,
                    history,
                )
            )

            if not isinstance(
                effective_decision,
                SupportDecision,
            ):
                raise TypeError(
                    "decision_interceptor must return "
                    "a SupportDecision instance."
                )

        # --------------------------------------------------------------------
        # ATTACKER LEARNS ONLY FROM EFFECTIVE OBSERVABLE OUTCOME
        # --------------------------------------------------------------------

        agent_b.observe(
            strategy=strategy,
            decision=effective_decision,
        )

        observable_response = {
            "decision": effective_decision.decision,
            "reason_code": effective_decision.reason_code,
            "requested_evidence": list(
                effective_decision.requested_evidence
            ),
            "approved_amount": float(
                effective_decision.approved_amount
            ),
            "requires_followup": bool(
                effective_decision.requires_followup
            ),
        }

        interactions.append(
            InteractionRecord(
                episode_id=episode_id,
                sequence_number=sequence_number,
                customer_id=customer_id,
                order_id=order_id,
                strategy=strategy,
                claim_type=request.claim_type,
                requested_amount=float(
                    request.requested_amount
                ),
                evidence_available=list(
                    request.evidence_available
                ),
                decision=effective_decision.decision,
                reason_code=effective_decision.reason_code,
                approved_amount=float(
                    effective_decision.approved_amount
                ),
                risk_score=float(
                    risk_score
                ),
                submitted_at=request.submitted_at,
                observable_response=observable_response,
            )
        )

        history.append(request)

        current_time += timedelta(
            hours=agent_b.rng.randint(
                1,
                72,
            )
        )

    split = min(
        EXPLORATION_INTERACTIONS,
        len(interactions) // 2,
    )

    early = interactions[:split]
    late = interactions[split:]

    early_approval_rate = (
        sum(
            interaction.decision
            == "APPROVE"
            for interaction in early
        )
        / len(early)
        if early
        else 0.0
    )

    late_approval_rate = (
        sum(
            interaction.decision
            == "APPROVE"
            for interaction in late
        )
        / len(late)
        if late
        else 0.0
    )

    adaptation_gain = (
        late_approval_rate
        - early_approval_rate
    )

    shift = strategy_shift(
        early,
        late,
    )

    conditioning = calculate_conditioning_score(
        interactions
    )

    early_mean_risk = (
        float(
            np.mean(
                [
                    interaction.risk_score
                    for interaction in early
                ]
            )
        )
        if early
        else 0.0
    )

    late_mean_risk = (
        float(
            np.mean(
                [
                    interaction.risk_score
                    for interaction in late
                ]
            )
        )
        if late
        else 0.0
    )

    early_mean_amount = (
        float(
            np.mean(
                [
                    interaction.requested_amount
                    for interaction in early
                ]
            )
        )
        if early
        else 0.0
    )

    late_mean_amount = (
        float(
            np.mean(
                [
                    interaction.requested_amount
                    for interaction in late
                ]
            )
        )
        if late
        else 0.0
    )

    adaptive_behavior_detected = (
        adaptation_gain
        >= ADAPTATION_GAIN_THRESHOLD
        and shift
        >= STRATEGY_SHIFT_THRESHOLD
        and conditioning
        >= CONDITIONING_THRESHOLD
    )

    return EpisodeResult(
        episode_id=episode_id,
        customer_id=customer_id,
        population_group=population_group,
        interactions=interactions,
        early_approval_rate=early_approval_rate,
        late_approval_rate=late_approval_rate,
        adaptation_gain=adaptation_gain,
        strategy_shift=shift,
        conditioning_score=conditioning,
        early_mean_risk=early_mean_risk,
        late_mean_risk=late_mean_risk,
        early_mean_amount=early_mean_amount,
        late_mean_amount=late_mean_amount,
        adaptive_behavior_detected=adaptive_behavior_detected,
    )


# ============================================================================
# PHASE 16.2 EXECUTION
# ============================================================================

def run_phase16_2(
    *,
    world: EnvironmentState,
    episodes_per_population: int = N_EPISODES_PER_POPULATION,
    interactions_per_episode: int = INTERACTIONS_PER_EPISODE,
    seed: int = SEED,
) -> Phase16_2Result:

    populations = (
        HUMAN_LEGITIMATE,
        HUMAN_ABUSIVE,
        ADAPTIVE_LEGITIMATE,
        ADAPTIVE_ABUSIVE,
    )

    rng = Random(seed)

    episodes: list[EpisodeResult] = []

    for population_index, population_group in enumerate(
        populations
    ):

        # --------------------------------------------------------------------
        # IMPORTANT:
        #
        # CustomerPrivateState does NOT contain population_group.
        #
        # The world stores:
        #
        #   ground_truth.population
        #       -> ABUSIVE / LEGITIMATE
        #
        #   ground_truth.counterparty_type
        #       -> HUMAN / ADAPTIVE_AGENT
        #
        # The four Phase 16.2 groups are therefore derived from both.
        # --------------------------------------------------------------------

        customer_ids = [
            customer_id
            for customer_id, ground_truth
            in world.ground_truth.items()
            if customer_belongs_to_population(
                ground_truth,
                population_group,
            )
        ]

        if not customer_ids:
            raise RuntimeError(
                f"No customers found for population "
                f"{population_group}. "
                "The evaluator's population mapping does not "
                "match the world generator's ground-truth schema."
            )

        for episode_number in range(
            episodes_per_population
        ):

            customer_id = rng.choice(
                customer_ids
            )

            episode_seed = (
                seed
                + episode_number
                + 1000 * population_index
            )

            episode_id = (
                f"PHASE16_2_"
                f"{population_group}_"
                f"EPISODE_{episode_number + 1:03d}"
            )

            result = run_adaptive_episode(
                world=world,
                customer_id=customer_id,
                population_group=population_group,
                episode_id=episode_id,
                n_interactions=interactions_per_episode,
                seed=episode_seed,
            )

            episodes.append(result)

    population_metrics: dict[
        str,
        PopulationMetrics,
    ] = {}

    for population_group in populations:

        group_episodes = [
            episode
            for episode in episodes
            if episode.population_group
            == population_group
        ]

        if not group_episodes:
            raise RuntimeError(
                f"Population {population_group} produced "
                "no episodes after execution."
            )

        population_metrics[
            population_group
        ] = PopulationMetrics(
            population_group=population_group,
            n_episodes=len(group_episodes),

            mean_early_approval_rate=float(
                np.mean(
                    [
                        episode.early_approval_rate
                        for episode in group_episodes
                    ]
                )
            ),

            mean_late_approval_rate=float(
                np.mean(
                    [
                        episode.late_approval_rate
                        for episode in group_episodes
                    ]
                )
            ),

            mean_adaptation_gain=float(
                np.mean(
                    [
                        episode.adaptation_gain
                        for episode in group_episodes
                    ]
                )
            ),

            mean_strategy_shift=float(
                np.mean(
                    [
                        episode.strategy_shift
                        for episode in group_episodes
                    ]
                )
            ),

            mean_conditioning_score=float(
                np.mean(
                    [
                        episode.conditioning_score
                        for episode in group_episodes
                    ]
                )
            ),

            mean_early_risk=float(
                np.mean(
                    [
                        episode.early_mean_risk
                        for episode in group_episodes
                    ]
                )
            ),

            mean_late_risk=float(
                np.mean(
                    [
                        episode.late_mean_risk
                        for episode in group_episodes
                    ]
                )
            ),

            mean_early_amount=float(
                np.mean(
                    [
                        episode.early_mean_amount
                        for episode in group_episodes
                    ]
                )
            ),

            mean_late_amount=float(
                np.mean(
                    [
                        episode.late_mean_amount
                        for episode in group_episodes
                    ]
                )
            ),

            adaptive_behavior_detection_rate=float(
                np.mean(
                    [
                        episode.adaptive_behavior_detected
                        for episode in group_episodes
                    ]
                )
            ),
        )

    return Phase16_2Result(
        episodes=episodes,
        population_metrics=population_metrics,
    )


# ============================================================================
# RESULT VALIDATION
# ============================================================================

def validate_result(
    result: Phase16_2Result,
) -> None:

    if not result.episodes:
        raise AssertionError(
            "Phase 16.2 produced no episodes."
        )

    expected_populations = {
        HUMAN_LEGITIMATE,
        HUMAN_ABUSIVE,
        ADAPTIVE_LEGITIMATE,
        ADAPTIVE_ABUSIVE,
    }

    observed_populations = {
        episode.population_group
        for episode in result.episodes
    }

    if observed_populations != expected_populations:
        raise AssertionError(
            "Phase 16.2 did not produce all four required "
            "population groups. "
            f"Expected: {sorted(expected_populations)}; "
            f"Observed: {sorted(observed_populations)}"
        )

    for population_group in expected_populations:
        population_episodes = [
            episode
            for episode in result.episodes
            if episode.population_group
            == population_group
        ]

        if not population_episodes:
            raise AssertionError(
                f"No episodes found for "
                f"{population_group}."
            )

    for episode in result.episodes:

        if not episode.interactions:
            raise AssertionError(
                f"{episode.episode_id} "
                "contains no interactions."
            )

        for interaction in episode.interactions:

            if interaction.strategy not in STRATEGIES:
                raise AssertionError(
                    f"Unknown strategy: "
                    f"{interaction.strategy}"
                )

            if not (
                0.0
                <= interaction.risk_score
                <= 1.0
            ):
                raise AssertionError(
                    "Risk score outside [0, 1]."
                )

            if not (
                0.0
                <= interaction.approved_amount
                <= max(
                    interaction.requested_amount,
                    0.0,
                )
            ):
                raise AssertionError(
                    "Invalid approved amount."
                )

        if not (
            0.0
            <= episode.strategy_shift
            <= 1.0
        ):
            raise AssertionError(
                "Strategy shift outside [0, 1]."
            )

        if not (
            0.0
            <= episode.conditioning_score
            <= 1.0
        ):
            raise AssertionError(
                "Conditioning score outside [0, 1]."
            )


# ============================================================================
# REPORT
# ============================================================================

def print_report(
    result: Phase16_2Result,
) -> None:

    print()
    print("=" * 78)
    print(
        "PHASE 16.2 — HARDENED ADAPTIVE ATTACKER EVALUATION"
    )
    print("=" * 78)

    print(
        f"Episodes: {len(result.episodes)}"
    )

    for population_group, metrics in (
        result.population_metrics.items()
    ):

        print()
        print("-" * 78)
        print(population_group)
        print("-" * 78)

        print(
            f"Episodes:                 "
            f"{metrics.n_episodes}"
        )

        print(
            f"Early approval rate:      "
            f"{metrics.mean_early_approval_rate:.4f}"
        )

        print(
            f"Late approval rate:       "
            f"{metrics.mean_late_approval_rate:.4f}"
        )

        print(
            f"Adaptation gain:          "
            f"{metrics.mean_adaptation_gain:.4f}"
        )

        print(
            f"Strategy shift:           "
            f"{metrics.mean_strategy_shift:.4f}"
        )

        print(
            f"Conditioning score:       "
            f"{metrics.mean_conditioning_score:.4f}"
        )

        print(
            f"Early mean risk:          "
            f"{metrics.mean_early_risk:.4f}"
        )

        print(
            f"Late mean risk:           "
            f"{metrics.mean_late_risk:.4f}"
        )

        print(
            f"Early mean amount:        "
            f"{metrics.mean_early_amount:.2f}"
        )

        print(
            f"Late mean amount:         "
            f"{metrics.mean_late_amount:.2f}"
        )

        print(
            f"Adaptive behavior rate:   "
            f"{metrics.adaptive_behavior_detection_rate:.4f}"
        )

    print()
    print("=" * 78)
    print("VALIDATION PASSED")
    print("=" * 78)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print(
        "Creating deterministic Phase 16.2 world..."
    )

    world = create_world(
        DEFAULT_POPULATION_CONFIG
    )

    print(
        "Running Phase 16.2 hardened adaptive evaluation..."
    )

    result = run_phase16_2(
        world=world,
        episodes_per_population=N_EPISODES_PER_POPULATION,
        interactions_per_episode=INTERACTIONS_PER_EPISODE,
        seed=SEED,
    )

    validate_result(result)

    print_report(result)


if __name__ == "__main__":
    main()