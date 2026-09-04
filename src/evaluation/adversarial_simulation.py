"""
Adversarial / Adaptive-Agent Environment
========================================

Steps 89 -> 95

89. Simulate adaptive customers/agents
90. Allow adaptive actors to observe support-agent responses
91. Model strategy adaptation
92. Model repeated interactions
93. Model manipulation of support-agent behavior
94. Test whether Governor can detect adaptation
95. Test network/graph-based manipulation

IMPORTANT
---------
This module creates a challenging adversarial environment.

It does NOT implement a second Governor.

The adaptive actor only sees observable support-agent outcomes.

Hidden signals such as:

    - GNN risk
    - network risk
    - strategic risk
    - fused risk
    - Governor internals

are NOT exposed to the adaptive actor.

IMPORTANT DETECTOR DESIGN
-------------------------
The adaptive actor's actual strategy is simulation-private.

Therefore the behavioral detector must NOT use:

    interaction.strategy_used

to determine whether adaptation occurred.

The detector instead uses observable consequences:

    - claim changes
    - amount changes
    - support response changes
    - evidence pressure
    - success-rate drift
    - request velocity

The strategy_switch_rate is retained as an evaluation metric so that
the simulation can prove that genuine strategy adaptation occurred,
but it is NOT used by the Governor-observable detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


# ============================================================
# CLAIM TYPES
# ============================================================


CLAIM_TYPES = (
    "item_not_received",
    "damaged_item",
    "wrong_item",
    "duplicate_charge",
    "quality_issue",
    "unauthorized_transaction",
)


# ============================================================
# STRATEGIES
# ============================================================


class Strategy(str, Enum):

    NORMAL = "normal"

    DIRECT_REFUND = "direct_refund"

    EVIDENCE_MANIPULATION = "evidence_manipulation"

    CLAIM_SWITCHING = "claim_switching"

    TIMING_MANIPULATION = "timing_manipulation"

    HIGH_VALUE_ESCALATION = "high_value_escalation"


# ============================================================
# SUPPORT RESPONSES
# ============================================================


class SupportResponse(str, Enum):

    APPROVE = "approve"

    REQUEST_EVIDENCE = "request_evidence"

    ESCALATE = "escalate"

    DENY = "deny"


# ============================================================
# INTERACTION
# ============================================================


@dataclass
class Interaction:

    interaction_number: int

    strategy_used: Strategy

    # Observable claim representation.
    claim_type: str

    support_response: SupportResponse

    evidence_requested: bool

    refund_amount: float

    requested_amount: float

    successful: bool

    support_sensitivity: float

    manipulation_pressure: float


# ============================================================
# ADAPTIVE CUSTOMER
# ============================================================


@dataclass
class AdaptiveCustomer:
    """
    Adaptive adversarial actor.

    The actor DOES NOT observe hidden Governor signals.

    It only observes:

        - support response
        - whether evidence was requested
        - whether the refund succeeded
        - refund amount
        - repeated outcomes

    The actor learns which observable presentation strategies
    are more successful.

    The actual strategy is private simulation state.
    """

    customer_id: str

    strategy: Strategy = Strategy.DIRECT_REFUND

    interactions: List[Interaction] = field(
        default_factory=list
    )

    strategy_scores: Dict[Strategy, float] = field(
        default_factory=lambda: {
            Strategy.NORMAL: 0.0,
            Strategy.DIRECT_REFUND: 0.0,
            Strategy.EVIDENCE_MANIPULATION: 0.0,
            Strategy.CLAIM_SWITCHING: 0.0,
            Strategy.TIMING_MANIPULATION: 0.0,
            Strategy.HIGH_VALUE_ESCALATION: 0.0,
        }
    )

    strategy_attempts: Dict[Strategy, int] = field(
        default_factory=dict
    )

    successful_strategies: List[Strategy] = field(
        default_factory=list
    )

    failed_strategies: List[Strategy] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # STEP 90
    # Observe support-agent response
    # --------------------------------------------------------

    def observe_support_response(
        self,
        interaction: Interaction,
    ) -> None:

        self.interactions.append(
            interaction
        )

        strategy = interaction.strategy_used

        self.strategy_attempts[strategy] = (
            self.strategy_attempts.get(
                strategy,
                0,
            )
            + 1
        )

        # ----------------------------------------------------
        # Learning signal
        # ----------------------------------------------------

        if interaction.successful:

            self.strategy_scores[strategy] += 3.0

            if strategy not in self.successful_strategies:

                self.successful_strategies.append(
                    strategy
                )

        else:

            self.strategy_scores[strategy] -= 2.0

            if strategy not in self.failed_strategies:

                self.failed_strategies.append(
                    strategy
                )

        # ----------------------------------------------------
        # Evidence request
        # ----------------------------------------------------

        if interaction.evidence_requested:

            self.strategy_scores[strategy] -= 1.0

        # ----------------------------------------------------
        # Escalation
        # ----------------------------------------------------

        if (
            interaction.support_response
            == SupportResponse.ESCALATE
        ):

            self.strategy_scores[strategy] -= 1.5

        # ----------------------------------------------------
        # Denial
        # ----------------------------------------------------

        if (
            interaction.support_response
            == SupportResponse.DENY
        ):

            self.strategy_scores[strategy] -= 2.0

    # --------------------------------------------------------
    # STEP 91
    # Strategy adaptation
    # --------------------------------------------------------

    def adapt_strategy(self) -> Strategy:
        """
        Adapt strategy using only observed outcomes.

        The actor deliberately explores alternatives when the
        current strategy becomes challenged.

        This prevents the environment from degenerating into:

            direct_refund -> direct_refund -> direct_refund

        forever.
        """

        if not self.interactions:

            return self.strategy

        last_interaction = self.interactions[-1]

        current_strategy = self.strategy

        current_score = self.strategy_scores[
            current_strategy
        ]

        current_attempts = self.strategy_attempts.get(
            current_strategy,
            0,
        )

        # ----------------------------------------------------
        # A challenged response is an explicit learning signal.
        #
        # If Agent A requests evidence, escalates, or denies,
        # the actor should test another presentation strategy.
        # ----------------------------------------------------

        challenged = (
            last_interaction.support_response
            in {
                SupportResponse.REQUEST_EVIDENCE,
                SupportResponse.ESCALATE,
                SupportResponse.DENY,
            }
        )

        # ----------------------------------------------------
        # First exploration opportunity.
        #
        # After a successful direct refund, test another
        # strategy rather than assuming the environment is
        # permanently exploitable.
        # ----------------------------------------------------

        if (
            current_strategy == Strategy.DIRECT_REFUND
            and current_attempts >= 2
        ):

            unexplored = [
                strategy
                for strategy in Strategy
                if (
                    strategy != current_strategy
                    and self.strategy_attempts.get(
                        strategy,
                        0,
                    )
                    == 0
                )
            ]

            if unexplored:

                # Evidence manipulation is a realistic next
                # attempt after direct requests are challenged.
                preferred_order = [
                    Strategy.EVIDENCE_MANIPULATION,
                    Strategy.CLAIM_SWITCHING,
                    Strategy.TIMING_MANIPULATION,
                    Strategy.HIGH_VALUE_ESCALATION,
                    Strategy.NORMAL,
                ]

                for candidate in preferred_order:

                    if candidate in unexplored:

                        self.strategy = candidate

                        return self.strategy

        # ----------------------------------------------------
        # If the current strategy is challenged, explore a
        # previously unseen strategy.
        # ----------------------------------------------------

        if challenged:

            unexplored = [
                strategy
                for strategy in Strategy
                if (
                    strategy != current_strategy
                    and self.strategy_attempts.get(
                        strategy,
                        0,
                    )
                    == 0
                )
            ]

            if unexplored:

                # Prefer a strategy materially different from
                # the current behavioral presentation.
                preferred_order = [
                    Strategy.CLAIM_SWITCHING,
                    Strategy.EVIDENCE_MANIPULATION,
                    Strategy.TIMING_MANIPULATION,
                    Strategy.HIGH_VALUE_ESCALATION,
                    Strategy.NORMAL,
                    Strategy.DIRECT_REFUND,
                ]

                for candidate in preferred_order:

                    if candidate in unexplored:

                        self.strategy = candidate

                        return self.strategy

        # ----------------------------------------------------
        # Repeated failure:
        # abandon the current strategy.
        # ----------------------------------------------------

        if current_score <= -2.0:

            candidates = [
                strategy
                for strategy in Strategy
                if strategy != current_strategy
            ]

            if candidates:

                self.strategy = max(
                    candidates,
                    key=lambda strategy:
                        self.strategy_scores[strategy],
                )

                return self.strategy

        # ----------------------------------------------------
        # After repeated success, occasionally explore a
        # different strategy to test whether the environment
        # can be exploited more effectively.
        # ----------------------------------------------------

        if (
            current_score >= 6.0
            and current_attempts >= 3
        ):

            candidates = [
                strategy
                for strategy in Strategy
                if strategy != current_strategy
            ]

            if candidates:

                unexplored = [
                    strategy
                    for strategy in candidates
                    if self.strategy_attempts.get(
                        strategy,
                        0,
                    )
                    == 0
                ]

                if unexplored:

                    self.strategy = unexplored[0]

                    return self.strategy

        # ----------------------------------------------------
        # Otherwise exploit the best-known strategy.
        # ----------------------------------------------------

        best_strategy = max(
            Strategy,
            key=lambda strategy:
                self.strategy_scores[strategy],
        )

        self.strategy = best_strategy

        return self.strategy


# ============================================================
# SUPPORT AGENT
# ============================================================


@dataclass
class SupportAgent:

    agent_id: str

    evidence_sensitivity: float = 0.35

    amount_sensitivity: float = 0.45

    escalation_threshold: float = 0.75

    manipulation_susceptibility: float = 0.65

    response_history: List[SupportResponse] = field(
        default_factory=list
    )

    suspicious_interactions: int = 0

    # --------------------------------------------------------
    # STEP 93
    # Support-agent behavioral manipulation
    # --------------------------------------------------------

    def receive_manipulation_pressure(
        self,
        strategy: Strategy,
    ) -> None:

        pressure = {
            Strategy.NORMAL: 0.00,
            Strategy.DIRECT_REFUND: 0.10,
            Strategy.EVIDENCE_MANIPULATION: 0.45,
            Strategy.CLAIM_SWITCHING: 0.60,
            Strategy.TIMING_MANIPULATION: 0.75,
            Strategy.HIGH_VALUE_ESCALATION: 0.90,
        }[strategy]

        self.manipulation_susceptibility = min(
            0.95,
            self.manipulation_susceptibility
            + pressure * 0.04,
        )

    # --------------------------------------------------------
    # Agent A response
    # --------------------------------------------------------

    def respond(
        self,
        strategy: Strategy,
        amount: float,
        interaction_number: int,
    ) -> SupportResponse:

        early_phase = (
            interaction_number <= 2
        )

        # ----------------------------------------------------
        # NORMAL
        # ----------------------------------------------------

        if strategy == Strategy.NORMAL:

            response = SupportResponse.APPROVE

        # ----------------------------------------------------
        # DIRECT REFUND
        # ----------------------------------------------------

        elif strategy == Strategy.DIRECT_REFUND:

            if early_phase:

                response = SupportResponse.APPROVE

            elif interaction_number <= 3:

                response = (
                    SupportResponse.REQUEST_EVIDENCE
                )

            else:

                response = SupportResponse.ESCALATE

        # ----------------------------------------------------
        # EVIDENCE MANIPULATION
        # ----------------------------------------------------

        elif strategy == Strategy.EVIDENCE_MANIPULATION:

            if (
                interaction_number <= 4
                and self.evidence_sensitivity < 0.70
            ):

                response = SupportResponse.APPROVE

            elif self.evidence_sensitivity < 0.85:

                response = (
                    SupportResponse.REQUEST_EVIDENCE
                )

            else:

                response = SupportResponse.ESCALATE

        # ----------------------------------------------------
        # CLAIM SWITCHING
        # ----------------------------------------------------

        elif strategy == Strategy.CLAIM_SWITCHING:

            if interaction_number <= 3:

                response = (
                    SupportResponse.REQUEST_EVIDENCE
                )

            elif interaction_number <= 5:

                response = SupportResponse.ESCALATE

            else:

                response = SupportResponse.DENY

        # ----------------------------------------------------
        # TIMING MANIPULATION
        # ----------------------------------------------------

        elif strategy == Strategy.TIMING_MANIPULATION:

            if (
                interaction_number <= 3
                and self.manipulation_susceptibility
                >= 0.60
            ):

                response = SupportResponse.APPROVE

            elif interaction_number <= 5:

                response = (
                    SupportResponse.REQUEST_EVIDENCE
                )

            else:

                response = SupportResponse.ESCALATE

        # ----------------------------------------------------
        # HIGH VALUE ESCALATION
        # ----------------------------------------------------

        elif strategy == Strategy.HIGH_VALUE_ESCALATION:

            if amount >= 5000:

                if (
                    interaction_number <= 2
                    and self.amount_sensitivity < 0.70
                ):

                    response = SupportResponse.APPROVE

                else:

                    response = SupportResponse.ESCALATE

            else:

                response = SupportResponse.APPROVE

        else:

            response = SupportResponse.ESCALATE

        # ----------------------------------------------------
        # Agent learns from suspicious behavior.
        # ----------------------------------------------------

        suspicious_strategies = {
            Strategy.EVIDENCE_MANIPULATION,
            Strategy.CLAIM_SWITCHING,
            Strategy.TIMING_MANIPULATION,
            Strategy.HIGH_VALUE_ESCALATION,
        }

        if strategy in suspicious_strategies:

            self.suspicious_interactions += 1

        # ----------------------------------------------------
        # Progressive defense.
        # ----------------------------------------------------

        if self.suspicious_interactions >= 2:

            self.evidence_sensitivity = min(
                0.95,
                self.evidence_sensitivity + 0.08,
            )

            self.amount_sensitivity = min(
                0.95,
                self.amount_sensitivity + 0.06,
            )

            self.manipulation_susceptibility = max(
                0.20,
                self.manipulation_susceptibility - 0.05,
            )

        self.response_history.append(
            response
        )

        return response


# ============================================================
# GRAPH
# ============================================================


@dataclass
class NetworkNode:

    node_id: str

    node_type: str


@dataclass
class NetworkEdge:

    source: str

    target: str

    edge_type: str


@dataclass
class InteractionGraph:

    nodes: List[NetworkNode] = field(
        default_factory=list
    )

    edges: List[NetworkEdge] = field(
        default_factory=list
    )

    def add_node(
        self,
        node_id: str,
        node_type: str,
    ) -> None:

        if not any(
            node.node_id == node_id
            for node in self.nodes
        ):

            self.nodes.append(
                NetworkNode(
                    node_id=node_id,
                    node_type=node_type,
                )
            )

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
    ) -> None:

        self.edges.append(
            NetworkEdge(
                source=source,
                target=target,
                edge_type=edge_type,
            )
        )

    def degree(
        self,
        node_id: str,
    ) -> int:

        return sum(
            1
            for edge in self.edges
            if (
                edge.source == node_id
                or edge.target == node_id
            )
        )

    def shared_identifier_count(
        self,
        node_id: str,
    ) -> int:

        identifier_types = {
            "shared_device",
            "shared_address",
            "shared_payment",
            "shared_identifier",
        }

        neighbors = set()

        for edge in self.edges:

            if edge.edge_type not in identifier_types:
                continue

            if edge.source == node_id:

                neighbors.add(
                    edge.target
                )

            elif edge.target == node_id:

                neighbors.add(
                    edge.source
                )

        return len(neighbors)


# ============================================================
# GOVERNOR-OBSERVABLE BEHAVIOR
# ============================================================


@dataclass
class GovernorObservableSignals:

    request_velocity: float

    # --------------------------------------------------------
    # Evaluation-only instrumentation.
    #
    # This is calculated from hidden simulation state.
    # It is NOT used by the behavioral detector.
    # --------------------------------------------------------

    strategy_switch_rate: float

    # --------------------------------------------------------
    # Observable behavioral signals.
    # --------------------------------------------------------

    repeated_claim_rate: float

    claim_change_rate: float

    amount_change_rate: float

    support_response_change_rate: float

    behavioral_drift: float

    network_degree_signal: float

    shared_identifier_signal: float

    behavioral_adaptation_score: float

    detected_adaptation: bool

    detected_network_manipulation: bool


# ============================================================
# CLAIM MAPPING
# ============================================================


def claim_for_strategy(
    strategy: Strategy,
    interaction_number: int,
) -> str:

    """
    Produce the observable claim presentation associated with
    the hidden strategy.

    The mapping creates observable consequences of adaptation
    without exposing the strategy itself to the detector.
    """

    if strategy == Strategy.NORMAL:

        return "quality_issue"

    if strategy == Strategy.DIRECT_REFUND:

        return "item_not_received"

    if strategy == Strategy.EVIDENCE_MANIPULATION:

        return "damaged_item"

    if strategy == Strategy.CLAIM_SWITCHING:

        claim_sequence = (
            "item_not_received",
            "damaged_item",
            "wrong_item",
            "quality_issue",
        )

        return claim_sequence[
            (interaction_number - 1)
            % len(claim_sequence)
        ]

    if strategy == Strategy.TIMING_MANIPULATION:

        return "duplicate_charge"

    if strategy == Strategy.HIGH_VALUE_ESCALATION:

        return "unauthorized_transaction"

    return "quality_issue"


# ============================================================
# GOVERNOR-OBSERVABLE SIGNAL CALCULATION
# ============================================================


def calculate_governor_observable_signals(
    customer: AdaptiveCustomer,
    graph: InteractionGraph,
) -> GovernorObservableSignals:

    interactions = customer.interactions

    if not interactions:

        return GovernorObservableSignals(
            request_velocity=0.0,
            strategy_switch_rate=0.0,
            repeated_claim_rate=0.0,
            claim_change_rate=0.0,
            amount_change_rate=0.0,
            support_response_change_rate=0.0,
            behavioral_drift=0.0,
            network_degree_signal=0.0,
            shared_identifier_signal=0.0,
            behavioral_adaptation_score=0.0,
            detected_adaptation=False,
            detected_network_manipulation=False,
        )

    # ========================================================
    # REQUEST VELOCITY
    # ========================================================

    recent = interactions[-4:]

    evidence_requests = sum(
        1
        for interaction in recent
        if interaction.evidence_requested
    )

    request_velocity = (
        evidence_requests
        / len(recent)
    )

    # ========================================================
    # STRATEGY SWITCH RATE
    #
    # IMPORTANT:
    # This is evaluation instrumentation only.
    #
    # It is intentionally NOT used by the detector.
    # ========================================================

    strategies = [
        interaction.strategy_used
        for interaction in interactions
    ]

    switches = sum(
        1
        for previous, current
        in zip(
            strategies,
            strategies[1:],
        )
        if previous != current
    )

    strategy_switch_rate = (
        switches
        / max(
            1,
            len(strategies) - 1,
        )
    )

    # ========================================================
    # REPEATED CLAIM RATE
    #
    # This is observable.
    # ========================================================

    claims = [
        interaction.claim_type
        for interaction in interactions
    ]

    repeated_claims = sum(
        1
        for previous, current
        in zip(
            claims,
            claims[1:],
        )
        if previous == current
    )

    repeated_claim_rate = (
        repeated_claims
        / max(
            1,
            len(claims) - 1,
        )
    )

    # ========================================================
    # CLAIM CHANGE RATE
    #
    # This is observable and is more useful for detecting
    # adaptive claim switching.
    # ========================================================

    claim_changes = sum(
        1
        for previous, current
        in zip(
            claims,
            claims[1:],
        )
        if previous != current
    )

    claim_change_rate = (
        claim_changes
        / max(
            1,
            len(claims) - 1,
        )
    )

    # ========================================================
    # AMOUNT CHANGE RATE
    # ========================================================

    amounts = [
        interaction.requested_amount
        for interaction in interactions
    ]

    amount_changes = 0

    for previous, current in zip(
        amounts,
        amounts[1:],
    ):

        if previous <= 0:

            continue

        relative_change = (
            abs(current - previous)
            / previous
        )

        if relative_change >= 0.25:

            amount_changes += 1

    amount_change_rate = (
        amount_changes
        / max(
            1,
            len(amounts) - 1,
        )
    )

    # ========================================================
    # SUPPORT RESPONSE CHANGES
    # ========================================================

    responses = [
        interaction.support_response
        for interaction in interactions
    ]

    response_changes = sum(
        1
        for previous, current
        in zip(
            responses,
            responses[1:],
        )
        if previous != current
    )

    support_response_change_rate = (
        response_changes
        / max(
            1,
            len(responses) - 1,
        )
    )

    # ========================================================
    # BEHAVIORAL DRIFT
    #
    # Compare early and late observable behavior.
    #
    # We combine:
    #
    #   - success rate
    #   - evidence rate
    #   - claim diversity
    #   - response severity
    #
    # This is entirely observable.
    # ========================================================

    midpoint = max(
        1,
        len(interactions) // 2,
    )

    early = interactions[:midpoint]
    late = interactions[midpoint:]

    if late:

        early_success_rate = (
            sum(
                interaction.successful
                for interaction in early
            )
            / len(early)
        )

        late_success_rate = (
            sum(
                interaction.successful
                for interaction in late
            )
            / len(late)
        )

        success_drift = abs(
            early_success_rate
            - late_success_rate
        )

        early_evidence_rate = (
            sum(
                interaction.evidence_requested
                for interaction in early
            )
            / len(early)
        )

        late_evidence_rate = (
            sum(
                interaction.evidence_requested
                for interaction in late
            )
            / len(late)
        )

        evidence_drift = abs(
            early_evidence_rate
            - late_evidence_rate
        )

        early_claim_diversity = (
            len(
                set(
                    interaction.claim_type
                    for interaction in early
                )
            )
            / max(
                1,
                len(early),
            )
        )

        late_claim_diversity = (
            len(
                set(
                    interaction.claim_type
                    for interaction in late
                )
            )
            / max(
                1,
                len(late),
            )
        )

        claim_diversity_drift = abs(
            early_claim_diversity
            - late_claim_diversity
        )

        response_severity = {
            SupportResponse.APPROVE: 0.0,
            SupportResponse.REQUEST_EVIDENCE: 0.33,
            SupportResponse.ESCALATE: 0.66,
            SupportResponse.DENY: 1.0,
        }

        early_response_severity = (
            sum(
                response_severity[
                    interaction.support_response
                ]
                for interaction in early
            )
            / len(early)
        )

        late_response_severity = (
            sum(
                response_severity[
                    interaction.support_response
                ]
                for interaction in late
            )
            / len(late)
        )

        response_drift = abs(
            early_response_severity
            - late_response_severity
        )

        behavioral_drift = min(
            1.0,
            (
                0.30 * success_drift
                + 0.25 * evidence_drift
                + 0.20 * claim_diversity_drift
                + 0.25 * response_drift
            ),
        )

    else:

        behavioral_drift = 0.0

    # ========================================================
    # NETWORK SIGNALS
    # ========================================================

    customer_id = customer.customer_id

    degree = graph.degree(
        customer_id
    )

    shared_identifiers = (
        graph.shared_identifier_count(
            customer_id
        )
    )

    network_degree_signal = min(
        1.0,
        degree / 10.0,
    )

    shared_identifier_signal = min(
        1.0,
        shared_identifiers / 4.0,
    )

    # ========================================================
    # GOVERNOR-OBSERVABLE ADAPTATION SCORE
    #
    # IMPORTANT:
    #
    # strategy_switch_rate is deliberately EXCLUDED.
    #
    # The detector does not know the actor's strategy.
    #
    # The score uses only observable behavioral consequences.
    # ========================================================

    behavioral_adaptation_score = min(
        1.0,
        (
            0.20 * claim_change_rate
            + 0.15 * amount_change_rate
            + 0.20 * support_response_change_rate
            + 0.20 * behavioral_drift
            + 0.15 * request_velocity
            + 0.10 * (1.0 - repeated_claim_rate)
        ),
    )

    # --------------------------------------------------------
    # Calibrated behavioral threshold.
    #
    # This is NOT the trained GNN threshold.
    #
    # It is a deterministic baseline detector used only to
    # establish that the environment produces observable
    # adaptation signals.
    # --------------------------------------------------------

    detected_adaptation = (
        behavioral_adaptation_score
        >= 0.15
    )

    # ========================================================
    # NETWORK MANIPULATION
    # ========================================================

    network_score = (
        0.50 * network_degree_signal
        + 0.50 * shared_identifier_signal
    )

    detected_network_manipulation = (
        network_score
        >= 0.55
    )

    return GovernorObservableSignals(
        request_velocity=request_velocity,
        strategy_switch_rate=strategy_switch_rate,
        repeated_claim_rate=repeated_claim_rate,
        claim_change_rate=claim_change_rate,
        amount_change_rate=amount_change_rate,
        support_response_change_rate=(
            support_response_change_rate
        ),
        behavioral_drift=behavioral_drift,
        network_degree_signal=network_degree_signal,
        shared_identifier_signal=(
            shared_identifier_signal
        ),
        behavioral_adaptation_score=(
            behavioral_adaptation_score
        ),
        detected_adaptation=(
            detected_adaptation
        ),
        detected_network_manipulation=(
            detected_network_manipulation
        ),
    )


# ============================================================
# NETWORK CONSTRUCTION
# ============================================================


def build_adversarial_network(
    customer: AdaptiveCustomer,
    graph: InteractionGraph,
) -> None:

    customer_id = customer.customer_id

    # --------------------------------------------------------
    # Shared device cluster
    # --------------------------------------------------------

    graph.add_node(
        "device_cluster_001",
        "device",
    )

    graph.add_edge(
        customer_id,
        "device_cluster_001",
        "shared_device",
    )

    # --------------------------------------------------------
    # Shared address
    # --------------------------------------------------------

    graph.add_node(
        "address_cluster_001",
        "address",
    )

    graph.add_edge(
        customer_id,
        "address_cluster_001",
        "shared_address",
    )

    # --------------------------------------------------------
    # Additional adaptive actors.
    # --------------------------------------------------------

    for index in range(1, 5):

        other_customer = (
            f"adaptive_customer_{index + 1:03d}"
        )

        graph.add_node(
            other_customer,
            "customer",
        )

        graph.add_edge(
            other_customer,
            "device_cluster_001",
            "shared_device",
        )

    for index in range(5, 8):

        other_customer = (
            f"adaptive_customer_{index + 1:03d}"
        )

        graph.add_node(
            other_customer,
            "customer",
        )

        graph.add_edge(
            other_customer,
            "address_cluster_001",
            "shared_address",
        )


# ============================================================
# REFUND AMOUNT
# ============================================================


def amount_for_strategy(
    strategy: Strategy,
    interaction_number: int,
) -> float:

    """
    Produce observable request amounts.

    The amounts deliberately change when the actor changes
    presentation strategy.
    """

    if strategy == Strategy.NORMAL:

        return 1000.0

    if strategy == Strategy.DIRECT_REFUND:

        return 1500.0

    if strategy == Strategy.EVIDENCE_MANIPULATION:

        return 2200.0

    if strategy == Strategy.CLAIM_SWITCHING:

        return 1800.0 + (
            250.0
            * ((interaction_number - 1) % 3)
        )

    if strategy == Strategy.TIMING_MANIPULATION:

        return 2500.0

    if strategy == Strategy.HIGH_VALUE_ESCALATION:

        return 10000.0

    return 1500.0


# ============================================================
# RUN ONE ADAPTIVE EPISODE
# ============================================================


def run_adaptive_episode(
    customer: AdaptiveCustomer,
    support_agent: SupportAgent,
    graph: InteractionGraph,
    episodes: int = 12,
) -> GovernorObservableSignals:

    if episodes <= 0:

        raise ValueError(
            "episodes must be greater than zero."
        )

    for interaction_number in range(
        1,
        episodes + 1,
    ):

        # ----------------------------------------------------
        # Current adaptive strategy
        # ----------------------------------------------------

        strategy = customer.strategy

        # ----------------------------------------------------
        # Strategy creates observable claim presentation.
        # ----------------------------------------------------

        claim_type = claim_for_strategy(
            strategy,
            interaction_number,
        )

        # ----------------------------------------------------
        # Strategy places pressure on Agent A.
        # ----------------------------------------------------

        support_agent.receive_manipulation_pressure(
            strategy
        )

        # ----------------------------------------------------
        # Refund amount.
        # ----------------------------------------------------

        amount = amount_for_strategy(
            strategy,
            interaction_number,
        )

        # ----------------------------------------------------
        # Agent A responds.
        # ----------------------------------------------------

        response = support_agent.respond(
            strategy=strategy,
            amount=amount,
            interaction_number=interaction_number,
        )

        # ----------------------------------------------------
        # Observable outcome.
        # ----------------------------------------------------

        evidence_requested = (
            response
            == SupportResponse.REQUEST_EVIDENCE
        )

        successful = (
            response
            == SupportResponse.APPROVE
        )

        refund_amount = (
            amount
            if successful
            else 0.0
        )

        pressure = {
            Strategy.NORMAL: 0.00,
            Strategy.DIRECT_REFUND: 0.10,
            Strategy.EVIDENCE_MANIPULATION: 0.45,
            Strategy.CLAIM_SWITCHING: 0.60,
            Strategy.TIMING_MANIPULATION: 0.75,
            Strategy.HIGH_VALUE_ESCALATION: 0.90,
        }[strategy]

        interaction = Interaction(
            interaction_number=interaction_number,

            strategy_used=strategy,

            claim_type=claim_type,

            support_response=response,

            evidence_requested=evidence_requested,

            refund_amount=refund_amount,

            requested_amount=amount,

            successful=successful,

            support_sensitivity=(
                support_agent.evidence_sensitivity
            ),

            manipulation_pressure=pressure,
        )

        # ----------------------------------------------------
        # STEP 90
        # Adaptive actor observes Agent A.
        # ----------------------------------------------------

        customer.observe_support_response(
            interaction
        )

        # ----------------------------------------------------
        # STEP 91
        # Adaptive actor updates strategy.
        # ----------------------------------------------------

        customer.adapt_strategy()

        # ----------------------------------------------------
        # STEP 92
        # Record repeated interaction.
        # ----------------------------------------------------

        graph.add_edge(
            source=customer.customer_id,
            target=support_agent.agent_id,
            edge_type="interaction",
        )

    # --------------------------------------------------------
    # STEP 94 + STEP 95
    # Produce Governor-observable signals.
    # --------------------------------------------------------

    return calculate_governor_observable_signals(
        customer,
        graph,
    )


# ============================================================
# MAIN VERIFICATION
# ============================================================


def main() -> None:

    print()
    print("=" * 70)
    print(
        "CHALLENGING ADVERSARIAL / ADAPTIVE ENVIRONMENT"
    )
    print("=" * 70)

    # ========================================================
    # STEP 89
    # ========================================================

    customer = AdaptiveCustomer(
        customer_id="adaptive_customer_001",
        strategy=Strategy.DIRECT_REFUND,
    )

    print()
    print(
        "STEP 89 — Adaptive actor created: PASSED"
    )

    # ========================================================
    # Support Agent
    # ========================================================

    support_agent = SupportAgent(
        agent_id="support_agent_001",
        evidence_sensitivity=0.35,
        amount_sensitivity=0.45,
        escalation_threshold=0.75,
        manipulation_susceptibility=0.65,
    )

    # ========================================================
    # STEP 95
    # Build non-trivial graph.
    # ========================================================

    graph = InteractionGraph()

    graph.add_node(
        customer.customer_id,
        "customer",
    )

    graph.add_node(
        support_agent.agent_id,
        "support_agent",
    )

    build_adversarial_network(
        customer,
        graph,
    )

    # ========================================================
    # Run episode
    # ========================================================

    signals = run_adaptive_episode(
        customer=customer,
        support_agent=support_agent,
        graph=graph,
        episodes=12,
    )

    # ========================================================
    # STEP 90
    # ========================================================

    assert len(
        customer.interactions
    ) == 12, (
        "Adaptive actor did not produce the expected "
        "number of interactions."
    )

    print(
        "STEP 90 — Actor observes support responses: PASSED"
    )

    # ========================================================
    # STEP 91
    # ========================================================

    unique_strategies = len(
        set(
            interaction.strategy_used
            for interaction
            in customer.interactions
        )
    )

    strategies = [
        interaction.strategy_used
        for interaction
        in customer.interactions
    ]

    strategy_switches = sum(
        1
        for previous, current
        in zip(
            strategies,
            strategies[1:],
        )
        if previous != current
    )

    print()
    print(
        "STEP 91 — Strategy adaptation"
    )

    print(
        f"  Unique strategies attempted: "
        f"{unique_strategies}"
    )

    print(
        f"  Strategy switches: "
        f"{strategy_switches}"
    )

    assert unique_strategies >= 2, (
        "Adaptive actor did not actually change strategy."
    )

    assert strategy_switches > 0, (
        "Adaptive actor did not perform a genuine "
        "strategy transition."
    )

    print(
        "STEP 91 — Genuine strategy adaptation: PASSED"
    )

    # ========================================================
    # STEP 92
    # ========================================================

    assert len(
        support_agent.response_history
    ) == 12, (
        "Repeated interaction history was not recorded."
    )

    print(
        "STEP 92 — Repeated interactions: PASSED"
    )

    # ========================================================
    # STEP 93
    # ========================================================

    response_changes = sum(
        1
        for previous, current
        in zip(
            support_agent.response_history,
            support_agent.response_history[1:],
        )
        if previous != current
    )

    print()
    print(
        "STEP 93 — Support-agent manipulation"
    )

    print(
        f"  Support responses changed: "
        f"{response_changes} times"
    )

    assert response_changes >= 1, (
        "Support-agent behavior never changed."
    )

    print(
        "STEP 93 — Non-stationary support behavior: PASSED"
    )

    # ========================================================
    # STEP 94
    # ========================================================

    print()
    print(
        "STEP 94 — Governor-observable adaptation signals"
    )

    print(
        f"  Request velocity: "
        f"{signals.request_velocity:.4f}"
    )

    print(
        f"  Strategy switch rate "
        f"(evaluation only): "
        f"{signals.strategy_switch_rate:.4f}"
    )

    print(
        f"  Repeated claim rate: "
        f"{signals.repeated_claim_rate:.4f}"
    )

    print(
        f"  Claim change rate: "
        f"{signals.claim_change_rate:.4f}"
    )

    print(
        f"  Amount change rate: "
        f"{signals.amount_change_rate:.4f}"
    )

    print(
        f"  Support response change rate: "
        f"{signals.support_response_change_rate:.4f}"
    )

    print(
        f"  Behavioral drift: "
        f"{signals.behavioral_drift:.4f}"
    )

    print(
        f"  Behavioral adaptation score: "
        f"{signals.behavioral_adaptation_score:.4f}"
    )

    print(
        f"  Adaptation detected: "
        f"{signals.detected_adaptation}"
    )

    # --------------------------------------------------------
    # Strategy adaptation actually occurred.
    # --------------------------------------------------------

    assert (
        signals.strategy_switch_rate > 0.0
    ), (
        "No genuine strategy adaptation occurred."
    )

    # --------------------------------------------------------
    # Observable consequences must exist.
    # --------------------------------------------------------

    observable_behavior_change = (
        signals.claim_change_rate > 0.0
        or signals.amount_change_rate > 0.0
        or signals.support_response_change_rate > 0.0
        or signals.behavioral_drift > 0.0
    )

    assert observable_behavior_change, (
        "Adaptive behavior occurred internally but produced "
        "no observable behavioral consequence."
    )

    # --------------------------------------------------------
    # Final detector validation.
    #
    # This threshold belongs to the behavioral baseline
    # detector, NOT the trained GNN Governor.
    # --------------------------------------------------------

    assert (
        signals.behavioral_adaptation_score >= 0.15
    ), (
        "Adaptive behavior was generated but the behavioral "
        "detector score remained below the calibrated threshold. "
        f"Score={signals.behavioral_adaptation_score:.4f}"
    )

    assert signals.detected_adaptation, (
        "Adaptive behavior was generated but not detected "
        "by the behavioral detector."
    )

    print(
        "STEP 94 — Adaptation signal detected: PASSED"
    )

    # ========================================================
    # STEP 95
    # ========================================================

    print()
    print(
        "STEP 95 — Network / graph manipulation"
    )

    degree = graph.degree(
        customer.customer_id
    )

    shared_identifiers = (
        graph.shared_identifier_count(
            customer.customer_id
        )
    )

    print(
        f"  Graph nodes: "
        f"{len(graph.nodes)}"
    )

    print(
        f"  Graph edges: "
        f"{len(graph.edges)}"
    )

    print(
        f"  Customer degree: "
        f"{degree}"
    )

    print(
        f"  Shared identifiers: "
        f"{shared_identifiers}"
    )

    print(
        f"  Network degree signal: "
        f"{signals.network_degree_signal:.4f}"
    )

    print(
        f"  Shared identifier signal: "
        f"{signals.shared_identifier_signal:.4f}"
    )

    print(
        f"  Network manipulation detected: "
        f"{signals.detected_network_manipulation}"
    )

    assert (
        shared_identifiers >= 2
    ), (
        "Network does not contain enough shared identifiers."
    )

    assert signals.detected_network_manipulation, (
        "Network manipulation was generated but not detected."
    )

    print(
        "STEP 95 — Network manipulation signal detected: PASSED"
    )

    # ========================================================
    # COMPLETE BEHAVIORAL TRACE
    # ========================================================

    print()
    print("-" * 70)
    print("ADAPTIVE ACTOR BEHAVIORAL TRACE")
    print("-" * 70)

    for interaction in customer.interactions:

        print(
            f"#{interaction.interaction_number:02d} | "
            f"strategy="
            f"{interaction.strategy_used.value:<25} | "
            f"claim="
            f"{interaction.claim_type:<25} | "
            f"amount="
            f"₹{interaction.requested_amount:>8.2f} | "
            f"response="
            f"{interaction.support_response.value:<18} | "
            f"success="
            f"{str(interaction.successful):<5} | "
            f"pressure="
            f"{interaction.manipulation_pressure:.2f}"
        )

    # ========================================================
    # STRATEGY SUMMARY
    # ========================================================

    print()
    print("-" * 70)
    print("STRATEGY LEARNING SUMMARY")
    print("-" * 70)

    for strategy in Strategy:

        attempts = customer.strategy_attempts.get(
            strategy,
            0,
        )

        score = customer.strategy_scores[
            strategy
        ]

        print(
            f"{strategy.value:<25} "
            f"attempts={attempts:<3} "
            f"score={score:>6.2f}"
        )

    # ========================================================
    # OBSERVABILITY BOUNDARY
    # ========================================================

    print()
    print("-" * 70)
    print("DETECTOR OBSERVABILITY BOUNDARY")
    print("-" * 70)

    print(
        "  Hidden attacker strategy used internally : YES"
    )

    print(
        "  Hidden strategy used by detector         : NO"
    )

    print(
        "  Claim behavior visible to detector      : YES"
    )

    print(
        "  Amount behavior visible to detector     : YES"
    )

    print(
        "  Support response visible to detector    : YES"
    )

    print(
        "  Evidence requests visible to detector   : YES"
    )

    print(
        "  GNN risk exposed to actor                : NO"
    )

    print(
        "  Governor internals exposed to actor     : NO"
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "STEPS 89 → 95 CHALLENGING ENVIRONMENT: PASSED"
    )
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The environment now generates genuine strategy "
        "adaptation rather than repeatedly executing one "
        "successful strategy."
    )

    print(
        "The adaptive actor learns only from observable "
        "support-agent outcomes."
    )

    print(
        "The behavioral detector does not use the hidden "
        "strategy label to detect adaptation."
    )

    print(
        "The detector instead relies on observable behavioral "
        "consequences."
    )

    print(
        "This still does NOT prove that the trained GNN "
        "Governor detects these patterns."
    )

    print(
        "The next integration test must feed the environment-"
        "generated observable state into the existing Governor "
        "and evaluate whether the actual Governor changes its "
        "action appropriately."
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()