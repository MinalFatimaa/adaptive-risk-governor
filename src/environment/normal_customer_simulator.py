from __future__ import annotations

"""
STEP 88 — NORMAL CUSTOMER SIMULATION

Purpose
-------
Establish the baseline behavior of normal human customers
before introducing adaptive customers/agents.

This module:

    - uses the existing EnvironmentState
    - uses the existing customer schemas
    - uses the existing InteractionEvent schema
    - generates normal customer interactions
    - records interactions in EnvironmentState.events
    - keeps private state and ground truth inside the simulator

This module does NOT:

    - modify the Governor
    - modify the GNN
    - modify Risk Fusion
    - introduce adaptive behavior
    - introduce strategy learning
    - allow customers to observe support-agent behavior
    - manipulate support agents
    - modify customer strategy
    - make Governor decisions

Roadmap position
----------------

88. Simulate normal customers              <- THIS MODULE

Next:

89. Simulate adaptive customers/agents
90. Allow adaptive actors to observe
   support-agent responses
91. Model strategy adaptation
92. Model repeated interactions
93. Model manipulation of support-agent behavior
94. Test whether Governor detects adaptation
95. Test network/graph-based manipulation
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from random import Random

from .world import EnvironmentState

from ..schemas.customer import (
    CustomerObservableState,
)

from ..schemas.interaction import (
    InteractionEvent,
)


# ============================================================
# CONSTANTS
# ============================================================

CLAIM_TYPES = (
    "PRODUCT_NOT_RECEIVED",
    "DAMAGED_ITEM",
    "WRONG_ITEM",
    "MISSING_ITEM",
    "REFUND",
)

SUPPORT_DECISIONS = (
    "APPROVE",
    "REQUEST_EVIDENCE",
    "ESCALATE",
    "DENY",
)

NORMAL_CUSTOMER_TYPE = "HUMAN"

NORMAL_BEHAVIOR = "NORMAL_CUSTOMER"


# ============================================================
# SUPPORT AGENT PROFILES
# ============================================================

SUPPORT_AGENT_PROFILES = (
    "BALANCED",
    "SATISFACTION_ORIENTED",
    "CONSERVATIVE",
    "EVIDENCE_HEAVY",
    "FAST_RESOLUTION",
    "BALANCED",
    "ESCALATION_HEAVY",
    "EVIDENCE_HEAVY",
)


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass(frozen=True)
class NormalInteractionResult:
    """
    Public result of one normal-customer interaction.

    This object deliberately contains only interaction-level
    information.

    Customer private state and ground truth are NOT exposed.
    """

    event_id: str

    episode_id: str

    customer_id: str

    support_agent_id: str

    claim_type: str

    requested_amount: float

    support_decision: str

    timestamp: datetime


# ============================================================
# SIMULATOR
# ============================================================

class NormalCustomerSimulator:
    """
    Simulator for normal human customers.

    A normal customer:

        - does not adapt its strategy
        - does not infer support-agent policy
        - does not optimize against the support agent
        - does not observe hidden simulator state
        - does not modify behavior based on outcomes

    This provides the baseline required before Step 89.
    """

    def __init__(
        self,
        environment: EnvironmentState,
        seed: int | None = None,
    ) -> None:

        self.environment = environment

        self.rng = Random(
            42 if seed is None else seed
        )

        self._event_counter = 0

    # ========================================================
    # ID GENERATION
    # ========================================================

    def _next_event_id(self) -> str:
        """
        Generate deterministic event IDs.
        """

        self._event_counter += 1

        return (
            f"EVENT_"
            f"{self._event_counter:06d}"
        )

    # ========================================================
    # EPISODE ID
    # ========================================================

    def _episode_id(
        self,
        customer_id: str,
    ) -> str:
        """
        Generate an episode identifier.

        For Step 88, one interaction is one episode.

        Later steps can reuse an episode ID for repeated
        interactions.
        """

        return (
            f"EPISODE_{customer_id}_"
            f"{self._event_counter:06d}"
        )

    # ========================================================
    # SUPPORT AGENTS
    # ========================================================

    def _support_agents(self) -> list[dict[str, str]]:
        """
        Return deterministic support-agent metadata.

        The current EnvironmentState does not yet contain a
        dedicated support-agent collection, so the existing
        support-agent configuration is represented locally.

        This does NOT alter EnvironmentState.
        """

        return [
            {
                "agent_id": (
                    f"SUPPORT_{index + 1:03d}"
                ),
                "profile": SUPPORT_AGENT_PROFILES[
                    index
                    % len(SUPPORT_AGENT_PROFILES)
                ],
            }
            for index in range(
                len(SUPPORT_AGENT_PROFILES)
            )
        ]

    # ========================================================
    # NORMAL CUSTOMERS
    # ========================================================

    def _normal_customers(
        self,
    ) -> list[CustomerObservableState]:
        """
        Select normal human customers.

        Step 88 deliberately excludes:

            - abusive customers
            - adaptive customers
        """

        customers = []

        for customer_id, customer in (
            self.environment.customers.items()
        ):

            truth = (
                self.environment.ground_truth.get(
                    customer_id
                )
            )

            if truth is None:
                continue

            # ------------------------------------------------
            # Human + legitimate = normal baseline
            # ------------------------------------------------

            if (
                truth.counterparty_type
                == NORMAL_CUSTOMER_TYPE
                and not truth.is_abusive
            ):
                customers.append(customer)

        return customers

    # ========================================================
    # CLAIM GENERATION
    # ========================================================

    def _generate_claim(
        self,
        customer: CustomerObservableState,
    ) -> tuple[str, float]:
        """
        Generate a normal customer claim.

        This is stochastic but deterministic under the
        simulator seed.

        The claim is NOT chosen strategically against the
        support agent.
        """

        claim_type = self.rng.choice(
            CLAIM_TYPES
        )

        # ----------------------------------------------------
        # Historical refund activity
        # ----------------------------------------------------

        if customer.refund_count > 0:

            historical_average = (
                customer.refund_amount_total
                / customer.refund_count
            )

        else:

            historical_average = 1000.0

        # ----------------------------------------------------
        # Add ordinary customer-level variation
        # ----------------------------------------------------

        lower = max(
            100.0,
            historical_average * 0.5,
        )

        upper = max(
            lower,
            historical_average * 1.5,
        )

        requested_amount = self.rng.uniform(
            lower,
            upper,
        )

        requested_amount = min(
            requested_amount,
            10000.0,
        )

        return (
            claim_type,
            round(
                requested_amount,
                2,
            ),
        )

    # ========================================================
    # SUPPORT DECISION
    # ========================================================

    def _support_decision(
        self,
        *,
        profile: str,
        requested_amount: float,
    ) -> str:
        """
        Produce the support-agent response.

        This is a baseline support policy.

        It is intentionally independent of:

            - customer private state
            - ground truth
            - adaptive strategy
            - Governor risk
            - GNN risk
            - strategic state

        This keeps Step 88 a clean baseline.
        """

        # ----------------------------------------------------
        # CONSERVATIVE
        # ----------------------------------------------------

        if profile == "CONSERVATIVE":

            if requested_amount >= 5000:
                return "ESCALATE"

            if requested_amount >= 2000:
                return "REQUEST_EVIDENCE"

            return "APPROVE"

        # ----------------------------------------------------
        # EVIDENCE HEAVY
        # ----------------------------------------------------

        if profile == "EVIDENCE_HEAVY":

            if requested_amount >= 1500:
                return "REQUEST_EVIDENCE"

            return "APPROVE"

        # ----------------------------------------------------
        # ESCALATION HEAVY
        # ----------------------------------------------------

        if profile == "ESCALATION_HEAVY":

            if requested_amount >= 3000:
                return "ESCALATE"

            if requested_amount >= 1500:
                return "REQUEST_EVIDENCE"

            return "APPROVE"

        # ----------------------------------------------------
        # FAST RESOLUTION
        # ----------------------------------------------------

        if profile == "FAST_RESOLUTION":

            if requested_amount <= 5000:
                return "APPROVE"

            return "ESCALATE"

        # ----------------------------------------------------
        # SATISFACTION ORIENTED
        # ----------------------------------------------------

        if profile == "SATISFACTION_ORIENTED":

            if requested_amount <= 3000:
                return "APPROVE"

            if requested_amount <= 5000:
                return "REQUEST_EVIDENCE"

            return "ESCALATE"

        # ----------------------------------------------------
        # BALANCED
        # ----------------------------------------------------

        if requested_amount >= 5000:
            return "ESCALATE"

        if requested_amount >= 2000:
            return "REQUEST_EVIDENCE"

        return "APPROVE"

    # ========================================================
    # CREATE INTERACTION
    # ========================================================

    def simulate_interaction(
        self,
        customer: CustomerObservableState,
        *,
        current_time: datetime,
    ) -> NormalInteractionResult:
        """
        Generate one normal customer interaction.
        """

        event_id = self._next_event_id()

        episode_id = self._episode_id(
            customer.customer_id
        )

        support_agents = (
            self._support_agents()
        )

        support_agent = self.rng.choice(
            support_agents
        )

        claim_type, requested_amount = (
            self._generate_claim(
                customer
            )
        )

        support_decision = (
            self._support_decision(
                profile=support_agent[
                    "profile"
                ],
                requested_amount=requested_amount,
            )
        )

        return NormalInteractionResult(
            event_id=event_id,

            episode_id=episode_id,

            customer_id=customer.customer_id,

            support_agent_id=support_agent[
                "agent_id"
            ],

            claim_type=claim_type,

            requested_amount=requested_amount,

            support_decision=support_decision,

            timestamp=current_time,
        )

    # ========================================================
    # INTERACTION EVENT
    # ========================================================

    def _create_event(
        self,
        result: NormalInteractionResult,
        *,
        sequence_number: int,
    ) -> InteractionEvent:
        """
        Convert a simulation result into the current
        InteractionEvent schema.

        IMPORTANT:
        Do not use the old interaction_id-based schema here.

        The current InteractionEvent schema requires exactly
        these top-level fields:

            event_id
            episode_id
            timestamp
            sequence_number
            actor
            event_type
            visibility
            payload
        """

        event = InteractionEvent(
            event_id=result.event_id,

            episode_id=result.episode_id,

            timestamp=result.timestamp,

            sequence_number=sequence_number,

            actor=result.customer_id,

            event_type="CUSTOMER_SUPPORT_INTERACTION",

            visibility="SIMULATION",

            payload={
                "customer_id": result.customer_id,

                "support_agent_id": (
                    result.support_agent_id
                ),

                "claim_type": result.claim_type,

                "requested_amount": (
                    result.requested_amount
                ),

                "support_decision": (
                    result.support_decision
                ),

                "customer_type": (
                    NORMAL_CUSTOMER_TYPE
                ),

                "behavior": (
                    NORMAL_BEHAVIOR
                ),
            },
        )

        return event

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        *,
        interactions_per_customer: int = 1,
        start_time: datetime | None = None,
    ) -> list[NormalInteractionResult]:
        """
        Simulate normal customers.

        Parameters
        ----------
        interactions_per_customer:
            Number of interactions per normal customer.

        start_time:
            Simulation start timestamp.

        Returns
        -------
        list[NormalInteractionResult]
        """

        if interactions_per_customer <= 0:
            raise ValueError(
                "interactions_per_customer "
                "must be greater than zero."
            )

        if start_time is None:

            start_time = datetime(
                2026,
                8,
                24,
                12,
                0,
                0,
            )

        normal_customers = (
            self._normal_customers()
        )

        results = []

        # ----------------------------------------------------
        # Generate interactions
        # ----------------------------------------------------

        for customer in normal_customers:

            for interaction_index in range(
                interactions_per_customer
            ):

                current_time = (
                    start_time
                    + timedelta(
                        hours=interaction_index
                    )
                )

                result = (
                    self.simulate_interaction(
                        customer,
                        current_time=current_time,
                    )
                )

                event = self._create_event(
                    result,
                    sequence_number=(
                        interaction_index + 1
                    ),
                )

                self.environment.events.append(
                    event
                )

                results.append(
                    result
                )

        return results


# ============================================================
# VALIDATION
# ============================================================

def validate_normal_simulation(
    environment: EnvironmentState,
    results: list[NormalInteractionResult],
) -> None:
    """
    Validate Step 88 invariants.

    These checks are intentionally strict so that later
    adaptive-agent experiments have a trustworthy baseline.
    """

    # --------------------------------------------------------
    # At least one interaction
    # --------------------------------------------------------

    if not results:
        raise AssertionError(
            "Normal-customer simulation produced "
            "no interactions."
        )

    # --------------------------------------------------------
    # Validate each interaction
    # --------------------------------------------------------

    for result in results:

        truth = environment.ground_truth.get(
            result.customer_id
        )

        if truth is None:
            raise AssertionError(
                "Interaction references a customer "
                "without ground truth."
            )

        # ----------------------------------------------------
        # Normal customer boundary
        # ----------------------------------------------------

        if truth.is_abusive:
            raise AssertionError(
                "Step 88 violation: abusive customer "
                "entered the normal-customer baseline."
            )

        if (
            truth.counterparty_type
            != NORMAL_CUSTOMER_TYPE
        ):
            raise AssertionError(
                "Step 88 violation: adaptive customer "
                "entered the normal-customer baseline."
            )

        # ----------------------------------------------------
        # Claim validation
        # ----------------------------------------------------

        if result.claim_type not in CLAIM_TYPES:
            raise AssertionError(
                "Invalid claim type: "
                f"{result.claim_type}"
            )

        # ----------------------------------------------------
        # Support decision validation
        # ----------------------------------------------------

        if (
            result.support_decision
            not in SUPPORT_DECISIONS
        ):
            raise AssertionError(
                "Invalid support decision: "
                f"{result.support_decision}"
            )

        # ----------------------------------------------------
        # Amount validation
        # ----------------------------------------------------

        if result.requested_amount <= 0:
            raise AssertionError(
                "Requested amount must be positive."
            )

        # ----------------------------------------------------
        # ID validation
        # ----------------------------------------------------

        if not result.event_id:
            raise AssertionError(
                "Interaction event_id is empty."
            )

        if not result.episode_id:
            raise AssertionError(
                "Interaction episode_id is empty."
            )

    # --------------------------------------------------------
    # Event recording
    # --------------------------------------------------------

    if len(environment.events) < len(results):
        raise AssertionError(
            "Generated interactions were not correctly "
            "recorded in EnvironmentState.events."
        )

    # --------------------------------------------------------
    # Validate actual InteractionEvent objects
    # --------------------------------------------------------

    recent_events = environment.events[
        -len(results):
    ]

    for event in recent_events:

        if not event.event_id:
            raise AssertionError(
                "InteractionEvent.event_id is empty."
            )

        if not event.episode_id:
            raise AssertionError(
                "InteractionEvent.episode_id is empty."
            )

        if not event.actor:
            raise AssertionError(
                "InteractionEvent.actor is empty."
            )

        if not event.event_type:
            raise AssertionError(
                "InteractionEvent.event_type is empty."
            )

        if not event.visibility:
            raise AssertionError(
                "InteractionEvent.visibility is empty."
            )

        if not isinstance(
            event.payload,
            dict,
        ):
            raise AssertionError(
                "InteractionEvent.payload "
                "must be a dictionary."
            )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def simulate_normal_customers(
    environment: EnvironmentState,
    *,
    interactions_per_customer: int = 1,
    seed: int = 42,
) -> list[NormalInteractionResult]:
    """
    Convenience wrapper for Step 88.
    """

    simulator = NormalCustomerSimulator(
        environment=environment,
        seed=seed,
    )

    results = simulator.run(
        interactions_per_customer=(
            interactions_per_customer
        )
    )

    validate_normal_simulation(
        environment,
        results,
    )

    return results


# ============================================================
# MANUAL TEST
# ============================================================

def main() -> None:

    from .world_generator import (
        create_world,
    )

    print()
    print("=" * 80)
    print(
        "STEP 88 — NORMAL CUSTOMER SIMULATION"
    )
    print("=" * 80)

    print()
    print(
        "Creating deterministic simulation world..."
    )

    environment = create_world(
        seed=42
    )

    # --------------------------------------------------------
    # Count normal customers
    # --------------------------------------------------------

    normal_customers = [
        customer_id
        for customer_id in environment.customers
        if (
            not environment.ground_truth[
                customer_id
            ].is_abusive
            and
            environment.ground_truth[
                customer_id
            ].counterparty_type
            == NORMAL_CUSTOMER_TYPE
        )
    ]

    print(
        f"Normal customers available : "
        f"{len(normal_customers)}"
    )

    # --------------------------------------------------------
    # Simulate
    # --------------------------------------------------------

    print()
    print(
        "Simulating one interaction per "
        "normal customer..."
    )

    results = simulate_normal_customers(
        environment,
        interactions_per_customer=1,
        seed=42,
    )

    print()
    print(
        f"Interactions generated : "
        f"{len(results)}"
    )

    print(
        f"Environment events      : "
        f"{len(environment.events)}"
    )

    # --------------------------------------------------------
    # Sample interactions
    # --------------------------------------------------------

    print()
    print(
        "Sample interactions:"
    )

    for result in results[:10]:

        print(
            f"  {result.customer_id:<15} "
            f"{result.claim_type:<22} "
            f"₹{result.requested_amount:>8.2f} "
            f"-> "
            f"{result.support_decision}"
        )

    # --------------------------------------------------------
    # Sample event
    # --------------------------------------------------------

    if environment.events:

        event = environment.events[0]

        print()
        print(
            "Sample InteractionEvent:"
        )

        print(
            f"  event_id        : "
            f"{event.event_id}"
        )

        print(
            f"  episode_id      : "
            f"{event.episode_id}"
        )

        print(
            f"  actor           : "
            f"{event.actor}"
        )

        print(
            f"  event_type      : "
            f"{event.event_type}"
        )

        print(
            f"  visibility      : "
            f"{event.visibility}"
        )

        print(
            f"  payload         : "
            f"{event.payload}"
        )

    # --------------------------------------------------------
    # Final boundary validation
    # --------------------------------------------------------

    for result in results:

        truth = environment.ground_truth[
            result.customer_id
        ]

        assert (
            truth.counterparty_type
            == NORMAL_CUSTOMER_TYPE
        )

        assert (
            truth.is_abusive is False
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print(
        "[PASS] Normal customers simulated."
    )

    print(
        "[PASS] No adaptive customers entered "
        "the baseline simulation."
    )

    print(
        "[PASS] No abusive customers entered "
        "the normal-customer baseline."
    )

    print(
        "[PASS] InteractionEvent objects conform "
        "to the existing project schema."
    )

    print(
        "[PASS] Interaction events recorded "
        "in EnvironmentState.events."
    )

    print()
    print(
        "STEP 88 COMPLETE."
    )

    print()
    print(
        "Next:"
    )

    print(
        "89. Simulate adaptive customers/agents"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()