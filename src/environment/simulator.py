from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from random import Random
from typing import Any

from ..agents.adaptive_customer import (
    AdaptiveCustomerAgent,
    AdaptiveCustomerState,
)
from ..agents.support_agent import SupportAgent
from ..schemas.customer import CustomerObservableState
from ..schemas.merchant import MerchantState
from ..schemas.order import OrderState
from ..schemas.support import RefundRequest, SupportDecision
from .event_log import EventLog
from .world import EnvironmentState


# ============================================================
# SIMULATION CONFIGURATION
# ============================================================

DEFAULT_EPISODE_LENGTH = 10

# Fixed simulation start time.
#
# Do NOT use datetime.now() here.
# Experimental evaluation should be reproducible.
SIMULATION_START_TIME = datetime(
    2026,
    8,
    31,
    12,
    0,
    0,
)

CLAIM_TYPES = [
    "SHORTAGE_CLAIM",
    "WRONG_ITEM_CLAIM",
    "NON_DELIVERY_CLAIM",
    "SUBSTITUTED_RETURN_CLAIM",
]

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
# TRAJECTORY RECORD
# ============================================================

@dataclass
class InteractionRecord:
    """
    One interaction between Agent B and Agent A.

    Only information explicitly exposed through the observation
    fields should be considered visible to Agent B.
    """

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

    # Observable customer/request information.
    customer_observation: dict[str, Any]

    # Observable Support Agent response.
    #
    # This deliberately contains no governor score,
    # ML features, or ground truth.
    support_observation: dict[str, Any]


# ============================================================
# EPISODE RESULT
# ============================================================

@dataclass
class EpisodeResult:
    """
    Complete adaptive trajectory.

    Agent B starts with an initial policy belief and updates it
    after every observed Agent A decision.
    """

    episode_id: str
    customer_id: str
    interactions: list[InteractionRecord]

    initial_policy_beliefs: dict[str, float]
    final_policy_beliefs: dict[str, float]

    @property
    def length(self) -> int:
        return len(
            self.interactions
        )


# ============================================================
# ADAPTIVE INTERACTION SIMULATOR
# ============================================================

class AdaptiveInteractionSimulator:
    """
    Simulates the adaptive-agent threat model.

    The loop is:

        Agent B submits request
                ↓
        Agent A evaluates request
                ↓
        Agent B observes external response
                ↓
        Agent B updates beliefs
                ↓
        Agent B changes next request

    IMPORTANT INFORMATION BOUNDARY
    -------------------------------

    Agent B may observe:

        - its own request
        - customer-visible information
        - Agent A's external decision
        - reason code
        - requested evidence
        - approved amount
        - whether follow-up is required

    Agent B does NOT receive:

        - Governor risk score
        - ML risk-fusion features
        - model probabilities
        - hidden model state
        - ground truth
        - other customers
        - internal Governor state
    """

    def __init__(
        self,
        world: EnvironmentState,
        merchant: MerchantState | None = None,
        seed: int = 42,
    ) -> None:

        self.world = world

        self.merchant = (
            merchant
            if merchant is not None
            else world.merchant
        )

        self.rng = Random(
            seed
        )

        # Agent A.
        #
        # This is the existing SupportAgent implementation.
        self.support_agent = SupportAgent(
            merchant=self.merchant
        )

        self.event_log = EventLog(
            env=self.world
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def run_episode(
        self,
        customer_id: str,
        episode_id: str,
        n_interactions: int = DEFAULT_EPISODE_LENGTH,
    ) -> EpisodeResult:

        if n_interactions < 1:

            raise ValueError(
                "n_interactions must be at least 1."
            )

        if customer_id not in self.world.customers:

            raise ValueError(
                f"Unknown customer_id: {customer_id}"
            )

        customer = self.world.customers[
            customer_id
        ]

        private_state = self.world.customer_private[
            customer_id
        ]

        # ----------------------------------------------------
        # Determine Agent B objective.
        #
        # Adaptive abusive customers attempt to maximize
        # illegitimate refund value.
        #
        # Adaptive legitimate customers attempt to maximize
        # successful legitimate resolution.
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

        # ----------------------------------------------------
        # Create Agent B.
        # ----------------------------------------------------

        agent_b_state = AdaptiveCustomerState(
            customer_id=customer_id,
            objective=objective,
        )

        agent_b = AdaptiveCustomerAgent(
            state=agent_b_state,
            seed=self.rng.randint(
                0,
                10_000_000,
            ),
        )

        # ----------------------------------------------------
        # Record initial policy beliefs.
        # ----------------------------------------------------

        initial_beliefs = (
            agent_b
            .get_inferred_policy()
            .copy()
        )

        interactions: list[
            InteractionRecord
        ] = []

        # ----------------------------------------------------
        # Deterministic simulated clock.
        # ----------------------------------------------------

        current_time = (
            SIMULATION_START_TIME
        )

        # ====================================================
        # INTERACTION LOOP
        # ====================================================

        for sequence_number in range(
            1,
            n_interactions + 1,
        ):

            # ------------------------------------------------
            # STEP 1
            # Agent B chooses a request.
            #
            # Its current policy beliefs influence:
            #
            # - claim type
            # - amount
            # - evidence
            # ------------------------------------------------

            request = self._generate_request(
                customer=customer,
                sequence_number=(
                    sequence_number
                ),
                episode_id=episode_id,
                current_time=current_time,
                agent_b=agent_b,
            )

            # ------------------------------------------------
            # STEP 2
            # Agent A evaluates the request.
            # ------------------------------------------------

            order = self.world.orders[
                request.order_id
            ]

            support_decision = (
                self.support_agent.decide(
                    customer=customer,
                    order=order,
                    request=request,
                )
            )

            # ------------------------------------------------
            # STEP 3
            # Construct the information visible to Agent B.
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
            # Agent B observes Agent A and learns.
            #
            # IMPORTANT:
            #
            # We pass only information that B could reasonably
            # observe from the interaction.
            #
            # No Governor score is passed.
            # No ML features are passed.
            # No ground truth is passed.
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
            # Record the complete trajectory.
            # ------------------------------------------------

            record = InteractionRecord(

                episode_id=episode_id,

                sequence_number=sequence_number,

                customer_id=customer_id,

                order_id=request.order_id,

                claim_type=request.claim_type,

                requested_amount=(
                    request.requested_amount
                ),

                evidence_available=(
                    list(
                        request.evidence_available
                    )
                ),

                support_decision=(
                    support_decision.decision
                ),

                reason_code=(
                    support_decision.reason_code
                ),

                approved_amount=(
                    support_decision.approved_amount
                ),

                requires_followup=(
                    support_decision.requires_followup
                ),

                submitted_at=(
                    request.submitted_at
                ),

                customer_observation=(
                    customer_observation
                ),

                support_observation=(
                    support_observation
                ),
            )

            interactions.append(
                record
            )

            # ------------------------------------------------
            # STEP 6
            # Advance simulated time.
            #
            # Agent B does not get to see or control this.
            # ------------------------------------------------

            current_time += timedelta(
                hours=self.rng.randint(
                    1,
                    72,
                )
            )

        # ----------------------------------------------------
        # Final Agent B policy beliefs.
        # ----------------------------------------------------

        final_beliefs = (
            agent_b
            .get_inferred_policy()
            .copy()
        )

        return EpisodeResult(

            episode_id=episode_id,

            customer_id=customer_id,

            interactions=interactions,

            initial_policy_beliefs=(
                initial_beliefs
            ),

            final_policy_beliefs=(
                final_beliefs
            ),
        )

    # ========================================================
    # REQUEST GENERATION
    # ========================================================

    def _generate_request(
        self,
        customer: CustomerObservableState,
        sequence_number: int,
        episode_id: str,
        current_time: datetime,
        agent_b: AdaptiveCustomerAgent,
    ) -> RefundRequest:

        # ----------------------------------------------------
        # Agent B uses its CURRENT learned policy.
        #
        # This is what creates adaptation.
        # ----------------------------------------------------

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

        claim_text = (
            self._build_claim_text(
                claim_type=claim_type
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

            claim_text=claim_text,

            requested_amount=amount,

            submitted_at=current_time,

            evidence_available=evidence,
        )

    # ========================================================
    # CLAIM SELECTION
    # ========================================================

    def _choose_claim_type(
        self,
        policy: dict[str, float],
    ) -> str:

        evidence_sensitivity = float(
            policy.get(
                "evidence_sensitivity",
                0.50,
            )
        )

        # ----------------------------------------------------
        # If Agent B learns that evidence-friendly claims
        # receive better treatment, it increasingly selects
        # claims for which it can provide evidence.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # If Agent B learns that high-value requests attract
        # scrutiny, it moves toward the automatic-refund
        # boundary instead of repeatedly requesting very high
        # amounts.
        # ----------------------------------------------------

        if (
            high_value_sensitivity > 0.65
        ):

            upper_bound = min(
                5000.0,
                (
                    0.95
                    * self.merchant
                    .refund_policy
                    .human_review_threshold
                ),
            )

        elif (
            amount_sensitivity > 0.65
        ):

            upper_bound = min(
                5000.0,
                (
                    0.80
                    * self.merchant
                    .refund_policy
                    .human_review_threshold
                ),
            )

        else:

            upper_bound = min(
                5000.0,
                9000.0,
            )

        lower_bound = 500.0

        if upper_bound < lower_bound:

            upper_bound = (
                lower_bound
            )

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

        # ----------------------------------------------------
        # Agent B increasingly provides evidence when it learns
        # that evidence improves its interaction outcome.
        # ----------------------------------------------------

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
    # CLAIM TEXT
    # ========================================================

    def _build_claim_text(
        self,
        claim_type: str,
    ) -> str:

        texts = {

            "SHORTAGE_CLAIM":
                "The delivered order was incomplete.",

            "WRONG_ITEM_CLAIM":
                "The item received was different "
                "from the item ordered.",

            "NON_DELIVERY_CLAIM":
                "The order was not received.",

            "SUBSTITUTED_RETURN_CLAIM":
                "The returned item does not match "
                "the expected item.",
        }

        if claim_type not in texts:

            raise ValueError(
                f"Unknown claim type: "
                f"{claim_type}"
            )

        return texts[
            claim_type
        ]

    # ========================================================
    # CUSTOMER OBSERVATION
    # ========================================================

    def _build_customer_observation(
        self,
        customer: CustomerObservableState,
        order: OrderState,
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
        decision: SupportDecision,
    ) -> dict[str, Any]:

        # ----------------------------------------------------
        # ONLY externally observable Support Agent information.
        #
        # DO NOT add:
        #
        #   governor_score
        #   risk_score
        #   ml_features
        #   feature_importance
        #   ground_truth
        #   hidden_policy
        # ----------------------------------------------------

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
# MULTI-EPISODE RUNNER
# ============================================================

def run_adaptive_simulation(
    world: EnvironmentState,
    n_episodes: int = 100,
    interactions_per_episode: int = 10,
    seed: int = 42,
) -> list[EpisodeResult]:

    if n_episodes < 1:

        raise ValueError(
            "n_episodes must be at least 1."
        )

    if interactions_per_episode < 1:

        raise ValueError(
            "interactions_per_episode "
            "must be at least 1."
        )

    simulator = AdaptiveInteractionSimulator(
        world=world,
        seed=seed,
    )

    # --------------------------------------------------------
    # Only adaptive-agent customers participate in this
    # helper.
    # --------------------------------------------------------

    adaptive_customers = [

        customer_id

        for customer_id, truth
        in world.ground_truth.items()

        if (
            str(
                truth.counterparty_type
            ).upper()
            == "ADAPTIVE_AGENT"
        )
    ]

    if not adaptive_customers:

        raise ValueError(
            "No adaptive-agent customers "
            "exist in the world."
        )

    results: list[
        EpisodeResult
    ] = []

    # --------------------------------------------------------
    # Cycle through adaptive customers.
    # --------------------------------------------------------

    for episode_number in range(
        1,
        n_episodes + 1,
    ):

        customer_id = (
            adaptive_customers[
                (episode_number - 1)
                % len(adaptive_customers)
            ]
        )

        episode_id = (
            f"EPISODE_"
            f"{episode_number:05d}"
        )

        result = (
            simulator.run_episode(
                customer_id=customer_id,
                episode_id=episode_id,
                n_interactions=(
                    interactions_per_episode
                ),
            )
        )

        results.append(
            result
        )

    return results