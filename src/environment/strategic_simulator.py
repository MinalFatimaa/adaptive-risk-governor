from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from random import Random
from uuid import uuid4

from ..agents.strategic_customer import StrategicCustomerAgent
from ..agents.support_agent import SupportAgent

from ..schemas.customer import CustomerObservableState
from ..schemas.order import OrderState
from ..schemas.support import RefundRequest, SupportDecision

from .world import EnvironmentState


# ============================================================
# HELPERS
# ============================================================

def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


# ============================================================
# STRATEGIC INTERACTION
# ============================================================

@dataclass
class StrategicInteraction:

    sequence_number: int

    claim_type: str

    requested_amount: float

    evidence_available: list[str]

    support_decision: str

    reason_code: str

    approved_amount: float

    predicted_decision: str | None

    prediction_correct: bool | None


# ============================================================
# STRATEGIC EPISODE
# ============================================================

@dataclass
class StrategicEpisode:

    episode_id: str

    customer_id: str

    initial_policy_beliefs: dict

    final_policy_beliefs: dict

    interactions: list[StrategicInteraction]

    initial_expected_success: float

    final_expected_success: float

    length: int

    def prediction_accuracy(self) -> float:

        predictions = [
            x
            for x in self.interactions
            if x.prediction_correct is not None
        ]

        if not predictions:
            return 0.0

        return sum(
            x.prediction_correct
            for x in predictions
        ) / len(predictions)


# ============================================================
# STRATEGIC SIMULATOR
# ============================================================

class StrategicInteractionSimulator:

    def __init__(
        self,
        world: EnvironmentState,
        merchant=None,
        seed: int = 42,
    ):

        self.world = world

        self.merchant = (
            merchant
            if merchant is not None
            else world.merchant
        )

        self.rng = Random(seed)

        # ----------------------------------------------------
        # Agent A
        # ----------------------------------------------------

        self.support_agent = SupportAgent(
            merchant=self.merchant
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def run_episode(
        self,
        customer_id: str,
        interactions: int = 20,
        episode_number: int = 1,
    ) -> StrategicEpisode:

        customer = self.world.customers[
            customer_id
        ]

        orders = self._get_customer_orders(
            customer_id
        )

        if not orders:
            raise ValueError(
                f"No orders found for {customer_id}"
            )

        # ----------------------------------------------------
        # CREATE STRATEGIC B
        # ----------------------------------------------------

        state = self._create_agent_state(
            customer_id=customer_id
        )

        agent_b = StrategicCustomerAgent(
            state=state,
            seed=self.rng.randint(
                0,
                1_000_000,
            ),
        )

        initial_beliefs = (
            self._safe_policy_beliefs(
                agent_b
            )
        )

        initial_expected_success = (
            self._estimate_expected_success(
                agent_b
            )
        )

        trajectory = []

        # ----------------------------------------------------
        # INTERACTION LOOP
        # ----------------------------------------------------

        for sequence_number in range(
            1,
            interactions + 1,
        ):

            order = self.rng.choice(
                orders
            )

            # -----------------------------------------------
            # B chooses an action
            # -----------------------------------------------

            action = self._choose_action(
                agent_b=agent_b,
                customer=customer,
                order=order,
                sequence_number=sequence_number,
            )

            claim_type = action[
                "claim_type"
            ]

            requested_amount = action[
                "requested_amount"
            ]

            evidence = action[
                "evidence_available"
            ]

            # -----------------------------------------------
            # B predicts Agent A
            # -----------------------------------------------

            predicted_decision = (
                self._predict_agent_a(
                    agent_b=agent_b,
                    claim_type=claim_type,
                    requested_amount=requested_amount,
                    evidence_available=evidence,
                )
            )

            # -----------------------------------------------
            # Create request
            # -----------------------------------------------

            request = RefundRequest(
                request_id=make_id(
                    "REQUEST"
                ),

                customer_id=customer_id,

                order_id=order.order_id,

                claim_type=claim_type,

                claim_text=(
                    self._claim_text(
                        claim_type
                    )
                ),

                requested_amount=(
                    requested_amount
                ),

                submitted_at=(
                    datetime.now()
                    + timedelta(
                        minutes=sequence_number
                    )
                ),

                evidence_available=evidence,
            )

            # -----------------------------------------------
            # Agent A responds
            # -----------------------------------------------

            decision = (
                self.support_agent.decide(
                    customer=customer,
                    order=order,
                    request=request,
                )
            )

            # -----------------------------------------------
            # Check prediction
            # -----------------------------------------------

            prediction_correct = (
                predicted_decision
                == decision.decision
                if predicted_decision is not None
                else None
            )

            # -----------------------------------------------
            # B learns from A
            # -----------------------------------------------

            self._observe_agent_a(
                agent_b=agent_b,
                decision=decision,
                claim_type=claim_type,
                requested_amount=requested_amount,
                evidence_available=evidence,
            )

            # -----------------------------------------------
            # Store interaction
            # -----------------------------------------------

            trajectory.append(
                StrategicInteraction(

                    sequence_number=(
                        sequence_number
                    ),

                    claim_type=claim_type,

                    requested_amount=(
                        requested_amount
                    ),

                    evidence_available=evidence,

                    support_decision=(
                        decision.decision
                    ),

                    reason_code=(
                        decision.reason_code
                    ),

                    approved_amount=(
                        decision.approved_amount
                    ),

                    predicted_decision=(
                        predicted_decision
                    ),

                    prediction_correct=(
                        prediction_correct
                    ),
                )
            )

        # ----------------------------------------------------
        # FINAL STATE
        # ----------------------------------------------------

        final_beliefs = (
            self._safe_policy_beliefs(
                agent_b
            )
        )

        final_expected_success = (
            self._estimate_expected_success(
                agent_b
            )
        )

        return StrategicEpisode(

            episode_id=(
                f"STRATEGIC_EPISODE_"
                f"{episode_number:05d}"
            ),

            customer_id=customer_id,

            initial_policy_beliefs=(
                initial_beliefs
            ),

            final_policy_beliefs=(
                final_beliefs
            ),

            interactions=trajectory,

            initial_expected_success=(
                initial_expected_success
            ),

            final_expected_success=(
                final_expected_success
            ),

            length=len(trajectory),
        )

    # ========================================================
    # CUSTOMER ORDERS
    # ========================================================

    def _get_customer_orders(
        self,
        customer_id: str,
    ) -> list[OrderState]:

        return [
            order
            for order in self.world.orders.values()
            if order.customer_id == customer_id
        ]

    # ========================================================
    # AGENT STATE
    # ========================================================

    def _create_agent_state(
        self,
        customer_id: str,
    ):

        # StrategicCustomerAgent is responsible
        # for maintaining its own learning state.

        return StrategicCustomerAgent.create_state(
            customer_id=customer_id,
            objective=(
                "maximize_illegitimate_refund_value"
            ),
        )

    # ========================================================
    # STRATEGIC ACTION
    # ========================================================

    def _choose_action(
        self,
        agent_b: StrategicCustomerAgent,
        customer: CustomerObservableState,
        order: OrderState,
        sequence_number: int,
    ) -> dict:

        # Delegate strategic behavior to B.

        action = agent_b.choose_action(
            customer=customer,
            order=order,
            sequence_number=sequence_number,
        )

        return action

    # ========================================================
    # PREDICTION
    # ========================================================

    def _predict_agent_a(
        self,
        agent_b: StrategicCustomerAgent,
        claim_type: str,
        requested_amount: float,
        evidence_available: list[str],
    ) -> str | None:

        try:

            return agent_b.predict_support_decision(
                claim_type=claim_type,
                requested_amount=requested_amount,
                evidence_available=evidence_available,
            )

        except AttributeError:

            return None

    # ========================================================
    # LEARNING
    # ========================================================

    def _observe_agent_a(
        self,
        agent_b: StrategicCustomerAgent,
        decision: SupportDecision,
        claim_type: str,
        requested_amount: float,
        evidence_available: list[str],
    ):

        agent_b.observe_support_decision(
            decision=decision,
            claim_type=claim_type,
            requested_amount=requested_amount,
            evidence_available=evidence_available,
        )

    # ========================================================
    # POLICY BELIEFS
    # ========================================================

    def _safe_policy_beliefs(
        self,
        agent_b: StrategicCustomerAgent,
    ) -> dict:

        try:

            beliefs = (
                agent_b.get_inferred_policy()
            )

            return dict(beliefs)

        except AttributeError:

            return {}

    # ========================================================
    # EXPECTED SUCCESS
    # ========================================================

    def _estimate_expected_success(
        self,
        agent_b: StrategicCustomerAgent,
    ) -> float:

        try:

            return float(
                agent_b.expected_success_probability()
            )

        except AttributeError:

            return 0.0

    # ========================================================
    # CLAIM TEXT
    # ========================================================

    @staticmethod
    def _claim_text(
        claim_type: str,
    ) -> str:

        texts = {

            "SHORTAGE_CLAIM":
                "Some items were missing from the package.",

            "WRONG_ITEM_CLAIM":
                "The received item was different from what was ordered.",

            "NON_DELIVERY_CLAIM":
                "The order was not received.",

            "SUBSTITUTED_RETURN_CLAIM":
                "The returned item was substituted.",

        }

        return texts.get(
            claim_type,
            "I want to request a refund.",
        )


# ============================================================
# MULTI-EPISODE RUNNER
# ============================================================

def run_strategic_simulation(
    world: EnvironmentState,
    customer_ids: list[str] | None = None,
    n_episodes: int = 100,
    interactions_per_episode: int = 20,
    seed: int = 42,
) -> list[StrategicEpisode]:

    simulator = StrategicInteractionSimulator(
        world=world,
        seed=seed,
    )

    rng = Random(seed)

    # --------------------------------------------------------
    # Default: choose from customers that actually exist
    # --------------------------------------------------------

    if customer_ids is None:

        customer_ids = list(
            world.customers.keys()
        )

    if not customer_ids:

        raise ValueError(
            "No customers available for strategic simulation."
        )

    episodes = []

    for episode_number in range(
        1,
        n_episodes + 1,
    ):

        customer_id = rng.choice(
            customer_ids
        )

        episode = simulator.run_episode(
            customer_id=customer_id,
            interactions=(
                interactions_per_episode
            ),
            episode_number=episode_number,
        )

        episodes.append(
            episode
        )

    return episodes