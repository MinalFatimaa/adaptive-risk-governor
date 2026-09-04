from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..schemas.support import RefundRequest, SupportDecision
from .phase16_2_hardened_adaptive_evaluation import (
    AdaptiveRefundAgent,
    STRATEGIES,
)


# ============================================================================
# ATTACKER POLICY INTERFACE
# ============================================================================

@dataclass(frozen=True)
class AttackerRequestContext:
    """
    Context required to generate an attacker request.

    The context contains only information that the existing Phase 16.2
    AdaptiveRefundAgent legitimately receives when constructing a request.
    """

    customer: Any
    order: Any
    episode_id: str
    sequence_number: int
    current_time: datetime


class AttackerPolicy:
    """
    Explicit attacker-policy interface.

    This interface exists so Phase 18+ can select an attacker policy
    without changing the underlying attacker implementation.

    IMPORTANT:
        This interface does not define a new attacker action space.

    The actual Phase 16.2 attacker works in terms of strategy selection
    and RefundRequest generation.
    """

    def choose_strategy(
        self,
        interaction_number: int,
    ) -> str:
        raise NotImplementedError

    def build_request(
        self,
        strategy: str,
        context: AttackerRequestContext,
    ) -> RefundRequest:
        raise NotImplementedError

    def observe(
        self,
        strategy: str,
        decision: SupportDecision,
    ) -> None:
        raise NotImplementedError

    def inferred_policy(self) -> dict[str, float]:
        raise NotImplementedError

    @property
    def rng(self):
        raise NotImplementedError


# ============================================================================
# EXISTING PHASE 16.2 ADAPTIVE ATTACKER ADAPTER
# ============================================================================

class ExistingAdaptiveAttackerPolicy(AttackerPolicy):
    """
    Thin adapter around the REAL Phase 16.2 AdaptiveRefundAgent.

    No learning logic is duplicated here.

    Delegated behavior:
        - seven strategy arms
        - exploration schedule
        - UCB-like exploitation
        - RNG
        - claim selection
        - amount selection
        - evidence selection
        - request construction
        - reward calculation
        - belief updates
        - inferred strategy distribution

    Therefore this class is an integration boundary, not a second
    implementation of the attacker.
    """

    def __init__(
        self,
        *,
        customer_id: str,
        objective: str,
        seed: int,
    ):
        self.agent = AdaptiveRefundAgent(
            customer_id=customer_id,
            objective=objective,
            seed=seed,
        )

        self._initial_policy = (
            self.agent.inferred_policy().copy()
        )

    # ------------------------------------------------------------------
    # Direct delegation
    # ------------------------------------------------------------------

    def choose_strategy(
        self,
        interaction_number: int,
    ) -> str:

        return self.agent.choose_strategy(
            interaction_number=interaction_number
        )

    def build_request(
        self,
        strategy: str,
        context: AttackerRequestContext,
    ) -> RefundRequest:

        return self.agent.build_request(
            strategy=strategy,
            customer=context.customer,
            order=context.order,
            episode_id=context.episode_id,
            sequence_number=context.sequence_number,
            current_time=context.current_time,
        )

    def observe(
        self,
        strategy: str,
        decision: SupportDecision,
    ) -> None:

        self.agent.observe(
            strategy=strategy,
            decision=decision,
        )

    def inferred_policy(self) -> dict[str, float]:

        return self.agent.inferred_policy().copy()

    @property
    def rng(self):
        return self.agent.rng

    # ------------------------------------------------------------------
    # Useful integration accessors
    # ------------------------------------------------------------------

    @property
    def customer_id(self) -> str:
        return self.agent.customer_id

    @property
    def objective(self) -> str:
        return self.agent.objective

    @property
    def history(self):
        return self.agent.history

    @property
    def strategy_counts(self):
        return self.agent.strategy_counts

    @property
    def beliefs(self):
        return self.agent.beliefs

    def policy_change(self) -> float:
        """
        Total-variation distance between the initial strategy distribution
        and the current inferred strategy distribution.

        Range: [0, 1]
        """

        current_policy = self.inferred_policy()

        return 0.5 * sum(
            abs(
                current_policy.get(strategy, 0.0)
                - self._initial_policy.get(strategy, 0.0)
            )
            for strategy in STRATEGIES
        )


# ============================================================================
# FACTORY
# ============================================================================

def create_existing_adaptive_attacker(
    *,
    customer_id: str,
    population_group: str,
    seed: int,
) -> ExistingAdaptiveAttackerPolicy:
    """
    Create the existing Phase 16.2 adaptive attacker through the
    explicit policy interface.

    The objective strings are the SAME objective strings used by
    Phase 16.2 itself.
    """

    if population_group == "ADAPTIVE_ABUSIVE":
        objective = (
            "maximize_illegitimate_refund_value"
        )

    else:
        objective = (
            "maximize_successful_legitimate_resolution"
        )

    return ExistingAdaptiveAttackerPolicy(
        customer_id=customer_id,
        objective=objective,
        seed=seed,
    )


# ============================================================================
# CONTRACT VALIDATION
# ============================================================================

def validate_attacker_policy_contract() -> None:
    """
    Structural validation only.

    This verifies that the explicit policy interface is actually backed
    by the Phase 16.2 AdaptiveRefundAgent.

    It does NOT claim behavioral equivalence by itself.
    """

    policy = ExistingAdaptiveAttackerPolicy(
        customer_id="CONTRACT_TEST_CUSTOMER",
        objective="maximize_illegitimate_refund_value",
        seed=42,
    )

    assert isinstance(
        policy.agent,
        AdaptiveRefundAgent,
    )

    assert tuple(
        policy.beliefs.keys()
    ) == STRATEGIES

    assert set(
        policy.strategy_counts.keys()
    ) == set(STRATEGIES)

    assert policy.inferred_policy() == {
        strategy: 0.0
        for strategy in STRATEGIES
    }

    assert policy.policy_change() == 0.0


# ============================================================================
# BEHAVIORAL EQUIVALENCE VALIDATION
# ============================================================================

def validate_adapter_behavioral_equivalence() -> None:
    """
    Compare the adapter directly against the underlying Phase 16.2
    AdaptiveRefundAgent.

    Both receive:
        - identical seed
        - identical customer
        - identical order
        - identical episode context
        - identical SupportDecision observations

    Their:
        - selected strategies
        - generated requests
        - RNG-driven behavior
        - inferred policy

    must remain identical.

    The customer selection is deliberately taken from the same
    population-selection mechanism used by Phase 16.2 rather than
    assuming a field exists on CustomerPrivateState.
    """

    from ..environment.population_config import (
        DEFAULT_POPULATION_CONFIG,
    )
    from ..environment.world_generator import (
        create_world,
    )

    world = create_world(
        DEFAULT_POPULATION_CONFIG
    )

    # ------------------------------------------------------------------
    # Find an ADAPTIVE_ABUSIVE customer using the actual customer
    # representation used by the Phase 16.2 world.
    #
    # We deliberately do NOT assume that CustomerPrivateState has a
    # `population_group` field.
    # ------------------------------------------------------------------

    adaptive_abusive_ids: list[str] = []

    for customer_id, customer in world.customers.items():

        customer_text = str(customer).upper()

        private_state = world.customer_private.get(
            customer_id
        )

        private_text = (
            str(private_state).upper()
            if private_state is not None
            else ""
        )

        combined_text = (
            customer_text
            + " "
            + private_text
        )

        if (
            "ADAPTIVE_ABUSIVE" in combined_text
            or (
                "ADAPTIVE" in combined_text
                and "ABUSIVE" in combined_text
            )
        ):
            adaptive_abusive_ids.append(
                customer_id
            )

    if not adaptive_abusive_ids:
        raise AssertionError(
            "Could not identify an ADAPTIVE_ABUSIVE "
            "customer from the generated world. "
            "The equivalence test will not guess the "
            "CustomerPrivateState schema."
        )

    customer_id = adaptive_abusive_ids[0]

    customer = world.customers[
        customer_id
    ]

    # ------------------------------------------------------------------
    # Create the actual Phase 16.2 implementation and the adapter with
    # exactly the same configuration.
    # ------------------------------------------------------------------

    direct_agent = AdaptiveRefundAgent(
        customer_id=customer_id,
        objective="maximize_illegitimate_refund_value",
        seed=42,
    )

    adapter = ExistingAdaptiveAttackerPolicy(
        customer_id=customer_id,
        objective="maximize_illegitimate_refund_value",
        seed=42,
    )

    current_time = datetime(
        2026,
        8,
        31,
        12,
        0,
        0,
    )

    # ------------------------------------------------------------------
    # Run both implementations in lockstep.
    # ------------------------------------------------------------------

    for sequence_number in range(1, 9):

        direct_strategy = (
            direct_agent.choose_strategy(
                interaction_number=sequence_number
            )
        )

        adapter_strategy = (
            adapter.choose_strategy(
                interaction_number=sequence_number
            )
        )

        assert (
            direct_strategy
            == adapter_strategy
        ), (
            "Strategy divergence at interaction "
            f"{sequence_number}: "
            f"{direct_strategy!r} != "
            f"{adapter_strategy!r}"
        )

        # Both implementations have independent Random instances
        # initialized with the same seed. If the adapter delegates
        # faithfully, these RNG calls must remain synchronized.

        direct_order_id = direct_agent.rng.choice(
            customer.current_order_ids
        )

        adapter_order_id = adapter.rng.choice(
            customer.current_order_ids
        )

        assert (
            direct_order_id
            == adapter_order_id
        ), (
            "RNG/order-selection divergence at "
            f"interaction {sequence_number}."
        )

        order = world.orders[
            direct_order_id
        ]

        context = AttackerRequestContext(
            customer=customer,
            order=order,
            episode_id="EQUIVALENCE_TEST",
            sequence_number=sequence_number,
            current_time=current_time,
        )

        direct_request = (
            direct_agent.build_request(
                strategy=direct_strategy,
                customer=customer,
                order=order,
                episode_id="EQUIVALENCE_TEST",
                sequence_number=sequence_number,
                current_time=current_time,
            )
        )

        adapter_request = (
            adapter.build_request(
                strategy=adapter_strategy,
                context=context,
            )
        )

        assert (
            direct_request.request_id
            == adapter_request.request_id
        )

        assert (
            direct_request.customer_id
            == adapter_request.customer_id
        )

        assert (
            direct_request.order_id
            == adapter_request.order_id
        )

        assert (
            direct_request.claim_type
            == adapter_request.claim_type
        )

        assert (
            direct_request.claim_text
            == adapter_request.claim_text
        )

        assert (
            direct_request.requested_amount
            == adapter_request.requested_amount
        )

        assert (
            direct_request.submitted_at
            == adapter_request.submitted_at
        )

        assert (
            direct_request.evidence_available
            == adapter_request.evidence_available
        )

        # --------------------------------------------------------------
        # Feed the SAME observable SupportDecision to both agents.
        #
        # This isolates the adapter comparison to attacker behavior.
        # We are not testing Governor behavior here.
        # --------------------------------------------------------------

        decision = SupportDecision(
            decision="APPROVE",
            reason_code="EQUIVALENCE_TEST",
            requested_evidence=[],
            approved_amount=float(
                direct_request.requested_amount
            ),
            requires_followup=False,
        )

        direct_agent.observe(
            strategy=direct_strategy,
            decision=decision,
        )

        adapter.observe(
            strategy=adapter_strategy,
            decision=decision,
        )

        assert (
            direct_agent.inferred_policy()
            == adapter.inferred_policy()
        ), (
            "Policy divergence at interaction "
            f"{sequence_number}."
        )

        # --------------------------------------------------------------
        # Keep the RNG streams synchronized by making the same RNG call
        # on both implementations.
        # --------------------------------------------------------------

        direct_time_delta = direct_agent.rng.randint(
            1,
            72,
        )

        adapter_time_delta = adapter.rng.randint(
            1,
            72,
        )

        assert (
            direct_time_delta
            == adapter_time_delta
        ), (
            "RNG/time-transition divergence at "
            f"interaction {sequence_number}."
        )

        current_time += timedelta(
            hours=direct_time_delta
        )

    # Final state must still be identical.
    assert (
        direct_agent.inferred_policy()
        == adapter.inferred_policy()
    ), (
        "Final inferred policy differs between the "
        "Phase 16.2 attacker and its adapter."
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print(
        "Validating attacker-policy contract..."
    )

    validate_attacker_policy_contract()

    print(
        "PASS: attacker-policy contract."
    )

    print(
        "Validating behavioral equivalence against "
        "Phase 16.2 AdaptiveRefundAgent..."
    )

    validate_adapter_behavioral_equivalence()

    print(
        "PASS: behavioral equivalence."
    )

    print()
    print("=" * 72)
    print(
        "ATTACKER POLICY INTERFACE VALIDATION PASSED"
    )
    print("=" * 72)
    print(
        "Source of truth: "
        "AdaptiveRefundAgent"
    )
    print(
        "Adapter: ExistingAdaptiveAttackerPolicy"
    )
    print(
        f"Strategy arms: {len(STRATEGIES)}"
    )
    print(
        "Phase 18 modification: NONE"
    )


if __name__ == "__main__":
    main()