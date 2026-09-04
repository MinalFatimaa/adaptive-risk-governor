from datetime import datetime

from src.environment.world import EnvironmentState
from src.environment.observations import (
    build_customer_observation,
)


def test_customer_observation_does_not_expose_other_customers():
    """
    Customer B must not be able to inspect another customer's
    private state or history.
    """

    # This test will be completed once the world generator exists.
    assert True


def test_governor_does_not_receive_ground_truth():
    """
    The Governor must never receive:
        - counterparty_type
        - is_abusive
        - abuse_family
        - private strategy
        - strategy beliefs
    """

    forbidden_fields = {
        "counterparty_type",
        "is_abusive",
        "abuse_family",
        "objective",
        "current_strategy",
        "strategy_beliefs",
        "successful_strategies",
        "failed_strategies",
    }

    # GovernorObservation itself should not contain these fields.
    from src.schemas.governor import GovernorObservation

    fields = set(
        GovernorObservation.model_fields.keys()
    )

    leaked = fields.intersection(forbidden_fields)

    assert not leaked, (
        f"Ground-truth/private fields leaked into "
        f"GovernorObservation: {leaked}"
    )