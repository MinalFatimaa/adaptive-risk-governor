from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class CustomerType(str, Enum):
    HUMAN = "human"
    ADAPTIVE_AGENT = "adaptive_agent"


class BehaviorClass(str, Enum):
    LEGITIMATE = "legitimate"
    ABUSIVE = "abusive"


class AbuseMechanism(str, Enum):
    SHORTAGE_CLAIM = "shortage_claim"
    WRONG_ITEM_CLAIM = "wrong_item_claim"
    NON_DELIVERY_CLAIM = "non_delivery_claim"
    SUBSTITUTED_RETURN = "substituted_return"


@dataclass(frozen=True)
class PopulationConfig:
    """
    Configuration for the merchant population simulation.

    IMPORTANT:
    These values are simulation parameters, not claims about
    real-world fraud prevalence.
    """

    # ---------------------------------------------------------
    # Population
    # ---------------------------------------------------------

    n_customers: int = 1000

    human_legitimate: int = 700
    human_abusive: int = 100

    adaptive_legitimate: int = 150
    adaptive_abusive: int = 50

    # ---------------------------------------------------------
    # Support environment
    # ---------------------------------------------------------

    n_support_agents: int = 8

    # ---------------------------------------------------------
    # Commerce volume
    # ---------------------------------------------------------

    target_orders: int = 6000

    simulation_days: int = 30

    # ---------------------------------------------------------
    # Infrastructure / relationship graph
    # ---------------------------------------------------------

    n_devices: int = 800
    n_addresses: int = 850
    n_payment_instruments: int = 900

    # ---------------------------------------------------------
    # Refund / dispute volume
    # ---------------------------------------------------------

    min_refund_episodes: int = 1500
    max_refund_episodes: int = 2500

    # ---------------------------------------------------------
    # Abuse mechanisms
    # ---------------------------------------------------------

    abuse_mechanisms: tuple[AbuseMechanism, ...] = (
        AbuseMechanism.SHORTAGE_CLAIM,
        AbuseMechanism.WRONG_ITEM_CLAIM,
        AbuseMechanism.NON_DELIVERY_CLAIM,
        AbuseMechanism.SUBSTITUTED_RETURN,
    )

    # ---------------------------------------------------------
    # Randomness
    # ---------------------------------------------------------

    seed: int = 42

    # ---------------------------------------------------------
    # Population sanity checks
    # ---------------------------------------------------------

    def validate(self) -> None:
        """Validate that the configuration is internally consistent."""

        assert self.n_customers > 0

        assert (
            self.human_legitimate
            + self.human_abusive
            + self.adaptive_legitimate
            + self.adaptive_abusive
            == self.n_customers
        ), (
            "Customer population counts must sum to n_customers."
        )

        assert self.n_support_agents > 0

        assert self.target_orders >= self.n_customers

        assert self.simulation_days > 0

        assert self.n_devices > 0
        assert self.n_addresses > 0
        assert self.n_payment_instruments > 0

        assert self.min_refund_episodes > 0
        assert self.max_refund_episodes >= self.min_refund_episodes

        assert len(self.abuse_mechanisms) > 0

    @property
    def human_customers(self) -> int:
        return self.human_legitimate + self.human_abusive

    @property
    def adaptive_customers(self) -> int:
        return self.adaptive_legitimate + self.adaptive_abusive

    @property
    def legitimate_customers(self) -> int:
        return self.human_legitimate + self.adaptive_legitimate

    @property
    def abusive_customers(self) -> int:
        return self.human_abusive + self.adaptive_abusive

    @property
    def abusive_rate(self) -> float:
        """
        Simulation parameter only.

        This must NOT be presented as the real-world prevalence
        of refund abuse.
        """
        return self.abusive_customers / self.n_customers

    @property
    def adaptive_rate(self) -> float:
        return self.adaptive_customers / self.n_customers


DEFAULT_POPULATION_CONFIG = PopulationConfig()

DEFAULT_POPULATION_CONFIG.validate()