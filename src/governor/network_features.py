from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime
from typing import Iterable


# ============================================================
# NETWORK OBSERVATION
# ============================================================

@dataclass(frozen=True)
class NetworkObservation:

    customer_id: str

    ip_address: str | None = None

    device_id: str | None = None

    payment_method_id: str | None = None

    shipping_address_id: str | None = None

    timestamp: datetime | None = None

    claim_type: str | None = None

    requested_amount: float = 0.0

    support_decision: str | None = None


# ============================================================
# NETWORK FEATURES
# ============================================================

class NetworkFeatureBuilder:

    """
    Builds Governor-only external observability features.

    These features are intentionally NOT exposed to Agent A.

    The builder does not decide whether a customer is fraudulent.
    It only converts raw network observations into behavioural
    signals.
    """

    def build(
        self,
        *,
        customer_id: str,
        observations: Iterable[NetworkObservation],
        current_ip: str | None = None,
        current_device_id: str | None = None,
        current_payment_method_id: str | None = None,
        current_shipping_address_id: str | None = None,
        current_claim_type: str | None = None,
        current_amount: float = 0.0,
        now: datetime | None = None,
    ) -> dict[str, float]:

        observations = list(observations)

        now = now or datetime.now()

        customer_observations = [
            x
            for x in observations
            if x.customer_id == customer_id
        ]

        # ----------------------------------------------------
        # Identity / network cardinality
        # ----------------------------------------------------

        customer_ips = self._unique(
            x.ip_address
            for x in customer_observations
        )

        customer_devices = self._unique(
            x.device_id
            for x in customer_observations
        )

        customer_payments = self._unique(
            x.payment_method_id
            for x in customer_observations
        )

        customer_addresses = self._unique(
            x.shipping_address_id
            for x in customer_observations
        )

        # ----------------------------------------------------
        # Cross-account sharing
        # ----------------------------------------------------

        shared_ip_accounts = self._linked_customers(
            observations,
            field="ip_address",
            value=current_ip,
            exclude_customer=customer_id,
        )

        shared_device_accounts = self._linked_customers(
            observations,
            field="device_id",
            value=current_device_id,
            exclude_customer=customer_id,
        )

        shared_payment_accounts = self._linked_customers(
            observations,
            field="payment_method_id",
            value=current_payment_method_id,
            exclude_customer=customer_id,
        )

        shared_address_accounts = self._linked_customers(
            observations,
            field="shipping_address_id",
            value=current_shipping_address_id,
            exclude_customer=customer_id,
        )

        # ----------------------------------------------------
        # Velocity
        # ----------------------------------------------------

        requests_24h = self._count_recent(
            customer_observations,
            now=now,
            hours=24,
        )

        requests_7d = self._count_recent(
            customer_observations,
            now=now,
            hours=24 * 7,
        )

        amount_24h = self._amount_recent(
            customer_observations,
            now=now,
            hours=24,
        )

        amount_7d = self._amount_recent(
            customer_observations,
            now=now,
            hours=24 * 7,
        )

        # ----------------------------------------------------
        # Network behaviour
        # ----------------------------------------------------

        network_observations = self._network_related(
            observations=observations,
            ip_address=current_ip,
            device_id=current_device_id,
            payment_method_id=current_payment_method_id,
            shipping_address_id=current_shipping_address_id,
        )

        network_customers = {
            x.customer_id
            for x in network_observations
        }

        network_refunds = sum(
            1
            for x in network_observations
            if x.support_decision == "APPROVE"
        )

        network_denials = sum(
            1
            for x in network_observations
            if x.support_decision == "DENY"
        )

        network_requests = len(
            network_observations
        )

        network_refund_rate = (
            network_refunds / network_requests
            if network_requests
            else 0.0
        )

        network_denial_rate = (
            network_denials / network_requests
            if network_requests
            else 0.0
        )

        # ----------------------------------------------------
        # Claim concentration
        # ----------------------------------------------------

        same_claim_count = sum(
            1
            for x in network_observations
            if (
                current_claim_type is not None
                and x.claim_type == current_claim_type
            )
        )

        claim_concentration = (
            same_claim_count / network_requests
            if network_requests
            else 0.0
        )

        # ----------------------------------------------------
        # Amount anomaly
        # ----------------------------------------------------

        network_amounts = [
            max(float(x.requested_amount), 0.0)
            for x in network_observations
            if x.requested_amount > 0
        ]

        average_network_amount = (
            sum(network_amounts)
            / len(network_amounts)
            if network_amounts
            else 0.0
        )

        amount_ratio = (
            current_amount / average_network_amount
            if average_network_amount > 0
            else 1.0
        )

        # ----------------------------------------------------
        # Shared identity score
        #
        # Sharing one attribute is not enough.
        # Multiple independent shared attributes increase
        # the signal.
        # ----------------------------------------------------

        shared_identity_count = sum(
            [
                int(bool(shared_ip_accounts)),
                int(bool(shared_device_accounts)),
                int(bool(shared_payment_accounts)),
                int(bool(shared_address_accounts)),
            ]
        )

        identity_link_strength = min(
            shared_identity_count / 4.0,
            1.0,
        )

        # ----------------------------------------------------
        # Network concentration
        # ----------------------------------------------------

        network_size_signal = min(
            len(network_customers) / 10.0,
            1.0,
        )

        # ----------------------------------------------------
        # Combined behavioural signal
        #
        # This deliberately requires multiple signals.
        # ----------------------------------------------------

        velocity_signal = min(
            requests_24h / 5.0,
            1.0,
        )

        claim_signal = claim_concentration

        amount_signal = min(
            max(amount_ratio - 1.0, 0.0) / 3.0,
            1.0,
        )

        network_behavior_score = min(
            1.0,
            (
                0.30 * identity_link_strength
                + 0.20 * network_size_signal
                + 0.20 * velocity_signal
                + 0.15 * claim_signal
                + 0.15 * amount_signal
            ),
        )

        return {
            # Identity
            "unique_ip_count": float(
                len(customer_ips)
            ),

            "unique_device_count": float(
                len(customer_devices)
            ),

            "unique_payment_method_count": float(
                len(customer_payments)
            ),

            "unique_address_count": float(
                len(customer_addresses)
            ),

            # Cross-account links
            "shared_ip_account_count": float(
                len(shared_ip_accounts)
            ),

            "shared_device_account_count": float(
                len(shared_device_accounts)
            ),

            "shared_payment_account_count": float(
                len(shared_payment_accounts)
            ),

            "shared_address_account_count": float(
                len(shared_address_accounts)
            ),

            # Velocity
            "requests_24h": float(
                requests_24h
            ),

            "requests_7d": float(
                requests_7d
            ),

            "refund_amount_24h": float(
                amount_24h
            ),

            "refund_amount_7d": float(
                amount_7d
            ),

            # Network behaviour
            "network_customer_count": float(
                len(network_customers)
            ),

            "network_request_count": float(
                network_requests
            ),

            "network_refund_rate": float(
                network_refund_rate
            ),

            "network_denial_rate": float(
                network_denial_rate
            ),

            "claim_concentration": float(
                claim_concentration
            ),

            "average_network_amount": float(
                average_network_amount
            ),

            "current_to_network_amount_ratio": float(
                amount_ratio
            ),

            # Derived signals
            "identity_link_strength": float(
                identity_link_strength
            ),

            "network_size_signal": float(
                network_size_signal
            ),

            "network_behavior_score": float(
                network_behavior_score
            ),
        }

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _unique(values):

        return {
            value
            for value in values
            if value is not None
        }

    @staticmethod
    def _linked_customers(
        observations,
        *,
        field: str,
        value: str | None,
        exclude_customer: str,
    ) -> set[str]:

        if value is None:
            return set()

        result = set()

        for observation in observations:

            if (
                getattr(observation, field, None)
                == value
                and observation.customer_id
                != exclude_customer
            ):

                result.add(
                    observation.customer_id
                )

        return result

    @staticmethod
    def _count_recent(
        observations,
        *,
        now: datetime,
        hours: int,
    ) -> int:

        count = 0

        for observation in observations:

            if observation.timestamp is None:
                continue

            delta = (
                now - observation.timestamp
            ).total_seconds()

            if (
                0 <= delta
                <= hours * 60 * 60
            ):

                count += 1

        return count

    @staticmethod
    def _amount_recent(
        observations,
        *,
        now: datetime,
        hours: int,
    ) -> float:

        amount = 0.0

        for observation in observations:

            if observation.timestamp is None:
                continue

            delta = (
                now - observation.timestamp
            ).total_seconds()

            if (
                0 <= delta
                <= hours * 60 * 60
            ):

                amount += max(
                    float(
                        observation.requested_amount
                    ),
                    0.0,
                )

        return amount

    @staticmethod
    def _network_related(
        *,
        observations,
        ip_address,
        device_id,
        payment_method_id,
        shipping_address_id,
    ):

        result = []

        for observation in observations:

            matches = False

            if (
                ip_address is not None
                and observation.ip_address
                == ip_address
            ):
                matches = True

            if (
                device_id is not None
                and observation.device_id
                == device_id
            ):
                matches = True

            if (
                payment_method_id is not None
                and observation.payment_method_id
                == payment_method_id
            ):
                matches = True

            if (
                shipping_address_id is not None
                and observation.shipping_address_id
                == shipping_address_id
            ):
                matches = True

            if matches:
                result.append(
                    observation
                )

        return result