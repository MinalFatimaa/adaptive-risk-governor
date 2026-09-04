from __future__ import annotations

from typing import Any

from src.governor.network_features import (
    NetworkFeatureBuilder,
    NetworkObservation,
)


# ============================================================
# GOVERNOR FEATURE BUILDER
# ============================================================

class GovernorFeatureBuilder:
    """
    Builds the complete set of external Governor features.

    This builder combines:

        1. Strategic/adaptive behavior features
        2. Network/coordination features

    These features are visible to the Risk Governor/Fusion
    layer but are NOT exposed to Agent A.

    Important:

        Agent A
            |
            | private policy
            v
        Support decision

        External observability
            |
            v
        GovernorFeatureBuilder
            |
            v
        Risk Governor / Risk Fusion
    """

    def __init__(
        self,
        network_builder: NetworkFeatureBuilder | None = None,
    ) -> None:

        self.network_builder = (
            network_builder
            if network_builder is not None
            else NetworkFeatureBuilder()
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def build(
        self,
        *,
        customer_id: str,
        customer_history: list[dict[str, Any]],
        current_claim_type: str,
        requested_amount: float,
        evidence_available: list[str],
        strategic_state: Any | None = None,
        network_observations: list[
            NetworkObservation
        ] | None = None,
        current_ip: str | None = None,
        current_device_id: str | None = None,
        current_payment_method_id: str | None = None,
        current_shipping_address_id: str | None = None,
    ) -> dict[str, float]:
        """
        Build all external Governor features.

        Returns
        -------
        dict[str, float]
            A flat dictionary containing strategic and
            network/observability features.
        """

        network_observations = (
            network_observations
            if network_observations is not None
            else []
        )

        features: dict[str, float] = {}

        # ====================================================
        # 1. STRATEGIC FEATURES
        # ====================================================

        (
            strategic_features
        ) = self._build_strategic_features(
            strategic_state=strategic_state,
        )

        features.update(
            strategic_features
        )

        # ====================================================
        # 2. NETWORK FEATURES
        # ====================================================

        network_features = (
            self.network_builder.build(
                customer_id=customer_id,
                observations=network_observations,
                current_ip=current_ip,
                current_device_id=current_device_id,
                current_payment_method_id=(
                    current_payment_method_id
                ),
                current_shipping_address_id=(
                    current_shipping_address_id
                ),
                current_claim_type=current_claim_type,
                current_amount=float(
                    requested_amount
                ),
            )
        )

        features.update(
            network_features
        )

        return features

    # ========================================================
    # STRATEGIC FEATURE BUILDER
    # ========================================================

    @staticmethod
    def _build_strategic_features(
        *,
        strategic_state: Any | None,
    ) -> dict[str, float]:
        """
        Extract strategic-agent beliefs and convert them
        into Governor features.

        Defaults are neutral (0.5).
        """

        evidence_sensitivity = 0.5
        amount_sensitivity = 0.5
        high_value_escalation = 0.5
        followup_sensitivity = 0.5

        strategic_history: list[Any] = []

        # ----------------------------------------------------
        # Extract learned policy
        # ----------------------------------------------------

        if strategic_state is not None:

            beliefs = getattr(
                strategic_state,
                "inferred_policy",
                {},
            )

            if not isinstance(
                beliefs,
                dict,
            ):
                beliefs = {}

            evidence_sensitivity = (
                GovernorFeatureBuilder._bounded(
                    beliefs.get(
                        "evidence_sensitivity",
                        0.5,
                    )
                )
            )

            amount_sensitivity = (
                GovernorFeatureBuilder._bounded(
                    beliefs.get(
                        "amount_sensitivity",
                        0.5,
                    )
                )
            )

            high_value_escalation = (
                GovernorFeatureBuilder._bounded(
                    beliefs.get(
                        "high_value_escalation",
                        0.5,
                    )
                )
            )

            followup_sensitivity = (
                GovernorFeatureBuilder._bounded(
                    beliefs.get(
                        "followup_sensitivity",
                        0.5,
                    )
                )
            )

            strategic_history = getattr(
                strategic_state,
                "interaction_history",
                [],
            )

            if strategic_history is None:
                strategic_history = []

        # ====================================================
        # LEARNING SIGNAL
        # ====================================================

        learning_signal = min(
            len(strategic_history) / 20.0,
            1.0,
        )

        # ====================================================
        # POLICY-BELIEF SHIFT
        # ====================================================

        belief_values = (
            evidence_sensitivity,
            amount_sensitivity,
            high_value_escalation,
            followup_sensitivity,
        )

        belief_shift = min(
            sum(
                abs(value - 0.5)
                for value in belief_values
            ) / 2.0,
            1.0,
        )

        # ====================================================
        # STRATEGIC ADAPTATION
        # ====================================================

        strategic_adaptation_score = min(
            1.0,
            (
                0.5 * learning_signal
                + 0.5 * belief_shift
            ),
        )

        return {
            "evidence_sensitivity": (
                evidence_sensitivity
            ),
            "amount_sensitivity": (
                amount_sensitivity
            ),
            "high_value_escalation": (
                high_value_escalation
            ),
            "followup_sensitivity": (
                followup_sensitivity
            ),
            "strategic_adaptation_score": (
                strategic_adaptation_score
            ),
        }

    # ========================================================
    # FEATURE ACCESSOR
    # ========================================================

    @staticmethod
    def get_feature(
        features: dict[str, float],
        name: str,
        default: float = 0.0,
    ) -> float:
        """
        Safely retrieve a Governor feature.

        Useful for downstream Governor/Fusion components.
        """

        try:

            value = float(
                features.get(
                    name,
                    default,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return default

        return GovernorFeatureBuilder._bounded(
            value
        )

    # ========================================================
    # BOUND VALUE
    # ========================================================

    @staticmethod
    def _bounded(
        value: Any,
    ) -> float:
        """
        Convert a value to a float in [0, 1].
        """

        try:

            value = float(value)

        except (
            TypeError,
            ValueError,
        ):

            return 0.5

        return max(
            0.0,
            min(
                value,
                1.0,
            ),
        )


# ============================================================
# MODULE SELF-CHECK
# ============================================================

if __name__ == "__main__":

    builder = GovernorFeatureBuilder()

    features = builder.build(
        customer_id="CUSTOMER_0001",
        customer_history=[],
        current_claim_type="ITEM_NOT_RECEIVED",
        requested_amount=1000.0,
        evidence_available=[],
        strategic_state=None,
        network_observations=[],
    )

    print("=" * 60)
    print("GOVERNOR FEATURE BUILDER SELF-CHECK")
    print("=" * 60)

    for name, value in features.items():
        print(
            f"{name:35s}: {value:.3f}"
        )

    print("=" * 60)
    print("SELF-CHECK PASSED")
    print("=" * 60)