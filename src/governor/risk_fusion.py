from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ============================================================
# RISK LEVELS
# ============================================================

RISK_LEVELS = (
    "LOW",
    "MEDIUM",
    "HIGH",
)


# ============================================================
# FUSED RISK DECISION
# ============================================================

@dataclass
class FusedRiskDecision:
    """
    Final decision produced by the Risk Fusion layer.

    The fusion layer combines four independent signals:

        1. Behavioral risk
        2. Strategic/adaptation risk
        3. External network/observability risk
        4. GNN graph-based risk

    The GNN is treated as an additional risk signal.

    It does NOT independently make the final decision.
    """

    behavioral_risk: float

    strategic_risk: float

    network_risk: float

    gnn_risk: float

    fused_risk_score: float

    risk_level: str

    reason_codes: list[str] = field(
        default_factory=list
    )

    action: str = "ALLOW_AGENT_A_DECISION"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# RISK FUSION ENGINE
# ============================================================

class RiskFusionEngine:
    """
    Combines:

        Behavioral Risk
        Strategic Risk
        Network Risk
        GNN Risk

    into one final Governor risk score.

    Architecture:

        Behavioral Signals
               |
               v
        Adaptive Governor
               |
               | behavioral risk
               |
               +--------------------+
                                    |
        Strategic State             |
               |                    |
               v                    |
        strategic risk              |
                                    |
        Network Observability       |
               |                    |
               v                    |
          network risk              |
                                    |
        Interaction Graph           |
               |                    |
               v                    |
            GNN risk                |
                                    |
               +--------------------+
                            |
                            v
                     Risk Fusion
                            |
                            v
                  Final Risk Decision


    Default weights:

        behavioral = 0.30
        strategic  = 0.20
        network    = 0.30
        gnn        = 0.20

    Total:

        0.30 + 0.20 + 0.30 + 0.20 = 1.00
    """

    def __init__(
        self,
        behavioral_weight: float = 0.30,
        strategic_weight: float = 0.20,
        network_weight: float = 0.30,
        gnn_weight: float = 0.20,
        low_risk_threshold: float = 0.30,
        high_risk_threshold: float = 0.70,
        coordination_threshold: float = 0.70,
        coordination_floor: float = 0.75,
        gnn_high_risk_threshold: float = 0.70,
    ):

        # ----------------------------------------------------
        # Validate weights
        # ----------------------------------------------------

        total_weight = (
            behavioral_weight
            + strategic_weight
            + network_weight
            + gnn_weight
        )

        if abs(total_weight - 1.0) > 1e-9:

            raise ValueError(
                "Risk fusion weights must sum to 1.0."
            )

        if any(
            weight < 0.0
            for weight in (
                behavioral_weight,
                strategic_weight,
                network_weight,
                gnn_weight,
            )
        ):

            raise ValueError(
                "Risk fusion weights cannot be negative."
            )

        # ----------------------------------------------------
        # Validate thresholds
        # ----------------------------------------------------

        if not (
            0.0
            <= low_risk_threshold
            < high_risk_threshold
            <= 1.0
        ):

            raise ValueError(
                "Risk thresholds must satisfy "
                "0 <= low < high <= 1."
            )

        if not (
            0.0
            <= coordination_threshold
            <= 1.0
        ):

            raise ValueError(
                "coordination_threshold must "
                "be between 0 and 1."
            )

        if not (
            0.0
            <= coordination_floor
            <= 1.0
        ):

            raise ValueError(
                "coordination_floor must "
                "be between 0 and 1."
            )

        if not (
            0.0
            <= gnn_high_risk_threshold
            <= 1.0
        ):

            raise ValueError(
                "gnn_high_risk_threshold must "
                "be between 0 and 1."
            )

        # ----------------------------------------------------
        # Store configuration
        # ----------------------------------------------------

        self.behavioral_weight = (
            behavioral_weight
        )

        self.strategic_weight = (
            strategic_weight
        )

        self.network_weight = (
            network_weight
        )

        self.gnn_weight = (
            gnn_weight
        )

        self.low_risk_threshold = (
            low_risk_threshold
        )

        self.high_risk_threshold = (
            high_risk_threshold
        )

        self.coordination_threshold = (
            coordination_threshold
        )

        self.coordination_floor = (
            coordination_floor
        )

        self.gnn_high_risk_threshold = (
            gnn_high_risk_threshold
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def fuse(
        self,
        *,
        governor_decision: Any,
        network_features: dict[str, float] | None = None,
        gnn_risk: float = 0.0,
    ) -> FusedRiskDecision:
        """
        Fuse Governor, network and GNN signals.

        Parameters
        ----------
        governor_decision:
            Existing GovernorDecision.

            Expected:

                governor_decision.risk_score

            and optionally:

                governor_decision.features[
                    "strategic_adaptation_score"
                ]

        network_features:
            Network/observability features.

        gnn_risk:
            Risk score generated by the GNN.

            Must be in [0, 1].

        The GNN does not independently make a decision.
        It contributes to the final fused score.
        """

        # ----------------------------------------------------
        # Safe defaults
        # ----------------------------------------------------

        network_features = (
            network_features or {}
        )

        # ----------------------------------------------------
        # Behavioral risk
        # ----------------------------------------------------

        behavioral_risk = self._bounded(
            getattr(
                governor_decision,
                "risk_score",
                0.0,
            )
        )

        # ----------------------------------------------------
        # Strategic risk
        # ----------------------------------------------------

        strategic_risk = (
            self._extract_strategic_risk(
                governor_decision
            )
        )

        # ----------------------------------------------------
        # Network risk
        # ----------------------------------------------------

        network_risk = (
            self.calculate_network_risk(
                network_features
            )
        )

        # ----------------------------------------------------
        # GNN risk
        # ----------------------------------------------------

        gnn_risk = self._bounded(
            gnn_risk
        )

        # ----------------------------------------------------
        # Network coordination
        # ----------------------------------------------------

        coordination_detected = (
            network_risk
            >= self.coordination_threshold
        )

        # ----------------------------------------------------
        # GNN high-risk signal
        # ----------------------------------------------------

        high_gnn_risk = (
            gnn_risk
            >= self.gnn_high_risk_threshold
        )

        # ----------------------------------------------------
        # Standard weighted fusion
        # ----------------------------------------------------
        #
        # behavioral * 0.30
        # strategic  * 0.20
        # network    * 0.30
        # gnn        * 0.20
        #
        # Total = 1.00
        #

        weighted_score = (
            self.behavioral_weight
            * behavioral_risk
            +
            self.strategic_weight
            * strategic_risk
            +
            self.network_weight
            * network_risk
            +
            self.gnn_weight
            * gnn_risk
        )

        # ----------------------------------------------------
        # Coordinated-network protection
        # ----------------------------------------------------
        #
        # A coordinated network should not be allowed to hide
        # behind a low individual behavioral score.
        #
        # Example:
        #
        # behavioral = 0.20
        # strategic  = 0.10
        # network    = 0.90
        # gnn        = 0.15
        #
        # The weighted score may still be below HIGH.
        #
        # Therefore a strong network coordination signal
        # establishes a minimum risk floor.
        #

        if coordination_detected:

            fused_score = max(
                weighted_score,
                self.coordination_floor,
            )

        else:

            fused_score = weighted_score

        # ----------------------------------------------------
        # Bound final score
        # ----------------------------------------------------

        fused_score = self._bounded(
            fused_score
        )

        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        risk_level = self._risk_level(
            fused_score
        )

        # ----------------------------------------------------
        # Reason codes
        # ----------------------------------------------------

        reason_codes = (
            self._build_reason_codes(
                network_features=network_features,
                network_risk=network_risk,
                coordination_detected=(
                    coordination_detected
                ),
                gnn_risk=gnn_risk,
                high_gnn_risk=high_gnn_risk,
            )
        )

        # ----------------------------------------------------
        # Recommended action
        # ----------------------------------------------------

        action = (
            self._recommended_action(
                risk_level
            )
        )

        # ----------------------------------------------------
        # Return final decision
        # ----------------------------------------------------

        return FusedRiskDecision(

            behavioral_risk=(
                behavioral_risk
            ),

            strategic_risk=(
                strategic_risk
            ),

            network_risk=(
                network_risk
            ),

            gnn_risk=(
                gnn_risk
            ),

            fused_risk_score=(
                fused_score
            ),

            risk_level=(
                risk_level
            ),

            reason_codes=(
                reason_codes
            ),

            action=(
                action
            ),

            metadata={

                "behavioral_weight": (
                    self.behavioral_weight
                ),

                "strategic_weight": (
                    self.strategic_weight
                ),

                "network_weight": (
                    self.network_weight
                ),

                "gnn_weight": (
                    self.gnn_weight
                ),

                "coordination_threshold": (
                    self.coordination_threshold
                ),

                "coordination_floor": (
                    self.coordination_floor
                ),

                "gnn_high_risk_threshold": (
                    self.gnn_high_risk_threshold
                ),

                "coordination_detected": (
                    coordination_detected
                ),

                "high_gnn_risk": (
                    high_gnn_risk
                ),
            },
        )

    # ========================================================
    # NETWORK RISK
    # ========================================================

    def calculate_network_risk(
        self,
        network_features: dict[str, float],
    ) -> float:
        """
        Convert raw network/observability features into one
        bounded network-risk score in [0, 1].

        Features:

            ip_reuse_score
            device_reuse_score
            payment_reuse_score
            address_reuse_score
            refund_velocity_score
            claim_similarity_score
            network_abnormality_score

        Equal weighting is used for the initial heuristic
        implementation.
        """

        feature_names = (

            "ip_reuse_score",

            "device_reuse_score",

            "payment_reuse_score",

            "address_reuse_score",

            "refund_velocity_score",

            "claim_similarity_score",

            "network_abnormality_score",
        )

        values = []

        for name in feature_names:

            value = network_features.get(
                name,
                0.0,
            )

            values.append(
                self._bounded(
                    value
                )
            )

        if not values:

            return 0.0

        score = (
            sum(values)
            / len(values)
        )

        return self._bounded(
            score
        )

    # ========================================================
    # STRATEGIC RISK EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_strategic_risk(
        governor_decision: Any,
    ) -> float:
        """
        Extract strategic adaptation risk.

        Preferred location:

            decision.features[
                "strategic_adaptation_score"
            ]

        Falls back safely to 0.0.
        """

        features = getattr(
            governor_decision,
            "features",
            {},
        )

        if not isinstance(
            features,
            dict,
        ):

            return 0.0

        value = features.get(
            "strategic_adaptation_score",
            0.0,
        )

        try:

            return RiskFusionEngine._bounded(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

    # ========================================================
    # REASON CODES
    # ========================================================

    def _build_reason_codes(
        self,
        *,
        network_features: dict[str, float],
        network_risk: float,
        coordination_detected: bool,
        gnn_risk: float,
        high_gnn_risk: bool,
    ) -> list[str]:

        reasons: list[str] = []

        # ----------------------------------------------------
        # Individual network signals
        # ----------------------------------------------------

        ip_reuse = self._feature(
            network_features,
            "ip_reuse_score",
        )

        device_reuse = self._feature(
            network_features,
            "device_reuse_score",
        )

        payment_reuse = self._feature(
            network_features,
            "payment_reuse_score",
        )

        address_reuse = self._feature(
            network_features,
            "address_reuse_score",
        )

        refund_velocity = self._feature(
            network_features,
            "refund_velocity_score",
        )

        claim_similarity = self._feature(
            network_features,
            "claim_similarity_score",
        )

        abnormality = self._feature(
            network_features,
            "network_abnormality_score",
        )

        # ----------------------------------------------------
        # High IP reuse
        # ----------------------------------------------------

        if ip_reuse >= 0.70:

            reasons.append(
                "HIGH_IP_REUSE"
            )

        # ----------------------------------------------------
        # High device reuse
        # ----------------------------------------------------

        if device_reuse >= 0.70:

            reasons.append(
                "HIGH_DEVICE_REUSE"
            )

        # ----------------------------------------------------
        # High payment reuse
        # ----------------------------------------------------

        if payment_reuse >= 0.70:

            reasons.append(
                "HIGH_PAYMENT_REUSE"
            )

        # ----------------------------------------------------
        # High address reuse
        # ----------------------------------------------------

        if address_reuse >= 0.70:

            reasons.append(
                "HIGH_ADDRESS_REUSE"
            )

        # ----------------------------------------------------
        # High refund velocity
        # ----------------------------------------------------

        if refund_velocity >= 0.70:

            reasons.append(
                "HIGH_REFUND_VELOCITY"
            )

        # ----------------------------------------------------
        # High claim similarity
        # ----------------------------------------------------

        if claim_similarity >= 0.70:

            reasons.append(
                "HIGH_CLAIM_SIMILARITY"
            )

        # ----------------------------------------------------
        # Network abnormality
        # ----------------------------------------------------

        if abnormality >= 0.70:

            reasons.append(
                "NETWORK_ABNORMALITY"
            )

        # ----------------------------------------------------
        # Coordinated behavior
        # ----------------------------------------------------

        if coordination_detected:

            reasons.append(
                "COORDINATED_NETWORK_BEHAVIOR"
            )

        # ----------------------------------------------------
        # Overall network risk
        # ----------------------------------------------------

        if network_risk >= 0.70:

            reasons.append(
                "HIGH_NETWORK_RISK"
            )

        # ----------------------------------------------------
        # GNN risk
        # ----------------------------------------------------

        if high_gnn_risk:

            reasons.append(
                "HIGH_GNN_RISK"
            )

        # ----------------------------------------------------
        # Very high GNN risk
        # ----------------------------------------------------
        #
        # This is deliberately separate from
        # HIGH_GNN_RISK so the system can distinguish
        # an extreme graph-based signal.
        #

        if gnn_risk >= 0.90:

            reasons.append(
                "VERY_HIGH_GNN_RISK"
            )

        return reasons

    # ========================================================
    # FEATURE HELPER
    # ========================================================

    @staticmethod
    def _feature(
        features: dict[str, float],
        name: str,
    ) -> float:

        try:

            return RiskFusionEngine._bounded(
                float(
                    features.get(
                        name,
                        0.0,
                    )
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

    # ========================================================
    # RISK LEVEL
    # ========================================================

    def _risk_level(
        self,
        score: float,
    ) -> str:

        if score < self.low_risk_threshold:

            return "LOW"

        if score < self.high_risk_threshold:

            return "MEDIUM"

        return "HIGH"

    # ========================================================
    # ACTION
    # ========================================================

    @staticmethod
    def _recommended_action(
        risk_level: str,
    ) -> str:

        if risk_level == "LOW":

            return (
                "ALLOW_AGENT_A_DECISION"
            )

        if risk_level == "MEDIUM":

            return (
                "REQUEST_ADDITIONAL_EVIDENCE"
            )

        return (
            "ESCALATE_TO_HUMAN_REVIEW"
        )

    # ========================================================
    # BOUND VALUE
    # ========================================================

    @staticmethod
    def _bounded(
        value: float,
    ) -> float:

        try:

            value = float(value)

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

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

    engine = RiskFusionEngine()

    # ========================================================
    # NORMAL NETWORK
    # ========================================================

    normal_network = {

        "ip_reuse_score": 0.05,

        "device_reuse_score": 0.05,

        "payment_reuse_score": 0.02,

        "address_reuse_score": 0.04,

        "refund_velocity_score": 0.10,

        "claim_similarity_score": 0.05,

        "network_abnormality_score": 0.03,
    }

    # ========================================================
    # COORDINATED NETWORK
    # ========================================================

    coordinated_network = {

        "ip_reuse_score": 0.95,

        "device_reuse_score": 0.90,

        "payment_reuse_score": 0.88,

        "address_reuse_score": 0.80,

        "refund_velocity_score": 0.92,

        "claim_similarity_score": 0.91,

        "network_abnormality_score": 0.93,
    }

    # ========================================================
    # MOCK GOVERNOR DECISION
    # ========================================================

    class MockGovernorDecision:

        risk_score = 0.20

        features = {
            "strategic_adaptation_score": 0.10,
        }

    governor_decision = (
        MockGovernorDecision()
    )

    # ========================================================
    # NORMAL CUSTOMER
    # ========================================================

    normal_result = engine.fuse(
        governor_decision=(
            governor_decision
        ),
        network_features=(
            normal_network
        ),
        gnn_risk=0.10,
    )

    # ========================================================
    # HIGH GNN CUSTOMER
    # ========================================================

    high_gnn_result = engine.fuse(
        governor_decision=(
            governor_decision
        ),
        network_features=(
            normal_network
        ),
        gnn_risk=1.00,
    )

    # ========================================================
    # COORDINATED CUSTOMER
    # ========================================================

    coordinated_result = engine.fuse(
        governor_decision=(
            governor_decision
        ),
        network_features=(
            coordinated_network
        ),
        gnn_risk=0.15,
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("=" * 70)
    print("RISK FUSION SELF-CHECK")
    print("=" * 70)

    print()

    print("-" * 70)
    print("NORMAL CUSTOMER")
    print("-" * 70)

    print(
        f"Behavioral risk : "
        f"{normal_result.behavioral_risk:.3f}"
    )

    print(
        f"Strategic risk  : "
        f"{normal_result.strategic_risk:.3f}"
    )

    print(
        f"Network risk    : "
        f"{normal_result.network_risk:.3f}"
    )

    print(
        f"GNN risk        : "
        f"{normal_result.gnn_risk:.3f}"
    )

    print(
        f"Fused risk      : "
        f"{normal_result.fused_risk_score:.3f}"
    )

    print(
        f"Risk level      : "
        f"{normal_result.risk_level}"
    )

    print(
        f"Action          : "
        f"{normal_result.action}"
    )

    print(
        f"Reasons         : "
        f"{normal_result.reason_codes}"
    )

    print()

    print("-" * 70)
    print("HIGH GNN RISK CUSTOMER")
    print("-" * 70)

    print(
        f"Behavioral risk : "
        f"{high_gnn_result.behavioral_risk:.3f}"
    )

    print(
        f"Strategic risk  : "
        f"{high_gnn_result.strategic_risk:.3f}"
    )

    print(
        f"Network risk    : "
        f"{high_gnn_result.network_risk:.3f}"
    )

    print(
        f"GNN risk        : "
        f"{high_gnn_result.gnn_risk:.3f}"
    )

    print(
        f"Fused risk      : "
        f"{high_gnn_result.fused_risk_score:.3f}"
    )

    print(
        f"Risk level      : "
        f"{high_gnn_result.risk_level}"
    )

    print(
        f"Action          : "
        f"{high_gnn_result.action}"
    )

    print(
        f"Reasons         : "
        f"{high_gnn_result.reason_codes}"
    )

    print()

    print("-" * 70)
    print("COORDINATED NETWORK CUSTOMER")
    print("-" * 70)

    print(
        f"Behavioral risk : "
        f"{coordinated_result.behavioral_risk:.3f}"
    )

    print(
        f"Strategic risk  : "
        f"{coordinated_result.strategic_risk:.3f}"
    )

    print(
        f"Network risk    : "
        f"{coordinated_result.network_risk:.3f}"
    )

    print(
        f"GNN risk        : "
        f"{coordinated_result.gnn_risk:.3f}"
    )

    print(
        f"Fused risk      : "
        f"{coordinated_result.fused_risk_score:.3f}"
    )

    print(
        f"Risk level      : "
        f"{coordinated_result.risk_level}"
    )

    print(
        f"Action          : "
        f"{coordinated_result.action}"
    )

    print(
        f"Reasons         : "
        f"{coordinated_result.reason_codes}"
    )

    print()

    print("=" * 70)
    print("RISK FUSION SELF-CHECK: PASSED")
    print("=" * 70)