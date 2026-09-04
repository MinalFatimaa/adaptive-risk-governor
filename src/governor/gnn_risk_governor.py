from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

from src.governor.gnn_governor_integration import (
    GNNGovernorAdapter,
    GNNObservation,
    GovernorObservation,
)

from src.governor.governor_features import (
    GovernorFeatureBuilder,
)

from src.governor.risk_fusion import (
    FusedRiskDecision,
    RiskFusionEngine,
)

from src.governor.risk_governor import (
    AdaptiveRiskGovernor,
    GovernorDecision,
)


# ============================================================
# GNN MODEL
# ============================================================

class GNNRiskModel(nn.Module):
    """
    Canonical GraphSAGE model used by the complete GNN pipeline.

    Architecture:

        Node Features
             |
             v
        SAGEConv(input_dim -> hidden_dim)
             |
             v
            ReLU
             |
             v
        SAGEConv(hidden_dim -> hidden_dim)
             |
             v
            ReLU
             |
             v
        Linear(hidden_dim -> 16)
             |
             v
            ReLU
             |
             v
        Linear(16 -> 1)
             |
             v
          Risk Logit

    This exact architecture is shared by:

        train_gnn.py
        gnn_inference.py
        gnn_governor_integration.py
        gnn_risk_governor.py
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
    ) -> None:

        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        # ----------------------------------------------------
        # GraphSAGE layer 1
        # ----------------------------------------------------

        self.conv1 = SAGEConv(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim,
            aggr="mean",
        )

        # ----------------------------------------------------
        # GraphSAGE layer 2
        # ----------------------------------------------------

        self.conv2 = SAGEConv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            aggr="mean",
        )

        # ----------------------------------------------------
        # Risk prediction head
        # ----------------------------------------------------

        self.risk_head = nn.Sequential(
            nn.Linear(
                self.hidden_dim,
                16,
            ),
            nn.ReLU(),
            nn.Linear(
                16,
                1,
            ),
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        data,
    ) -> torch.Tensor:

        if data is None:
            raise ValueError(
                "Graph data cannot be None."
            )

        if not hasattr(data, "x"):
            raise ValueError(
                "Graph data must contain node features 'x'."
            )

        if not hasattr(data, "edge_index"):
            raise ValueError(
                "Graph data must contain 'edge_index'."
            )

        x = data.x
        edge_index = data.edge_index

        if x is None:
            raise ValueError(
                "Graph node features cannot be None."
            )

        if x.ndim != 2:
            raise ValueError(
                "Graph node features must be 2-dimensional."
            )

        if x.shape[1] != self.input_dim:
            raise ValueError(
                "Graph feature dimension does not match "
                "the model.\n"
                f"Expected: {self.input_dim}\n"
                f"Received: {x.shape[1]}"
            )

        # ----------------------------------------------------
        # GraphSAGE layer 1
        # ----------------------------------------------------

        x = self.conv1(
            x,
            edge_index,
        )

        x = torch.relu(x)

        # ----------------------------------------------------
        # GraphSAGE layer 2
        # ----------------------------------------------------

        x = self.conv2(
            x,
            edge_index,
        )

        x = torch.relu(x)

        # ----------------------------------------------------
        # Risk head
        # ----------------------------------------------------

        logits = self.risk_head(x)

        return logits


# ============================================================
# MODEL FACTORY
# ============================================================

def create_gnn_risk_governor(
    *,
    input_dim: int,
    hidden_dim: int = 32,
) -> GNNRiskModel:
    """
    Public factory for creating the canonical GNN model.

    IMPORTANT:
        gnn_inference.py imports this function.

    Do not remove or rename it.
    """

    return GNNRiskModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    )


# ============================================================
# GNN-AWARE RISK DECISION
# ============================================================

@dataclass
class GNNRiskGovernorDecision:
    """
    Final decision produced by the GNN-aware Governor.

    IMPORTANT INFORMATION BOUNDARY:

    The Governor internally maintains many signals, including:

        - GNN risk
        - strategic adaptation
        - network risk
        - fused risk
        - network behavior
        - identity-link information

    These are Governor-internal signals.

    `features` is deliberately restricted to the
    Agent-A-visible feature contract.

    `internal_features` contains Governor-only features and
    must never be passed to Agent A.
    """

    behavioral_risk: float

    strategic_risk: float

    network_risk: float

    gnn_risk: float

    fused_risk_score: float

    risk_level: str

    action: str

    reason_codes: list[str] = field(
        default_factory=list
    )

    governor_decision: GovernorDecision | None = None

    gnn_observation: GNNObservation | None = None

    governor_observation: GovernorObservation | None = None

    fusion_decision: FusedRiskDecision | None = None

    # --------------------------------------------------------
    # Agent-A-visible features.
    #
    # IMPORTANT:
    # This is an explicit allowlist.
    # Do NOT put Governor-internal signals here.
    # --------------------------------------------------------

    features: dict[str, float] = field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Governor-internal feature store.
    #
    # This contains the complete internal risk state.
    # --------------------------------------------------------

    internal_features: dict[str, float] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# GNN-AWARE RISK GOVERNOR
# ============================================================

class GNNRiskGovernor:
    """
    Complete GNN-aware Adaptive Risk Governor.

    Pipeline:

        World / Customer
              |
              v
        Interaction Graph
              |
              v
        GraphSAGE
              |
              v
           gnn_risk
              |
              +----------------------+
              |                      |
              v                      v
        GNN Observation       Governor Features
              |                      |
              +----------+-----------+
                         |
                         v
                Adaptive Governor
                         |
                         v
                 Risk Fusion
                         |
                         v
                GNN-aware decision

    INFORMATION ISOLATION:

        Agent A must NOT receive:

            - GNN risk
            - strategic risk
            - network risk
            - fused risk
            - strategic adaptation score
            - network behavior score
            - network size signal
            - identity-link strength
            - GNN override reason codes

    Those remain inside the Governor.
    """

    # ========================================================
    # AGENT A INFORMATION CONTRACT
    # ========================================================

    # --------------------------------------------------------
    # Explicit allowlist.
    #
    # IMPORTANT:
    # Never construct Agent-A-visible features by removing
    # forbidden fields from a larger dictionary.
    #
    # New internal Governor signals will therefore remain
    # hidden by default.
    # --------------------------------------------------------

    AGENT_A_ALLOWED_FEATURES = frozenset(
        {
            "evidence_sensitivity",
            "amount_sensitivity",
            "high_value_escalation",
            "followup_sensitivity",
        }
    )

    # --------------------------------------------------------
    # Explicitly forbidden internal signals.
    #
    # This is documentation and an internal safety check.
    # --------------------------------------------------------

    AGENT_A_FORBIDDEN_FEATURES = frozenset(
        {
            "gnn_risk",
            "strategic_risk",
            "network_risk",
            "fused_risk",
            "final_fused_risk",
            "strategic_adaptation_score",
            "network_behavior_score",
            "network_size_signal",
            "identity_link_strength",
            "network_abnormality_score",
            "network_customer_count",
            "network_request_count",
            "network_refund_rate",
            "network_denial_rate",
            "shared_ip_account_count",
            "shared_device_account_count",
            "shared_payment_account_count",
            "shared_address_account_count",
            "ip_reuse_score",
            "device_reuse_score",
            "payment_reuse_score",
            "address_reuse_score",
            "claim_concentration",
            "average_network_amount",
            "current_to_network_amount_ratio",
        }
    )

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        *,
        gnn_adapter: GNNGovernorAdapter | None = None,
        governor: AdaptiveRiskGovernor | None = None,
        feature_builder: GovernorFeatureBuilder | None = None,
        fusion_engine: RiskFusionEngine | None = None,
        gnn_weight: float = 0.20,
    ) -> None:

        if not 0.0 <= gnn_weight <= 1.0:
            raise ValueError(
                "gnn_weight must be between 0 and 1."
            )

        # ----------------------------------------------------
        # GNN adapter
        # ----------------------------------------------------

        self.gnn_adapter = (
            gnn_adapter
            if gnn_adapter is not None
            else GNNGovernorAdapter()
        )

        # ----------------------------------------------------
        # Existing behavioral Governor
        # ----------------------------------------------------

        self.governor = (
            governor
            if governor is not None
            else AdaptiveRiskGovernor()
        )

        # ----------------------------------------------------
        # Feature builder
        # ----------------------------------------------------

        self.feature_builder = (
            feature_builder
            if feature_builder is not None
            else GovernorFeatureBuilder()
        )

        # ----------------------------------------------------
        # Fusion engine
        # ----------------------------------------------------

        self.fusion_engine = (
            fusion_engine
            if fusion_engine is not None
            else RiskFusionEngine()
        )

        # ----------------------------------------------------
        # GNN fusion weight
        # ----------------------------------------------------

        self.gnn_weight = float(
            gnn_weight
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def evaluate(
        self,
        *,
        data: Any,
        customer_id: str,
        customer_node_index: int,
        customer_history: list[
            dict[str, Any]
        ] | None = None,
        current_claim_type: str,
        requested_amount: float,
        evidence_available: list[str] | None = None,
        support_decision: str | None = None,
        strategic_state: Any | None = None,
        network_observations: list[Any] | None = None,
        current_ip: str | None = None,
        current_device_id: str | None = None,
        current_payment_method_id: str | None = None,
        current_shipping_address_id: str | None = None,
    ) -> GNNRiskGovernorDecision:

        # ----------------------------------------------------
        # Normalize optional inputs
        # ----------------------------------------------------

        customer_history = (
            []
            if customer_history is None
            else customer_history
        )

        evidence_available = (
            []
            if evidence_available is None
            else evidence_available
        )

        network_observations = (
            []
            if network_observations is None
            else network_observations
        )

        # ====================================================
        # 1. GNN RISK
        # ====================================================

        gnn_observation = (
            self.gnn_adapter.observe_customer(
                data=data,
                customer_id=customer_id,
                customer_node_index=customer_node_index,
            )
        )

        gnn_risk = self._bounded(
            gnn_observation.gnn_risk
        )

        # ====================================================
        # 2. EXTERNAL GOVERNOR FEATURES
        # ====================================================

        external_features = (
            self.feature_builder.build(
                customer_id=customer_id,
                customer_history=customer_history,
                current_claim_type=current_claim_type,
                requested_amount=requested_amount,
                evidence_available=evidence_available,
                strategic_state=strategic_state,
                network_observations=network_observations,
                current_ip=current_ip,
                current_device_id=current_device_id,
                current_payment_method_id=(
                    current_payment_method_id
                ),
                current_shipping_address_id=(
                    current_shipping_address_id
                ),
            )
        )

        # ====================================================
        # 3. BASE BEHAVIORAL GOVERNOR
        # ====================================================

        governor_decision = (
            self.governor.evaluate(
                customer_history=customer_history,
                current_claim_type=current_claim_type,
                requested_amount=requested_amount,
                evidence_available=evidence_available,
                support_decision=support_decision,
                strategic_state=strategic_state,
            )
        )

        # ====================================================
        # 4. GOVERNOR OBSERVATION
        # ====================================================

        governor_observation = (
            self._build_governor_observation(
                governor_decision=governor_decision,
                gnn_observation=gnn_observation,
                external_features=external_features,
                customer_id=customer_id,
            )
        )

        # ====================================================
        # 5. RISK FUSION
        # ====================================================

        fusion_decision = (
            self._fuse(
                governor_decision=governor_decision,
                network_features=external_features,
                gnn_risk=gnn_risk,
            )
        )

        # ====================================================
        # 6. INTERNAL REASON CODES
        # ====================================================

        internal_reason_codes = list(
            governor_decision.reason_codes
        )

        for reason in fusion_decision.reason_codes:

            if reason not in internal_reason_codes:

                internal_reason_codes.append(
                    reason
                )

        # ====================================================
        # 7. INTERNAL GOVERNOR FEATURES
        # ====================================================

        #
        # IMPORTANT:
        #
        # All signals can continue to exist internally.
        #
        # This preserves the existing Governor behavior.
        #

        internal_features: dict[str, float] = {}

        for key, value in governor_decision.features.items():

            try:

                internal_features[key] = self._bounded(
                    float(value)
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

        for key, value in external_features.items():

            try:

                internal_features[key] = self._bounded(
                    float(value)
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

        internal_features["gnn_risk"] = gnn_risk

        internal_features[
            "final_fused_risk"
        ] = self._bounded(
            fusion_decision.fused_risk_score
        )

        internal_features[
            "strategic_risk"
        ] = self._bounded(
            fusion_decision.strategic_risk
        )

        internal_features[
            "network_risk"
        ] = self._bounded(
            fusion_decision.network_risk
        )

        internal_features[
            "behavioral_risk"
        ] = self._bounded(
            fusion_decision.behavioral_risk
        )

        # ====================================================
        # 8. AGENT-A-SAFE FEATURES
        # ====================================================

        #
        # IMPORTANT:
        #
        # This is an ALLOWLIST.
        #
        # We intentionally select only the fields Agent A is
        # permitted to receive.
        #
        # We do NOT copy all features and remove forbidden keys.
        #

        agent_a_features = (
            self._build_agent_a_features(
                governor_decision=governor_decision,
                external_features=external_features,
            )
        )

        # ====================================================
        # 9. FINAL SAFETY VALIDATION
        # ====================================================

        self._validate_agent_a_features(
            agent_a_features
        )

        # ====================================================
        # 10. FINAL DECISION OBJECT
        # ====================================================

        return GNNRiskGovernorDecision(

            behavioral_risk=(
                fusion_decision.behavioral_risk
            ),

            strategic_risk=(
                fusion_decision.strategic_risk
            ),

            network_risk=(
                fusion_decision.network_risk
            ),

            gnn_risk=gnn_risk,

            fused_risk_score=(
                fusion_decision.fused_risk_score
            ),

            risk_level=(
                fusion_decision.risk_level
            ),

            action=(
                fusion_decision.action
            ),

            # ------------------------------------------------
            # IMPORTANT:
            #
            # reason_codes remain the internal Governor
            # decision because these contain GNN/network/
            # strategic reasons.
            #
            # Agent A must not receive this complete object.
            # ------------------------------------------------

            reason_codes=internal_reason_codes,

            governor_decision=governor_decision,

            gnn_observation=gnn_observation,

            governor_observation=governor_observation,

            fusion_decision=fusion_decision,

            # ------------------------------------------------
            # Agent-A-visible features ONLY
            # ------------------------------------------------

            features=agent_a_features,

            # ------------------------------------------------
            # Governor-only features
            # ------------------------------------------------

            internal_features=internal_features,

            metadata={
                "gnn_weight": self.gnn_weight,
                "gnn_enabled": True,
                "customer_id": customer_id,
                "customer_node_index": (
                    customer_node_index
                ),

                # Explicitly mark the boundary.
                "agent_a_information_isolated": True,
            },
        )

    # ========================================================
    # AGENT A FEATURE BUILDER
    # ========================================================

    def _build_agent_a_features(
        self,
        *,
        governor_decision: GovernorDecision,
        external_features: dict[str, float],
    ) -> dict[str, float]:

        """
        Construct the ONLY feature dictionary that is safe
        for Agent A.

        This method uses an explicit allowlist.

        Internal Governor signals are never copied here.
        """

        result: dict[str, float] = {}

        # ----------------------------------------------------
        # First source: external features
        # ----------------------------------------------------

        for feature_name in self.AGENT_A_ALLOWED_FEATURES:

            if feature_name not in external_features:
                continue

            result[feature_name] = self._feature(
                external_features,
                feature_name,
            )

        # ----------------------------------------------------
        # Second source: existing Governor features
        #
        # This is only used when the explicitly allowed field
        # exists there.
        # ----------------------------------------------------

        for feature_name in self.AGENT_A_ALLOWED_FEATURES:

            if feature_name in result:
                continue

            if feature_name not in governor_decision.features:
                continue

            result[feature_name] = self._feature(
                governor_decision.features,
                feature_name,
            )

        # ----------------------------------------------------
        # Ensure only explicitly allowed fields exist.
        # ----------------------------------------------------

        result = {
            key: self._bounded(value)
            for key, value in result.items()
            if key in self.AGENT_A_ALLOWED_FEATURES
        }

        return result

    # ========================================================
    # AGENT A FEATURE VALIDATION
    # ========================================================

    @classmethod
    def _validate_agent_a_features(
        cls,
        features: dict[str, float],
    ) -> None:

        """
        Safety check for the Agent-A information boundary.

        Any forbidden signal reaching this dictionary causes
        an immediate failure.
        """

        exposed_forbidden = (
            set(features.keys())
            & cls.AGENT_A_FORBIDDEN_FEATURES
        )

        if exposed_forbidden:

            raise RuntimeError(
                "Agent A information-isolation violation.\n"
                "Forbidden signals exposed: "
                f"{sorted(exposed_forbidden)}"
            )

        unexpected = (
            set(features.keys())
            - cls.AGENT_A_ALLOWED_FEATURES
        )

        if unexpected:

            raise RuntimeError(
                "Agent A information contract violation.\n"
                "Unexpected features exposed: "
                f"{sorted(unexpected)}"
            )

    # ========================================================
    # GOVERNOR OBSERVATION
    # ========================================================

    def _build_governor_observation(
        self,
        *,
        governor_decision: GovernorDecision,
        gnn_observation: GNNObservation,
        external_features: dict[str, float],
        customer_id: str,
    ) -> GovernorObservation:

        features = governor_decision.features

        return GovernorObservation(

            customer_id=customer_id,

            gnn_risk=self._bounded(
                gnn_observation.gnn_risk
            ),

            governor_risk=self._bounded(
                governor_decision.risk_score
            ),

            network_risk=self._feature(
                external_features,
                "network_abnormality_score",
            ),

            temporal_abnormality=self._feature(
                external_features,
                "temporal_abnormality",
            ),

            semantic_paraphrase_score=self._feature(
                external_features,
                "semantic_paraphrase_score",
            ),

            semantic_claim_switch=self._feature(
                external_features,
                "semantic_claim_switch",
            ),

            evidence_consistency=self._feature(
                external_features,
                "evidence_consistency",
            ),

            claim_similarity=self._feature(
                external_features,
                "claim_similarity_score",
            ),

            request_velocity=self._feature(
                external_features,
                "refund_velocity_score",
            ),

            amount_acceleration=self._feature(
                external_features,
                "amount_acceleration",
            ),

            shared_identifier_strength=(
                self._shared_identifier_strength(
                    external_features
                )
            ),

            agent_decision_anomaly=self._feature(
                external_features,
                "agent_decision_anomaly",
            ),

            strategic_behavior=self._bounded(
                features.get(
                    "strategic_adaptation_score",
                    external_features.get(
                        "strategic_adaptation_score",
                        0.0,
                    ),
                )
            ),
        )

    # ========================================================
    # GNN-AWARE FUSION
    # ========================================================

    def _fuse(
        self,
        *,
        governor_decision: GovernorDecision,
        network_features: dict[str, float],
        gnn_risk: float,
    ) -> FusedRiskDecision:

        # ----------------------------------------------------
        # Existing fusion engine
        # ----------------------------------------------------

        base_fusion = (
            self.fusion_engine.fuse(
                governor_decision=governor_decision,
                network_features=network_features,
            )
        )

        base_score = self._bounded(
            base_fusion.fused_risk_score
        )

        gnn_risk = self._bounded(
            gnn_risk
        )

        # ----------------------------------------------------
        # Weighted GNN fusion
        # ----------------------------------------------------

        fused_score = (
            (1.0 - self.gnn_weight)
            * base_score
            +
            self.gnn_weight
            * gnn_risk
        )

        fused_score = self._bounded(
            fused_score
        )

        # ----------------------------------------------------
        # Internal reason codes
        # ----------------------------------------------------

        reason_codes = list(
            base_fusion.reason_codes
        )

        if gnn_risk >= 0.70:

            reason_codes.append(
                "HIGH_GNN_RISK"
            )

        elif gnn_risk >= 0.40:

            reason_codes.append(
                "ELEVATED_GNN_RISK"
            )

        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        risk_level = self._risk_level(
            fused_score
        )

        # ----------------------------------------------------
        # Strong GNN override
        # ----------------------------------------------------

        if gnn_risk >= 0.85:

            if fused_score < 0.70:
                fused_score = 0.70

            risk_level = "HIGH"

            reason_codes.append(
                "GNN_HIGH_RISK_OVERRIDE"
            )

        elif gnn_risk >= 0.70:

            if risk_level == "LOW":
                risk_level = "MEDIUM"

        # ----------------------------------------------------
        # Final action
        # ----------------------------------------------------

        action = self._recommended_action(
            risk_level
        )

        # ----------------------------------------------------
        # Fusion metadata
        # ----------------------------------------------------

        metadata = dict(
            base_fusion.metadata
        )

        metadata.update(
            {
                "gnn_risk": gnn_risk,
                "gnn_weight": self.gnn_weight,
                "pre_gnn_fused_score": base_score,
                "gnn_aware_fused_score": fused_score,
            }
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # FusedRiskDecision in the current project requires
        # gnn_risk.
        #
        # Keep this field here.
        # ----------------------------------------------------

        return FusedRiskDecision(

            behavioral_risk=(
                base_fusion.behavioral_risk
            ),

            strategic_risk=(
                base_fusion.strategic_risk
            ),

            network_risk=(
                base_fusion.network_risk
            ),

            gnn_risk=gnn_risk,

            fused_risk_score=fused_score,

            risk_level=risk_level,

            reason_codes=self._unique(
                reason_codes
            ),

            action=action,

            metadata=metadata,
        )

    # ========================================================
    # RISK LEVEL
    # ========================================================

    @staticmethod
    def _risk_level(
        score: float,
    ) -> str:

        if score < 0.30:
            return "LOW"

        if score < 0.70:
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
            return "ALLOW_AGENT_A_DECISION"

        if risk_level == "MEDIUM":
            return "REQUEST_ADDITIONAL_EVIDENCE"

        return "ESCALATE_TO_HUMAN_REVIEW"

    # ========================================================
    # FEATURE ACCESSOR
    # ========================================================

    @staticmethod
    def _feature(
        features: dict[str, float],
        name: str,
        default: float = 0.0,
    ) -> float:

        try:

            return GNNRiskGovernor._bounded(
                float(
                    features.get(
                        name,
                        default,
                    )
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return GNNRiskGovernor._bounded(
                default
            )

    # ========================================================
    # SHARED IDENTIFIER STRENGTH
    # ========================================================

    @staticmethod
    def _shared_identifier_strength(
        features: dict[str, float],
    ) -> float:

        values = [

            GNNRiskGovernor._feature(
                features,
                "ip_reuse_score",
            ),

            GNNRiskGovernor._feature(
                features,
                "device_reuse_score",
            ),

            GNNRiskGovernor._feature(
                features,
                "payment_reuse_score",
            ),

            GNNRiskGovernor._feature(
                features,
                "address_reuse_score",
            ),
        ]

        return GNNRiskGovernor._bounded(
            sum(values) / len(values)
        )

    # ========================================================
    # UNIQUE REASONS
    # ========================================================

    @staticmethod
    def _unique(
        values: list[str],
    ) -> list[str]:

        result: list[str] = []

        for value in values:

            if value not in result:

                result.append(
                    value
                )

        return result

    # ========================================================
    # BOUND VALUE
    # ========================================================

    @staticmethod
    def _bounded(
        value: Any,
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

    print()
    print("=" * 70)
    print("GNN RISK GOVERNOR MODULE")
    print("=" * 70)

    # --------------------------------------------------------
    # Verify canonical model independently
    # --------------------------------------------------------

    test_model = create_gnn_risk_governor(
        input_dim=12,
        hidden_dim=32,
    )

    print(
        "GNN model initialized."
    )

    print(
        f"Input dimension : "
        f"{test_model.input_dim}"
    )

    print(
        f"Hidden dimension: "
        f"{test_model.hidden_dim}"
    )

    print()
    print("Model architecture:")
    print(test_model)

    # --------------------------------------------------------
    # Verify actual Governor integration.
    # --------------------------------------------------------

    try:

        governor = GNNRiskGovernor(
            gnn_weight=0.20
        )

        print()
        print(
            "GNN Governor initialized successfully."
        )

        print(
            f"GNN weight: "
            f"{governor.gnn_weight:.2f}"
        )

        print()
        print(
            "GNN → Governor integration: PASSED"
        )

    except FileNotFoundError as exc:

        print()
        print(
            "WARNING: trained GNN checkpoint "
            "was not found."
        )

        print(exc)

        print()
        print(
            "Model architecture self-check: PASSED"
        )

    except Exception as exc:

        print()
        print(
            "GNN → Governor integration: FAILED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise

    print("=" * 70)