from __future__ import annotations

import pytest
import torch

from src.environment.population_config import (
    DEFAULT_POPULATION_CONFIG,
)

from src.environment.world_generator import (
    create_world,
)

from src.governor.interaction_graph import (
    build_interaction_graph,
)

from src.governor.gnn_inference import (
    GNNRiskInference,
)

from src.governor.gnn_governor_integration import (
    GNNGovernorAdapter,
)

from src.governor.risk_governor import (
    AdaptiveRiskGovernor,
)

from src.governor.governor_features import (
    GovernorFeatureBuilder,
)

from src.governor.risk_fusion import (
    RiskFusionEngine,
)


# ============================================================
# END-TO-END GNN → GOVERNOR TEST
# ============================================================


def test_end_to_end_gnn_governor_pipeline():
    """
    Verify the complete:

        World
          ↓
        Interaction Graph
          ↓
        Trained GNN
          ↓
        GNN Risk
          ↓
        Governor Observation
          ↓
        Behavioral Governor
          ↓
        Risk Fusion
          ↓
        Final Governor Decision

    pipeline.

    This test verifies integration, not model performance.
    """


    # ========================================================
    # 1. CREATE DETERMINISTIC WORLD
    # ========================================================

    world = create_world(
        config=DEFAULT_POPULATION_CONFIG,
        seed=42,
    )

    assert world is not None


    # ========================================================
    # 2. BUILD INTERACTION GRAPH
    # ========================================================

    graph = build_interaction_graph(
        world
    )

    assert graph is not None

    data = graph.data

    assert data.num_nodes > 0
    assert data.num_edges > 0
    assert data.x is not None

    assert data.x.ndim == 2
    assert data.x.shape[1] == 12


    # ========================================================
    # 3. FIND CUSTOMER
    # ========================================================

    customer_id = "CUSTOMER_0001"

    assert customer_id in graph.node_ids

    customer_node_index = (
        graph.node_ids.index(
            customer_id
        )
    )

    assert customer_node_index >= 0
    assert (
        customer_node_index
        < data.num_nodes
    )

    assert (
        graph.node_types[
            customer_node_index
        ]
        == "CUSTOMER"
    )


    # ========================================================
    # 4. LOAD TRAINED GNN
    # ========================================================

    gnn_inference = (
        GNNRiskInference()
    )

    assert gnn_inference is not None


    # ========================================================
    # 5. GENERATE GNN RISK
    # ========================================================

    gnn_risk = (
        gnn_inference.predict_customer_risk(
            data=data,
            customer_node_index=(
                customer_node_index
            ),
        )
    )

    assert isinstance(
        gnn_risk,
        float,
    )

    assert 0.0 <= gnn_risk <= 1.0


    # ========================================================
    # 6. GNN → GOVERNOR ADAPTER
    # ========================================================

    adapter = GNNGovernorAdapter(
        gnn_inference=gnn_inference,
    )

    assert adapter is not None


    # ========================================================
    # 7. GENERATE GNN OBSERVATION
    # ========================================================

    gnn_observation = (
        adapter.observe_customer(
            data=data,
            customer_id=customer_id,
            customer_node_index=(
                customer_node_index
            ),
        )
    )

    assert (
        gnn_observation.customer_id
        == customer_id
    )

    assert (
        gnn_observation.customer_node_index
        == customer_node_index
    )

    assert isinstance(
        gnn_observation.gnn_risk,
        float,
    )

    assert (
        0.0
        <= gnn_observation.gnn_risk
        <= 1.0
    )


    # ========================================================
    # 8. BUILD BASE GOVERNOR
    #    OBSERVATION
    # ========================================================

    governor = (
        AdaptiveRiskGovernor()
    )

    base_observation = (
        governor.evaluate(
            customer_history=[],
            current_claim_type=(
                "ITEM_NOT_RECEIVED"
            ),
            requested_amount=1000.0,
            evidence_available=[],
            support_decision=(
                "REQUEST_EVIDENCE"
            ),
            strategic_state=None,
        )
    )

    assert base_observation is not None

    assert 0.0 <= (
        base_observation.risk_score
    ) <= 1.0


    # ========================================================
    # 9. AUGMENT GOVERNOR OBSERVATION
    # ========================================================

    governor_observation = (
        adapter.augment_observation(
            observation=base_observation,
            data=data,
            customer_id=customer_id,
            customer_node_index=(
                customer_node_index
            ),
        )
    )

    assert (
        governor_observation.customer_id
        == customer_id
    )

    assert isinstance(
        governor_observation.gnn_risk,
        float,
    )

    assert (
        0.0
        <= governor_observation.gnn_risk
        <= 1.0
    )

    assert isinstance(
        governor_observation.governor_risk,
        float,
    )

    assert (
        0.0
        <= governor_observation.governor_risk
        <= 1.0
    )


    # ========================================================
    # 10. VERIFY ALL EXPECTED GOVERNOR FEATURES
    # ========================================================

    expected_features = {
        "governor_risk",
        "network_risk",
        "temporal_abnormality",
        "semantic_paraphrase_score",
        "semantic_claim_switch",
        "evidence_consistency",
        "claim_similarity",
        "request_velocity",
        "amount_acceleration",
        "shared_identifier_strength",
        "agent_decision_anomaly",
        "strategic_behavior",
    }

    # The adapter produces these as explicit fields.
    for feature_name in expected_features:

        assert hasattr(
            governor_observation,
            feature_name,
        )


    # ========================================================
    # 11. BUILD NETWORK FEATURES
    #
    # For this integration test we use neutral network
    # observations. The dedicated network tests already verify
    # the actual network feature extraction.
    # ========================================================

    network_features = {
        "ip_reuse_score": 0.05,
        "device_reuse_score": 0.05,
        "payment_reuse_score": 0.05,
        "address_reuse_score": 0.05,
        "refund_velocity_score": 0.05,
        "claim_similarity_score": 0.05,
        "network_abnormality_score": 0.05,
    }


    # ========================================================
    # 12. RISK FUSION
    # ========================================================

    fusion_engine = RiskFusionEngine()

    assert fusion_engine is not None

    # --------------------------------------------------------
    # Extract network features
    #
    # For this integration test, use the network feature
    # builder already present in the project.
    # --------------------------------------------------------

    network_features = {
        "ip_reuse_score": 0.05,
        "device_reuse_score": 0.05,
        "payment_reuse_score": 0.02,
        "address_reuse_score": 0.04,
        "refund_velocity_score": 0.10,
        "claim_similarity_score": 0.05,
        "network_abnormality_score": 0.03,
    }

    # --------------------------------------------------------
    # Fuse Governor + Network + GNN
    # --------------------------------------------------------

    fused_decision = fusion_engine.fuse(
        governor_decision=base_observation,
        network_features=network_features,
        gnn_risk=gnn_observation.gnn_risk,
    )

    assert fused_decision is not None

    # --------------------------------------------------------
    # Verify individual signals
    # --------------------------------------------------------

    assert 0.0 <= (
        fused_decision.behavioral_risk
    ) <= 1.0

    assert 0.0 <= (
        fused_decision.strategic_risk
    ) <= 1.0

    assert 0.0 <= (
        fused_decision.network_risk
    ) <= 1.0

    assert 0.0 <= (
        fused_decision.gnn_risk
    ) <= 1.0

    # --------------------------------------------------------
    # Verify final fused score
    # --------------------------------------------------------

    assert 0.0 <= (
        fused_decision.fused_risk_score
    ) <= 1.0

    # --------------------------------------------------------
    # Verify risk level
    # --------------------------------------------------------

    assert (
        fused_decision.risk_level
        in (
            "LOW",
            "MEDIUM",
            "HIGH",
        )
    )

    # --------------------------------------------------------
    # Verify Governor action
    # --------------------------------------------------------

    assert (
        fused_decision.action
        in (
            "ALLOW_AGENT_A_DECISION",
            "REQUEST_ADDITIONAL_EVIDENCE",
            "ESCALATE_TO_HUMAN_REVIEW",
        )
    )

    # --------------------------------------------------------
    # Verify GNN is actually represented in final result
    # --------------------------------------------------------

    assert (
        fused_decision.gnn_risk
        == gnn_observation.gnn_risk
    )


    # ========================================================
    # 13. VERIFY FUSED RISK
    # ========================================================

    assert isinstance(
        fused_decision.fused_risk_score,
        float,
    )

    assert (
        0.0
        <= fused_decision.fused_risk_score
        <= 1.0
    )


    # ========================================================
    # 14. VERIFY RISK COMPONENTS
    # ========================================================

    assert (
        0.0
        <= fused_decision.behavioral_risk
        <= 1.0
    )

    assert (
        0.0
        <= fused_decision.strategic_risk
        <= 1.0
    )

    assert (
        0.0
        <= fused_decision.network_risk
        <= 1.0
    )


    # ========================================================
    # 15. VERIFY RISK LEVEL
    # ========================================================

    assert fused_decision.risk_level in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }


    # ========================================================
    # 16. VERIFY GOVERNOR ACTION
    # ========================================================

    valid_actions = {
        "ALLOW_AGENT_A_DECISION",
        "REQUEST_ADDITIONAL_EVIDENCE",
        "ESCALATE_TO_HUMAN_REVIEW",
    }

    assert (
        fused_decision.action
        in valid_actions
    )


    # ========================================================
    # 17. VERIFY INFORMATION BOUNDARY
    # ========================================================
    #
    # Agent A's private decision process should not receive
    # the GNN risk directly.
    #
    # GNN risk belongs to the Governor layer.
    #

    assert hasattr(
        governor_observation,
        "gnn_risk",
    )

    assert not hasattr(
        base_observation,
        "gnn_risk",
    )


    # ========================================================
    # 18. VERIFY FUSION METADATA
    # ========================================================

    assert isinstance(
        fused_decision.metadata,
        dict,
    )

    assert (
        "behavioral_weight"
        in fused_decision.metadata
    )

    assert (
        "strategic_weight"
        in fused_decision.metadata
    )

    assert (
        "network_weight"
        in fused_decision.metadata
    )


    # ========================================================
    # 19. FINAL PIPELINE ASSERTION
    # ========================================================

    assert (
        gnn_observation.gnn_risk
        == governor_observation.gnn_risk
    )