from __future__ import annotations

from src.environment.world_generator import (
    create_world,
)

from src.environment.population_config import (
    DEFAULT_POPULATION_CONFIG,
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

from src.governor.gnn_risk_governor import (
    GNNRiskGovernor,
)


# ============================================================
# END-TO-END GNN RISK GOVERNOR TEST
# ============================================================

def test_gnn_risk_governor_end_to_end():

    # ========================================================
    # 1. CREATE WORLD
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

    assert data is not None
    assert data.num_nodes > 0
    assert data.num_edges > 0
    assert data.x is not None

    assert data.x.ndim == 2
    assert data.x.shape[1] == 12

    # ========================================================
    # 3. FIND CUSTOMER
    # ========================================================

    customer_id = "CUSTOMER_0001"

    assert (
        customer_id
        in graph.node_ids
    )

    customer_node_index = (
        graph.node_ids.index(
            customer_id
        )
    )

    assert (
        graph.node_types[
            customer_node_index
        ]
        == "CUSTOMER"
    )

    # ========================================================
    # 4. CREATE GNN INFERENCE
    # ========================================================

    gnn_inference = (
        GNNRiskInference()
    )

    assert gnn_inference is not None

    # ========================================================
    # 5. CREATE GNN ADAPTER
    # ========================================================

    adapter = GNNGovernorAdapter(
        gnn_inference=gnn_inference
    )

    assert adapter is not None

    # ========================================================
    # 6. CREATE GNN RISK GOVERNOR
    # ========================================================

    governor = GNNRiskGovernor(
        gnn_adapter=adapter,
        gnn_weight=0.20,
    )

    assert governor is not None

    # ========================================================
    # 7. RUN COMPLETE PIPELINE
    # ========================================================

    decision = governor.evaluate(

        data=data,

        customer_id=customer_id,

        customer_node_index=(
            customer_node_index
        ),

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

        network_observations=[],

    )

    # ========================================================
    # 8. VERIFY GNN RISK
    # ========================================================

    assert isinstance(
        decision.gnn_risk,
        float,
    )

    assert (
        0.0
        <= decision.gnn_risk
        <= 1.0
    )

    # ========================================================
    # 9. VERIFY BEHAVIORAL RISK
    # ========================================================

    assert isinstance(
        decision.behavioral_risk,
        float,
    )

    assert (
        0.0
        <= decision.behavioral_risk
        <= 1.0
    )

    # ========================================================
    # 10. VERIFY STRATEGIC RISK
    # ========================================================

    assert isinstance(
        decision.strategic_risk,
        float,
    )

    assert (
        0.0
        <= decision.strategic_risk
        <= 1.0
    )

    # ========================================================
    # 11. VERIFY NETWORK RISK
    # ========================================================

    assert isinstance(
        decision.network_risk,
        float,
    )

    assert (
        0.0
        <= decision.network_risk
        <= 1.0
    )

    # ========================================================
    # 12. VERIFY FINAL FUSED RISK
    # ========================================================

    assert isinstance(
        decision.fused_risk_score,
        float,
    )

    assert (
        0.0
        <= decision.fused_risk_score
        <= 1.0
    )

    # ========================================================
    # 13. VERIFY RISK LEVEL
    # ========================================================

    assert decision.risk_level in (
        "LOW",
        "MEDIUM",
        "HIGH",
    )

    # ========================================================
    # 14. VERIFY ACTION
    # ========================================================

    assert decision.action in (
        "ALLOW_AGENT_A_DECISION",
        "REQUEST_ADDITIONAL_EVIDENCE",
        "ESCALATE_TO_HUMAN_REVIEW",
    )

    # ========================================================
    # 15. VERIFY OBSERVATIONS
    # ========================================================

    assert (
        decision.gnn_observation
        is not None
    )

    assert (
        decision.governor_observation
        is not None
    )

    assert (
        decision.governor_decision
        is not None
    )

    assert (
        decision.fusion_decision
        is not None
    )

    # ========================================================
    # 16. VERIFY FEATURES
    # ========================================================

    assert (
        "gnn_risk"
        in decision.features
    )

    assert (
        "final_fused_risk"
        in decision.features
    )

    # ========================================================
    # 17. VERIFY METADATA
    # ========================================================

    assert (
        decision.metadata[
            "gnn_enabled"
        ]
        is True
    )

    assert (
        decision.metadata[
            "gnn_weight"
        ]
        == 0.20
    )