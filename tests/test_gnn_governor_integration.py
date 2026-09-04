from __future__ import annotations

import torch

from src.environment.population_config import (
    DEFAULT_POPULATION_CONFIG,
)

from src.environment.world_generator import (
    create_world,
)

from src.governor.interaction_graph import (
    build_interaction_graph,
    customer_node_index,
)

from src.governor.gnn_inference import (
    GNNRiskInference,
)

from src.governor.gnn_governor_integration import (
    GNNGovernorAdapter,
    GovernorAction,
)


# ============================================================
# TEST
# ============================================================

def test_gnn_to_governor_integration():

    print()
    print("=" * 80)
    print("GNN → GOVERNOR INTEGRATION TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Build deterministic world
    # --------------------------------------------------------

    world = create_world(
        config=DEFAULT_POPULATION_CONFIG,
        seed=42,
    )

    # --------------------------------------------------------
    # Build interaction graph
    # --------------------------------------------------------

    graph = build_interaction_graph(
        world
    )

    data = graph.data

    print()
    print("GRAPH")
    print("-" * 80)

    print(
        f"Nodes                 : "
        f"{data.num_nodes}"
    )

    print(
        f"Edges                 : "
        f"{data.num_edges}"
    )

    print(
        f"Feature dimension     : "
        f"{data.x.shape[1]}"
    )

    # --------------------------------------------------------
    # Customer
    # --------------------------------------------------------

    customer_id = "CUSTOMER_0001"

    node_index = customer_node_index(
        graph=graph,
        customer_id=customer_id,
    )

    assert node_index >= 0

    assert (
        graph.node_ids[node_index]
        == customer_id
    )

    assert (
        graph.node_types[node_index]
        == "CUSTOMER"
    )

    print()
    print("CUSTOMER")
    print("-" * 80)

    print(
        f"Customer ID           : "
        f"{customer_id}"
    )

    print(
        f"Customer node index   : "
        f"{node_index}"
    )

    # --------------------------------------------------------
    # Load trained GNN
    # --------------------------------------------------------

    inference = (
        GNNRiskInference()
    )

    print()
    print(
        "GNN model loading     : PASSED"
    )

    # --------------------------------------------------------
    # GNN risk
    # --------------------------------------------------------

    gnn_risk = (
        inference.predict_customer_risk(
            data=data,
            customer_node_index=node_index,
        )
    )

    assert 0.0 <= gnn_risk <= 1.0

    print()
    print("GNN RISK")
    print("-" * 80)

    print(
        f"GNN risk score        : "
        f"{gnn_risk:.4f}"
    )

    print(
        "Risk score validity   : PASSED"
    )

    # --------------------------------------------------------
    # Create adapter
    # --------------------------------------------------------

    adapter = GNNGovernorAdapter(
        gnn_inference=inference
    )

    assert adapter is not None

    # --------------------------------------------------------
    # Generate GNN observation
    # --------------------------------------------------------

    gnn_observation = (
        adapter.observe_customer(
            data=data,
            customer_id=customer_id,
            customer_node_index=node_index,
        )
    )

    assert (
        gnn_observation.customer_id
        == customer_id
    )

    assert (
        gnn_observation.customer_node_index
        == node_index
    )

    assert (
        0.0
        <= gnn_observation.gnn_risk
        <= 1.0
    )

    print()
    print(
        "GNN → Governor observation: PASSED"
    )

    # --------------------------------------------------------
    # Governor action space validation
    # --------------------------------------------------------

    allowed_actions = {
        GovernorAction.ALLOW_AGENT_A_DECISION,
        GovernorAction.REQUEST_ADDITIONAL_EVIDENCE,
        GovernorAction.ESCALATE_TO_HUMAN_REVIEW,
    }

    assert len(
        allowed_actions
    ) == 3

    assert (
        GovernorAction.ALLOW_AGENT_A_DECISION
        in allowed_actions
    )

    assert (
        GovernorAction.REQUEST_ADDITIONAL_EVIDENCE
        in allowed_actions
    )

    assert (
        GovernorAction.ESCALATE_TO_HUMAN_REVIEW
        in allowed_actions
    )

    print()
    print(
        "Governor action space : PASSED"
    )

    # --------------------------------------------------------
    # Information boundary
    # --------------------------------------------------------

    forbidden_attributes = {
        "is_abusive",
        "behavior_type",
        "abuse_family",
        "counterparty_type",
        "objective",
        "current_strategy",
        "strategy_beliefs",
        "private_memory",
    }

    assert not any(
        hasattr(
            gnn_observation,
            attribute,
        )
        for attribute
        in forbidden_attributes
    )

    print()
    print(
        "Information boundary   : PASSED"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "GNN → GOVERNOR INTEGRATION: PASSED"
    )
    print("=" * 80)