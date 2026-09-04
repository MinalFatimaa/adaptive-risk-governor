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

from src.governor.gnn_risk_governor import (
    create_gnn_risk_governor,
)


def test_gnn_scores_real_world_customer():

    # --------------------------------------------------------
    # 1. Create the actual simulation world
    # --------------------------------------------------------

    world = create_world(
        config=DEFAULT_POPULATION_CONFIG,
        seed=42,
    )

    # --------------------------------------------------------
    # 2. Convert the world into the interaction graph
    # --------------------------------------------------------

    graph = build_interaction_graph(
        world
    )

    # --------------------------------------------------------
    # 3. Select an actual customer
    # --------------------------------------------------------

    customer_id = "CUSTOMER_0001"

    node_index = customer_node_index(
        graph,
        customer_id,
    )

    # --------------------------------------------------------
    # 4. Create the GNN
    #
    # The input dimension must match the graph feature
    # dimension.
    # --------------------------------------------------------

    model = create_gnn_risk_governor(
        input_dim=graph.data.x.shape[1],
        hidden_dim=32,
    )

    # --------------------------------------------------------
    # 5. Evaluate the customer
    # --------------------------------------------------------

    result = model.evaluate_node(
        data=graph.data,
        node_index=node_index,
        intervention_threshold=0.70,
    )

    # --------------------------------------------------------
    # 6. Validate the result
    # --------------------------------------------------------

    assert 0.0 <= result.risk_score <= 1.0

    assert isinstance(
        result.intervention_required,
        bool,
    )

    # --------------------------------------------------------
    # 7. Display result
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("REAL WORLD → GNN RISK GOVERNOR TEST")
    print("=" * 80)

    print(
        f"Customer ID            : {customer_id}"
    )

    print(
        f"Customer node index    : {node_index}"
    )

    print(
        f"Graph nodes             : "
        f"{graph.data.num_nodes}"
    )

    print(
        f"Graph edges             : "
        f"{graph.data.num_edges}"
    )

    print(
        f"Feature dimension       : "
        f"{graph.data.x.shape[1]}"
    )

    print(
        f"GNN risk score          : "
        f"{result.risk_score:.4f}"
    )

    print(
        f"Intervention required   : "
        f"{result.intervention_required}"
    )

    print()
    print(
        "Real-world GNN integration: PASSED"
    )