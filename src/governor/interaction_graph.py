from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch_geometric.data import Data

from ..environment.world import EnvironmentState
from ..schemas.graph import GraphNode, GraphEdge


# ============================================================
# CONFIGURATION
# ============================================================

NODE_TYPES = {
    "CUSTOMER",
    "ORDER",
    "DEVICE",
    "ADDRESS",
    "PAYMENT",
    "REFUND",
}

# Feature vector layout for every node:
#
# [0] is_customer
# [1] is_order
# [2] is_device
# [3] is_address
# [4] is_payment
# [5] is_refund
# [6] normalized_amount
# [7] activity_count
# [8] refund_count
# [9] refund_amount
# [10] chargeback_count
# [11] account_age_normalized
#
# The same dimensional representation is used for every node.
# Features that do not apply to a node type are zero.
#
# IMPORTANT:
# Ground truth and private customer state are deliberately
# excluded from these features.


NUM_NODE_FEATURES = 12


# ============================================================
# GRAPH RESULT
# ============================================================

@dataclass(frozen=True)
class InteractionGraph:

    data: Data

    node_ids: tuple[str, ...]

    node_types: tuple[str, ...]

    node_index: dict[str, int]


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def _safe_float(
    value,
) -> float:

    try:

        value = float(value)

    except (
        TypeError,
        ValueError,
    ):

        return 0.0

    if not torch.isfinite(
        torch.tensor(value)
    ):

        return 0.0

    return value


def _normalize_amount(
    amount: float,
    scale: float = 10_000.0,
) -> float:

    amount = max(
        0.0,
        _safe_float(amount),
    )

    return min(
        amount / scale,
        1.0,
    )


def _normalize_count(
    value: int | float,
    scale: float = 20.0,
) -> float:

    value = max(
        0.0,
        _safe_float(value),
    )

    return min(
        value / scale,
        1.0,
    )


def _normalize_account_age(
    days: int | float,
) -> float:

    days = max(
        0.0,
        _safe_float(days),
    )

    # One year is used as a practical normalization scale.
    return min(
        days / 365.0,
        1.0,
    )


# ============================================================
# NODE TYPE NORMALIZATION
# ============================================================

def normalize_node_type(
    node_type: str,
) -> str:

    value = str(
        node_type
    ).strip().upper()

    aliases = {

        "CUSTOMER_NODE":
            "CUSTOMER",

        "ORDER_NODE":
            "ORDER",

        "DEVICE_NODE":
            "DEVICE",

        "ADDRESS_NODE":
            "ADDRESS",

        "PAYMENT_NODE":
            "PAYMENT",

        "PAYMENT_INSTRUMENT":
            "PAYMENT",

        "REFUND_NODE":
            "REFUND",
    }

    value = aliases.get(
        value,
        value,
    )

    return value


# ============================================================
# EDGE VALIDATION
# ============================================================

def validate_graph_edges(
    node_ids: set[str],
    edges: Iterable[GraphEdge],
) -> None:

    for edge in edges:

        if edge.source_id not in node_ids:

            raise ValueError(
                "Graph edge references unknown "
                f"source node: {edge.source_id}"
            )

        if edge.target_id not in node_ids:

            raise ValueError(
                "Graph edge references unknown "
                f"target node: {edge.target_id}"
            )


# ============================================================
# NODE FEATURE BUILDING
# ============================================================

def _base_node_features(
    node_type: str,
) -> list[float]:

    node_type = normalize_node_type(
        node_type
    )

    features = [
        0.0
        for _ in range(
            NUM_NODE_FEATURES
        )
    ]

    type_index = {

        "CUSTOMER": 0,
        "ORDER": 1,
        "DEVICE": 2,
        "ADDRESS": 3,
        "PAYMENT": 4,
        "REFUND": 5,
    }

    if node_type in type_index:

        features[
            type_index[node_type]
        ] = 1.0

    return features


def build_customer_features(
    customer,
) -> list[float]:

    features = _base_node_features(
        "CUSTOMER"
    )

    features[6] = 0.0

    features[7] = _normalize_count(
        customer.order_count
    )

    features[8] = _normalize_count(
        customer.refund_count
    )

    features[9] = _normalize_amount(
        customer.refund_amount_total
    )

    features[10] = _normalize_count(
        customer.chargeback_count
    )

    features[11] = _normalize_account_age(
        customer.account_age_days
    )

    return features


def build_order_features(
    order,
) -> list[float]:

    features = _base_node_features(
        "ORDER"
    )

    features[6] = _normalize_amount(
        order.order_amount
    )

    features[7] = _normalize_count(
        order.quantity
    )

    return features


def build_refund_features(
    refund,
) -> list[float]:

    features = _base_node_features(
        "REFUND"
    )

    features[6] = _normalize_amount(
        refund.requested_amount
    )

    features[9] = _normalize_amount(
        refund.approved_amount
    )

    return features


def build_generic_features(
    node: GraphNode,
) -> list[float]:

    features = _base_node_features(
        node.node_type
    )

    attributes = node.attributes or {}

    # Only observable/non-ground-truth attributes
    # are considered here.

    if "amount" in attributes:

        features[6] = _normalize_amount(
            attributes["amount"]
        )

    elif "order_amount" in attributes:

        features[6] = _normalize_amount(
            attributes["order_amount"]
        )

    elif "requested_amount" in attributes:

        features[6] = _normalize_amount(
            attributes["requested_amount"]
        )

    if "activity_count" in attributes:

        features[7] = _normalize_count(
            attributes["activity_count"]
        )

    if "refund_count" in attributes:

        features[8] = _normalize_count(
            attributes["refund_count"]
        )

    if "refund_amount_total" in attributes:

        features[9] = _normalize_amount(
            attributes["refund_amount_total"]
        )

    if "chargeback_count" in attributes:

        features[10] = _normalize_count(
            attributes["chargeback_count"]
        )

    if "account_age_days" in attributes:

        features[11] = _normalize_account_age(
            attributes["account_age_days"]
        )

    return features


# ============================================================
# WORLD → NODE FEATURES
# ============================================================

def build_node_features(
    world: EnvironmentState,
    nodes: list[GraphNode],
) -> list[list[float]]:

    feature_rows = []

    for node in nodes:

        node_type = normalize_node_type(
            node.node_type
        )

        # ----------------------------------------------------
        # Customer
        # ----------------------------------------------------

        if node_type == "CUSTOMER":

            customer = world.customers.get(
                node.node_id
            )

            if customer is not None:

                features = build_customer_features(
                    customer
                )

            else:

                features = build_generic_features(
                    node
                )

        # ----------------------------------------------------
        # Order
        # ----------------------------------------------------

        elif node_type == "ORDER":

            order = world.orders.get(
                node.node_id
            )

            if order is not None:

                features = build_order_features(
                    order
                )

            else:

                features = build_generic_features(
                    node
                )

        # ----------------------------------------------------
        # Refund
        # ----------------------------------------------------

        elif node_type == "REFUND":

            refund = world.refunds.get(
                node.node_id
            )

            if refund is not None:

                features = build_refund_features(
                    refund
                )

            else:

                features = build_generic_features(
                    node
                )

        # ----------------------------------------------------
        # Device / Address / Payment
        # ----------------------------------------------------

        else:

            features = build_generic_features(
                node
            )

        feature_rows.append(
            features
        )

    return feature_rows


# ============================================================
# EDGE → PYTORCH GEOMETRIC FORMAT
# ============================================================

def build_edge_index(
    node_index: dict[str, int],
    edges: list[GraphEdge],
) -> torch.Tensor:

    edge_pairs = []

    for edge in edges:

        source = node_index[
            edge.source_id
        ]

        target = node_index[
            edge.target_id
        ]

        # ----------------------------------------------------
        # Add both directions.
        #
        # This allows information to propagate through the
        # relationship regardless of the original direction.
        # ----------------------------------------------------

        edge_pairs.append(
            [source, target]
        )

        edge_pairs.append(
            [target, source]
        )

    if not edge_pairs:

        return torch.empty(
            (2, 0),
            dtype=torch.long,
        )

    return torch.tensor(
        edge_pairs,
        dtype=torch.long,
    ).t().contiguous()


# ============================================================
# EDGE WEIGHTS
# ============================================================

def build_edge_weights(
    edges: list[GraphEdge],
) -> torch.Tensor:

    weights = []

    for edge in edges:

        weight = max(
            0.0,
            _safe_float(
                edge.weight
            ),
        )

        # Add the same weight to the reverse edge.
        weights.append(
            weight
        )

        weights.append(
            weight
        )

    if not weights:

        return torch.empty(
            (0,),
            dtype=torch.float32,
        )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# BUILD INTERACTION GRAPH
# ============================================================

def build_interaction_graph(
    world: EnvironmentState,
) -> InteractionGraph:

    nodes = list(
        world.graph_nodes
    )

    edges = list(
        world.graph_edges
    )

    # --------------------------------------------------------
    # If the world has no explicit graph nodes, construct a
    # minimal graph from the actual world entities.
    #
    # This is a fallback only. The preferred path is to use
    # graph_nodes / graph_edges generated by world_generator.
    # --------------------------------------------------------

    if not nodes:

        nodes = []

        # Customers
        for customer_id in world.customers:

            nodes.append(
                GraphNode(
                    node_id=customer_id,
                    node_type="CUSTOMER",
                    attributes={},
                )
            )

        # Orders
        for order_id in world.orders:

            nodes.append(
                GraphNode(
                    node_id=order_id,
                    node_type="ORDER",
                    attributes={},
                )
            )

        # Refunds
        for refund_id in world.refunds:

            nodes.append(
                GraphNode(
                    node_id=refund_id,
                    node_type="REFUND",
                    attributes={},
                )
            )

    if not nodes:

        raise ValueError(
            "Cannot build interaction graph: "
            "world contains no graph nodes."
        )

    # --------------------------------------------------------
    # Ensure node IDs are unique.
    # --------------------------------------------------------

    node_ids = [
        node.node_id
        for node in nodes
    ]

    if len(node_ids) != len(
        set(node_ids)
    ):

        raise ValueError(
            "Duplicate graph node IDs detected."
        )

    node_index = {
        node_id: index
        for index, node_id
        in enumerate(node_ids)
    }

    validate_graph_edges(
        node_ids=set(node_ids),
        edges=edges,
    )

    # --------------------------------------------------------
    # Build node features.
    # --------------------------------------------------------

    feature_rows = build_node_features(
        world=world,
        nodes=nodes,
    )

    x = torch.tensor(
        feature_rows,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Build graph connectivity.
    # --------------------------------------------------------

    edge_index = build_edge_index(
        node_index=node_index,
        edges=edges,
    )

    edge_weight = build_edge_weights(
        edges=edges,
    )

    # --------------------------------------------------------
    # Build PyG Data object.
    # --------------------------------------------------------

    data = Data(
        x=x,
        edge_index=edge_index,
    )

    if edge_weight.numel() > 0:

        data.edge_weight = edge_weight

    data.num_nodes = len(
        nodes
    )

    return InteractionGraph(
        data=data,
        node_ids=tuple(
            node_ids
        ),
        node_types=tuple(
            normalize_node_type(
                node.node_type
            )
            for node in nodes
        ),
        node_index=node_index,
    )


# ============================================================
# CUSTOMER SUBGRAPH
# ============================================================

def customer_node_index(
    graph: InteractionGraph,
    customer_id: str,
) -> int:

    if customer_id not in graph.node_index:

        raise KeyError(
            "Customer is not present in graph: "
            f"{customer_id}"
        )

    index = graph.node_index[
        customer_id
    ]

    if graph.node_types[index] != "CUSTOMER":

        raise ValueError(
            f"Node {customer_id} is not a "
            "CUSTOMER node."
        )

    return index


# ============================================================
# GRAPH SUMMARY
# ============================================================

def summarize_interaction_graph(
    graph: InteractionGraph,
) -> dict[str, int]:

    summary = {}

    for node_type in graph.node_types:

        summary[node_type] = (
            summary.get(
                node_type,
                0,
            )
            + 1
        )

    summary[
        "TOTAL_NODES"
    ] = graph.data.num_nodes

    summary[
        "TOTAL_EDGES"
    ] = graph.data.num_edges

    return summary


# ============================================================
# INFORMATION BOUNDARY AUDIT
# ============================================================

def audit_graph_information_boundary(
    world: EnvironmentState,
    graph: InteractionGraph,
) -> dict[str, bool]:

    """
    Verify that graph construction has not directly used
    customer private state or ground truth.

    The graph is allowed to represent observable structure.

    It must NOT expose:

        - is_abusive
        - population
        - behavior_type
        - abuse_family
        - counterparty_type
        - objective
        - current_strategy
        - strategy_beliefs
        - private_memory
    """

    forbidden_customer_fields = {
        "is_abusive",
        "population",
        "behavior_type",
        "abuse_family",
        "counterparty_type",
        "objective",
        "current_strategy",
        "strategy_beliefs",
        "private_memory",
    }

    violations = []

    for node in world.graph_nodes:

        attributes = node.attributes or {}

        for field in forbidden_customer_fields:

            if field in attributes:

                violations.append(
                    (
                        node.node_id,
                        field,
                    )
                )

    return {
        "no_private_or_ground_truth_attributes":
            len(violations) == 0,

        "graph_has_nodes":
            graph.data.num_nodes > 0,

        "feature_dimension_correct":
            graph.data.x.shape[1]
            == NUM_NODE_FEATURES,
    }


# ============================================================
# SMOKE TEST
# ============================================================

def _smoke_test() -> None:

    from ..environment.population_config import (
        DEFAULT_POPULATION_CONFIG,
    )

    from ..environment.world_generator import (
        create_world,
    )

    print()
    print("=" * 80)
    print(
        "INTERACTION GRAPH BUILDER SMOKE TEST"
    )
    print("=" * 80)

    world = create_world(
        config=DEFAULT_POPULATION_CONFIG,
        seed=42,
    )

    graph = build_interaction_graph(
        world
    )

    summary = summarize_interaction_graph(
        graph
    )

    audit = audit_graph_information_boundary(
        world,
        graph,
    )

    print()
    print("GRAPH")

    for key, value in summary.items():

        print(
            f"{key:<30}: {value}"
        )

    print()
    print(
        f"Node feature shape        : "
        f"{tuple(graph.data.x.shape)}"
    )

    print(
        f"Edge index shape          : "
        f"{tuple(graph.data.edge_index.shape)}"
    )

    print()
    print("INFORMATION BOUNDARY")

    for key, value in audit.items():

        print(
            f"{key:<45}: "
            f"{'PASSED' if value else 'FAILED'}"
        )

    assert audit[
        "no_private_or_ground_truth_attributes"
    ]

    assert audit[
        "graph_has_nodes"
    ]

    assert audit[
        "feature_dimension_correct"
    ]

    print()
    print(
        "Interaction graph construction: PASSED"
    )


if __name__ == "__main__":
    _smoke_test()