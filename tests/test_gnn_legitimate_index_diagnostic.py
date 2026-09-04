from __future__ import annotations

"""
GNN LEGITIMATE-CUSTOMER / INDEX-ARTIFACT DIAGNOSTIC

READ-ONLY DIAGNOSTIC.

Purpose:
    1. Run the existing trained GNN on multiple CUSTOMER nodes.
    2. Compare customers with different graph positions.
    3. Check whether GNN predictions are suspiciously constant.
    4. Check whether node index appears strongly associated with GNN risk.
    5. Inspect whether legitimate-looking customers receive near-identical
       / near-certain risk scores.

IMPORTANT:
    - Does NOT modify src/
    - Does NOT retrain the GNN
    - Does NOT modify the checkpoint
    - Does NOT modify the graph
    - Does NOT modify the Governor
    - Does NOT change the existing architecture

This is a diagnostic, not a formal statistical validation.
"""

from pathlib import Path
from statistics import mean, pstdev
from typing import Any

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


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "evaluation"
    / "gnn_risk_governor.pt"
)

# Number of CUSTOMER nodes to inspect.
# Keep this small so the diagnostic finishes quickly.
N_CUSTOMERS = 20

# We specifically inspect:
#   - first customer
#   - last customer
#   - several low-index customers
#   - several high-index customers
#
# This helps detect position/index artifacts.
LOW_INDEX_COUNT = 5
HIGH_INDEX_COUNT = 5

# Suspiciously high risk.
HIGH_RISK_THRESHOLD = 0.95

# If almost every customer receives approximately the same
# prediction, that is suspicious.
CONSTANT_RISK_TOLERANCE = 0.01

# If risk changes strongly with node position, flag it.
POSITION_CORRELATION_THRESHOLD = 0.70


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def subsection(title: str) -> None:
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def bounded(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(
        0.0,
        min(
            value,
            1.0,
        ),
    )


def safe_correlation(
    xs: list[float],
    ys: list[float],
) -> float | None:

    if len(xs) != len(ys):
        return None

    if len(xs) < 2:
        return None

    x_mean = mean(xs)
    y_mean = mean(ys)

    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, ys)
    )

    x_variance = sum(
        (x - x_mean) ** 2
        for x in xs
    )

    y_variance = sum(
        (y - y_mean) ** 2
        for y in ys
    )

    denominator = (
        x_variance * y_variance
    ) ** 0.5

    if denominator == 0:
        return None

    return numerator / denominator


def get_customer_nodes(graph) -> list[tuple[str, int]]:

    if not hasattr(graph, "node_ids"):
        raise AssertionError(
            "Graph does not expose node_ids."
        )

    if not hasattr(graph, "node_types"):
        raise AssertionError(
            "Graph does not expose node_types."
        )

    customer_nodes = []

    for index, (
        node_id,
        node_type,
    ) in enumerate(
        zip(
            graph.node_ids,
            graph.node_types,
        )
    ):

        if node_type == "CUSTOMER":

            customer_nodes.append(
                (
                    str(node_id),
                    int(index),
                )
            )

    if not customer_nodes:
        raise AssertionError(
            "No CUSTOMER nodes found."
        )

    return customer_nodes


def get_node_features(
    graph,
    node_index: int,
):

    x = graph.data.x

    if x is None:
        raise AssertionError(
            "Graph contains no node features."
        )

    if node_index < 0:
        raise AssertionError(
            "Negative node index."
        )

    if node_index >= x.shape[0]:
        raise AssertionError(
            "Node index outside feature matrix."
        )

    return x[node_index]


# ============================================================
# CHECK 1
# CHECKPOINT
# ============================================================

def check_checkpoint() -> None:

    subsection(
        "CHECK 1 — EXISTING TRAINED CHECKPOINT"
    )

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            "Existing GNN checkpoint not found:\n"
            f"{MODEL_FILE}"
        )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise AssertionError(
            "Checkpoint is not a dictionary."
        )

    required = {
        "model_state_dict",
        "input_dim",
        "hidden_dim",
    }

    missing = (
        required
        - set(checkpoint.keys())
    )

    if missing:

        raise AssertionError(
            "Checkpoint missing fields: "
            f"{sorted(missing)}"
        )

    print(
        f"[PASS] Checkpoint exists."
    )

    print(
        f"       Input dim  : "
        f"{checkpoint['input_dim']}"
    )

    print(
        f"       Hidden dim : "
        f"{checkpoint['hidden_dim']}"
    )


# ============================================================
# CHECK 2
# EXISTING WORLD + GRAPH
# ============================================================

def build_existing_graph():

    subsection(
        "CHECK 2 — EXISTING WORLD → INTERACTION GRAPH"
    )

    world = create_world(
        config=DEFAULT_POPULATION_CONFIG,
        seed=SEED,
    )

    graph = build_interaction_graph(
        world
    )

    if graph.data is None:
        raise AssertionError(
            "Graph contains no PyG data object."
        )

    if graph.data.x is None:
        raise AssertionError(
            "Graph contains no node features."
        )

    print(
        f"[PASS] Existing graph constructed."
    )

    print(
        f"       Nodes   : "
        f"{graph.data.num_nodes}"
    )

    print(
        f"       Edges   : "
        f"{graph.data.num_edges}"
    )

    print(
        f"       Features: "
        f"{tuple(graph.data.x.shape)}"
    )

    return graph


# ============================================================
# CHECK 3
# CUSTOMER NODE DISCOVERY
# ============================================================

def inspect_customer_nodes(graph):

    subsection(
        "CHECK 3 — CUSTOMER NODE DISCOVERY"
    )

    customer_nodes = get_customer_nodes(
        graph
    )

    print(
        f"[PASS] CUSTOMER nodes found: "
        f"{len(customer_nodes)}"
    )

    print()
    print(
        "First 10 CUSTOMER nodes:"
    )

    for customer_id, index in (
        customer_nodes[:10]
    ):

        print(
            f"  index={index:<6} "
            f"id={customer_id}"
        )

    return customer_nodes


# ============================================================
# CHECK 4
# NODE FEATURE / POSITION AUDIT
# ============================================================

def inspect_position_features(
    graph,
    customer_nodes,
):

    subsection(
        "CHECK 4 — NODE INDEX / FEATURE AUDIT"
    )

    selected = []

    if not customer_nodes:
        raise AssertionError(
            "No customers available."
        )

    # First few
    selected.extend(
        customer_nodes[
            :LOW_INDEX_COUNT
        ]
    )

    # Last few
    if len(customer_nodes) > (
        LOW_INDEX_COUNT
        + HIGH_INDEX_COUNT
    ):

        selected.extend(
            customer_nodes[
                -HIGH_INDEX_COUNT:
            ]
        )

    # Remove duplicates while preserving order.
    selected = list(
        dict.fromkeys(selected)
    )

    print(
        "Selected nodes for position audit:"
    )

    for customer_id, index in selected:

        features = get_node_features(
            graph,
            index,
        )

        print()
        print(
            f"Customer : {customer_id}"
        )

        print(
            f"Index    : {index}"
        )

        print(
            f"Features : "
            f"{features.detach().cpu().tolist()}"
        )

    print()
    print(
        "[PASS] Node features inspected "
        "without modifying the graph."
    )


# ============================================================
# CHECK 5
# MULTI-CUSTOMER GNN INFERENCE
# ============================================================

def run_multi_customer_inference(
    graph,
    customer_nodes,
    inference,
):

    subsection(
        "CHECK 5 — MULTI-CUSTOMER GNN OUTPUT AUDIT"
    )

    # Select evenly distributed customers.
    if len(customer_nodes) <= N_CUSTOMERS:

        selected = customer_nodes

    else:

        step = (
            len(customer_nodes)
            / N_CUSTOMERS
        )

        selected = []

        for i in range(N_CUSTOMERS):

            position = int(
                i * step
            )

            selected.append(
                customer_nodes[position]
            )

    results = []

    for customer_id, node_index in selected:

        risk = inference.predict_customer_risk(
            data=graph.data,
            customer_node_index=node_index,
        )

        risk = bounded(risk)

        results.append(
            {
                "customer_id": customer_id,
                "node_index": node_index,
                "risk": risk,
            }
        )

    print()
    print(
        f"{'Index':<10}"
        f"{'Customer':<25}"
        f"{'GNN Risk':<12}"
    )

    print(
        "-" * 47
    )

    for result in results:

        print(
            f"{result['node_index']:<10}"
            f"{result['customer_id']:<25}"
            f"{result['risk']:<12.6f}"
        )

    return results


# ============================================================
# CHECK 6
# CONSTANT-PREDICTION TEST
# ============================================================

def check_prediction_variance(
    results,
):

    subsection(
        "CHECK 6 — CONSTANT / NEAR-CONSTANT PREDICTION TEST"
    )

    risks = [
        result["risk"]
        for result in results
    ]

    if not risks:
        raise AssertionError(
            "No GNN predictions available."
        )

    minimum = min(risks)
    maximum = max(risks)

    average = mean(risks)

    std = pstdev(risks)

    spread = maximum - minimum

    print(
        f"Minimum risk : {minimum:.6f}"
    )

    print(
        f"Maximum risk : {maximum:.6f}"
    )

    print(
        f"Mean risk    : {average:.6f}"
    )

    print(
        f"Std deviation: {std:.6f}"
    )

    print(
        f"Risk spread  : {spread:.6f}"
    )

    if spread <= CONSTANT_RISK_TOLERANCE:

        print()
        print(
            "[WARNING] GNN predictions are "
            "near-constant across sampled customers."
        )

        print(
            "[WARNING] This is suspicious and "
            "requires investigation of training/data."
        )

    else:

        print()
        print(
            "[PASS] GNN predictions vary across "
            "sampled customers."
        )

    high_risk_count = sum(
        risk >= HIGH_RISK_THRESHOLD
        for risk in risks
    )

    high_risk_fraction = (
        high_risk_count
        / len(risks)
    )

    print()
    print(
        f"Customers >= {HIGH_RISK_THRESHOLD:.2f}: "
        f"{high_risk_count}/{len(risks)} "
        f"({high_risk_fraction:.1%})"
    )

    if high_risk_fraction >= 0.90:

        print(
            "[WARNING] More than 90% of sampled "
            "customers receive near-certain high risk."
        )


# ============================================================
# CHECK 7
# NODE POSITION CORRELATION
# ============================================================

def check_position_artifact(
    results,
):

    subsection(
        "CHECK 7 — NODE POSITION / INDEX ARTIFACT"
    )

    indices = [
        float(result["node_index"])
        for result in results
    ]

    risks = [
        float(result["risk"])
        for result in results
    ]

    correlation = safe_correlation(
        indices,
        risks,
    )

    if correlation is None:

        print(
            "[INFO] Position/risk correlation "
            "cannot be calculated."
        )

        return

    print(
        f"Pearson correlation "
        f"(node index vs GNN risk): "
        f"{correlation:.6f}"
    )

    if abs(correlation) >= POSITION_CORRELATION_THRESHOLD:

        print()
        print(
            "[WARNING] Strong correlation detected "
            "between node position and GNN risk."
        )

        print(
            "[WARNING] This may indicate a "
            "synthetic-data / node-order artifact."
        )

    else:

        print()
        print(
            "[PASS] No strong linear relationship "
            "between node position and GNN risk."
        )


# ============================================================
# CHECK 8
# LOW/HIGH INDEX COMPARISON
# ============================================================

def check_low_vs_high_index(
    results,
):

    subsection(
        "CHECK 8 — LOW-INDEX VS HIGH-INDEX COMPARISON"
    )

    if len(results) < 6:

        print(
            "[WARNING] Not enough customers "
            "for low/high index comparison."
        )

        return

    ordered = sorted(
        results,
        key=lambda x: x["node_index"],
    )

    n = min(
        3,
        len(ordered) // 2,
    )

    low_group = ordered[:n]
    high_group = ordered[-n:]

    low_risks = [
        x["risk"]
        for x in low_group
    ]

    high_risks = [
        x["risk"]
        for x in high_group
    ]

    low_mean = mean(
        low_risks
    )

    high_mean = mean(
        high_risks
    )

    difference = (
        high_mean
        - low_mean
    )

    print(
        "Low-index customers:"
    )

    for result in low_group:

        print(
            f"  index={result['node_index']:<6} "
            f"risk={result['risk']:.6f}"
        )

    print()
    print(
        "High-index customers:"
    )

    for result in high_group:

        print(
            f"  index={result['node_index']:<6} "
            f"risk={result['risk']:.6f}"
        )

    print()

    print(
        f"Low-index mean risk : "
        f"{low_mean:.6f}"
    )

    print(
        f"High-index mean risk: "
        f"{high_mean:.6f}"
    )

    print(
        f"Difference          : "
        f"{difference:.6f}"
    )

    if abs(difference) >= 0.25:

        print()
        print(
            "[WARNING] Large risk difference between "
            "low-index and high-index customers."
        )

        print(
            "[WARNING] Investigate node ordering "
            "and synthetic label generation."
        )

    else:

        print()
        print(
            "[PASS] No large low-index/high-index "
            "risk separation detected."
        )


# ============================================================
# CHECK 9
# SAME FEATURES / DIFFERENT POSITION
# ============================================================

def check_feature_position_relationship(
    graph,
    results,
):

    subsection(
        "CHECK 9 — NODE FEATURES VS POSITION"
    )

    # This is deliberately descriptive.
    # We do NOT mutate features or create a new graph.

    selected = results[:10]

    if len(selected) < 2:

        print(
            "[WARNING] Not enough samples."
        )

        return

    print(
        "Comparing feature vectors for sampled "
        "customers at different positions."
    )

    for result in selected:

        index = result["node_index"]

        features = get_node_features(
            graph,
            index,
        )

        feature_sum = float(
            features.detach()
            .cpu()
            .float()
            .sum()
            .item()
        )

        feature_mean = float(
            features.detach()
            .cpu()
            .float()
            .mean()
            .item()
        )

        print(
            f"index={index:<6} "
            f"risk={result['risk']:.6f} "
            f"feature_mean={feature_mean:.6f} "
            f"feature_sum={feature_sum:.6f}"
        )

    print()
    print(
        "[PASS] Feature/position relationship "
        "inspected descriptively."
    )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

def final_summary(
    results,
):

    section(
        "DIAGNOSTIC SUMMARY"
    )

    risks = [
        result["risk"]
        for result in results
    ]

    spread = (
        max(risks)
        - min(risks)
    )

    average = mean(
        risks
    )

    high_fraction = (
        sum(
            risk >= HIGH_RISK_THRESHOLD
            for risk in risks
        )
        / len(risks)
    )

    correlation = safe_correlation(
        [
            float(x["node_index"])
            for x in results
        ],
        risks,
    )

    suspicious = []

    if spread <= CONSTANT_RISK_TOLERANCE:

        suspicious.append(
            "near-constant GNN predictions"
        )

    if high_fraction >= 0.90:

        suspicious.append(
            "near-universal high GNN risk"
        )

    if (
        correlation is not None
        and abs(correlation)
        >= POSITION_CORRELATION_THRESHOLD
    ):

        suspicious.append(
            "strong node-index/risk correlation"
        )

    print(
        f"Customers tested       : "
        f"{len(results)}"
    )

    print(
        f"Mean GNN risk          : "
        f"{average:.6f}"
    )

    print(
        f"GNN risk spread        : "
        f"{spread:.6f}"
    )

    if correlation is not None:

        print(
            f"Index/risk correlation : "
            f"{correlation:.6f}"
        )

    print()

    if suspicious:

        print(
            "[RESULT] INVESTIGATION REQUIRED"
        )

        print(
            "Suspicious indicators:"
        )

        for item in suspicious:

            print(
                f"  - {item}"
            )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "This does NOT prove that the GNN is invalid."
        )

        print(
            "It means the existing checkpoint needs "
            "training-data / label / feature investigation "
            "before its predictions are treated as evidence."
        )

    else:

        print(
            "[RESULT] NO OBVIOUS GNN POSITION ARTIFACT "
            "DETECTED IN THIS SAMPLE."
        )

        print(
            "This is not proof of model validity; "
            "it only means these simple diagnostics "
            "did not expose an obvious artifact."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "GNN LEGITIMATE-CUSTOMER / INDEX-ARTIFACT DIAGNOSTIC"
    )

    print(
        "READ-ONLY diagnostic."
    )

    print(
        "No source architecture or checkpoint "
        "will be modified."
    )

    print(
        f"Seed       : {SEED}"
    )

    print(
        f"Customers  : {N_CUSTOMERS}"
    )

    # --------------------------------------------------------
    # 1. Checkpoint
    # --------------------------------------------------------

    check_checkpoint()

    # --------------------------------------------------------
    # 2. Existing inference
    # --------------------------------------------------------

    subsection(
        "CHECK 2B — EXISTING GNN INFERENCE"
    )

    inference = GNNRiskInference(
        model_file=MODEL_FILE
    )

    print(
        "[PASS] Existing GNNRiskInference initialized."
    )

    # --------------------------------------------------------
    # 3. Existing world + graph
    # --------------------------------------------------------

    graph = build_existing_graph()

    # --------------------------------------------------------
    # 4. Customer nodes
    # --------------------------------------------------------

    customer_nodes = (
        inspect_customer_nodes(
            graph
        )
    )

    # --------------------------------------------------------
    # 5. Feature / position inspection
    # --------------------------------------------------------

    inspect_position_features(
        graph,
        customer_nodes,
    )

    # --------------------------------------------------------
    # 6. GNN inference
    # --------------------------------------------------------

    results = (
        run_multi_customer_inference(
            graph=graph,
            customer_nodes=customer_nodes,
            inference=inference,
        )
    )

    # --------------------------------------------------------
    # 7. Constant prediction
    # --------------------------------------------------------

    check_prediction_variance(
        results
    )

    # --------------------------------------------------------
    # 8. Index artifact
    # --------------------------------------------------------

    check_position_artifact(
        results
    )

    # --------------------------------------------------------
    # 9. Low vs high index
    # --------------------------------------------------------

    check_low_vs_high_index(
        results
    )

    # --------------------------------------------------------
    # 10. Feature/position inspection
    # --------------------------------------------------------

    check_feature_position_relationship(
        graph,
        results,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    final_summary(
        results
    )

    print()
    print("=" * 80)
    print(
        "GNN LEGITIMATE-CUSTOMER / INDEX-ARTIFACT "
        "DIAGNOSTIC COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()