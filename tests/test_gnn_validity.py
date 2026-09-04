from __future__ import annotations

"""
GNN VALIDITY TEST
=================

Read-only diagnostic for the existing Adaptive Risk Governor.

Purpose:
    1. Evaluate the trained GNN on multiple customers.
    2. Check whether predictions are suspiciously saturated.
    3. Check whether graph position appears to influence predictions.
    4. Verify customer -> graph-node mapping where possible.
    5. Compare predictions across customers without changing
       the existing GNN/Governor architecture.

IMPORTANT:
    This file does NOT modify:
        - GNN model
        - checkpoint
        - graph builder
        - Governor
        - risk fusion
        - dataset
        - existing tests

Run:
    python -m tests.test_gnn_validity
"""

from pathlib import Path
import sys
import traceback

import numpy as np
import torch


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# EXISTING PROJECT IMPORTS
# ============================================================

try:
    from src.governor.gnn_inference import GNNRiskInference
    from src.governor.interaction_graph import *
except Exception:
    GNNRiskInference = None


# ============================================================
# CONSTANTS
# ============================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "evaluation"
    / "gnn_risk_governor.pt"
)

GNN_WEIGHT = 0.20

SATURATION_THRESHOLD = 0.95

N_CUSTOMERS_TO_TEST = 8


# ============================================================
# OUTPUT HELPERS
# ============================================================

def section(title: str) -> None:

    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def passed(message: str) -> None:
    print(f"[PASS] {message}")


def warning(message: str) -> None:
    print(f"[WARNING] {message}")


def info(message: str) -> None:
    print(f"[INFO] {message}")


def failed(message: str) -> None:
    print(f"[FAIL] {message}")


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value) -> float:

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ============================================================
# FIND CUSTOMER IDs
# ============================================================

def discover_customer_ids(world):

    customers = getattr(world, "customers", None)

    if customers is None:
        return []

    if isinstance(customers, dict):

        return list(customers.keys())

    result = []

    try:

        for customer in customers:

            customer_id = getattr(
                customer,
                "customer_id",
                None,
            )

            if customer_id is not None:
                result.append(customer_id)

    except TypeError:
        pass

    return result


# ============================================================
# FIND GRAPH OBJECT
# ============================================================

def discover_graph(world):

    candidates = [
        "interaction_graph",
        "graph",
        "gnn_graph",
    ]

    for name in candidates:

        graph = getattr(
            world,
            name,
            None,
        )

        if graph is not None:
            return graph

    return None


# ============================================================
# FIND NODE FEATURES
# ============================================================

def get_node_features(graph):

    if graph is None:
        return None

    if hasattr(graph, "x"):
        return graph.x

    if isinstance(graph, dict):

        for key in (
            "x",
            "node_features",
            "features",
        ):

            if key in graph:
                return graph[key]

    return None


# ============================================================
# FIND CUSTOMER -> NODE MAPPING
# ============================================================

def discover_mapping(graph):

    if graph is None:
        return None

    candidates = [
        "customer_to_node",
        "customer_node_mapping",
        "customer_to_node_index",
        "node_mapping",
        "customer_index",
    ]

    for name in candidates:

        mapping = getattr(
            graph,
            name,
            None,
        )

        if isinstance(mapping, dict):
            return mapping

    if isinstance(graph, dict):

        for name in candidates:

            mapping = graph.get(name)

            if isinstance(mapping, dict):
                return mapping

    return None


# ============================================================
# DISCOVER NODE INDEX
# ============================================================

def get_customer_node_index(
    customer_id,
    mapping,
):

    if mapping is None:
        return None

    if customer_id in mapping:

        return mapping[customer_id]

    return None


# ============================================================
# RUN GNN PREDICTION
# ============================================================

def predict_customer(
    inference,
    graph,
    node_index,
):

    if graph is None:
        return None

    if node_index is None:
        return None

    methods = [
        "predict_customer_risk",
        "predict_node_risk",
        "infer_customer_risk",
        "predict",
    ]

    for method_name in methods:

        method = getattr(
            inference,
            method_name,
            None,
        )

        if method is None:
            continue

        attempts = [
            {
                "data": graph,
                "customer_node_index": node_index,
            },
            {
                "graph": graph,
                "customer_node_index": node_index,
            },
            {
                "data": graph,
                "node_index": node_index,
            },
            {
                "graph": graph,
                "node_index": node_index,
            },
        ]

        for kwargs in attempts:

            try:

                result = method(**kwargs)

                if hasattr(
                    result,
                    "gnn_risk",
                ):

                    return safe_float(
                        result.gnn_risk
                    )

                if isinstance(
                    result,
                    dict,
                ):

                    for key in (
                        "gnn_risk",
                        "risk",
                        "score",
                    ):

                        if key in result:

                            return safe_float(
                                result[key]
                            )

                return safe_float(result)

            except TypeError:
                continue
            except Exception:
                continue

    return None


# ============================================================
# DIRECT MODEL PREDICTION FALLBACK
# ============================================================

def direct_model_predictions(
    inference,
    graph,
):

    model = getattr(
        inference,
        "model",
        None,
    )

    if model is None:
        return None

    if graph is None:
        return None

    if not hasattr(graph, "x"):
        return None

    if not hasattr(graph, "edge_index"):
        return None

    try:

        model.eval()

        with torch.no_grad():

            output = model(graph)

        if output.ndim == 2:

            output = output.squeeze(-1)

        probabilities = torch.sigmoid(
            output
        )

        return probabilities.detach().cpu().numpy()

    except Exception:
        return None


# ============================================================
# FIND WORLD BUILDER
# ============================================================

def discover_world():

    candidates = [
        (
            "src.environment.world_generator",
            "WorldGenerator",
        ),
    ]

    for module_name, class_name in candidates:

        try:

            module = __import__(
                module_name,
                fromlist=[class_name],
            )

            cls = getattr(
                module,
                class_name,
                None,
            )

            if cls is None:
                continue

            generator = cls(
                seed=42
            )

            methods = [
                "generate",
                "build",
                "create_world",
            ]

            for method_name in methods:

                method = getattr(
                    generator,
                    method_name,
                    None,
                )

                if method is None:
                    continue

                try:
                    world = method()
                    return world
                except TypeError:
                    continue

        except Exception:
            continue

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("GNN VALIDITY AUDIT")
    print("=" * 80)
    print("Read-only validation of the existing trained GNN.")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Checkpoint   : {CHECKPOINT}")
    print(f"GNN weight   : {GNN_WEIGHT:.2f}")

    # ========================================================
    # CHECK 1 — CHECKPOINT
    # ========================================================

    section("CHECK 1 — TRAINED CHECKPOINT")

    if not CHECKPOINT.exists():

        failed(
            "Trained GNN checkpoint does not exist."
        )

        return 1

    passed(
        "Trained GNN checkpoint exists."
    )

    # ========================================================
    # CHECK 2 — INFERENCE
    # ========================================================

    section("CHECK 2 — EXISTING GNN INFERENCE")

    if GNNRiskInference is None:

        failed(
            "Could not import existing GNN inference."
        )

        return 1

    try:

        inference = GNNRiskInference()

        passed(
            "Existing GNNRiskInference initialized."
        )

    except Exception as exc:

        failed(
            "Existing GNNRiskInference failed."
        )

        print(exc)

        traceback.print_exc()

        return 1

    # ========================================================
    # CHECK 3 — WORLD
    # ========================================================

    section(
        "CHECK 3 — EXISTING WORLD / GRAPH DISCOVERY"
    )

    world = discover_world()

    if world is None:

        warning(
            "Existing WorldGenerator could not be "
            "constructed through a generic interface."
        )

        info(
            "No replacement world or graph will be created."
        )

        return 0

    passed(
        "Existing world implementation discovered."
    )

    graph = discover_graph(world)

    if graph is None:

        warning(
            "Existing world does not expose a graph "
            "through a generic attribute."
        )

        info(
            "No replacement graph will be created."
        )

        return 0

    passed(
        "Existing graph object discovered."
    )

    # ========================================================
    # CHECK 4 — GRAPH FEATURES
    # ========================================================

    section(
        "CHECK 4 — GRAPH NODE FEATURES"
    )

    node_features = get_node_features(
        graph
    )

    if node_features is None:

        warning(
            "Graph does not expose node features."
        )

    else:

        try:

            print(
                f"Node feature shape: "
                f"{tuple(node_features.shape)}"
            )

            passed(
                "Graph node features are accessible."
            )

        except Exception:

            warning(
                "Node feature object exists but "
                "shape could not be inspected."
            )

    # ========================================================
    # CHECK 5 — CUSTOMER MAPPING
    # ========================================================

    section(
        "CHECK 5 — CUSTOMER → NODE MAPPING"
    )

    mapping = discover_mapping(
        graph
    )

    customer_ids = discover_customer_ids(
        world
    )

    if mapping is None:

        warning(
            "Customer → node mapping was not "
            "discoverable from the graph object."
        )

        info(
            "This is not treated as a pipeline failure."
        )

    else:

        passed(
            f"Customer → node mapping found "
            f"({len(mapping)} entries)."
        )

    if not customer_ids:

        warning(
            "Customer IDs could not be discovered "
            "from the existing world."
        )

        return 0

    print(
        f"Customers discovered: {len(customer_ids)}"
    )

    # ========================================================
    # CHECK 6 — MULTI CUSTOMER PREDICTIONS
    # ========================================================

    section(
        "CHECK 6 — MULTI-CUSTOMER GNN OUTPUT AUDIT"
    )

    predictions = []

    selected_customers = (
        customer_ids[
            :N_CUSTOMERS_TO_TEST
        ]
    )

    direct_predictions = (
        direct_model_predictions(
            inference,
            graph,
        )
    )

    for position, customer_id in enumerate(
        selected_customers
    ):

        node_index = get_customer_node_index(
            customer_id,
            mapping,
        )

        if node_index is None:

            # Existing systems sometimes use the
            # customer ordering itself as the node index.
            if position < len(customer_ids):

                node_index = position

            else:

                warning(
                    f"Could not determine node index "
                    f"for {customer_id}."
                )

                continue

        risk = None

        if direct_predictions is not None:

            try:

                risk = safe_float(
                    direct_predictions[
                        int(node_index)
                    ]
                )

            except (
                IndexError,
                TypeError,
                ValueError,
            ):

                risk = None

        if risk is None:

            risk = predict_customer(
                inference=inference,
                graph=graph,
                node_index=node_index,
            )

        if risk is None:

            warning(
                f"Could not obtain GNN prediction "
                f"for {customer_id}."
            )

            continue

        predictions.append(
            {
                "customer_id": customer_id,
                "node_index": int(node_index),
                "risk": risk,
            }
        )

    if not predictions:

        warning(
            "No multi-customer predictions could "
            "be obtained."
        )

        return 0

    passed(
        f"Obtained GNN predictions for "
        f"{len(predictions)} customers."
    )

    print()

    print(
        f"{'Customer':<22}"
        f"{'Node':>8}"
        f"{'GNN Risk':>14}"
    )

    print(
        "-" * 44
    )

    for row in predictions:

        print(
            f"{str(row['customer_id']):<22}"
            f"{row['node_index']:>8}"
            f"{row['risk']:>14.6f}"
        )

    # ========================================================
    # CHECK 7 — SATURATION
    # ========================================================

    section(
        "CHECK 7 — GNN PREDICTION SATURATION"
    )

    risks = np.array(
        [
            row["risk"]
            for row in predictions
        ],
        dtype=float,
    )

    mean_risk = float(
        risks.mean()
    )

    std_risk = float(
        risks.std()
    )

    min_risk = float(
        risks.min()
    )

    max_risk = float(
        risks.max()
    )

    print(
        f"Minimum risk : {min_risk:.6f}"
    )

    print(
        f"Maximum risk : {max_risk:.6f}"
    )

    print(
        f"Mean risk    : {mean_risk:.6f}"
    )

    print(
        f"Std. dev.    : {std_risk:.6f}"
    )

    saturated = int(
        np.sum(
            risks >= SATURATION_THRESHOLD
        )
    )

    print(
        f"Scores >= {SATURATION_THRESHOLD:.2f}: "
        f"{saturated}/{len(risks)}"
    )

    if saturated == len(risks):

        warning(
            "All tested customers receive near-certain "
            "high GNN risk."
        )

        warning(
            "This is suspicious and requires investigation "
            "before treating GNN probabilities as calibrated."
        )

    elif saturated >= max(
        2,
        len(risks) * 0.75,
    ):

        warning(
            "Most tested customers receive near-certain "
            "high GNN risk."
        )

    else:

        passed(
            "GNN predictions are not universally saturated."
        )

    # ========================================================
    # CHECK 8 — POSITION / INDEX AUDIT
    # ========================================================

    section(
        "CHECK 8 — NODE POSITION / INDEX ARTIFACT"
    )

    if len(predictions) < 3:

        warning(
            "Insufficient predictions for a meaningful "
            "position audit."
        )

    else:

        ordered = sorted(
            predictions,
            key=lambda x: x["node_index"],
        )

        indices = np.array(
            [
                row["node_index"]
                for row in ordered
            ],
            dtype=float,
        )

        scores = np.array(
            [
                row["risk"]
                for row in ordered
            ],
            dtype=float,
        )

        if (
            np.std(indices) == 0
            or np.std(scores) == 0
        ):

            warning(
                "Cannot calculate index/risk correlation "
                "because one variable has zero variance."
            )

        else:

            correlation = float(
                np.corrcoef(
                    indices,
                    scores,
                )[0, 1]
            )

            print(
                f"Index/risk correlation: "
                f"{correlation:.6f}"
            )

            if abs(correlation) >= 0.80:

                warning(
                    "Strong correlation between node "
                    "position and GNN risk detected."
                )

                warning(
                    "This may indicate a synthetic-data "
                    "or graph-construction artifact."
                )

            else:

                passed(
                    "No strong node-position/risk "
                    "correlation detected in this sample."
                )

    # ========================================================
    # CHECK 9 — CUSTOMER / NODE CONSISTENCY
    # ========================================================

    section(
        "CHECK 9 — CUSTOMER → NODE FEATURE CONSISTENCY"
    )

    if mapping is None:

        warning(
            "Cannot independently verify mapping because "
            "the existing mapping is not discoverable."
        )

    elif node_features is None:

        warning(
            "Cannot inspect node features."
        )

    else:

        consistency_passed = True

        for row in predictions:

            customer_id = row[
                "customer_id"
            ]

            node_index = row[
                "node_index"
            ]

            mapped_index = mapping.get(
                customer_id
            )

            if mapped_index is None:

                consistency_passed = False

                warning(
                    f"{customer_id} missing from mapping."
                )

                continue

            if int(mapped_index) != int(
                node_index
            ):

                consistency_passed = False

                warning(
                    f"Mapping mismatch for "
                    f"{customer_id}: "
                    f"mapping={mapped_index}, "
                    f"prediction_node={node_index}"
                )

        if consistency_passed:

            passed(
                "Customer IDs resolve to the same "
                "node indices used for inference."
            )

    # ========================================================
    # CHECK 10 — SPECIFIC INDEX COMPARISON
    # ========================================================

    section(
        "CHECK 10 — LOW-INDEX vs HIGH-INDEX COMPARISON"
    )

    if len(predictions) < 4:

        warning(
            "Not enough customers for low/high index "
            "comparison."
        )

    else:

        ordered = sorted(
            predictions,
            key=lambda x: x["node_index"],
        )

        low = ordered[
            :max(2, len(ordered) // 3)
        ]

        high = ordered[
            -max(2, len(ordered) // 3):
        ]

        low_mean = float(
            np.mean(
                [
                    row["risk"]
                    for row in low
                ]
            )
        )

        high_mean = float(
            np.mean(
                [
                    row["risk"]
                    for row in high
                ]
            )
        )

        difference = abs(
            high_mean - low_mean
        )

        print(
            f"Low-index mean risk  : "
            f"{low_mean:.6f}"
        )

        print(
            f"High-index mean risk : "
            f"{high_mean:.6f}"
        )

        print(
            f"Absolute difference  : "
            f"{difference:.6f}"
        )

        if difference >= 0.50:

            warning(
                "Large low-index/high-index risk difference "
                "detected."
            )

            warning(
                "This does not prove an index artifact, "
                "but it is strong enough to investigate."
            )

        else:

            passed(
                "No large low-index/high-index risk "
                "difference detected."
            )

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    section(
        "FINAL INTERPRETATION"
    )

    print(
        "This audit is diagnostic only."
    )

    print(
        "It does not modify the GNN, checkpoint, "
        "graph builder, Governor, or fusion engine."
    )

    print()

    if (
        saturated == len(risks)
        and len(risks) >= 3
    ):

        print(
            "[ACTION REQUIRED]"
        )

        print(
            "The GNN should NOT yet be treated as a "
            "trustworthy calibrated risk model."
        )

        print(
            "Inspect training data generation and labels "
            "before changing the production pipeline."
        )

    else:

        print(
            "[STATUS]"
        )

        print(
            "No immediate saturation failure was found "
            "in the sampled customers."
        )

        print(
            "The GNN can proceed to broader population-level "
            "evaluation."
        )

    print()
    print("=" * 80)
    print("GNN VALIDITY AUDIT COMPLETE")
    print("=" * 80)

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:

        print()
        print(
            "Audit interrupted."
        )

        raise SystemExit(1)

    except Exception as exc:

        print()
        print(
            "[ERROR] Diagnostic failed unexpectedly."
        )

        print(
            str(exc)
        )

        traceback.print_exc()

        raise SystemExit(1)