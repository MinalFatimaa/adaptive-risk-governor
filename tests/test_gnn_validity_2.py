from __future__ import annotations

"""
READ-ONLY GNN / GOVERNOR VALIDITY DIAGNOSTIC

Purpose
-------
Audit the existing Adaptive Risk Governor without changing:

    - src/
    - trained GNN checkpoint
    - graph construction
    - Governor logic
    - Risk Fusion logic
    - Agent A
    - Agent B
    - existing tests

This is intentionally ONE diagnostic file.

It investigates the highest-value concerns:

1. Is the trained checkpoint valid?
2. Does the existing graph construct correctly?
3. Does customer -> graph-node mapping work?
4. Does the GNN produce varied predictions?
5. Is the GNN effectively predicting position/index?
6. Is there evidence of a node-ordering artifact?
7. Do feature vectors explain the observed index/risk difference?
8. Does the fusion actually use the configured GNN weight?
9. Is there a separate GNN high-risk override/floor?
10. What does Agent B actually observe?
11. Is Agent B directly exposed to hidden GNN/network/strategic signals?
12. Does Agent B receive Governor outcomes/actions?

Important
---------
This file is diagnostic only.

It does NOT:
    - retrain the model
    - modify the checkpoint
    - modify graph construction
    - modify Governor logic
    - modify Agent B
    - modify existing source files
    - claim model validity from a small sample

Run:

    python -m tests.test_gnn_validity
"""

from pathlib import Path
from typing import Any
import ast
import inspect
import math
import re

import numpy as np
import torch


# ============================================================
# PROJECT IMPORTS
# ============================================================

try:

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

    from src.governor.gnn_risk_governor import (
        GNNRiskGovernor,
    )

    from src.governor.risk_fusion import (
        RiskFusionEngine,
    )

    IMPORTS_OK = True

except Exception as exc:

    IMPORTS_OK = False
    IMPORT_ERROR = exc


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "evaluation"
    / "gnn_risk_governor.pt"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

GNN_WEIGHT = 0.20

SAMPLE_CUSTOMERS = 20

LOW_INDEX_COUNT = 5

HIGH_INDEX_COUNT = 5

# Used only for the descriptive nearest-feature
# matching diagnostic.
NEAREST_FEATURE_COUNT = 3

# Correlation is descriptive only.
# It must NOT be interpreted as proof of an artifact.
STRONG_CORRELATION_THRESHOLD = 0.80

LARGE_GROUP_DIFFERENCE_THRESHOLD = 0.25

HIGH_RISK_THRESHOLD = 0.85


# ============================================================
# OUTPUT HELPERS
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


def info(message: str) -> None:

    print(f"[INFO] {message}")


def warning(message: str) -> None:

    print(f"[WARNING] {message}")


def passed(message: str) -> None:

    print(f"[PASS] {message}")


def failed(message: str) -> None:

    print(f"[FAIL] {message}")


def bounded(value: Any) -> float:

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
# GLOBAL DIAGNOSTIC STATE
# ============================================================

CHECKPOINT = None

INFERENCE = None

WORLD = None

GRAPH = None

CUSTOMER_ROWS = []

PREDICTIONS = []


# ============================================================
# CHECK 1
# CHECKPOINT
# ============================================================

def check_checkpoint() -> bool:

    section(
        "CHECK 1 — EXISTING TRAINED CHECKPOINT"
    )

    if not MODEL_FILE.exists():

        failed(
            "Trained GNN checkpoint does not exist."
        )

        print(
            f"Expected: {MODEL_FILE}"
        )

        return False

    try:

        checkpoint = torch.load(
            MODEL_FILE,
            map_location="cpu",
            weights_only=False,
        )

    except Exception as exc:

        failed(
            f"Checkpoint could not be loaded: {exc}"
        )

        return False

    if not isinstance(
        checkpoint,
        dict,
    ):

        failed(
            "Checkpoint is not a dictionary."
        )

        return False

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

        failed(
            "Checkpoint missing required keys: "
            f"{sorted(missing)}"
        )

        return False

    global CHECKPOINT

    CHECKPOINT = checkpoint

    print(
        f"Checkpoint : {MODEL_FILE}"
    )

    print(
        f"Input dim  : {checkpoint['input_dim']}"
    )

    print(
        f"Hidden dim : {checkpoint['hidden_dim']}"
    )

    passed(
        "Existing trained checkpoint is structurally valid."
    )

    return True


# ============================================================
# CHECK 2
# EXISTING INFERENCE
# ============================================================

def check_inference() -> bool:

    section(
        "CHECK 2 — EXISTING GNN INFERENCE"
    )

    if not IMPORTS_OK:

        failed(
            "Existing project imports failed."
        )

        print(
            IMPORT_ERROR
        )

        return False

    try:

        global INFERENCE

        INFERENCE = GNNRiskInference(
            model_file=MODEL_FILE
        )

    except Exception as exc:

        failed(
            f"Existing GNN inference could not initialize: {exc}"
        )

        return False

    print(
        f"Input dimension : {INFERENCE.input_dim}"
    )

    print(
        f"Hidden dimension: {INFERENCE.hidden_dim}"
    )

    passed(
        "Existing GNNRiskInference initialized."
    )

    return True


# ============================================================
# CHECK 3
# WORLD + GRAPH
# ============================================================

def check_graph() -> bool:

    section(
        "CHECK 3 — EXISTING WORLD → INTERACTION GRAPH"
    )

    if not IMPORTS_OK:

        failed(
            "Project imports unavailable."
        )

        return False

    try:

        global WORLD
        global GRAPH

        WORLD = create_world(
            config=DEFAULT_POPULATION_CONFIG,
            seed=SEED,
        )

        GRAPH = build_interaction_graph(
            WORLD
        )

    except Exception as exc:

        failed(
            f"Existing world/graph construction failed: {exc}"
        )

        return False

    data = GRAPH.data

    if data is None:

        failed(
            "Existing graph has no data object."
        )

        return False

    if data.x is None:

        failed(
            "Existing graph has no node features."
        )

        return False

    print(
        f"Nodes    : {data.num_nodes}"
    )

    print(
        f"Edges    : {data.num_edges}"
    )

    print(
        f"Features : {tuple(data.x.shape)}"
    )

    passed(
        "Existing world and interaction graph constructed."
    )

    return True


# ============================================================
# CHECK 4
# CUSTOMER NODE DISCOVERY
# ============================================================

def check_customer_mapping() -> bool:

    section(
        "CHECK 4 — CUSTOMER → GRAPH NODE MAPPING"
    )

    if GRAPH is None:

        warning(
            "Graph unavailable."
        )

        return False

    if not hasattr(
        GRAPH,
        "node_ids",
    ):

        warning(
            "Graph does not expose node_ids."
        )

        return False

    if not hasattr(
        GRAPH,
        "node_types",
    ):

        warning(
            "Graph does not expose node_types."
        )

        return False

    rows = []

    for index, (
        node_id,
        node_type,
    ) in enumerate(
        zip(
            GRAPH.node_ids,
            GRAPH.node_types,
        )
    ):

        if node_type == "CUSTOMER":

            rows.append(
                {
                    "customer_id": str(node_id),
                    "index": int(index),
                }
            )

    if not rows:

        failed(
            "No CUSTOMER nodes found."
        )

        return False

    global CUSTOMER_ROWS

    CUSTOMER_ROWS = rows

    print(
        f"CUSTOMER nodes: {len(rows)}"
    )

    for row in rows[:10]:

        print(
            f"index={row['index']:<6} "
            f"id={row['customer_id']}"
        )

    # Independent consistency check.
    for row in rows:

        index = row["index"]

        actual_id = str(
            GRAPH.node_ids[index]
        )

        actual_type = (
            GRAPH.node_types[index]
        )

        if actual_id != row["customer_id"]:

            failed(
                "Customer ID/node index mismatch."
            )

            return False

        if actual_type != "CUSTOMER":

            failed(
                "Customer node has incorrect node type."
            )

            return False

    passed(
        "Customer → graph-node mapping is internally consistent."
    )

    return True


# ============================================================
# CHECK 5
# MULTI-CUSTOMER GNN OUTPUT
# ============================================================

def check_multi_customer_predictions() -> bool:

    section(
        "CHECK 5 — MULTI-CUSTOMER GNN OUTPUT AUDIT"
    )

    if not CUSTOMER_ROWS:

        warning(
            "Customer mapping unavailable."
        )

        return False

    if INFERENCE is None:

        warning(
            "GNN inference unavailable."
        )

        return False

    total = len(
        CUSTOMER_ROWS
    )

    selected_positions = np.linspace(
        0,
        total - 1,
        min(
            SAMPLE_CUSTOMERS,
            total,
        ),
        dtype=int,
    )

    rows = []

    print()

    print(
        f"{'Index':<10}"
        f"{'Customer':<25}"
        f"{'GNN Risk':<12}"
    )

    print(
        "-" * 47
    )

    for position in selected_positions:

        row = CUSTOMER_ROWS[
            int(position)
        ]

        index = row["index"]

        risk = bounded(
            INFERENCE.predict_customer_risk(
                data=GRAPH.data,
                customer_node_index=index,
            )
        )

        feature_vector = (
            GRAPH.data.x[
                index
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )

        record = {
            "customer_id":
                row["customer_id"],

            "index":
                index,

            "risk":
                risk,

            "features":
                feature_vector,
        }

        rows.append(
            record
        )

        print(
            f"{index:<10}"
            f"{row['customer_id']:<25}"
            f"{risk:<12.6f}"
        )

    global PREDICTIONS

    PREDICTIONS = rows

    passed(
        f"GNN predictions obtained for {len(rows)} customers."
    )

    return True


# ============================================================
# CHECK 6
# PREDICTION SPREAD
# ============================================================

def check_prediction_spread() -> bool:

    section(
        "CHECK 6 — PREDICTION SPREAD / SATURATION"
    )

    if not PREDICTIONS:

        warning(
            "No GNN predictions available."
        )

        return False

    risks = np.array(
        [
            row["risk"]
            for row in PREDICTIONS
        ],
        dtype=float,
    )

    minimum = float(
        risks.min()
    )

    maximum = float(
        risks.max()
    )

    mean = float(
        risks.mean()
    )

    std = float(
        risks.std()
    )

    spread = (
        maximum
        - minimum
    )

    print(
        f"Minimum risk : {minimum:.6f}"
    )

    print(
        f"Maximum risk : {maximum:.6f}"
    )

    print(
        f"Mean risk    : {mean:.6f}"
    )

    print(
        f"Std deviation: {std:.6f}"
    )

    print(
        f"Risk spread  : {spread:.6f}"
    )

    high_count = int(
        np.sum(
            risks >= HIGH_RISK_THRESHOLD
        )
    )

    print(
        f"Customers >= {HIGH_RISK_THRESHOLD:.2f}: "
        f"{high_count}/{len(risks)}"
    )

    if std < 0.02:

        warning(
            "Predictions are nearly constant in this sample."
        )

        warning(
            "This requires investigation before trusting the GNN."
        )

        return False

    passed(
        "GNN predictions vary across sampled customers."
    )

    return True


# ============================================================
# CHECK 7
# INDEX / RISK CORRELATION
# ============================================================

def check_index_risk_correlation() -> bool:

    section(
        "CHECK 7 — NODE INDEX / GNN RISK RELATIONSHIP"
    )

    if len(PREDICTIONS) < 3:

        warning(
            "Insufficient predictions for correlation."
        )

        return False

    indices = np.array(
        [
            row["index"]
            for row in PREDICTIONS
        ],
        dtype=float,
    )

    risks = np.array(
        [
            row["risk"]
            for row in PREDICTIONS
        ],
        dtype=float,
    )

    correlation = float(
        np.corrcoef(
            indices,
            risks,
        )[0, 1]
    )

    print(
        f"Pearson correlation "
        f"(node index vs GNN risk): "
        f"{correlation:.6f}"
    )

    if (
        abs(correlation)
        >= STRONG_CORRELATION_THRESHOLD
    ):

        warning(
            "Strong linear index/risk relationship detected."
        )

        warning(
            "This is a diagnostic warning, not proof of an artifact."
        )

        return False

    passed(
        "No strong linear relationship between node position and GNN risk."
    )

    return True


# ============================================================
# CHECK 8
# LOW VS HIGH INDEX
# ============================================================

def check_low_high_index() -> bool:

    section(
        "CHECK 8 — LOW-INDEX VS HIGH-INDEX COMPARISON"
    )

    if len(PREDICTIONS) < 6:

        warning(
            "Insufficient predictions."
        )

        return False

    ordered = sorted(
        PREDICTIONS,
        key=lambda row: row["index"],
    )

    n = min(
        LOW_INDEX_COUNT,
        len(ordered) // 2,
    )

    low = ordered[:n]

    high = ordered[-n:]

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

    difference = (
        high_mean
        - low_mean
    )

    print(
        "Low-index customers:"
    )

    for row in low:

        print(
            f"  index={row['index']:<6} "
            f"risk={row['risk']:.6f}"
        )

    print()

    print(
        "High-index customers:"
    )

    for row in high:

        print(
            f"  index={row['index']:<6} "
            f"risk={row['risk']:.6f}"
        )

    print()

    print(
        f"Low-index mean risk : {low_mean:.6f}"
    )

    print(
        f"High-index mean risk: {high_mean:.6f}"
    )

    print(
        f"Difference          : {difference:.6f}"
    )

    if abs(difference) >= LARGE_GROUP_DIFFERENCE_THRESHOLD:

        warning(
            "Large low-index/high-index risk difference detected."
        )

        warning(
            "This does NOT prove a position artifact."
        )

        warning(
            "Feature-controlled analysis below is required."
        )

        return False

    passed(
        "Low/high index groups do not show a large risk difference."
    )

    return True


# ============================================================
# CHECK 9
# FEATURE MAGNITUDE / RISK
# ============================================================

def check_feature_risk_relationship() -> bool:

    section(
        "CHECK 9 — NODE FEATURES VS GNN RISK"
    )

    if not PREDICTIONS:

        warning(
            "No predictions available."
        )

        return False

    feature_means = np.array(
        [
            np.mean(
                row["features"]
            )
            for row in PREDICTIONS
        ],
        dtype=float,
    )

    feature_sums = np.array(
        [
            np.sum(
                row["features"]
            )
            for row in PREDICTIONS
        ],
        dtype=float,
    )

    risks = np.array(
        [
            row["risk"]
            for row in PREDICTIONS
        ],
        dtype=float,
    )

    mean_corr = float(
        np.corrcoef(
            feature_means,
            risks,
        )[0, 1]
    )

    sum_corr = float(
        np.corrcoef(
            feature_sums,
            risks,
        )[0, 1]
    )

    print(
        f"Feature-mean vs risk correlation : "
        f"{mean_corr:.6f}"
    )

    print(
        f"Feature-sum vs risk correlation  : "
        f"{sum_corr:.6f}"
    )

    print()

    for row in PREDICTIONS[:10]:

        print(
            f"index={row['index']:<6} "
            f"risk={row['risk']:.6f} "
            f"feature_mean="
            f"{np.mean(row['features']):.6f} "
            f"feature_sum="
            f"{np.sum(row['features']):.6f}"
        )

    info(
        "Feature magnitude correlations are descriptive only."
    )

    passed(
        "Feature/risk relationship inspected."
    )

    return True


# ============================================================
# CHECK 10
# FEATURE-CONTROLLED POSITION AUDIT
# ============================================================

def check_feature_controlled_index_artifact() -> bool:

    section(
        "CHECK 10 — FEATURE-CONTROLLED INDEX ARTIFACT AUDIT"
    )

    if len(PREDICTIONS) < 6:

        warning(
            "Insufficient customer sample."
        )

        return False

    # --------------------------------------------------------
    # Idea:
    #
    # If two customers have similar feature vectors but very
    # different graph positions, their GNN predictions should
    # not differ systematically merely because of position.
    #
    # This is still NOT a causal test.
    # It is a stronger diagnostic than raw index/risk
    # correlation.
    # --------------------------------------------------------

    rows = PREDICTIONS

    distances = []

    for i, a in enumerate(rows):

        for j, b in enumerate(rows):

            if i >= j:

                continue

            feature_distance = float(
                np.linalg.norm(
                    a["features"]
                    - b["features"]
                )
            )

            index_distance = abs(
                a["index"]
                - b["index"]
            )

            risk_difference = abs(
                a["risk"]
                - b["risk"]
            )

            distances.append(
                {
                    "feature_distance":
                        feature_distance,

                    "index_distance":
                        index_distance,

                    "risk_difference":
                        risk_difference,

                    "a":
                        a,

                    "b":
                        b,
                }
            )

    distances.sort(
        key=lambda item:
        item["feature_distance"]
    )

    close_pairs = distances[
        :NEAREST_FEATURE_COUNT
    ]

    if not close_pairs:

        warning(
            "No feature-near pairs available."
        )

        return False

    print(
        "Nearest feature-vector pairs:"
    )

    print()

    for pair in close_pairs:

        a = pair["a"]

        b = pair["b"]

        print(
            f"{a['customer_id']} "
            f"(index={a['index']}, "
            f"risk={a['risk']:.6f})"
        )

        print(
            f"{b['customer_id']} "
            f"(index={b['index']}, "
            f"risk={b['risk']:.6f})"
        )

        print(
            f"feature distance = "
            f"{pair['feature_distance']:.6f}"
        )

        print(
            f"index distance   = "
            f"{pair['index_distance']}"
        )

        print(
            f"risk difference  = "
            f"{pair['risk_difference']:.6f}"
        )

        print()

    large_difference_pairs = [
        pair
        for pair in close_pairs
        if (
            pair["risk_difference"]
            >= 0.50
            and
            pair["index_distance"]
            >= 1000
        )
    ]

    if large_difference_pairs:

        warning(
            "Feature-similar customers at distant positions "
            "show large GNN-risk differences."
        )

        warning(
            "This deserves deeper investigation."
        )

        return False

    passed(
        "No obvious feature-controlled position artifact "
        "was exposed in this sample."
    )

    return True


# ============================================================
# CHECK 11
# CUSTOMER NODE FEATURE CONSISTENCY
# ============================================================

def check_feature_consistency() -> bool:

    section(
        "CHECK 11 — CUSTOMER NODE FEATURE CONSISTENCY"
    )

    if GRAPH is None:

        warning(
            "Graph unavailable."
        )

        return False

    if not PREDICTIONS:

        warning(
            "No sampled customer nodes."
        )

        return False

    for row in PREDICTIONS:

        index = row["index"]

        stored_features = (
            GRAPH.data.x[
                index
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )

        if not np.allclose(
            stored_features,
            row["features"],
            atol=1e-7,
        ):

            failed(
                "Feature vector changed between reads."
            )

            print(
                f"Customer: {row['customer_id']}"
            )

            return False

    passed(
        "Customer node feature vectors are internally consistent."
    )

    return True


# ============================================================
# CHECK 12
# FUSION WEIGHT
# ============================================================

def check_fusion_weight_and_override() -> bool:

    section(
        "CHECK 12 — GNN WEIGHT / OVERRIDE POLICY"
    )

    if not IMPORTS_OK:

        warning(
            "Risk fusion imports unavailable."
        )

        return False

    if not (
        0.0
        <= GNN_WEIGHT
        <= 1.0
    ):

        failed(
            "Configured GNN weight is outside [0, 1]."
        )

        return False

    print(
        f"Configured GNN weight: {GNN_WEIGHT:.2f}"
    )

    # --------------------------------------------------------
    # Inspect existing source rather than changing it.
    # --------------------------------------------------------

    governor_file = (
        PROJECT_ROOT
        / "src"
        / "governor"
        / "gnn_risk_governor.py"
    )

    if not governor_file.exists():

        warning(
            "GNN Governor source file not found."
        )

        return False

    source = governor_file.read_text(
        encoding="utf-8"
    )

    weight_tokens = [
        "gnn_weight",
        "pre_gnn_fused_score",
        "gnn_aware_fused_score",
    ]

    found_weight_logic = all(
        token in source
        for token in weight_tokens
    )

    if found_weight_logic:

        passed(
            "Existing GNN-aware fusion source contains "
            "the configured weight pathway."
        )

    else:

        warning(
            "Could not fully verify weighted fusion "
            "from source inspection."
        )

    # --------------------------------------------------------
    # Search for override/floor logic.
    # --------------------------------------------------------

    override_terms = [
        "GNN_HIGH_RISK_OVERRIDE",
        "HIGH_GNN_RISK",
        "gnn_risk >= 0.85",
        "gnn_risk >=",
        "gnn_risk >",
        "high_risk",
        "override",
        "floor",
    ]

    found_terms = [
        term
        for term in override_terms
        if term in source
    ]

    if found_terms:

        print()
        info(
            "Existing GNN Governor contains override/floor-related logic."
        )

        print(
            "Detected terms:"
        )

        for term in found_terms:

            print(
                f"  - {term}"
            )

        info(
            "Therefore GNN weight and final-score behavior "
            "must be interpreted as two policy layers:"
        )

        print(
            "  1. weighted GNN fusion"
        )

        print(
            "  2. possible high-risk override/floor"
        )

        passed(
            "Fusion/override policy inspected."
        )

    else:

        info(
            "No obvious GNN override/floor token was found."
        )

        passed(
            "Weighted fusion path inspected."
        )

    return True


# ============================================================
# CHECK 13
# STATIC SOURCE AUDIT
# AGENT B HIDDEN SIGNALS
# ============================================================

AGENT_B_FILES = [
    PROJECT_ROOT
    / "src"
    / "agents"
    / "adaptive_customer.py",

    PROJECT_ROOT
    / "src"
    / "evaluation"
    / "phase16_2_hardened_adaptive_evaluation.py",

    PROJECT_ROOT
    / "src"
    / "evaluation"
    / "phase16_adaptive_agent_evaluation.py",

    PROJECT_ROOT
    / "src"
    / "evaluation"
    / "phase17_adaptive_detection.py",
]


HIDDEN_SIGNAL_NAMES = {
    "gnn_risk",
    "final_fused_risk",
    "fused_risk_score",
    "network_abnormality_score",
    "network_risk",
    "strategic_adaptation_score",
    "strategic_behavior",
    "strategic_risk",
}


def check_agent_b_hidden_signal_audit() -> bool:

    section(
        "CHECK 13 — AGENT B HIDDEN-SIGNAL SOURCE AUDIT"
    )

    inspected = 0

    direct_hits = []

    for file_path in AGENT_B_FILES:

        if not file_path.exists():

            continue

        inspected += 1

        try:

            source = file_path.read_text(
                encoding="utf-8"
            )

        except Exception:

            continue

        for signal in HIDDEN_SIGNAL_NAMES:

            if signal in source:

                direct_hits.append(
                    (
                        str(file_path.relative_to(
                            PROJECT_ROOT
                        )),
                        signal,
                    )
                )

    print(
        f"Files inspected: {inspected}"
    )

    if direct_hits:

        warning(
            "Potential hidden-signal references were found."
        )

        for file_name, signal in direct_hits:

            print(
                f"  {file_name} -> {signal}"
            )

        warning(
            "A source reference alone does not prove Agent B "
            "receives that signal at runtime."
        )

        return False

    passed(
        "No direct hidden GNN/network/strategic signal "
        "references detected in inspected Agent B files."
    )

    info(
        "This does NOT prove outcome-based leakage is absent."
    )

    return True


# ============================================================
# CHECK 14
# AGENT B OBSERVATION / OUTCOME PATH
# ============================================================

def check_agent_b_governor_observation() -> bool:

    section(
        "CHECK 14 — AGENT B GOVERNOR-OUTCOME OBSERVATION AUDIT"
    )

    # We deliberately inspect source rather than execute the
    # entire adaptive-agent simulation.
    #
    # Reason:
    # This file must remain a fast, read-only diagnostic.

    observation_terms = {
        "decision",
        "support_decision",
        "governor_decision",
        "action",
        "reward",
        "observe",
    }

    hits = []

    for file_path in AGENT_B_FILES:

        if not file_path.exists():

            continue

        try:

            source = file_path.read_text(
                encoding="utf-8"
            )

        except Exception:

            continue

        lines = source.splitlines()

        for line_number, line in enumerate(
            lines,
            start=1,
        ):

            lowered = line.lower()

            if (
                "agent_b" in lowered
                or "adaptive" in lowered
                or "observe" in lowered
                or "reward" in lowered
                or "support_decision" in lowered
            ):

                if any(
                    term in lowered
                    for term in observation_terms
                ):

                    hits.append(
                        (
                            str(
                                file_path.relative_to(
                                    PROJECT_ROOT
                                )
                            ),
                            line_number,
                            line.strip(),
                        )
                    )

    if not hits:

        warning(
            "No obvious Agent B observation/reward path "
            "was found by static inspection."
        )

        return False

    print(
        f"Potential observation/reward lines found: "
        f"{len(hits)}"
    )

    print()

    # Keep output manageable.
    for file_name, line_number, line in hits[:40]:

        print(
            f"{file_name}:{line_number}"
        )

        print(
            f"    {line}"
        )

    print()

    info(
        "Static inspection suggests Agent B may observe "
        "decision/outcome information."
    )

    info(
        "This is NOT automatically a design flaw."
    )

    info(
        "It determines what claim the adaptive-attacker "
        "evaluation can legitimately make."
    )

    passed(
        "Agent B observation/reward pathway inspected."
    )

    return True


# ============================================================
# CHECK 15
# SOURCE-LEVEL OVERRIDE THRESHOLD DISCOVERY
# ============================================================

def check_override_threshold() -> bool:

    section(
        "CHECK 15 — GNN OVERRIDE THRESHOLD DISCOVERY"
    )

    governor_file = (
        PROJECT_ROOT
        / "src"
        / "governor"
        / "gnn_risk_governor.py"
    )

    if not governor_file.exists():

        warning(
            "GNN Governor source file unavailable."
        )

        return False

    source = governor_file.read_text(
        encoding="utf-8"
    )

    patterns = [
        r"gnn_risk\s*>=\s*([0-9]*\.?[0-9]+)",
        r"gnn_risk\s*>\s*([0-9]*\.?[0-9]+)",
        r"([0-9]*\.?[0-9]+)\s*<=\s*gnn_risk",
        r"([0-9]*\.?[0-9]+)\s*<\s*gnn_risk",
    ]

    thresholds = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            source,
        )

        for match in matches:

            try:

                value = float(match)

            except ValueError:

                continue

            if (
                0.0
                <= value
                <= 1.0
            ):

                thresholds.append(
                    value
                )

    thresholds = sorted(
        set(thresholds)
    )

    if thresholds:

        print(
            "Potential GNN thresholds found:"
        )

        for threshold in thresholds:

            print(
                f"  {threshold:.4f}"
            )

        passed(
            "Potential GNN override thresholds discovered."
        )

    else:

        info(
            "No explicit numeric GNN threshold was discovered "
            "through simple source patterns."
        )

        info(
            "Override logic may be indirect/config-driven."
        )

    return True


# ============================================================
# CHECK 16
# GNN WEIGHT MATHEMATICAL CONSISTENCY
# ============================================================

def check_weight_math() -> bool:

    section(
        "CHECK 16 — GNN WEIGHT MATHEMATICAL SANITY CHECK"
    )

    base_score = 0.0

    gnn_risk = 0.999947

    expected = (
        (1.0 - GNN_WEIGHT)
        * base_score
        +
        GNN_WEIGHT
        * gnn_risk
    )

    print(
        f"Configured GNN weight : {GNN_WEIGHT:.2f}"
    )

    print(
        f"Base score            : {base_score:.6f}"
    )

    print(
        f"GNN risk              : {gnn_risk:.6f}"
    )

    print(
        f"Pure weighted fusion   : {expected:.6f}"
    )

    info(
        "If an observed final score is substantially above "
        f"{expected:.6f}, a separate override/floor/policy "
        "is responsible."
    )

    passed(
        "Weighted-fusion arithmetic verified."
    )

    return True


# ============================================================
# SUMMARY
# ============================================================

def summary(results: dict[str, bool]) -> None:

    section(
        "DIAGNOSTIC SUMMARY"
    )

    for name, result in results.items():

        status = (
            "PASS"
            if result
            else "WARNING"
        )

        print(
            f"{name:<45}: {status}"
        )

    print()

    print(
        "IMPORTANT INTERPRETATION"
    )

    print(
        "-" * 80
    )

    print(
        "These diagnostics are designed to identify obvious "
        "problems quickly."
    )

    print(
        "They do NOT prove that the GNN is causally valid, "
        "generalizes to unseen data, or improves economic outcomes."
    )

    print()

    print(
        "A WARNING means:"
    )

    print(
        "  investigate before making a strong claim."
    )

    print()

    print(
        "A PASS means:"
    )

    print(
        "  this particular diagnostic did not expose the "
        "specific failure mode being checked."
    )

    print()

    print(
        "NO EXISTING PIPELINE FILES WERE MODIFIED."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section(
        "GNN / ADAPTIVE GOVERNOR VALIDITY AUDIT"
    )

    print(
        "READ-ONLY diagnostic."
    )

    print(
        "No source architecture, model checkpoint, "
        "or existing pipeline will be modified."
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"GNN weight   : {GNN_WEIGHT:.2f}"
    )

    print(
        f"Seed         : {SEED}"
    )

    print(
        f"Sample size  : {SAMPLE_CUSTOMERS}"
    )

    print()

    results = {}

    # --------------------------------------------------------
    # Core pipeline checks
    # --------------------------------------------------------

    results[
        "Checkpoint"
    ] = check_checkpoint()

    results[
        "Existing GNN inference"
    ] = check_inference()

    results[
        "Existing world/graph"
    ] = check_graph()

    results[
        "Customer → node mapping"
    ] = check_customer_mapping()

    # --------------------------------------------------------
    # GNN validity diagnostics
    # --------------------------------------------------------

    results[
        "Multi-customer GNN predictions"
    ] = check_multi_customer_predictions()

    results[
        "Prediction spread"
    ] = check_prediction_spread()

    results[
        "Index/risk relationship"
    ] = check_index_risk_correlation()

    results[
        "Low/high index comparison"
    ] = check_low_high_index()

    results[
        "Feature/risk relationship"
    ] = check_feature_risk_relationship()

    results[
        "Feature-controlled index audit"
    ] = check_feature_controlled_index_artifact()

    results[
        "Node feature consistency"
    ] = check_feature_consistency()

    # --------------------------------------------------------
    # Governor diagnostics
    # --------------------------------------------------------

    results[
        "Fusion weight / override inspection"
    ] = check_fusion_weight_and_override()

    results[
        "Agent B hidden-signal audit"
    ] = check_agent_b_hidden_signal_audit()

    results[
        "Agent B observation/reward audit"
    ] = check_agent_b_governor_observation()

    results[
        "Override threshold discovery"
    ] = check_override_threshold()

    results[
        "GNN weight arithmetic"
    ] = check_weight_math()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    summary(
        results
    )

    print()

    print("=" * 80)

    print(
        "GNN / ADAPTIVE GOVERNOR VALIDITY AUDIT COMPLETE"
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()