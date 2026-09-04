"""
GNN DIAGNOSTIC AUDIT
====================

Purpose
-------
This file is READ-ONLY with respect to the existing architecture.

It does NOT modify:

    - gnn_risk_governor.py
    - gnn_inference.py
    - gnn_governor_integration.py
    - train_gnn.py
    - risk_fusion.py
    - risk_governor.py
    - the trained checkpoint
    - the existing end-to-end pipeline

It only performs diagnostics for:

    1. Suspiciously high GNN outputs.
    2. Possible node-index / position artifacts.
    3. Customer -> graph-node mapping.
    4. GNN fusion / override behavior.
    5. Agent B reward / observation leakage.

IMPORTANT
---------
A diagnostic WARNING does not fail the existing architecture.

This script is intentionally conservative:
    - PASS = evidence looks healthy.
    - WARNING = worth investigating.
    - FAIL = a concrete integrity problem was found.

Run from project root:

    python -m tests.test_gnn_diagnostics

or:

    python tests/test_gnn_diagnostics.py
"""

from __future__ import annotations

import ast
import inspect
import math
import os
import sys
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONSTANTS
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "evaluation"
    / "gnn_risk_governor.pt"
)

GNN_WEIGHT = 0.20

HIGH_GNN_THRESHOLD = 0.85

NEAR_CERTAIN_THRESHOLD = 0.95

POSITION_CORRELATION_WARNING = 0.50

POSITION_MEAN_GAP_WARNING = 0.25

MAX_CUSTOMERS_TO_SAMPLE = 12


# ============================================================
# DISPLAY HELPERS
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


def passed(message: str) -> None:

    print(f"[PASS] {message}")


def warning(message: str) -> None:

    print(f"[WARNING] {message}")


def failed(message: str) -> None:

    print(f"[FAIL] {message}")


def info(message: str) -> None:

    print(f"[INFO] {message}")


# ============================================================
# SAFE CONVERSION
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except (
        TypeError,
        ValueError,
    ):

        return default


# ============================================================
# IMPORT EXISTING PIPELINE
# ============================================================

def import_existing_pipeline():

    """
    Import the EXISTING working pipeline.

    Nothing is recreated here.

    Returns:
        tuple containing the existing modules/classes.
    """

    try:

        from src.governor.gnn_inference import (
            GNNRiskInference,
        )

        from src.governor.gnn_risk_governor import (
            GNNRiskGovernor,
            GNNRiskModel,
        )

        passed(
            "Existing GNN inference/governor modules imported."
        )

        return (
            GNNRiskInference,
            GNNRiskGovernor,
            GNNRiskModel,
        )

    except Exception as exc:

        failed(
            "Could not import the existing GNN pipeline."
        )

        print(
            f"Import error: {type(exc).__name__}: {exc}"
        )

        return (
            None,
            None,
            None,
        )


# ============================================================
# CHECKPOINT
# ============================================================

def check_checkpoint() -> bool:

    subsection(
        "CHECK 1 — TRAINED CHECKPOINT"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    if not CHECKPOINT_PATH.exists():

        failed(
            "Trained GNN checkpoint was not found."
        )

        return False

    passed(
        "Trained GNN checkpoint exists."
    )

    return True


# ============================================================
# GNN INITIALIZATION
# ============================================================

def initialize_inference():

    subsection(
        "CHECK 2 — EXISTING GNN INFERENCE"
    )

    try:

        from src.governor.gnn_inference import (
            GNNRiskInference,
        )

        inference = GNNRiskInference()

        passed(
            "Existing GNNRiskInference initialized."
        )

        return inference

    except Exception as exc:

        failed(
            "Existing GNN inference could not be initialized."
        )

        print(
            f"Error: {type(exc).__name__}: {exc}"
        )

        return None


# ============================================================
# FIND GRAPH BUILDER
# ============================================================

def find_graph_builder():

    """
    Locate the graph construction function/class without
    modifying or replacing it.

    This searches the existing src/ tree for likely graph
    construction definitions.
    """

    candidates = []

    src_root = PROJECT_ROOT / "src"

    if not src_root.exists():
        return candidates

    keywords = (
        "graph",
        "interaction",
        "customer",
        "node",
    )

    for path in src_root.rglob("*.py"):

        try:

            text = path.read_text(
                encoding="utf-8"
            )

        except (
            OSError,
            UnicodeDecodeError,
        ):

            continue

        lower = text.lower()

        score = 0

        for keyword in keywords:

            if keyword in lower:
                score += 1

        if (
            "edge_index" in lower
            and "customer" in lower
            and "node" in lower
        ):

            score += 5

        if score >= 4:

            candidates.append(
                (
                    score,
                    path,
                )
            )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        path
        for _, path in candidates
    ]


# ============================================================
# STATIC GRAPH API DISCOVERY
# ============================================================

def inspect_graph_builder():

    subsection(
        "CHECK 3 — EXISTING GRAPH BUILDER DISCOVERY"
    )

    builders = find_graph_builder()

    if not builders:

        warning(
            "Could not automatically identify the graph builder."
        )

        warning(
            "No graph code was changed. "
            "Mapping diagnostics will therefore be limited."
        )

        return None

    print(
        "Candidate existing graph files:"
    )

    for path in builders[:8]:

        print(
            f"  {path.relative_to(PROJECT_ROOT)}"
        )

    passed(
        "Existing graph implementation located."
    )

    return builders[0]


# ============================================================
# GRAPH OBJECT EXTRACTION
# ============================================================

def try_build_existing_graph():

    """
    Attempt to use the existing end-to-end test's graph
    construction helpers.

    We deliberately reuse the project's own test code rather
    than creating a second graph implementation.
    """

    try:

        import src.governor.test_gnn_governor_pipeline as pipeline

    except Exception as exc:

        warning(
            "Could not import existing pipeline test module."
        )

        info(
            f"Reason: {type(exc).__name__}: {exc}"
        )

        return None

    # --------------------------------------------------------
    # Look for obvious graph construction functions.
    # --------------------------------------------------------

    preferred_names = [
        "build_interaction_graph",
        "build_graph",
        "create_interaction_graph",
        "construct_interaction_graph",
        "build_world_graph",
        "create_graph",
    ]

    for name in preferred_names:

        function = getattr(
            pipeline,
            name,
            None,
        )

        if callable(function):

            try:

                graph = function()

                if graph is not None:

                    passed(
                        f"Reused existing graph builder: {name}()"
                    )

                    return graph

            except TypeError:

                # Function requires arguments.
                continue

            except Exception as exc:

                warning(
                    f"Existing graph builder {name}() "
                    f"requires a different setup: {exc}"
                )

    warning(
        "The existing test module does not expose a "
        "zero-argument graph builder."
    )

    info(
        "No replacement graph is created."
    )

    return None


# ============================================================
# NODE MAPPING DISCOVERY
# ============================================================

def discover_customer_mapping(graph: Any):

    """
    Attempt to discover the existing customer -> node mapping
    from the graph/test objects.

    No new mapping is constructed.
    """

    if graph is None:

        return None

    candidate_names = [
        "customer_to_node",
        "customer_node_mapping",
        "customer_to_node_index",
        "node_mapping",
        "customer_mapping",
    ]

    for name in candidate_names:

        mapping = getattr(
            graph,
            name,
            None,
        )

        if isinstance(mapping, dict):

            return mapping

    # Sometimes mappings are stored in graph metadata.
    metadata = getattr(
        graph,
        "metadata",
        None,
    )

    if isinstance(metadata, dict):

        for name in candidate_names:

            mapping = metadata.get(name)

            if isinstance(mapping, dict):

                return mapping

    return None


# ============================================================
# CUSTOMER -> NODE MAPPING AUDIT
# ============================================================

def audit_customer_mapping(graph: Any):

    subsection(
        "CHECK 4 — CUSTOMER → GRAPH NODE MAPPING"
    )

    mapping = discover_customer_mapping(
        graph
    )

    if mapping is None:

        warning(
            "Existing graph object does not expose a "
            "customer→node mapping in a discoverable form."
        )

        info(
            "This is NOT treated as a pipeline failure."
        )

        return None

    if not mapping:

        failed(
            "Customer→node mapping exists but is empty."
        )

        return False

    passed(
        f"Customer→node mapping contains "
        f"{len(mapping)} entries."
    )

    # --------------------------------------------------------
    # Basic structural validation.
    # --------------------------------------------------------

    indices = []

    for customer_id, node_index in mapping.items():

        try:

            index = int(node_index)

        except (
            TypeError,
            ValueError,
        ):

            failed(
                f"Invalid node index for {customer_id}: "
                f"{node_index!r}"
            )

            return False

        indices.append(index)

    # Duplicate node indices can be legitimate in some
    # architectures, but for a customer-node identity mapping
    # they are suspicious.
    duplicates = (
        len(indices)
        - len(set(indices))
    )

    if duplicates > 0:

        warning(
            f"{duplicates} duplicate node indices detected."
        )

    else:

        passed(
            "Customer node indices are unique."
        )

    # --------------------------------------------------------
    # Check index range if graph features exist.
    # --------------------------------------------------------

    data = getattr(
        graph,
        "data",
        graph,
    )

    x = getattr(
        data,
        "x",
        None,
    )

    if x is not None:

        node_count = int(
            x.shape[0]
        )

        invalid = [
            index
            for index in indices
            if index < 0
            or index >= node_count
        ]

        if invalid:

            failed(
                "Mapping contains out-of-range node indices."
            )

            print(
                f"Invalid indices: {invalid[:10]}"
            )

            return False

        passed(
            f"All mapped node indices are within "
            f"0..{node_count - 1}."
        )

    return True


# ============================================================
# GNN SINGLE-NODE INFERENCE
# ============================================================

def predict_customer(
    inference: Any,
    graph: Any,
    node_index: int,
) -> float | None:

    if inference is None:
        return None

    if graph is None:
        return None

    # --------------------------------------------------------
    # Preferred existing inference APIs.
    # --------------------------------------------------------

    method_names = [
        "predict_customer_risk",
        "predict_customer",
        "infer_customer_risk",
        "get_customer_risk",
        "predict",
    ]

    for method_name in method_names:

        method = getattr(
            inference,
            method_name,
            None,
        )

        if not callable(method):
            continue

        # Try common signatures.
        signatures = [
            {
                "data": getattr(
                    graph,
                    "data",
                    graph,
                ),
                "node_index": node_index,
            },
            {
                "graph": graph,
                "node_index": node_index,
            },
            {
                "data": getattr(
                    graph,
                    "data",
                    graph,
                ),
                "customer_node_index": node_index,
            },
        ]

        for kwargs in signatures:

            try:

                result = method(
                    **kwargs
                )

                # Handle scalar.
                if isinstance(
                    result,
                    (float, int),
                ):

                    return max(
                        0.0,
                        min(
                            1.0,
                            float(result),
                        ),
                    )

                # Handle object carrying gnn_risk.
                value = getattr(
                    result,
                    "gnn_risk",
                    None,
                )

                if value is not None:

                    return max(
                        0.0,
                        min(
                            1.0,
                            safe_float(value),
                        ),
                    )

            except (
                TypeError,
                AttributeError,
            ):

                continue

            except Exception:

                continue

    # --------------------------------------------------------
    # Last-resort direct model inference.
    #
    # This uses the EXISTING inference model and graph.
    # It does not train or modify anything.
    # --------------------------------------------------------

    model = getattr(
        inference,
        "model",
        None,
    )

    data = getattr(
        graph,
        "data",
        graph,
    )

    if model is None or data is None:

        return None

    try:

        import torch

        model.eval()

        with torch.no_grad():

            logits = model(
                data
            )

            if logits.ndim == 2:

                logit = logits[
                    node_index,
                    0,
                ]

            else:

                logit = logits[
                    node_index
                ]

            risk = torch.sigmoid(
                logit
            ).item()

        return max(
            0.0,
            min(
                1.0,
                float(risk),
            ),
        )

    except Exception:

        return None


# ============================================================
# MULTI-CUSTOMER GNN OUTPUT AUDIT
# ============================================================

def audit_gnn_outputs(
    inference: Any,
    graph: Any,
    mapping: dict[Any, Any] | None,
):

    subsection(
        "CHECK 5 — MULTI-CUSTOMER GNN OUTPUT AUDIT"
    )

    if inference is None:

        warning(
            "GNN inference unavailable."
        )

        return None

    if graph is None:

        warning(
            "Existing graph unavailable."
        )

        return None

    if not mapping:

        warning(
            "Customer→node mapping unavailable."
        )

        return None

    customers = list(
        mapping.items()
    )

    # Prefer a varied spread through the graph rather than
    # only taking consecutive indices.
    if len(customers) > MAX_CUSTOMERS_TO_SAMPLE:

        positions = [
            0,
            len(customers) // 8,
            len(customers) // 4,
            len(customers) * 3 // 8,
            len(customers) // 2,
            len(customers) * 5 // 8,
            len(customers) * 3 // 4,
            len(customers) * 7 // 8,
            len(customers) - 1,
        ]

        selected = []

        for position in positions:

            if (
                0 <= position < len(customers)
                and customers[position]
                not in selected
            ):

                selected.append(
                    customers[position]
                )

        customers = selected

    results = []

    for customer_id, node_index in customers:

        risk = predict_customer(
            inference=inference,
            graph=graph,
            node_index=int(node_index),
        )

        if risk is None:

            warning(
                f"Could not infer GNN risk for "
                f"{customer_id}."
            )

            continue

        results.append(
            {
                "customer_id": str(
                    customer_id
                ),
                "node_index": int(
                    node_index
                ),
                "risk": float(risk),
            }
        )

        print(
            f"{str(customer_id):<25} "
            f"node={int(node_index):<6} "
            f"risk={risk:.6f}"
        )

    if not results:

        warning(
            "No customer predictions could be obtained."
        )

        return None

    risks = [
        item["risk"]
        for item in results
    ]

    high_count = sum(
        risk >= HIGH_GNN_THRESHOLD
        for risk in risks
    )

    near_certain_count = sum(
        risk >= NEAR_CERTAIN_THRESHOLD
        for risk in risks
    )

    mean_risk = (
        sum(risks)
        / len(risks)
    )

    minimum = min(risks)
    maximum = max(risks)

    print()
    print(
        f"Samples evaluated       : {len(risks)}"
    )

    print(
        f"Mean GNN risk            : {mean_risk:.6f}"
    )

    print(
        f"Minimum GNN risk         : {minimum:.6f}"
    )

    print(
        f"Maximum GNN risk         : {maximum:.6f}"
    )

    print(
        f"Risk >= {HIGH_GNN_THRESHOLD:.2f}         : "
        f"{high_count}/{len(risks)}"
    )

    print(
        f"Risk >= {NEAR_CERTAIN_THRESHOLD:.2f}         : "
        f"{near_certain_count}/{len(risks)}"
    )

    # --------------------------------------------------------
    # Interpretation.
    # --------------------------------------------------------

    if (
        len(risks) >= 4
        and near_certain_count == len(risks)
    ):

        warning(
            "Every sampled customer receives near-certain "
            "GNN risk."
        )

        warning(
            "This is strong evidence that the GNN output "
            "needs further investigation before being treated "
            "as calibrated risk."
        )

        status = "WARNING"

    elif (
        len(risks) >= 4
        and high_count / len(risks) >= 0.80
    ):

        warning(
            "A large majority of sampled customers receive "
            "very high GNN risk."
        )

        warning(
            "This does not prove a bug, but it is suspicious."
        )

        status = "WARNING"

    else:

        passed(
            "GNN outputs vary across sampled customers."
        )

        status = "PASS"

    return {
        "status": status,
        "results": results,
        "mean": mean_risk,
        "minimum": minimum,
        "maximum": maximum,
        "high_count": high_count,
        "near_certain_count": near_certain_count,
    }


# ============================================================
# NODE INDEX ARTIFACT AUDIT
# ============================================================

def pearson_correlation(
    x: list[float],
    y: list[float],
) -> float | None:

    if len(x) != len(y):
        return None

    if len(x) < 3:
        return None

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    numerator = sum(
        (
            a - mean_x
        )
        *
        (
            b - mean_y
        )
        for a, b in zip(x, y)
    )

    denominator_x = math.sqrt(
        sum(
            (a - mean_x) ** 2
            for a in x
        )
    )

    denominator_y = math.sqrt(
        sum(
            (b - mean_y) ** 2
            for b in y
        )
    )

    denominator = (
        denominator_x
        * denominator_y
    )

    if denominator == 0.0:
        return None

    return (
        numerator
        / denominator
    )


def audit_index_artifact(
    output_audit: dict[str, Any] | None,
):

    subsection(
        "CHECK 6 — NODE INDEX / POSITION ARTIFACT"
    )

    if not output_audit:

        warning(
            "Insufficient GNN predictions for position audit."
        )

        return None

    results = output_audit[
        "results"
    ]

    if len(results) < 4:

        warning(
            "Fewer than four samples available."
        )

        return None

    indices = [
        float(item["node_index"])
        for item in results
    ]

    risks = [
        float(item["risk"])
        for item in results
    ]

    correlation = pearson_correlation(
        indices,
        risks,
    )

    if correlation is None:

        warning(
            "Could not calculate index/risk correlation."
        )

        return None

    print(
        f"Node-index ↔ GNN-risk correlation: "
        f"{correlation:.6f}"
    )

    # --------------------------------------------------------
    # Compare early and late graph positions.
    # --------------------------------------------------------

    ordered = sorted(
        results,
        key=lambda item: item[
            "node_index"
        ],
    )

    split = max(
        1,
        len(ordered) // 2,
    )

    early = [
        item["risk"]
        for item in ordered[:split]
    ]

    late = [
        item["risk"]
        for item in ordered[split:]
    ]

    early_mean = (
        sum(early) / len(early)
    )

    late_mean = (
        sum(late) / len(late)
    )

    mean_gap = abs(
        late_mean
        - early_mean
    )

    print(
        f"Early-node mean risk     : {early_mean:.6f}"
    )

    print(
        f"Late-node mean risk      : {late_mean:.6f}"
    )

    print(
        f"Absolute mean difference : {mean_gap:.6f}"
    )

    # --------------------------------------------------------
    # This is deliberately a WARNING, not automatic failure.
    #
    # A correlation can arise legitimately because graph
    # ordering may correlate with data generation.
    # --------------------------------------------------------

    if (
        abs(correlation)
        >= POSITION_CORRELATION_WARNING
        or mean_gap
        >= POSITION_MEAN_GAP_WARNING
    ):

        warning(
            "Node position appears strongly associated "
            "with GNN risk."
        )

        warning(
            "This is a diagnostic warning, not proof of "
            "data leakage."
        )

        warning(
            "Inspect graph/data generation before trusting "
            "the GNN as a risk model."
        )

        return {
            "status": "WARNING",
            "correlation": correlation,
            "mean_gap": mean_gap,
        }

    passed(
        "No strong node-position/risk association detected "
        "in the sampled customers."
    )

    return {
        "status": "PASS",
        "correlation": correlation,
        "mean_gap": mean_gap,
    }


# ============================================================
# MAPPING CONSISTENCY AUDIT
# ============================================================

def audit_mapping_consistency(
    graph: Any,
    mapping: dict[Any, Any] | None,
):

    subsection(
        "CHECK 7 — CUSTOMER → NODE FEATURE CONSISTENCY"
    )

    if graph is None:

        warning(
            "Graph unavailable."
        )

        return None

    if not mapping:

        warning(
            "Customer→node mapping unavailable."
        )

        return None

    data = getattr(
        graph,
        "data",
        graph,
    )

    x = getattr(
        data,
        "x",
        None,
    )

    if x is None:

        warning(
            "Graph node features are not accessible."
        )

        return None

    node_count = int(
        x.shape[0]
    )

    checked = 0

    for customer_id, node_index in list(
        mapping.items()
    )[:MAX_CUSTOMERS_TO_SAMPLE]:

        try:

            index = int(node_index)

        except (
            TypeError,
            ValueError,
        ):

            failed(
                f"Invalid node index for {customer_id}."
            )

            return False

        if index < 0 or index >= node_count:

            failed(
                f"{customer_id} maps to invalid node "
                f"{index}."
            )

            return False

        feature = x[index]

        if feature is None:

            failed(
                f"{customer_id} maps to a node without "
                f"features."
            )

            return False

        if len(feature.shape) != 1:

            failed(
                f"{customer_id} maps to malformed node "
                f"features."
            )

            return False

        checked += 1

    if checked == 0:

        warning(
            "No mapping entries could be checked."
        )

        return None

    passed(
        f"Verified {checked} customer→node mappings "
        f"against actual graph node features."
    )

    return True


# ============================================================
# FUSION / OVERRIDE AUDIT
# ============================================================

def audit_fusion_policy():

    subsection(
        "CHECK 8 — GNN WEIGHT AND OVERRIDE POLICY"
    )

    try:

        from src.governor.gnn_risk_governor import (
            GNNRiskGovernor,
        )

        source = inspect.getsource(
            GNNRiskGovernor._fuse
        )

    except Exception as exc:

        warning(
            "Could not inspect existing _fuse() implementation."
        )

        info(
            f"Reason: {type(exc).__name__}: {exc}"
        )

        return None

    print(
        f"Configured GNN weight: {GNN_WEIGHT:.2f}"
    )

    weighted_formula_found = (
        "self.gnn_weight"
        in source
        and "gnn_risk"
        in source
    )

    override_found = (
        "gnn_risk >= 0.85"
        in source
        or "gnn_risk >= HIGH"
        in source
        or "GNN_HIGH_RISK_OVERRIDE"
        in source
    )

    if weighted_formula_found:

        passed(
            "Existing fusion code uses the configured "
            "GNN weight."
        )

    else:

        warning(
            "Could not confirm the weighted fusion formula "
            "from source inspection."
        )

    if override_found:

        print()
        info(
            "Existing code contains a high-GNN-risk override."
        )

        info(
            "Therefore a GNN weight of 0.20 does NOT mean "
            "the final score can never exceed 0.20 when the "
            "base score is zero."
        )

        info(
            "The override is a separate policy layer."
        )

    else:

        info(
            "No explicit GNN high-risk override was detected "
            "by source inspection."
        )

    return {
        "weighted_formula_found": weighted_formula_found,
        "override_found": override_found,
    }


# ============================================================
# AGENT B SOURCE AUDIT
# ============================================================

def find_agent_b_files() -> list[Path]:

    """
    Locate likely Agent B / attacker files.

    This is static inspection only.
    """

    roots = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "tests",
    ]

    matches = []

    names = (
        "agent_b",
        "attacker",
        "adversary",
        "adaptive",
    )

    for root in roots:

        if not root.exists():
            continue

        for path in root.rglob("*.py"):

            name = path.name.lower()

            if any(
                token in name
                for token in names
            ):

                matches.append(path)

    return sorted(
        set(matches)
    )


def audit_agent_b():

    subsection(
        "CHECK 9 — AGENT B REWARD / OBSERVATION AUDIT"
    )

    files = find_agent_b_files()

    if not files:

        warning(
            "No obvious Agent B / attacker Python file "
            "was automatically located."
        )

        info(
            "No code was changed."
        )

        return None

    print(
        "Candidate Agent B files:"
    )

    for path in files:

        print(
            f"  {path.relative_to(PROJECT_ROOT)}"
        )

    # --------------------------------------------------------
    # Search source text for relevant concepts.
    # --------------------------------------------------------

    reward_terms = {
        "reward",
        "observation",
        "observe",
        "agent_a",
        "governor",
        "fused_risk",
        "gnn_risk",
        "network_risk",
        "strategic",
        "final_decision",
        "action",
        "refund",
        "outcome",
    }

    findings = []

    for path in files:

        try:

            text = path.read_text(
                encoding="utf-8"
            )

        except (
            OSError,
            UnicodeDecodeError,
        ):

            continue

        try:

            tree = ast.parse(
                text
            )

        except SyntaxError:

            warning(
                f"Could not parse {path.name}."
            )

            continue

        for node in ast.walk(tree):

            if not isinstance(
                node,
                ast.Call,
            ):

                continue

            function_name = ""

            if isinstance(
                node.func,
                ast.Name,
            ):

                function_name = (
                    node.func.id
                )

            elif isinstance(
                node.func,
                ast.Attribute,
            ):

                function_name = (
                    node.func.attr
                )

            lower_name = (
                function_name.lower()
            )

            if not any(
                term in lower_name
                for term in reward_terms
            ):

                continue

            source_segment = ast.get_source_segment(
                text,
                node,
            )

            if source_segment:

                findings.append(
                    (
                        path,
                        function_name,
                        source_segment[:500],
                    )
                )

    # --------------------------------------------------------
    # Report relevant calls.
    # --------------------------------------------------------

    if not findings:

        warning(
            "No obvious Agent B reward/observation calls "
            "were detected automatically."
        )

        return {
            "status": "UNKNOWN",
            "files": files,
            "findings": [],
        }

    print()
    print(
        "Potential reward/observation pathways:"
    )

    for (
        path,
        function_name,
        source_segment,
    ) in findings[:30]:

        print()
        print(
            f"FILE   : "
            f"{path.relative_to(PROJECT_ROOT)}"
        )

        print(
            f"CALL   : {function_name}"
        )

        print(
            f"SOURCE : {source_segment}"
        )

    # --------------------------------------------------------
    # Look specifically for hidden Governor signals.
    # --------------------------------------------------------

    hidden_terms = (
        "gnn_risk",
        "network_risk",
        "strategic_risk",
        "strategic_adaptation_score",
        "fused_risk",
        "pre_gnn",
        "gnn_aware",
    )

    hidden_hits = []

    for (
        path,
        _,
        source_segment,
    ) in findings:

        lower = source_segment.lower()

        for term in hidden_terms:

            if term.lower() in lower:

                hidden_hits.append(
                    (
                        path,
                        term,
                        source_segment,
                    )
                )

    if hidden_hits:

        warning(
            "Potential direct access to hidden Governor "
            "signals was found in Agent B-related code."
        )

        for (
            path,
            term,
            source_segment,
        ) in hidden_hits[:20]:

            print(
                f"  {path.relative_to(PROJECT_ROOT)} "
                f"→ {term}"
            )

        return {
            "status": "WARNING",
            "files": files,
            "findings": findings,
            "hidden_hits": hidden_hits,
        }

    passed(
        "No direct hidden GNN/network/strategic signal "
        "reference was detected in the inspected Agent B "
        "source snippets."
    )

    info(
        "This does NOT prove outcome-based leakage is absent."
    )

    info(
        "It only establishes that direct hidden-signal "
        "references were not detected by this static audit."
    )

    return {
        "status": "PASS",
        "files": files,
        "findings": findings,
        "hidden_hits": [],
    }


# ============================================================
# OVERALL REPORT
# ============================================================

def print_summary(
    checkpoint_ok: bool,
    mapping_result: Any,
    output_result: Any,
    index_result: Any,
    consistency_result: Any,
    fusion_result: Any,
    agent_b_result: Any,
) -> None:

    section(
        "DIAGNOSTIC SUMMARY"
    )

    checks = [
        (
            "Checkpoint exists",
            checkpoint_ok,
        ),
        (
            "Customer→node mapping",
            mapping_result is not False,
        ),
        (
            "GNN multi-customer outputs",
            (
                output_result is None
                or output_result.get("status")
                != "FAIL"
            ),
        ),
        (
            "Node-index artifact",
            (
                index_result is None
                or index_result.get("status")
                != "FAIL"
            ),
        ),
        (
            "Node feature consistency",
            consistency_result is not False,
        ),
        (
            "Fusion/override inspection",
            fusion_result is not None,
        ),
        (
            "Agent B audit",
            (
                agent_b_result is None
                or agent_b_result.get("status")
                != "FAIL"
            ),
        ),
    ]

    for name, result in checks:

        print(
            f"{name:<42}: "
            f"{'PASS' if result else 'FAIL'}"
        )

    print()

    # --------------------------------------------------------
    # Important interpretation.
    # --------------------------------------------------------

    if (
        output_result is not None
        and output_result.get("status")
        == "WARNING"
    ):

        print(
            "IMPORTANT: The GNN output distribution is suspicious."
        )

        print(
            "Do NOT change the GNN yet."
        )

        print(
            "Use this result to inspect training/data generation."
        )

    if (
        index_result is not None
        and index_result.get("status")
        == "WARNING"
    ):

        print(
            "IMPORTANT: Node position may be associated with risk."
        )

        print(
            "This is a warning, not proof of leakage."
        )

    if (
        agent_b_result is not None
        and agent_b_result.get("status")
        == "WARNING"
    ):

        print(
            "IMPORTANT: Agent B may have a direct hidden-signal "
            "reference."
        )

        print(
            "Inspect the reported source before the adaptive "
            "attacker evaluation."
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
        "GNN DIAGNOSTIC AUDIT"
    )

    print(
        "Read-only diagnostic for the existing "
        "Adaptive Risk Governor architecture."
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"GNN weight : {GNN_WEIGHT:.2f}"
    )

    # --------------------------------------------------------
    # Existing imports.
    # --------------------------------------------------------

    (
        GNNRiskInference,
        GNNRiskGovernor,
        GNNRiskModel,
    ) = import_existing_pipeline()

    # --------------------------------------------------------
    # Check checkpoint.
    # --------------------------------------------------------

    checkpoint_ok = check_checkpoint()

    if not checkpoint_ok:

        print()
        failed(
            "Diagnostic cannot continue without the "
            "existing checkpoint."
        )

        return

    # --------------------------------------------------------
    # Initialize existing inference.
    # --------------------------------------------------------

    inference = initialize_inference()

    # --------------------------------------------------------
    # Discover graph implementation.
    # --------------------------------------------------------

    inspect_graph_builder()

    # --------------------------------------------------------
    # Reuse existing test graph if possible.
    # --------------------------------------------------------

    graph = try_build_existing_graph()

    # --------------------------------------------------------
    # Mapping audit.
    # --------------------------------------------------------

    mapping = None

    if graph is not None:

        mapping = discover_customer_mapping(
            graph
        )

    mapping_result = audit_customer_mapping(
        graph
    )

    # --------------------------------------------------------
    # GNN outputs.
    # --------------------------------------------------------

    output_result = audit_gnn_outputs(
        inference=inference,
        graph=graph,
        mapping=mapping,
    )

    # --------------------------------------------------------
    # Node position artifact.
    # --------------------------------------------------------

    index_result = audit_index_artifact(
        output_result
    )

    # --------------------------------------------------------
    # Mapping → actual features.
    # --------------------------------------------------------

    consistency_result = audit_mapping_consistency(
        graph=graph,
        mapping=mapping,
    )

    # --------------------------------------------------------
    # Fusion policy.
    # --------------------------------------------------------

    fusion_result = audit_fusion_policy()

    # --------------------------------------------------------
    # Agent B.
    # --------------------------------------------------------

    agent_b_result = audit_agent_b()

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    print_summary(
        checkpoint_ok=checkpoint_ok,
        mapping_result=mapping_result,
        output_result=output_result,
        index_result=index_result,
        consistency_result=consistency_result,
        fusion_result=fusion_result,
        agent_b_result=agent_b_result,
    )

    print()
    print("=" * 80)
    print(
        "GNN DIAGNOSTIC AUDIT COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":

    main()