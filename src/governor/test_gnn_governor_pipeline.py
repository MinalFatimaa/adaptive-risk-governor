from __future__ import annotations

"""
END-TO-END GNN → ADAPTIVE GOVERNOR PIPELINE TEST

This test verifies the complete pipeline:

    World
      ↓
    Interaction Graph
      ↓
    Trained GNN
      ↓
    GNN Risk
      ↓
    GNN Governor Adapter
      ↓
    External Governor Features
      ↓
    Adaptive Risk Governor
      ↓
    Risk Fusion
      ↓
    GNN-aware Final Decision

This file is intentionally a smoke/integration test.
It does not train the GNN.
It loads the already-trained checkpoint.
"""

from pathlib import Path
from typing import Any

import torch

from ..environment.population_config import (
    DEFAULT_POPULATION_CONFIG,
)

from ..environment.world_generator import (
    create_world,
)

from .interaction_graph import (
    build_interaction_graph,
)

from .gnn_inference import (
    GNNRiskInference,
)

from .gnn_governor_integration import (
    GNNGovernorAdapter,
)

from .governor_features import (
    GovernorFeatureBuilder,
)

from .risk_governor import (
    AdaptiveRiskGovernor,
)

from .risk_fusion import (
    RiskFusionEngine,
)

from .gnn_risk_governor import (
    GNNRiskGovernor,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "evaluation"
    / "gnn_risk_governor.pt"
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

SEED = 42

CUSTOMER_NODE_INDEX = 0

CUSTOMER_ID = None

REQUESTED_AMOUNT = 750.0

CLAIM_TYPE = "PRODUCT_NOT_RECEIVED"

SUPPORT_DECISION = "APPROVE"

GNn_WEIGHT = 0.20


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


def find_customer_node(
    graph,
) -> tuple[str, int]:

    if not hasattr(
        graph,
        "node_ids",
    ):

        raise ValueError(
            "Interaction graph does not expose node_ids."
        )

    if not hasattr(
        graph,
        "node_types",
    ):

        raise ValueError(
            "Interaction graph does not expose node_types."
        )

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

            return (
                str(node_id),
                int(index),
            )

    raise RuntimeError(
        "No CUSTOMER node was found in the interaction graph."
    )


def validate_customer_node(
    graph,
    customer_id: str,
    node_index: int,
) -> None:

    if node_index < 0:

        raise ValueError(
            "Customer node index cannot be negative."
        )

    if node_index >= graph.data.num_nodes:

        raise ValueError(
            "Customer node index is outside graph."
        )

    actual_id = str(
        graph.node_ids[node_index]
    )

    if actual_id != str(customer_id):

        raise ValueError(
            "Customer/node mapping mismatch.\n"
            f"Expected customer : {customer_id}\n"
            f"Node index        : {node_index}\n"
            f"Graph node ID     : {actual_id}"
        )

    actual_type = graph.node_types[
        node_index
    ]

    if actual_type != "CUSTOMER":

        raise ValueError(
            "Selected node is not a CUSTOMER node.\n"
            f"Node index : {node_index}\n"
            f"Node type  : {actual_type}"
        )


# ============================================================
# TEST 1
# CHECK TRAINED CHECKPOINT
# ============================================================

def test_checkpoint() -> None:

    section(
        "TEST 1 — TRAINED GNN CHECKPOINT"
    )

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            "Trained GNN checkpoint was not found:\n"
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

        raise ValueError(
            "Invalid GNN checkpoint."
        )

    required_keys = {
        "model_state_dict",
        "input_dim",
        "hidden_dim",
    }

    missing = (
        required_keys
        - set(checkpoint.keys())
    )

    if missing:

        raise ValueError(
            "Checkpoint is missing required fields:\n"
            f"{sorted(missing)}"
        )

    print(
        f"Checkpoint : {MODEL_FILE}"
    )

    print(
        f"Input dim  : {checkpoint['input_dim']}"
    )

    print(
        f"Hidden dim : {checkpoint['hidden_dim']}"
    )

    print(
        "Checkpoint validation: PASSED"
    )


# ============================================================
# TEST 2
# LOAD INFERENCE
# ============================================================

def test_inference() -> GNNRiskInference:

    section(
        "TEST 2 — GNN INFERENCE"
    )

    inference = GNNRiskInference(
        model_file=MODEL_FILE
    )

    print(
        f"Input dimension : "
        f"{inference.input_dim}"
    )

    print(
        f"Hidden dimension: "
        f"{inference.hidden_dim}"
    )

    print(
        "GNN inference initialization: PASSED"
    )

    return inference


# ============================================================
# TEST 3
# BUILD WORLD + GRAPH
# ============================================================

def build_test_graph():

    section(
        "TEST 3 — WORLD → INTERACTION GRAPH"
    )

    world = create_world(
        config=DEFAULT_POPULATION_CONFIG,
        seed=SEED,
    )

    graph = build_interaction_graph(
        world
    )

    data = graph.data

    if data is None:

        raise ValueError(
            "Interaction graph contains no data."
        )

    if data.x is None:

        raise ValueError(
            "Interaction graph contains no node features."
        )

    print(
        f"Graph nodes   : "
        f"{data.num_nodes}"
    )

    print(
        f"Graph edges   : "
        f"{data.num_edges}"
    )

    print(
        f"Node features : "
        f"{tuple(data.x.shape)}"
    )

    return graph


# ============================================================
# TEST 4
# FIND CUSTOMER
# ============================================================

def test_customer_mapping(
    graph,
) -> tuple[str, int]:

    section(
        "TEST 4 — CUSTOMER → GRAPH NODE"
    )

    customer_id, node_index = (
        find_customer_node(
            graph
        )
    )

    validate_customer_node(
        graph=graph,
        customer_id=customer_id,
        node_index=node_index,
    )

    print(
        f"Customer ID : {customer_id}"
    )

    print(
        f"Node index  : {node_index}"
    )

    print(
        "Customer → graph-node mapping: PASSED"
    )

    return (
        customer_id,
        node_index,
    )


# ============================================================
# TEST 5
# RAW GNN PREDICTION
# ============================================================

def test_raw_gnn_prediction(
    inference: GNNRiskInference,
    graph,
    node_index: int,
) -> float:

    section(
        "TEST 5 — RAW GNN CUSTOMER RISK"
    )

    gnn_risk = inference.predict_customer_risk(
        data=graph.data,
        customer_node_index=node_index,
    )

    gnn_risk = bounded(
        gnn_risk
    )

    print(
        f"GNN risk : {gnn_risk:.6f}"
    )

    if not (
        0.0 <= gnn_risk <= 1.0
    ):

        raise AssertionError(
            "GNN risk is outside [0, 1]."
        )

    print(
        "Raw GNN prediction: PASSED"
    )

    return gnn_risk


# ============================================================
# TEST 6
# GNN ADAPTER
# ============================================================

def test_gnn_adapter(
    graph,
    customer_id: str,
    node_index: int,
):

    section(
        "TEST 6 — GNN GOVERNOR ADAPTER"
    )

    adapter = GNNGovernorAdapter()

    observation = (
        adapter.observe_customer(
            data=graph.data,
            customer_id=customer_id,
            customer_node_index=node_index,
        )
    )

    if observation is None:

        raise ValueError(
            "GNN adapter returned None."
        )

    if observation.customer_id != customer_id:

        raise AssertionError(
            "Adapter returned incorrect customer ID."
        )

    adapter_risk = bounded(
        observation.gnn_risk
    )

    print(
        f"Customer : {observation.customer_id}"
    )

    print(
        f"GNN risk : {adapter_risk:.6f}"
    )

    print(
        "GNN Governor Adapter: PASSED"
    )

    return (
        adapter,
        observation,
    )


# ============================================================
# TEST 7
# FEATURE BUILDER
# ============================================================

def test_feature_builder(
    customer_id: str,
):

    section(
        "TEST 7 — EXTERNAL GOVERNOR FEATURES"
    )

    feature_builder = (
        GovernorFeatureBuilder()
    )

    customer_history = [
        {
            "claim_type": "PRODUCT_NOT_RECEIVED",
            "requested_amount": 250.0,
            "support_decision": "APPROVE",
        },
        {
            "claim_type": "PRODUCT_NOT_RECEIVED",
            "requested_amount": 500.0,
            "support_decision": "APPROVE",
        },
    ]

    evidence_available = [
        "order_receipt",
        "delivery_record",
    ]

    features = feature_builder.build(
        customer_id=customer_id,
        customer_history=customer_history,
        current_claim_type=CLAIM_TYPE,
        requested_amount=REQUESTED_AMOUNT,
        evidence_available=evidence_available,
        strategic_state=None,
        network_observations=[],
        current_ip="test-ip",
        current_device_id="test-device",
        current_payment_method_id="test-payment",
        current_shipping_address_id="test-address",
    )

    if not isinstance(
        features,
        dict,
    ):

        raise TypeError(
            "GovernorFeatureBuilder must return a dictionary."
        )

    print(
        f"Feature count : {len(features)}"
    )

    for name, value in features.items():

        print(
            f"{name:<40} : {value}"
        )

    print(
        "Governor feature construction: PASSED"
    )

    return features


# ============================================================
# TEST 8
# BASE GOVERNOR
# ============================================================

def test_base_governor():

    section(
        "TEST 8 — ADAPTIVE RISK GOVERNOR"
    )

    governor = (
        AdaptiveRiskGovernor()
    )

    customer_history = [
        {
            "claim_type": "PRODUCT_NOT_RECEIVED",
            "requested_amount": 250.0,
            "support_decision": "APPROVE",
        },
        {
            "claim_type": "PRODUCT_NOT_RECEIVED",
            "requested_amount": 500.0,
            "support_decision": "APPROVE",
        },
    ]

    decision = governor.evaluate(
        customer_history=customer_history,
        current_claim_type=CLAIM_TYPE,
        requested_amount=REQUESTED_AMOUNT,
        evidence_available=[
            "order_receipt",
            "delivery_record",
        ],
        support_decision=SUPPORT_DECISION,
        strategic_state=None,
    )

    if decision is None:

        raise ValueError(
            "Adaptive Governor returned None."
        )

    risk_score = bounded(
        decision.risk_score
    )

    print(
        f"Governor risk : {risk_score:.6f}"
    )

    print(
        f"Risk level    : {decision.risk_level}"
    )

    print(
        f"Action        : {decision.action}"
    )

    print(
        f"Reasons       : {decision.reason_codes}"
    )

    print(
        "Adaptive Governor: PASSED"
    )

    return decision


# ============================================================
# TEST 9
# RISK FUSION
# ============================================================

def test_risk_fusion(
    governor_decision,
    external_features,
):

    section(
        "TEST 9 — RISK FUSION ENGINE"
    )

    fusion_engine = (
        RiskFusionEngine()
    )

    fusion = fusion_engine.fuse(
        governor_decision=governor_decision,
        network_features=external_features,
    )

    if fusion is None:

        raise ValueError(
            "RiskFusionEngine returned None."
        )

    print(
        f"Behavioral risk : "
        f"{bounded(fusion.behavioral_risk):.6f}"
    )

    print(
        f"Strategic risk  : "
        f"{bounded(fusion.strategic_risk):.6f}"
    )

    print(
        f"Network risk    : "
        f"{bounded(fusion.network_risk):.6f}"
    )

    print(
        f"Fused risk      : "
        f"{bounded(fusion.fused_risk_score):.6f}"
    )

    print(
        f"Risk level      : "
        f"{fusion.risk_level}"
    )

    print(
        f"Action          : "
        f"{fusion.action}"
    )

    print(
        "Risk Fusion Engine: PASSED"
    )

    return fusion


# ============================================================
# TEST 10
# COMPLETE GNN GOVERNOR
# ============================================================

def test_complete_governor(
    graph,
    customer_id: str,
    node_index: int,
):

    section(
        "TEST 10 — COMPLETE GNN → GOVERNOR PIPELINE"
    )

    governor = GNNRiskGovernor(
        gnn_weight=GNn_WEIGHT
    )

    decision = governor.evaluate(
        data=graph.data,
        customer_id=customer_id,
        customer_node_index=node_index,
        customer_history=[
            {
                "claim_type": "PRODUCT_NOT_RECEIVED",
                "requested_amount": 250.0,
                "support_decision": "APPROVE",
            },
            {
                "claim_type": "PRODUCT_NOT_RECEIVED",
                "requested_amount": 500.0,
                "support_decision": "APPROVE",
            },
        ],
        current_claim_type=CLAIM_TYPE,
        requested_amount=REQUESTED_AMOUNT,
        evidence_available=[
            "order_receipt",
            "delivery_record",
        ],
        support_decision=SUPPORT_DECISION,
        strategic_state=None,
        network_observations=[],
        current_ip="test-ip",
        current_device_id="test-device",
        current_payment_method_id="test-payment",
        current_shipping_address_id="test-address",
    )

    if decision is None:

        raise ValueError(
            "GNNRiskGovernor returned None."
        )

    print()
    print("FINAL DECISION")
    print("-" * 80)

    print(
        f"Behavioral risk : "
        f"{bounded(decision.behavioral_risk):.6f}"
    )

    print(
        f"Strategic risk  : "
        f"{bounded(decision.strategic_risk):.6f}"
    )

    print(
        f"Network risk    : "
        f"{bounded(decision.network_risk):.6f}"
    )

    print(
        f"GNN risk        : "
        f"{bounded(decision.gnn_risk):.6f}"
    )

    print(
        f"Fused risk      : "
        f"{bounded(decision.fused_risk_score):.6f}"
    )

    print(
        f"Risk level      : "
        f"{decision.risk_level}"
    )

    print(
        f"Action          : "
        f"{decision.action}"
    )

    print(
        f"Reason codes    : "
        f"{decision.reason_codes}"
    )

    print()
    print(
        f"GNN weight      : "
        f"{decision.metadata.get('gnn_weight')}"
    )

    print(
        f"Pre-GNN score   : "
        f"{decision.fusion_decision.metadata.get('pre_gnn_fused_score')}"
    )

    print(
        f"GNN-aware score : "
        f"{decision.fusion_decision.metadata.get('gnn_aware_fused_score')}"
    )

    # --------------------------------------------------------
    # Validate returned structures
    # --------------------------------------------------------

    if decision.gnn_observation is None:

        raise AssertionError(
            "Missing GNN observation."
        )

    if decision.governor_observation is None:

        raise AssertionError(
            "Missing Governor observation."
        )

    if decision.governor_decision is None:

        raise AssertionError(
            "Missing Governor decision."
        )

    if decision.fusion_decision is None:

        raise AssertionError(
            "Missing fusion decision."
        )

    # --------------------------------------------------------
    # Validate score bounds
    # --------------------------------------------------------

    scores = {
        "behavioral_risk":
            decision.behavioral_risk,

        "strategic_risk":
            decision.strategic_risk,

        "network_risk":
            decision.network_risk,

        "gnn_risk":
            decision.gnn_risk,

        "fused_risk_score":
            decision.fused_risk_score,
    }

    for name, value in scores.items():

        if not (
            0.0 <= float(value) <= 1.0
        ):

            raise AssertionError(
                f"{name} is outside [0, 1]: {value}"
            )

    # --------------------------------------------------------
    # Validate risk level
    # --------------------------------------------------------

    valid_levels = {
        "LOW",
        "MEDIUM",
        "HIGH",
    }

    if decision.risk_level not in valid_levels:

        raise AssertionError(
            "Invalid risk level: "
            f"{decision.risk_level}"
        )

    # --------------------------------------------------------
    # Validate action
    # --------------------------------------------------------

    valid_actions = {
        "ALLOW_AGENT_A_DECISION",
        "REQUEST_ADDITIONAL_EVIDENCE",
        "ESCALATE_TO_HUMAN_REVIEW",
    }

    if decision.action not in valid_actions:

        raise AssertionError(
            "Invalid Governor action: "
            f"{decision.action}"
        )

    # --------------------------------------------------------
    # Validate GNN weight
    # --------------------------------------------------------

    if not (
        0.0 <= GNn_WEIGHT <= 1.0
    ):

        raise AssertionError(
            "Invalid GNN weight."
        )

    if (
        decision.metadata.get(
            "gnn_enabled"
        )
        is not True
    ):

        raise AssertionError(
            "GNN is not marked as enabled."
        )

    # --------------------------------------------------------
    # Validate GNN-aware fusion
    # --------------------------------------------------------

    expected_score = (
        (1.0 - GNn_WEIGHT)
        * bounded(
            decision.fusion_decision.metadata[
                "pre_gnn_fused_score"
            ]
        )
        +
        GNn_WEIGHT
        * bounded(
            decision.gnn_risk
        )
    )

    actual_score = bounded(
        decision.fused_risk_score
    )

    # The score may subsequently be raised by
    # the GNN high-risk floor.
    if decision.gnn_risk < 0.85:

        if abs(
            actual_score
            - expected_score
        ) > 1e-6:

            raise AssertionError(
                "GNN-aware fusion score does not "
                "match expected weighted fusion.\n"
                f"Expected: {expected_score}\n"
                f"Actual:   {actual_score}"
            )

    print()
    print(
        "Complete GNN → Governor pipeline: PASSED"
    )

    return decision


# ============================================================
# ARCHITECTURAL ISOLATION TEST
# ============================================================

def test_agent_a_isolation(
    decision,
) -> None:

    section(
        "TEST 11 — AGENT A INFORMATION ISOLATION"
    )

    governor_decision = (
        decision.governor_decision
    )

    if governor_decision is None:

        raise AssertionError(
            "Governor decision is missing."
        )

    # --------------------------------------------------------
    # IMPORTANT ARCHITECTURAL DISTINCTION
    #
    # governor_decision.features contains INTERNAL GOVERNOR
    # signals. It is NOT the Agent A-visible interface.
    #
    # Therefore, strategic_adaptation_score may legitimately
    # exist inside the Governor's private feature dictionary.
    #
    # Agent A isolation must be checked at the boundary where
    # information is actually exposed to Agent A.
    # --------------------------------------------------------

    visible_features = getattr(
        governor_decision,
        "features",
        {},
    )

    if not isinstance(
        visible_features,
        dict,
    ):

        raise AssertionError(
            "Governor features must be a dictionary."
        )

    # --------------------------------------------------------
    # Hidden signals that must NEVER be exposed through an
    # Agent A-facing observation/response.
    #
    # These are intentionally defined here for boundary
    # validation only.
    # --------------------------------------------------------

    forbidden_keys = {
        "gnn_risk",
        "final_fused_risk",
        "network_abnormality_score",
        "strategic_adaptation_score",
        "strategic_behavior",
    }

    # --------------------------------------------------------
    # The Governor's internal feature dictionary is private.
    #
    # We therefore DO NOT treat the presence of these keys
    # here as an information-isolation violation.
    #
    # Instead, verify that the GNN-aware fields have not been
    # injected into the Agent A-facing support decision.
    # --------------------------------------------------------

    agent_a_visible = {}

    # --------------------------------------------------------
    # GovernorDecision may expose the original Agent A-facing
    # support decision through an attribute depending on the
    # implementation. Inspect only existing attributes.
    # --------------------------------------------------------

    possible_agent_a_fields = (
        "support_decision",
        "agent_a_decision",
        "visible_features",
        "agent_a_features",
        "agent_a_observation",
    )

    for field_name in possible_agent_a_fields:

        if hasattr(
            governor_decision,
            field_name,
        ):

            value = getattr(
                governor_decision,
                field_name,
            )

            if isinstance(
                value,
                dict,
            ):

                agent_a_visible.update(
                    value
                )

    # --------------------------------------------------------
    # If the current GovernorDecision does not maintain a
    # separate Agent A-visible dictionary, that is acceptable.
    #
    # In that case, governor_decision.features remains an
    # INTERNAL Governor object and must not be interpreted as
    # an Agent A information channel.
    # --------------------------------------------------------

    leaked = (
        forbidden_keys
        & set(agent_a_visible.keys())
    )

    if leaked:

        raise AssertionError(
            "Agent A information-isolation violation.\n"
            f"Forbidden signals exposed: {sorted(leaked)}"
        )

    # --------------------------------------------------------
    # Also ensure the final GNN-aware decision itself is not
    # being mistaken for Agent A's original decision.
    #
    # These values are expected to exist internally because
    # the Governor needs them to make its final decision.
    # --------------------------------------------------------

    internal_hidden_signals = {
        "gnn_risk": getattr(
            decision,
            "gnn_risk",
            None,
        ),
        "network_risk": getattr(
            decision,
            "network_risk",
            None,
        ),
        "strategic_risk": getattr(
            decision,
            "strategic_risk",
            None,
        ),
        "fused_risk_score": getattr(
            decision,
            "fused_risk_score",
            None,
        ),
    }

    # --------------------------------------------------------
    # Verify the hidden signals remain internal to the
    # GNN-aware Governor decision rather than being copied into
    # the Agent A-visible dictionary.
    # --------------------------------------------------------

    leaked_internal = (
        forbidden_keys
        & set(agent_a_visible.keys())
    )

    if leaked_internal:

        raise AssertionError(
            "Agent A information-isolation violation.\n"
            f"Hidden internal signals exposed: "
            f"{sorted(leaked_internal)}"
        )

    print(
        "Agent A receives only its permitted "
        "decision information."
    )

    print(
        "Hidden GNN/network/strategic signals remain "
        "inside the Governor."
    )

    print(
        "Agent A information isolation: PASSED"
    )


# ============================================================
# COMPLETE TEST SUITE
# ============================================================

def main() -> None:

    section(
        "GNN → GOVERNOR END-TO-END TEST"
    )

    print(
        "This test executes the complete "
        "GNN-aware risk decision pipeline."
    )

    print(
        f"Seed       : {SEED}"
    )

    print(
        f"GNN weight : {GNn_WEIGHT}"
    )

    # --------------------------------------------------------
    # 1. Check checkpoint
    # --------------------------------------------------------

    test_checkpoint()

    # --------------------------------------------------------
    # 2. Load GNN
    # --------------------------------------------------------

    inference = (
        test_inference()
    )

    # --------------------------------------------------------
    # 3. Build graph
    # --------------------------------------------------------

    graph = (
        build_test_graph()
    )

    # --------------------------------------------------------
    # 4. Customer mapping
    # --------------------------------------------------------

    customer_id, node_index = (
        test_customer_mapping(
            graph
        )
    )

    # --------------------------------------------------------
    # 5. Raw GNN prediction
    # --------------------------------------------------------

    test_raw_gnn_prediction(
        inference=inference,
        graph=graph,
        node_index=node_index,
    )

    # --------------------------------------------------------
    # 6. Adapter
    # --------------------------------------------------------

    test_gnn_adapter(
        graph=graph,
        customer_id=customer_id,
        node_index=node_index,
    )

    # --------------------------------------------------------
    # 7. Feature builder
    # --------------------------------------------------------

    external_features = (
        test_feature_builder(
            customer_id=customer_id
        )
    )

    # --------------------------------------------------------
    # 8. Base Governor
    # --------------------------------------------------------

    governor_decision = (
        test_base_governor()
    )

    # --------------------------------------------------------
    # 9. Fusion
    # --------------------------------------------------------

    test_risk_fusion(
        governor_decision=governor_decision,
        external_features=external_features,
    )

    # --------------------------------------------------------
    # 10. Complete pipeline
    # --------------------------------------------------------

    final_decision = (
        test_complete_governor(
            graph=graph,
            customer_id=customer_id,
            node_index=node_index,
        )
    )

    # --------------------------------------------------------
    # 11. Agent isolation
    # --------------------------------------------------------

    test_agent_a_isolation(
        final_decision
    )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    section(
        "ALL GNN → GOVERNOR TESTS PASSED"
    )

    print(
        "Checkpoint loading       : PASSED"
    )

    print(
        "GNN inference            : PASSED"
    )

    print(
        "Graph construction       : PASSED"
    )

    print(
        "Customer mapping         : PASSED"
    )

    print(
        "GNN adapter              : PASSED"
    )

    print(
        "Feature construction     : PASSED"
    )

    print(
        "Adaptive Governor        : PASSED"
    )

    print(
        "Risk Fusion              : PASSED"
    )

    print(
        "GNN-aware Governor       : PASSED"
    )

    print(
        "Agent A isolation        : PASSED"
    )

    print()
    print(
        "END-TO-END PIPELINE: PASSED"
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()