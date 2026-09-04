from __future__ import annotations

from src.evaluation.phase9_network_evaluation import (
    CoordinatedStrategicState,
    NormalStrategicState,
    NetworkEvaluationEngine,
    calculate_metrics,
    coordinated_network_features,
    generate_evaluation_cases,
    normal_network_features,
    validate_evaluation,
)


# ============================================================
# NETWORK FEATURES
# ============================================================


def test_normal_network_features_are_bounded():

    features = normal_network_features()

    assert features

    for value in features.values():

        assert 0.0 <= value <= 1.0


def test_coordinated_network_features_are_bounded():

    features = coordinated_network_features()

    assert features

    for value in features.values():

        assert 0.0 <= value <= 1.0


def test_coordinated_network_is_stronger_than_normal():

    normal = normal_network_features()

    coordinated = coordinated_network_features()

    assert (
        coordinated["ip_reuse_score"]
        > normal["ip_reuse_score"]
    )

    assert (
        coordinated["device_reuse_score"]
        > normal["device_reuse_score"]
    )

    assert (
        coordinated["payment_reuse_score"]
        > normal["payment_reuse_score"]
    )

    assert (
        coordinated["network_abnormality_score"]
        > normal["network_abnormality_score"]
    )


# ============================================================
# STRATEGIC STATES
# ============================================================


def test_normal_strategic_state_is_weak():

    state = NormalStrategicState()

    assert (
        len(state.interaction_history)
        == 0
    )

    assert (
        state.inferred_policy[
            "evidence_sensitivity"
        ]
        < 0.70
    )


def test_coordinated_strategic_state_has_learning_history():

    state = CoordinatedStrategicState()

    assert (
        len(state.interaction_history)
        == 20
    )

    assert (
        state.inferred_policy[
            "evidence_sensitivity"
        ]
        > 0.80
    )


# ============================================================
# SINGLE CASE
# ============================================================


def test_engine_can_evaluate_normal_case():

    engine = NetworkEvaluationEngine(
        seed=42
    )

    result = engine.evaluate_case(

        case_id="NORMAL_TEST",

        scenario="NORMAL",

        customer_history=[],

        claim_type="WRONG_ITEM_CLAIM",

        requested_amount=1500,

        evidence_available=[
            "package_photo"
        ],

        strategic_state=(
            NormalStrategicState()
        ),

        network_features=(
            normal_network_features()
        ),
    )

    assert result.case_id == "NORMAL_TEST"

    assert result.scenario == "NORMAL"

    assert 0.0 <= result.network_risk <= 1.0

    assert 0.0 <= result.fused_risk <= 1.0


def test_engine_can_evaluate_coordinated_case():

    engine = NetworkEvaluationEngine(
        seed=42
    )

    result = engine.evaluate_case(

        case_id="COORDINATED_TEST",

        scenario="COORDINATED",

        customer_history=[],

        claim_type="WRONG_ITEM_CLAIM",

        requested_amount=6000,

        evidence_available=[
            "package_photo"
        ],

        strategic_state=(
            CoordinatedStrategicState()
        ),

        network_features=(
            coordinated_network_features()
        ),
    )

    assert result.scenario == "COORDINATED"

    assert result.network_risk > 0.70

    assert result.fused_risk > 0.70

    assert result.risk_level == "HIGH"

    assert result.coordination_detected is True


# ============================================================
# DATASET
# ============================================================


def test_evaluation_dataset_contains_both_classes():

    cases = generate_evaluation_cases(
        n_normal=10,
        n_coordinated=10,
        seed=42,
    )

    assert len(cases) == 20

    assert any(
        x.scenario == "NORMAL"
        for x in cases
    )

    assert any(
        x.scenario == "COORDINATED"
        for x in cases
    )


def test_evaluation_case_ids_are_unique():

    cases = generate_evaluation_cases(
        n_normal=10,
        n_coordinated=10,
        seed=42,
    )

    ids = [
        x.case_id
        for x in cases
    ]

    assert len(ids) == len(set(ids))


# ============================================================
# METRICS
# ============================================================


def test_metrics_are_generated():

    cases = generate_evaluation_cases(
        n_normal=10,
        n_coordinated=10,
        seed=42,
    )

    metrics = calculate_metrics(
        cases
    )

    required = {
        "normal_high_risk_rate",
        "coordinated_high_risk_rate",
        "coordination_detection_rate",
        "false_coordination_rate",
        "normal_mean_fused_risk",
        "coordinated_mean_fused_risk",
    }

    assert required.issubset(
        metrics.keys()
    )


def test_coordinated_risk_is_higher_than_normal():

    cases = generate_evaluation_cases(
        n_normal=25,
        n_coordinated=25,
        seed=42,
    )

    metrics = calculate_metrics(
        cases
    )

    assert (
        metrics[
            "coordinated_mean_fused_risk"
        ]
        >
        metrics[
            "normal_mean_fused_risk"
        ]
    )


def test_coordinated_customers_are_detected():

    cases = generate_evaluation_cases(
        n_normal=25,
        n_coordinated=25,
        seed=42,
    )

    metrics = calculate_metrics(
        cases
    )

    assert (
        metrics[
            "coordination_detection_rate"
        ]
        >= 0.90
    )


def test_normal_customers_do_not_trigger_coordination():

    cases = generate_evaluation_cases(
        n_normal=25,
        n_coordinated=25,
        seed=42,
    )

    metrics = calculate_metrics(
        cases
    )

    assert (
        metrics[
            "false_coordination_rate"
        ]
        <= 0.10
    )


# ============================================================
# VALIDATION
# ============================================================


def test_phase9_validation_passes():

    cases = generate_evaluation_cases(
        n_normal=25,
        n_coordinated=25,
        seed=42,
    )

    validation = validate_evaluation(
        cases
    )

    assert all(
        validation.values()
    )