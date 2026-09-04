from __future__ import annotations

import numpy as np

from src.evaluation.phase14_ml_risk_fusion import (
    FEATURE_NAMES,
    generate_dataset,
    split_dataset,
)

from src.ml.risk_fusion_models import (
    build_models,
    compare_models,
)


def test_dataset_is_generated():

    X, y = generate_dataset(
        count=200,
    )

    assert X.shape[0] == 200
    assert y.shape[0] == 200


def test_feature_count_matches_names():

    X, _ = generate_dataset(
        count=100,
    )

    assert X.shape[1] == len(
        FEATURE_NAMES
    )


def test_binary_target():

    _, y = generate_dataset(
        count=200,
    )

    assert set(
        np.unique(y)
    ).issubset({0, 1})


def test_both_classes_are_present():

    _, y = generate_dataset(
        count=500,
    )

    assert len(
        np.unique(y)
    ) == 2


def test_features_are_finite():

    X, _ = generate_dataset(
        count=200,
    )

    assert np.isfinite(
        X
    ).all()


def test_train_validation_split():

    X, y = generate_dataset(
        count=500,
    )

    (
        X_train,
        y_train,
        X_val,
        y_val,
    ) = split_dataset(
        X,
        y,
    )

    assert len(X_train) > 0
    assert len(X_val) > 0

    assert (
        len(X_train)
        + len(X_val)
        == len(X)
    )

    assert (
        len(y_train)
        + len(y_val)
        == len(y)
    )


def test_models_are_available():

    models = build_models()

    assert (
        "LOGISTIC_REGRESSION"
        in models
    )

    assert (
        "RANDOM_FOREST"
        in models
    )

    assert (
        "HIST_GRADIENT_BOOSTING"
        in models
    )


def test_multiple_models_can_be_compared():

    X, y = generate_dataset(
        count=300,
    )

    (
        X_train,
        y_train,
        X_val,
        y_val,
    ) = split_dataset(
        X,
        y,
    )

    results = compare_models(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    assert len(results) >= 3


def test_model_metrics_are_bounded():

    X, y = generate_dataset(
        count=300,
    )

    (
        X_train,
        y_train,
        X_val,
        y_val,
    ) = split_dataset(
        X,
        y,
    )

    results = compare_models(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    for result in results:

        assert (
            0.0
            <= result.roc_auc
            <= 1.0
        )

        assert (
            0.0
            <= result.average_precision
            <= 1.0
        )

        assert (
            0.0
            <= result.precision
            <= 1.0
        )

        assert (
            0.0
            <= result.recall
            <= 1.0
        )

        assert np.isfinite(
            result.log_loss_value
        )

        assert np.isfinite(
            result.brier_score
        )