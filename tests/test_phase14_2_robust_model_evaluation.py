from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.phase14_2_robust_model_evaluation import (
    FEATURE_NAMES,
    RANDOM_SEEDS,
    N_SPLITS,
    compare_models_robustly,
    evaluate_model_robustly,
    generate_dataset,
    select_robust_best_model,
    validate_dataset,
)
from src.ml.risk_fusion_models import build_models


# ============================================================
# DATASET
# ============================================================

def test_dataset_has_expected_shape():

    X, y = generate_dataset(
        count=300,
    )

    assert X.shape == (
        300,
        len(FEATURE_NAMES),
    )

    assert y.shape == (
        300,
    )


def test_dataset_is_binary():

    X, y = generate_dataset(
        count=300,
    )

    validate_dataset(
        X,
        y,
    )

    assert set(
        np.unique(y)
    ).issubset({0, 1})


def test_both_classes_exist():

    X, y = generate_dataset(
        count=300,
    )

    assert len(
        np.unique(y)
    ) == 2


# ============================================================
# MODEL EVALUATION
# ============================================================

def test_multiple_models_available():

    models = build_models()

    assert len(models) >= 3


def test_single_model_robust_evaluation():

    X, y = generate_dataset(
        count=300,
    )

    models = build_models()

    model_name = next(
        iter(models)
    )

    result = evaluate_model_robustly(
        model_name=model_name,
        model=models[model_name],
        X=X,
        y=y,
        n_splits=3,
        seeds=(42,),
    )

    assert (
        result.folds_evaluated
        == 3
    )

    assert (
        result.seeds_evaluated
        == 1
    )


# ============================================================
# METRIC BOUNDS
# ============================================================

def test_robust_metrics_are_bounded():

    X, y = generate_dataset(
        count=300,
    )

    results = compare_models_robustly(
        X,
        y,
        n_splits=3,
        seeds=(42,),
    )

    for result in results:

        assert (
            0.0
            <= result.roc_auc_mean
            <= 1.0
        )

        assert (
            0.0
            <= result.average_precision_mean
            <= 1.0
        )

        assert (
            result.average_precision_std
            >= 0.0
        )

        assert (
            result.log_loss_mean
            >= 0.0
        )

        assert (
            0.0
            <= result.brier_score_mean
            <= 1.0
        )

        assert (
            0.0
            <= result.precision_mean
            <= 1.0
        )

        assert (
            0.0
            <= result.recall_mean
            <= 1.0
        )

        assert (
            0.0
            <= result.probability_mean
            <= 1.0
        )


# ============================================================
# CROSS VALIDATION
# ============================================================

def test_all_folds_are_evaluated():

    X, y = generate_dataset(
        count=300,
    )

    results = compare_models_robustly(
        X,
        y,
        n_splits=4,
        seeds=(42, 123),
    )

    for result in results:

        assert (
            result.folds_evaluated
            == 4
        )

        assert (
            result.seeds_evaluated
            == 2
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

def test_multiple_models_are_compared():

    X, y = generate_dataset(
        count=300,
    )

    results = compare_models_robustly(
        X,
        y,
        n_splits=3,
        seeds=(42,),
    )

    assert len(results) >= 3


def test_results_are_ranked():

    X, y = generate_dataset(
        count=300,
    )

    results = compare_models_robustly(
        X,
        y,
        n_splits=3,
        seeds=(42,),
    )

    average_precisions = [
        result.average_precision_mean
        for result in results
    ]

    assert (
        average_precisions
        == sorted(
            average_precisions,
            reverse=True,
        )
    )


# ============================================================
# BEST MODEL
# ============================================================

def test_best_model_is_first_result():

    X, y = generate_dataset(
        count=300,
    )

    results = compare_models_robustly(
        X,
        y,
        n_splits=3,
        seeds=(42,),
    )

    best = select_robust_best_model(
        results
    )

    assert best == results[0]


def test_empty_results_are_rejected():

    with pytest.raises(
        ValueError
    ):

        select_robust_best_model(
            []
        )


# ============================================================
# INPUT VALIDATION
# ============================================================

def test_invalid_feature_count_is_rejected():

    X = np.zeros(
        (10, 5)
    )

    y = np.zeros(
        10,
        dtype=int,
    )

    with pytest.raises(
        ValueError
    ):

        validate_dataset(
            X,
            y,
        )


def test_nan_features_are_rejected():

    X = np.zeros(
        (10, len(FEATURE_NAMES))
    )

    X[0, 0] = np.nan

    y = np.zeros(
        10,
        dtype=int,
    )

    y[0] = 1

    with pytest.raises(
        ValueError
    ):

        validate_dataset(
            X,
            y,
        )


def test_non_binary_target_is_rejected():

    X = np.zeros(
        (10, len(FEATURE_NAMES))
    )

    y = np.array(
        [0, 1, 2, 0, 1, 0, 1, 0, 1, 0]
    )

    with pytest.raises(
        ValueError
    ):

        validate_dataset(
            X,
            y,
        )