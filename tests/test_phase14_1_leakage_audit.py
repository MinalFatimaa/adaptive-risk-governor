from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.phase14_1_leakage_audit import (
    FEATURE_NAMES,
    FORBIDDEN_FEATURE_NAMES,
    generate_audit_dataset,
    run_leakage_audit,
    validate_customer_split,
    validate_feature_contract,
    validate_no_duplicate_observations,
    validate_preprocessing_boundary,
    validate_split_indices,
    validate_target,
    validate_target_feature_boundary,
    validate_temporal_information,
)


# ============================================================
# DATASET
# ============================================================

def test_audit_dataset_is_generated():

    X, y = generate_audit_dataset(
        count=100,
    )

    assert X.shape == (
        100,
        len(FEATURE_NAMES),
    )

    assert y.shape == (
        100,
    )


# ============================================================
# FEATURE CONTRACT
# ============================================================

def test_feature_contract_is_valid():

    validate_feature_contract(
        FEATURE_NAMES
    )


def test_feature_count_is_correct():

    assert len(
        FEATURE_NAMES
    ) == 12


def test_forbidden_features_are_not_in_feature_contract():

    feature_set = {
        name.lower()
        for name in FEATURE_NAMES
    }

    forbidden_set = {
        name.lower()
        for name in FORBIDDEN_FEATURE_NAMES
    }

    assert not (
        feature_set
        & forbidden_set
    )


def test_feature_contract_rejects_wrong_order():

    wrong_order = tuple(
        reversed(FEATURE_NAMES)
    )

    with pytest.raises(
        AssertionError
    ):
        validate_feature_contract(
            wrong_order
        )


# ============================================================
# TARGET
# ============================================================

def test_target_is_binary():

    _, y = generate_audit_dataset()

    validate_target(
        y
    )


def test_target_feature_boundary_is_valid():

    X, y = generate_audit_dataset()

    validate_target_feature_boundary(
        X,
        y,
    )


def test_target_feature_boundary_rejects_extra_column():

    X, y = generate_audit_dataset()

    leaked_X = np.column_stack(
        [
            X,
            y,
        ]
    )

    with pytest.raises(
        AssertionError
    ):
        validate_target_feature_boundary(
            leaked_X,
            y,
        )


# ============================================================
# TRAIN / VALIDATION LEAKAGE
# ============================================================

def test_train_validation_observations_are_disjoint():

    X, y = generate_audit_dataset(
        count=100,
    )

    validate_no_duplicate_observations(
        X[:70],
        y[:70],
        X[70:],
        y[70:],
    )


def test_duplicate_train_validation_observation_is_rejected():

    X, y = generate_audit_dataset(
        count=100,
    )

    X_val = X[70:].copy()
    y_val = y[70:].copy()

    X_val[0] = X[0]
    y_val[0] = y[0]

    with pytest.raises(
        AssertionError
    ):
        validate_no_duplicate_observations(
            X[:70],
            y[:70],
            X_val,
            y_val,
        )


# ============================================================
# SPLIT INDICES
# ============================================================

def test_split_indices_are_valid():

    validate_split_indices(
        range(70),
        range(70, 100),
        100,
    )


def test_overlapping_split_indices_are_rejected():

    with pytest.raises(
        AssertionError
    ):
        validate_split_indices(
            range(0, 70),
            range(60, 100),
            100,
        )


# ============================================================
# PREPROCESSING
# ============================================================

def test_preprocessing_training_only_is_valid():

    validate_preprocessing_boundary(
        fitted_on_training_only=True
    )


def test_preprocessing_fitted_on_validation_is_rejected():

    with pytest.raises(
        AssertionError
    ):
        validate_preprocessing_boundary(
            fitted_on_training_only=False
        )


# ============================================================
# TEMPORAL LEAKAGE
# ============================================================

def test_feature_before_decision_is_valid():

    validate_temporal_information(
        feature_timestamp=90.0,
        decision_timestamp=100.0,
    )


def test_feature_after_decision_is_rejected():

    with pytest.raises(
        AssertionError
    ):
        validate_temporal_information(
            feature_timestamp=110.0,
            decision_timestamp=100.0,
        )


# ============================================================
# CUSTOMER LEAKAGE
# ============================================================

def test_customer_groups_are_disjoint():

    validate_customer_split(
        [
            "CUSTOMER_001",
            "CUSTOMER_002",
        ],
        [
            "CUSTOMER_003",
            "CUSTOMER_004",
        ],
    )


def test_customer_overlap_is_rejected():

    with pytest.raises(
        AssertionError
    ):
        validate_customer_split(
            [
                "CUSTOMER_001",
                "CUSTOMER_002",
            ],
            [
                "CUSTOMER_002",
                "CUSTOMER_003",
            ],
        )


# ============================================================
# COMPLETE AUDIT
# ============================================================

def test_complete_leakage_audit_passes():

    result = run_leakage_audit()

    assert result.overall_passed is True


def test_complete_audit_has_all_checks():

    result = run_leakage_audit()

    assert result.feature_count_valid
    assert result.feature_names_valid
    assert result.forbidden_features_absent
    assert result.target_shape_valid
    assert result.train_validation_disjoint
    assert result.duplicate_observations_absent
    assert result.preprocessing_boundary_valid
    assert result.temporal_information_valid
    assert result.customer_information_valid