from __future__ import annotations

import random

import numpy as np

from src.ml.risk_fusion_models import (
    compare_models,
    select_best_model,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
NUM_CASES = 1000

# Canonical feature count.
NUM_FEATURES = 12


# ============================================================
# CANONICAL FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "governor_risk",
    "network_risk",
    "temporal_abnormality",
    "semantic_paraphrase_score",
    "semantic_claim_switch",
    "evidence_consistency",
    "claim_similarity",
    "request_velocity",
    "amount_acceleration",
    "shared_identifier_strength",
    "agent_decision_anomaly",
    "strategic_behavior",
]


# Keep the feature contract explicit.
assert len(FEATURE_NAMES) == NUM_FEATURES


# ============================================================
# SYNTHETIC RISK-FUSION DATASET
# ============================================================

def generate_dataset(
    *,
    count: int = NUM_CASES,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic supervised-learning dataset
    representing signals available to the Governor.

    IMPORTANT:
    ----------------
    The target `y` represents fraud ground truth.

    Ground truth is used ONLY to construct the target.

    It is NEVER inserted into X.

    This is critical because the ML risk-fusion model
    must learn from observations available to the Governor,
    not from the answer it is supposed to predict.

    Feature layout
    --------------

    0  : governor_risk
    1  : network_risk
    2  : temporal_abnormality
    3  : semantic_paraphrase_score
    4  : semantic_claim_switch
    5  : evidence_consistency
    6  : claim_similarity
    7  : request_velocity
    8  : amount_acceleration
    9  : shared_identifier_strength
    10 : agent_decision_anomaly
    11 : strategic_behavior
    """

    if count < 10:
        raise ValueError(
            "count must be at least 10."
        )

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Generate only observable Governor features.
    # --------------------------------------------------------

    X = rng.uniform(
        0.0,
        1.0,
        size=(count, NUM_FEATURES),
    )

    # --------------------------------------------------------
    # Construct latent fraud-risk signal.
    #
    # IMPORTANT:
    # y is generated from a latent mechanism.
    #
    # The actual fraud label is NOT inserted into X.
    # --------------------------------------------------------

    latent_score = (
        0.16 * X[:, 0]
        + 0.18 * X[:, 1]
        + 0.12 * X[:, 2]
        + 0.08 * X[:, 3]
        + 0.08 * X[:, 4]
        + 0.04 * (1.0 - X[:, 5])
        + 0.06 * X[:, 6]
        + 0.08 * X[:, 7]
        + 0.05 * X[:, 8]
        + 0.06 * X[:, 9]
        + 0.04 * X[:, 10]
        + 0.05 * X[:, 11]
    )

    # --------------------------------------------------------
    # Add noise to prevent artificial perfect separation.
    # --------------------------------------------------------

    noise = rng.normal(
        loc=0.0,
        scale=0.12,
        size=count,
    )

    latent_score = (
        latent_score
        + noise
    )

    # --------------------------------------------------------
    # Convert latent signal to probability.
    # --------------------------------------------------------

    probability = (
        1.0
        / (
            1.0
            + np.exp(
                -(
                    (latent_score - 0.50)
                    * 10.0
                )
            )
        )
    )

    # --------------------------------------------------------
    # Generate fraud ground truth.
    #
    # This variable is the target only.
    # --------------------------------------------------------

    y = (
        rng.random(count)
        < probability
    ).astype(int)

    # Guarantee both classes exist.
    if np.all(y == 0):
        y[0] = 1

    if np.all(y == 1):
        y[0] = 0

    # --------------------------------------------------------
    # Defensive checks against accidental leakage.
    # --------------------------------------------------------

    assert X.shape == (
        count,
        NUM_FEATURES,
    )

    assert len(y) == count

    assert set(
        np.unique(y)
    ).issubset({0, 1})

    return X, y


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    validation_fraction: float = 0.30,
    seed: int = SEED,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Create a reproducible stratified train/validation split.

    Important leakage guarantee
    ----------------------------
    Each original row is assigned to exactly one split.

    No row appears in both training and validation data.
    """

    if not (
        0.0
        < validation_fraction
        < 1.0
    ):
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    if not isinstance(X, np.ndarray):
        raise TypeError(
            "X must be a numpy array."
        )

    if not isinstance(y, np.ndarray):
        raise TypeError(
            "y must be a numpy array."
        )

    if X.ndim != 2:
        raise ValueError(
            "X must be two-dimensional."
        )

    if y.ndim != 1:
        raise ValueError(
            "y must be one-dimensional."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of rows."
        )

    if X.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"Expected {NUM_FEATURES} features, "
            f"received {X.shape[1]}."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "X contains NaN or infinite values."
        )

    unique_labels = set(
        np.unique(y).tolist()
    )

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            "y must contain only binary labels 0 and 1."
        )

    rng = random.Random(seed)

    # --------------------------------------------------------
    # Stratify by target class.
    # --------------------------------------------------------

    positive_indices = [
        index
        for index, value in enumerate(y)
        if value == 1
    ]

    negative_indices = [
        index
        for index, value in enumerate(y)
        if value == 0
    ]

    if len(positive_indices) < 2:
        raise ValueError(
            "Not enough positive samples for splitting."
        )

    if len(negative_indices) < 2:
        raise ValueError(
            "Not enough negative samples for splitting."
        )

    rng.shuffle(
        positive_indices
    )

    rng.shuffle(
        negative_indices
    )

    positive_validation_count = max(
        1,
        int(
            len(positive_indices)
            * validation_fraction
        ),
    )

    negative_validation_count = max(
        1,
        int(
            len(negative_indices)
            * validation_fraction
        ),
    )

    # Keep at least one sample in training.
    positive_validation_count = min(
        positive_validation_count,
        len(positive_indices) - 1,
    )

    negative_validation_count = min(
        negative_validation_count,
        len(negative_indices) - 1,
    )

    validation_indices = (
        positive_indices[
            :positive_validation_count
        ]
        + negative_indices[
            :negative_validation_count
        ]
    )

    validation_set = set(
        validation_indices
    )

    train_indices = [
        index
        for index in range(len(X))
        if index not in validation_set
    ]

    rng.shuffle(
        train_indices
    )

    rng.shuffle(
        validation_indices
    )

    # --------------------------------------------------------
    # Leakage check at the index level.
    # --------------------------------------------------------

    train_index_set = set(
        train_indices
    )

    validation_index_set = set(
        validation_indices
    )

    assert train_index_set.isdisjoint(
        validation_index_set
    )

    assert (
        len(train_index_set)
        + len(validation_index_set)
        == len(X)
    )

    # --------------------------------------------------------
    # Construct splits.
    # --------------------------------------------------------

    X_train = X[
        train_indices
    ]

    y_train = y[
        train_indices
    ]

    X_val = X[
        validation_indices
    ]

    y_val = y[
        validation_indices
    ]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
    )


# ============================================================
# DATASET LEAKAGE VALIDATION
# ============================================================

def validate_no_feature_target_leakage(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Validate that the feature matrix does not contain
    the target as an explicit feature.

    This is a structural check.

    It cannot prove that a feature is causally independent
    from the target, but it catches obvious target-column
    leakage and dimensionality mistakes.
    """

    if X.ndim != 2:
        raise ValueError(
            "X must be two-dimensional."
        )

    if y.ndim != 1:
        raise ValueError(
            "y must be one-dimensional."
        )

    if X.shape[1] != len(FEATURE_NAMES):
        raise AssertionError(
            "Feature count does not match FEATURE_NAMES."
        )

    if len(X) != len(y):
        raise AssertionError(
            "X and y row counts do not match."
        )

    if not np.isfinite(X).all():
        raise AssertionError(
            "Feature matrix contains non-finite values."
        )

    if not set(
        np.unique(y)
    ).issubset({0, 1}):
        raise AssertionError(
            "Target contains labels outside {0, 1}."
        )


def validate_train_validation_separation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    """
    Validate that no identical observations occur
    in both training and validation sets.

    This protects against accidental row duplication.
    """

    if len(X_train) != len(y_train):
        raise AssertionError(
            "Training X/y lengths do not match."
        )

    if len(X_val) != len(y_val):
        raise AssertionError(
            "Validation X/y lengths do not match."
        )

    train_rows = {
        (
            tuple(row.tolist()),
            int(label),
        )
        for row, label
        in zip(X_train, y_train)
    }

    validation_rows = {
        (
            tuple(row.tolist()),
            int(label),
        )
        for row, label
        in zip(X_val, y_val)
    }

    overlap = (
        train_rows
        & validation_rows
    )

    if overlap:
        raise AssertionError(
            "Data leakage detected: "
            "identical observations occur "
            "in both train and validation sets."
        )


# ============================================================
# PRINT DATASET INFORMATION
# ============================================================

def print_dataset_summary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:

    print()
    print("=" * 80)
    print("PHASE 14 — ML RISK-FUSION MODEL COMPARISON")
    print("=" * 80)

    print()

    print(
        f"Total cases              : "
        f"{len(X_train) + len(X_val)}"
    )

    print(
        f"Training cases           : "
        f"{len(X_train)}"
    )

    print(
        f"Validation cases        : "
        f"{len(X_val)}"
    )

    print()

    print(
        f"Training fraud rate      : "
        f"{np.mean(y_train) * 100:.2f}%"
    )

    print(
        f"Validation fraud rate    : "
        f"{np.mean(y_val) * 100:.2f}%"
    )

    print()

    print(
        "FEATURE CONTRACT"
    )

    for index, name in enumerate(
        FEATURE_NAMES
    ):

        print(
            f"  {index:>2} : {name}"
        )


# ============================================================
# PRINT MODEL COMPARISON
# ============================================================

def print_model_results(
    results,
) -> None:

    print()
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print()

    print(
        f"{'MODEL':<26}"
        f"{'ROC-AUC':>12}"
        f"{'AVG PRECISION':>17}"
        f"{'LOG LOSS':>14}"
        f"{'BRIER':>12}"
        f"{'PRECISION':>13}"
        f"{'RECALL':>11}"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{result.model_name:<26}"
            f"{result.roc_auc:>12.4f}"
            f"{result.average_precision:>17.4f}"
            f"{result.log_loss_value:>14.4f}"
            f"{result.brier_score:>12.4f}"
            f"{result.precision:>13.4f}"
            f"{result.recall:>11.4f}"
        )


# ============================================================
# PRINT BEST MODEL
# ============================================================

def print_best_model(
    best_model,
) -> None:

    print()
    print("=" * 80)
    print("CURRENT BEST MODEL")
    print("=" * 80)

    print()

    print(
        f"Model                 : "
        f"{best_model.model_name}"
    )

    print(
        f"Average Precision     : "
        f"{best_model.average_precision:.4f}"
    )

    print(
        f"ROC-AUC               : "
        f"{best_model.roc_auc:.4f}"
    )

    print(
        f"Log Loss              : "
        f"{best_model.log_loss_value:.4f}"
    )

    print(
        f"Brier Score           : "
        f"{best_model.brier_score:.4f}"
    )

    print(
        f"Precision             : "
        f"{best_model.precision:.4f}"
    )

    print(
        f"Recall                : "
        f"{best_model.recall:.4f}"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_results(
    results,
    X_train,
    y_train,
    X_val,
    y_val,
) -> None:

    assert len(results) >= 3

    assert len(X_train) > 0
    assert len(X_val) > 0

    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)

    assert X_train.shape[1] == NUM_FEATURES
    assert X_val.shape[1] == NUM_FEATURES

    assert set(
        np.unique(y_train)
    ).issubset({0, 1})

    assert set(
        np.unique(y_val)
    ).issubset({0, 1})

    # --------------------------------------------------------
    # Leakage checks.
    # --------------------------------------------------------

    validate_no_feature_target_leakage(
        X_train,
        y_train,
    )

    validate_no_feature_target_leakage(
        X_val,
        y_val,
    )

    validate_train_validation_separation(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    # --------------------------------------------------------
    # Model metric validation.
    # --------------------------------------------------------

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
            result.log_loss_value
            >= 0.0
        )

        assert (
            0.0
            <= result.brier_score
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

        assert (
            0.0
            <= result.probability_mean
            <= 1.0
        )

        assert (
            result.train_size
            == len(X_train)
        )

        assert (
            result.validation_size
            == len(X_val)
        )

        assert np.isfinite(
            result.log_loss_value
        )

    # --------------------------------------------------------
    # Ensure ranking actually happened.
    # --------------------------------------------------------

    assert (
        results[0].average_precision
        >= results[-1].average_precision
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Generate data.
    # --------------------------------------------------------

    X, y = generate_dataset()

    # --------------------------------------------------------
    # Validate feature/target boundary BEFORE splitting.
    # --------------------------------------------------------

    validate_no_feature_target_leakage(
        X,
        y,
    )

    # --------------------------------------------------------
    # Split data.
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
    ) = split_dataset(
        X,
        y,
    )

    # --------------------------------------------------------
    # Explicit train/validation leakage check.
    # --------------------------------------------------------

    validate_train_validation_separation(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    print_dataset_summary(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    # --------------------------------------------------------
    # Compare candidate ML models.
    # --------------------------------------------------------

    results = compare_models(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    # --------------------------------------------------------
    # Validate.
    # --------------------------------------------------------

    validate_results(
        results,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    # --------------------------------------------------------
    # Print comparison.
    # --------------------------------------------------------

    print_model_results(
        results
    )

    # --------------------------------------------------------
    # Select current best model.
    # --------------------------------------------------------

    best_model = select_best_model(
        results
    )

    print_best_model(
        best_model
    )

    # --------------------------------------------------------
    # Validation summary.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    checks = {
        "dataset_generated": (
            len(X)
            == NUM_CASES
        ),

        "feature_contract_valid": (
            X.shape[1]
            == len(FEATURE_NAMES)
            == NUM_FEATURES
        ),

        "binary_target": (
            set(
                np.unique(y)
            ).issubset({0, 1})
        ),

        "train_validation_split": (
            len(X_train)
            + len(X_val)
            == NUM_CASES
        ),

        "train_validation_disjoint": (
            True
        ),

        "target_not_in_feature_matrix": (
            X.shape[1]
            == len(FEATURE_NAMES)
        ),

        "multiple_models_compared": (
            len(results) >= 3
        ),

        "metrics_bounded": all(
            (
                0.0
                <= result.roc_auc
                <= 1.0
            )
            and (
                0.0
                <= result.average_precision
                <= 1.0
            )
            and (
                result.log_loss_value
                >= 0.0
            )
            and (
                0.0
                <= result.brier_score
                <= 1.0
            )
            for result in results
        ),

        "best_model_selected": (
            best_model
            == results[0]
        ),
    }

    all_passed = True

    for name, passed in checks.items():

        status = (
            "PASSED"
            if passed
            else "FAILED"
        )

        print(
            f"{name:<40}: {status}"
        )

        if not passed:
            all_passed = False

    print()

    print(
        "Overall validation       : "
        + (
            "PASSED"
            if all_passed
            else "FAILED"
        )
    )

    if not all_passed:
        raise RuntimeError(
            "Phase 14 validation failed."
        )

    print()

    print(
        "PHASE 14 ML MODEL COMPARISON COMPLETE"
    )


if __name__ == "__main__":
    main()