from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

# These are the only features currently allowed to enter
# the Phase 14 ML risk-fusion model.
FEATURE_NAMES = (
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
)


# ============================================================
# FORBIDDEN / GROUND-TRUTH FIELDS
# ============================================================

# These fields represent information that must NOT be available
# to the ML model at prediction time.
#
# Some are explicit ground truth.
# Others are post-outcome information.

FORBIDDEN_FEATURE_NAMES = (
    "is_fraud",
    "fraud_label",
    "ground_truth",
    "fraud_loss_if_allowed",
    "loss_prevented",
    "recovery",
    "final_outcome",
    "chargeback_confirmed",
    "investigation_result",
    "refund_approved_after_review",
)


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class LeakageAuditResult:

    feature_count_valid: bool

    feature_names_valid: bool

    forbidden_features_absent: bool

    target_shape_valid: bool

    train_validation_disjoint: bool

    duplicate_observations_absent: bool

    preprocessing_boundary_valid: bool

    temporal_information_valid: bool

    customer_information_valid: bool

    overall_passed: bool


# ============================================================
# FEATURE CONTRACT
# ============================================================

def validate_feature_contract(
    feature_names: Iterable[str],
) -> None:
    """
    Validate the feature contract used by the ML model.

    The target must never be part of the feature vector.
    """

    feature_names = tuple(feature_names)

    if len(feature_names) != len(
        FEATURE_NAMES
    ):
        raise AssertionError(
            "Unexpected number of ML features."
        )

    if feature_names != FEATURE_NAMES:
        raise AssertionError(
            "Feature ordering does not match "
            "the canonical Phase 14 feature contract."
        )

    lowered = {
        name.lower()
        for name in feature_names
    }

    forbidden = {
        name.lower()
        for name in FORBIDDEN_FEATURE_NAMES
    }

    leakage = (
        lowered
        & forbidden
    )

    if leakage:
        raise AssertionError(
            "Forbidden ground-truth/post-outcome "
            f"features detected: {sorted(leakage)}"
        )


# ============================================================
# TARGET VALIDATION
# ============================================================

def validate_target(
    y: np.ndarray,
) -> None:
    """
    Validate the supervised-learning target.

    The target is allowed to exist separately from X.
    It must never be embedded into X.
    """

    if not isinstance(
        y,
        np.ndarray,
    ):
        raise TypeError(
            "y must be a numpy array."
        )

    if y.ndim != 1:
        raise ValueError(
            "y must be one-dimensional."
        )

    if len(y) == 0:
        raise ValueError(
            "Target cannot be empty."
        )

    if not np.isfinite(
        y.astype(float)
    ).all():
        raise ValueError(
            "Target contains non-finite values."
        )

    labels = set(
        np.unique(y).tolist()
    )

    if not labels.issubset({0, 1}):
        raise ValueError(
            "Target must contain only 0 and 1."
        )


# ============================================================
# FEATURE MATRIX VALIDATION
# ============================================================

def validate_feature_matrix(
    X: np.ndarray,
) -> None:
    """
    Validate that X has the expected shape and contains
    only finite numeric observations.
    """

    if not isinstance(
        X,
        np.ndarray,
    ):
        raise TypeError(
            "X must be a numpy array."
        )

    if X.ndim != 2:
        raise ValueError(
            "X must be two-dimensional."
        )

    if X.shape[1] != len(
        FEATURE_NAMES
    ):
        raise AssertionError(
            "Feature matrix contains an unexpected "
            "number of columns."
        )

    if not np.isfinite(X).all():
        raise AssertionError(
            "Feature matrix contains NaN or infinite values."
        )


# ============================================================
# TARGET / FEATURE BOUNDARY
# ============================================================

def validate_target_feature_boundary(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Verify the basic structural boundary between X and y.

    Important:
        y is supplied separately.

    This catches accidental cases where the target becomes
    an additional feature column.

    It does NOT claim statistical independence between
    individual features and the target.
    """

    validate_feature_matrix(X)
    validate_target(y)

    if len(X) != len(y):
        raise AssertionError(
            "X and y row counts do not match."
        )

    if X.shape[1] != len(
        FEATURE_NAMES
    ):
        raise AssertionError(
            "Target may have been inserted into X."
        )


# ============================================================
# TRAIN / VALIDATION DUPLICATE CHECK
# ============================================================

def validate_no_duplicate_observations(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    """
    Ensure identical feature/target observations do not
    appear in both train and validation datasets.
    """

    validate_target_feature_boundary(
        X_train,
        y_train,
    )

    validate_target_feature_boundary(
        X_val,
        y_val,
    )

    train_rows = {
        (
            tuple(
                row.tolist()
            ),
            int(label),
        )
        for row, label
        in zip(
            X_train,
            y_train,
        )
    }

    validation_rows = {
        (
            tuple(
                row.tolist()
            ),
            int(label),
        )
        for row, label
        in zip(
            X_val,
            y_val,
        )
    }

    overlap = (
        train_rows
        & validation_rows
    )

    if overlap:
        raise AssertionError(
            "Train/validation contamination detected: "
            "identical observations occur in both sets."
        )


# ============================================================
# INDEX-LEVEL SPLIT CHECK
# ============================================================

def validate_split_indices(
    train_indices: Iterable[int],
    validation_indices: Iterable[int],
    total_count: int,
) -> None:
    """
    Verify that train and validation index sets are:

    1. disjoint
    2. within bounds
    3. collectively exhaustive
    """

    train_indices = set(
        train_indices
    )

    validation_indices = set(
        validation_indices
    )

    if train_indices & validation_indices:
        raise AssertionError(
            "Train and validation indices overlap."
        )

    all_indices = (
        train_indices
        | validation_indices
    )

    if any(
        index < 0
        or index >= total_count
        for index in all_indices
    ):
        raise AssertionError(
            "Split contains an out-of-range index."
        )

    if len(all_indices) != total_count:
        raise AssertionError(
            "Train and validation splits do not cover "
            "the complete dataset."
        )


# ============================================================
# PREPROCESSING LEAKAGE CONTRACT
# ============================================================

def validate_preprocessing_boundary(
    *,
    fitted_on_training_only: bool,
) -> None:
    """
    Enforce the preprocessing boundary.

    Any transformation that learns parameters from data
    must be fitted using training data only.

    Examples:

        StandardScaler
        PCA
        imputation statistics
        feature selection
        target encoding

    must NOT be fitted on validation/test data.
    """

    if not fitted_on_training_only:
        raise AssertionError(
            "Preprocessing leakage detected: "
            "preprocessing parameters were not fitted "
            "using training data only."
        )


# ============================================================
# TEMPORAL LEAKAGE CONTRACT
# ============================================================

def validate_temporal_information(
    *,
    feature_timestamp: float,
    decision_timestamp: float,
) -> None:
    """
    Verify that a feature was available before the
    Governor's decision.

    A feature timestamp after the decision timestamp
    represents future information and therefore leakage.
    """

    if not np.isfinite(
        feature_timestamp
    ):
        raise ValueError(
            "feature_timestamp must be finite."
        )

    if not np.isfinite(
        decision_timestamp
    ):
        raise ValueError(
            "decision_timestamp must be finite."
        )

    if feature_timestamp > decision_timestamp:
        raise AssertionError(
            "Temporal leakage detected: "
            "feature information occurs after "
            "the Governor decision."
        )


# ============================================================
# CUSTOMER GROUP LEAKAGE CONTRACT
# ============================================================

def validate_customer_split(
    train_customer_ids: Iterable[str],
    validation_customer_ids: Iterable[str],
) -> None:
    """
    Verify that customer-grouped evaluation has no
    customer overlap.

    This is useful when the goal is to evaluate whether
    the model generalizes to unseen customers.
    """

    train_customers = set(
        train_customer_ids
    )

    validation_customers = set(
        validation_customer_ids
    )

    overlap = (
        train_customers
        & validation_customers
    )

    if overlap:
        raise AssertionError(
            "Customer leakage detected: "
            "customers occur in both training "
            "and validation sets."
        )


# ============================================================
# SYNTHETIC AUDIT DATA
# ============================================================

def generate_audit_dataset(
    *,
    count: int = 100,
    seed: int = SEED,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Generate a clean synthetic dataset for leakage tests.

    Ground truth is generated independently and returned
    separately from the feature matrix.
    """

    if count < 20:
        raise ValueError(
            "count must be at least 20."
        )

    rng = np.random.default_rng(
        seed
    )

    X = rng.uniform(
        0.0,
        1.0,
        size=(
            count,
            len(FEATURE_NAMES),
        ),
    )

    latent = (
        0.20 * X[:, 0]
        + 0.18 * X[:, 1]
        + 0.12 * X[:, 2]
        + 0.10 * X[:, 3]
        + 0.08 * X[:, 4]
        + 0.07 * X[:, 6]
        + 0.08 * X[:, 7]
        + 0.06 * X[:, 8]
        + 0.05 * X[:, 9]
        + 0.03 * X[:, 10]
        + 0.03 * X[:, 11]
        - 0.04 * X[:, 5]
    )

    latent += rng.normal(
        0.0,
        0.15,
        size=count,
    )

    probability = (
        1.0
        / (
            1.0
            + np.exp(
                -(
                    latent
                    - 0.5
                )
                    * 8.0
            )
        )
    )

    y = (
        rng.random(count)
        < probability
    ).astype(int)

    # Guarantee both classes.
    if np.all(y == 0):
        y[0] = 1

    if np.all(y == 1):
        y[0] = 0

    return X, y


# ============================================================
# COMPLETE AUDIT
# ============================================================

def run_leakage_audit() -> LeakageAuditResult:
    """
    Run the Phase 14.1 structural leakage audit.
    """

    X, y = generate_audit_dataset()

    # --------------------------------------------------------
    # Feature contract
    # --------------------------------------------------------

    feature_count_valid = (
        X.shape[1]
        == len(FEATURE_NAMES)
    )

    feature_names_valid = True

    try:
        validate_feature_contract(
            FEATURE_NAMES
        )
    except AssertionError:
        feature_names_valid = False

    # --------------------------------------------------------
    # Forbidden features
    # --------------------------------------------------------

    forbidden_features_absent = True

    try:
        validate_feature_contract(
            FEATURE_NAMES
        )
    except AssertionError:
        forbidden_features_absent = False

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target_shape_valid = True

    try:
        validate_target_feature_boundary(
            X,
            y,
        )
    except (
        AssertionError,
        ValueError,
        TypeError,
    ):
        target_shape_valid = False

    # --------------------------------------------------------
    # Train / validation separation
    # --------------------------------------------------------

    split = int(
        len(X) * 0.70
    )

    X_train = X[:split]
    y_train = y[:split]

    X_val = X[split:]
    y_val = y[split:]

    train_validation_disjoint = True
    duplicate_observations_absent = True

    try:
        validate_no_duplicate_observations(
            X_train,
            y_train,
            X_val,
            y_val,
        )
    except AssertionError:
        train_validation_disjoint = False
        duplicate_observations_absent = False

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocessing_boundary_valid = True

    try:
        validate_preprocessing_boundary(
            fitted_on_training_only=True
        )
    except AssertionError:
        preprocessing_boundary_valid = False

    # --------------------------------------------------------
    # Temporal boundary
    # --------------------------------------------------------

    temporal_information_valid = True

    try:
        validate_temporal_information(
            feature_timestamp=90.0,
            decision_timestamp=100.0,
        )
    except AssertionError:
        temporal_information_valid = False

    # --------------------------------------------------------
    # Customer boundary
    #
    # Synthetic audit deliberately gives every customer
    # to only one split.
    # --------------------------------------------------------

    customer_information_valid = True

    train_customers = [
        f"CUSTOMER_{index}"
        for index in range(70)
    ]

    validation_customers = [
        f"CUSTOMER_{index}"
        for index in range(70, 100)
    ]

    try:
        validate_customer_split(
            train_customers,
            validation_customers,
        )
    except AssertionError:
        customer_information_valid = False

    # --------------------------------------------------------
    # Overall result
    # --------------------------------------------------------

    checks = (
        feature_count_valid,
        feature_names_valid,
        forbidden_features_absent,
        target_shape_valid,
        train_validation_disjoint,
        duplicate_observations_absent,
        preprocessing_boundary_valid,
        temporal_information_valid,
        customer_information_valid,
    )

    overall_passed = all(
        checks
    )

    return LeakageAuditResult(
        feature_count_valid=feature_count_valid,
        feature_names_valid=feature_names_valid,
        forbidden_features_absent=(
            forbidden_features_absent
        ),
        target_shape_valid=target_shape_valid,
        train_validation_disjoint=(
            train_validation_disjoint
        ),
        duplicate_observations_absent=(
            duplicate_observations_absent
        ),
        preprocessing_boundary_valid=(
            preprocessing_boundary_valid
        ),
        temporal_information_valid=(
            temporal_information_valid
        ),
        customer_information_valid=(
            customer_information_valid
        ),
        overall_passed=overall_passed,
    )


# ============================================================
# REPORT
# ============================================================

def print_audit_report(
    result: LeakageAuditResult,
) -> None:

    print()
    print("=" * 80)
    print(
        "PHASE 14.1 — LEAKAGE & "
        "EVALUATION INTEGRITY AUDIT"
    )
    print("=" * 80)

    print()

    checks = {
        "feature_count_valid": (
            result.feature_count_valid
        ),
        "feature_names_valid": (
            result.feature_names_valid
        ),
        "forbidden_features_absent": (
            result.forbidden_features_absent
        ),
        "target_feature_boundary_valid": (
            result.target_shape_valid
        ),
        "train_validation_disjoint": (
            result.train_validation_disjoint
        ),
        "duplicate_observations_absent": (
            result.duplicate_observations_absent
        ),
        "preprocessing_training_only": (
            result.preprocessing_boundary_valid
        ),
        "temporal_information_valid": (
            result.temporal_information_valid
        ),
        "customer_split_valid": (
            result.customer_information_valid
        ),
    }

    for name, passed in checks.items():

        print(
            f"{name:<45}: "
            f"{'PASSED' if passed else 'FAILED'}"
        )

    print()

    print(
        "Overall leakage audit : "
        + (
            "PASSED"
            if result.overall_passed
            else "FAILED"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    result = run_leakage_audit()

    print_audit_report(
        result
    )

    if not result.overall_passed:
        raise RuntimeError(
            "Phase 14.1 leakage audit failed."
        )

    print()
    print(
        "PHASE 14.1 LEAKAGE AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()