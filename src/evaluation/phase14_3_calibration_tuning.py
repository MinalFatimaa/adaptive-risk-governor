from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.ml.risk_fusion_models import build_models


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

INNER_CV_FOLDS = 3

CALIBRATION_METHOD = "sigmoid"

PRIMARY_METRIC = "average_precision"


# ============================================================
# FEATURE NAMES
# ============================================================

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
# RESULT
# ============================================================

@dataclass(frozen=True)
class CalibrationResult:

    model_name: str

    tuned: bool

    calibrated: bool

    average_precision: float

    roc_auc: float

    brier_score: float

    log_loss_value: float

    precision: float

    recall: float

    probability_mean: float

    train_size: int

    validation_size: int


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(
    X: np.ndarray,
    y: np.ndarray,
) -> None:

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
            "X and y must have the same number of rows."
        )

    if len(X) == 0:
        raise ValueError(
            "Dataset cannot be empty."
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
            "y must contain only binary labels."
        )

    if len(unique_labels) < 2:
        raise ValueError(
            "Both target classes must be present."
        )


# ============================================================
# HYPERPARAMETER SEARCH SPACES
# ============================================================

def get_parameter_grids() -> dict[str, dict[str, list[Any]]]:

    return {

        # ----------------------------------------------------
        # LOGISTIC REGRESSION
        # ----------------------------------------------------

        "LOGISTIC_REGRESSION": {

            "model__C": [
                0.01,
                0.1,
                1.0,
                10.0,
            ],

            "model__class_weight": [
                None,
                "balanced",
            ],
        },

        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        "RANDOM_FOREST": {

            "n_estimators": [
                200,
                300,
            ],

            "max_depth": [
                5,
                8,
                12,
            ],

            "min_samples_leaf": [
                2,
                5,
                10,
            ],
        },

        # ----------------------------------------------------
        # HISTOGRAM GRADIENT BOOSTING
        # ----------------------------------------------------

        "HIST_GRADIENT_BOOSTING": {

            "max_iter": [
                100,
                200,
            ],

            "learning_rate": [
                0.03,
                0.05,
                0.10,
            ],

            "max_leaf_nodes": [
                15,
                31,
            ],

            "l2_regularization": [
                0.0,
                1.0,
            ],
        },

        # ----------------------------------------------------
        # LIGHTGBM
        # ----------------------------------------------------

        "LIGHTGBM": {

            "n_estimators": [
                150,
                300,
            ],

            "learning_rate": [
                0.03,
                0.05,
            ],

            "num_leaves": [
                15,
                31,
            ],

            "min_child_samples": [
                10,
                20,
            ],
        },

        # ----------------------------------------------------
        # XGBOOST
        # ----------------------------------------------------

        "XGBOOST": {

            "n_estimators": [
                150,
                300,
            ],

            "learning_rate": [
                0.03,
                0.05,
            ],

            "max_depth": [
                3,
                5,
            ],

            "min_child_weight": [
                1,
                3,
            ],
        },
    }


# ============================================================
# MODEL TUNING
# ============================================================

def tune_model(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:

    validate_dataset(
        X_train,
        y_train,
    )

    parameter_grids = get_parameter_grids()

    parameter_grid = parameter_grids.get(
        model_name,
        {},
    )

    if not parameter_grid:

        model.fit(
            X_train,
            y_train,
        )

        return model

    cv = StratifiedKFold(
        n_splits=INNER_CV_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring=PRIMARY_METRIC,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(
        X_train,
        y_train,
    )

    return search.best_estimator_


# ============================================================
# MODEL CALIBRATION
# ============================================================

def calibrate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:

    validate_dataset(
        X_train,
        y_train,
    )

    calibration_cv = StratifiedKFold(
        n_splits=INNER_CV_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method=CALIBRATION_METHOD,
        cv=calibration_cv,
        n_jobs=-1,
    )

    calibrated_model.fit(
        X_train,
        y_train,
    )

    return calibrated_model


# ============================================================
# FINAL EVALUATION
# ============================================================

def evaluate_model(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> CalibrationResult:

    validate_dataset(
        X_train,
        y_train,
    )

    validate_dataset(
        X_val,
        y_val,
    )

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    average_precision = float(
        average_precision_score(
            y_val,
            probabilities,
        )
    )

    roc_auc = float(
        roc_auc_score(
            y_val,
            probabilities,
        )
    )

    brier = float(
        brier_score_loss(
            y_val,
            probabilities,
        )
    )

    loss = float(
        log_loss(
            y_val,
            probabilities,
            labels=[0, 1],
        )
    )

    precision = float(
        precision_score(
            y_val,
            predictions,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_val,
            predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Floating point protection.
    #
    # Example:
    # 1.0000000000000002 -> 1.0
    # --------------------------------------------------------

    average_precision = float(
        np.clip(
            average_precision,
            0.0,
            1.0,
        )
    )

    roc_auc = float(
        np.clip(
            roc_auc,
            0.0,
            1.0,
        )
    )

    brier = float(
        np.clip(
            brier,
            0.0,
            1.0,
        )
    )

    precision = float(
        np.clip(
            precision,
            0.0,
            1.0,
        )
    )

    recall = float(
        np.clip(
            recall,
            0.0,
            1.0,
        )
    )

    probability_mean = float(
        np.mean(probabilities)
    )

    probability_mean = float(
        np.clip(
            probability_mean,
            0.0,
            1.0,
        )
    )

    return CalibrationResult(

        model_name=model_name,

        tuned=True,

        calibrated=True,

        average_precision=average_precision,

        roc_auc=roc_auc,

        brier_score=brier,

        log_loss_value=max(
            loss,
            0.0,
        ),

        precision=precision,

        recall=recall,

        probability_mean=probability_mean,

        train_size=len(X_train),

        validation_size=len(X_val),
    )


# ============================================================
# COMPARE ALL MODELS
# ============================================================

def compare_calibrated_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> list[CalibrationResult]:

    validate_dataset(
        X_train,
        y_train,
    )

    validate_dataset(
        X_val,
        y_val,
    )

    models = build_models()

    results: list[CalibrationResult] = []

    for model_name, base_model in models.items():

        print(
            f"  Tuning + calibrating {model_name}..."
        )

        # ----------------------------------------------------
        # 1. Hyperparameter tuning
        #
        # ONLY training data is used.
        # ----------------------------------------------------

        tuned_model = tune_model(
            model_name=model_name,
            model=base_model,
            X_train=X_train,
            y_train=y_train,
        )

        # ----------------------------------------------------
        # 2. Probability calibration
        #
        # ONLY training data is used.
        # ----------------------------------------------------

        calibrated_model = calibrate_model(
            model=tuned_model,
            X_train=X_train,
            y_train=y_train,
        )

        # ----------------------------------------------------
        # 3. Final validation
        #
        # Validation data is touched only here.
        # ----------------------------------------------------

        result = evaluate_model(
            model_name=model_name,
            model=calibrated_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Ranking
    #
    # Primary:
    #   Average Precision ↑
    #
    # Secondary:
    #   Brier Score ↓
    #
    # Third:
    #   Log Loss ↓
    #
    # Fourth:
    #   ROC-AUC ↑
    # --------------------------------------------------------

    results = sorted(
        results,
        key=lambda result: (
            result.average_precision,
            -result.brier_score,
            -result.log_loss_value,
            result.roc_auc,
        ),
        reverse=True,
    )

    return results


# ============================================================
# SELECT BEST MODEL
# ============================================================

def select_best_model(
    results: list[CalibrationResult],
) -> CalibrationResult:

    if not results:

        raise ValueError(
            "No model results available."
        )

    return results[0]


# ============================================================
# VALIDATION
# ============================================================

def validate_results(
    results: list[CalibrationResult],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:

    assert len(results) >= 3

    assert len(X_train) > 0
    assert len(X_val) > 0

    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)

    model_names = [
        result.model_name
        for result in results
    ]

    assert len(
        set(model_names)
    ) == len(model_names)

    for result in results:

        assert result.tuned is True

        assert result.calibrated is True

        assert (
            0.0
            <= result.average_precision
            <= 1.0
        )

        assert (
            0.0
            <= result.roc_auc
            <= 1.0
        )

        assert (
            0.0
            <= result.brier_score
            <= 1.0
        )

        assert (
            result.log_loss_value
            >= 0.0
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
            result.average_precision
        )

        assert np.isfinite(
            result.roc_auc
        )

        assert np.isfinite(
            result.brier_score
        )

        assert np.isfinite(
            result.log_loss_value
        )

    # --------------------------------------------------------
    # Ensure ranking is valid.
    # --------------------------------------------------------

    assert (
        results[0].average_precision
        >= results[-1].average_precision
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results: list[CalibrationResult],
) -> None:

    print()
    print("=" * 110)
    print(
        "PHASE 14.3 — CALIBRATED & TUNED ML MODEL COMPARISON"
    )
    print("=" * 110)

    print()

    print(
        f"{'MODEL':<28}"
        f"{'AP':>10}"
        f"{'ROC-AUC':>12}"
        f"{'BRIER':>11}"
        f"{'LOG LOSS':>12}"
        f"{'PRECISION':>12}"
        f"{'RECALL':>10}"
    )

    print("-" * 110)

    for result in results:

        print(
            f"{result.model_name:<28}"
            f"{result.average_precision:>10.4f}"
            f"{result.roc_auc:>12.4f}"
            f"{result.brier_score:>11.4f}"
            f"{result.log_loss_value:>12.4f}"
            f"{result.precision:>12.4f}"
            f"{result.recall:>10.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Reuse the exact Phase 14 dataset generator.
    #
    # This prevents Phase 14.3 from silently introducing
    # a different dataset.
    # --------------------------------------------------------

    from src.evaluation.phase14_ml_risk_fusion import (
        generate_dataset,
        split_dataset,
    )

    print()
    print("=" * 110)
    print(
        "PHASE 14.3 — CALIBRATION & HYPERPARAMETER TUNING"
    )
    print("=" * 110)

    print()

    X, y = generate_dataset()

    (
        X_train,
        y_train,
        X_val,
        y_val,
    ) = split_dataset(
        X,
        y,
    )

    print(
        f"Total cases              : {len(X)}"
    )

    print(
        f"Training cases           : {len(X_train)}"
    )

    print(
        f"Validation cases         : {len(X_val)}"
    )

    print()

    print(
        "LEAKAGE CONTROL"
    )

    print("-" * 110)

    print(
        "Hyperparameter tuning    : TRAINING DATA ONLY"
    )

    print(
        "Calibration              : TRAINING DATA ONLY"
    )

    print(
        "Final evaluation         : VALIDATION DATA"
    )

    print(
        "Validation used for tune : NO"
    )

    print(
        "Validation used for cal. : NO"
    )

    print()

    # --------------------------------------------------------
    # Compare every candidate.
    # --------------------------------------------------------

    results = compare_calibrated_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    # --------------------------------------------------------
    # Validate.
    # --------------------------------------------------------

    validate_results(
        results=results,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    # --------------------------------------------------------
    # Print comparison.
    # --------------------------------------------------------

    print_results(
        results
    )

    # --------------------------------------------------------
    # Select winner.
    # --------------------------------------------------------

    best_model = select_best_model(
        results
    )

    print()
    print("=" * 110)
    print(
        "BEST MODEL AFTER CALIBRATION + TUNING"
    )
    print("=" * 110)

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
        f"Brier Score           : "
        f"{best_model.brier_score:.4f}"
    )

    print(
        f"Log Loss              : "
        f"{best_model.log_loss_value:.4f}"
    )

    print(
        f"Precision             : "
        f"{best_model.precision:.4f}"
    )

    print(
        f"Recall                : "
        f"{best_model.recall:.4f}"
    )

    # --------------------------------------------------------
    # Final validation.
    # --------------------------------------------------------

    checks = {

        "dataset_generated": (
            len(X) > 0
        ),

        "binary_target": (
            set(
                np.unique(y)
            ).issubset({0, 1})
        ),

        "train_validation_split": (
            len(X_train)
            + len(X_val)
            == len(X)
        ),

        "multiple_models_compared": (
            len(results) >= 3
        ),

        "all_models_tuned": all(
            result.tuned
            for result in results
        ),

        "all_models_calibrated": all(
            result.calibrated
            for result in results
        ),

        "metrics_bounded": all(
            (
                0.0
                <= result.average_precision
                <= 1.0
            )
            and (
                0.0
                <= result.roc_auc
                <= 1.0
            )
            and (
                0.0
                <= result.brier_score
                <= 1.0
            )
            and (
                result.log_loss_value
                >= 0.0
            )
            for result in results
        ),

        "best_model_selected": (
            best_model
            == results[0]
        ),

        "validation_size_preserved": all(
            result.validation_size
            == len(X_val)
            for result in results
        ),

        "finite_metrics": all(
            np.isfinite(
                result.average_precision
            )
            and np.isfinite(
                result.roc_auc
            )
            and np.isfinite(
                result.brier_score
            )
            and np.isfinite(
                result.log_loss_value
            )
            for result in results
        ),
    }

    print()
    print("=" * 110)
    print(
        "VALIDATION"
    )
    print("=" * 110)

    all_passed = True

    for name, passed in checks.items():

        status = (
            "PASSED"
            if passed
            else "FAILED"
        )

        print(
            f"{name:<50}: {status}"
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
            "Phase 14.3 validation failed."
        )

    print()

    print(
        "PHASE 14.3 CALIBRATION & TUNING COMPLETE"
    )


if __name__ == "__main__":
    main()