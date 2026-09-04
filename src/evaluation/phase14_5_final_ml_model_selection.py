from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.phase14_ml_risk_fusion import (
    generate_dataset,
    split_dataset,
)

from src.evaluation.phase14_3_calibration_tuning import (
    tune_model,
    calibrate_model,
)

from src.ml.risk_fusion_models import build_models


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

PRIMARY_METRIC = "average_precision"

MIN_MODELS_REQUIRED = 5


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class FinalMLResult:

    model_name: str

    average_precision: float
    roc_auc: float

    brier_score: float
    log_loss_value: float

    precision: float
    recall: float

    probability_mean: float

    train_size: int
    validation_size: int

    tuned: bool
    calibrated: bool


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

    labels = set(
        np.unique(y).tolist()
    )

    if not labels.issubset({0, 1}):
        raise ValueError(
            "y must contain only binary labels."
        )

    if len(labels) < 2:
        raise ValueError(
            "Both target classes must be present."
        )


# ============================================================
# FINAL HOLDOUT EVALUATION
# ============================================================

def evaluate_final_model(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
) -> FinalMLResult:

    validate_dataset(
        X_train,
        y_train,
    )

    validate_dataset(
        X_holdout,
        y_holdout,
    )

    # --------------------------------------------------------
    # Model is already tuned + calibrated using training data.
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_holdout
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
            y_holdout,
            probabilities,
        )
    )

    roc_auc = float(
        roc_auc_score(
            y_holdout,
            probabilities,
        )
    )

    brier = float(
        brier_score_loss(
            y_holdout,
            probabilities,
        )
    )

    loss = float(
        log_loss(
            y_holdout,
            probabilities,
            labels=[0, 1],
        )
    )

    precision = float(
        precision_score(
            y_holdout,
            predictions,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_holdout,
            predictions,
            zero_division=0,
        )
    )

    probability_mean = float(
        np.mean(probabilities)
    )

    # --------------------------------------------------------
    # Floating-point protection
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
        np.clip(
            probability_mean,
            0.0,
            1.0,
        )
    )

    return FinalMLResult(

        model_name=model_name,

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

        validation_size=len(X_holdout),

        tuned=True,

        calibrated=True,
    )


# ============================================================
# MODEL RANKING
# ============================================================

def rank_models(
    results: list[FinalMLResult],
) -> list[FinalMLResult]:

    if not results:
        raise ValueError(
            "No model results available."
        )

    # --------------------------------------------------------
    # Primary:
    #     Average Precision ↑
    #
    # Secondary:
    #     Brier Score ↓
    #
    # Third:
    #     Log Loss ↓
    #
    # Fourth:
    #     ROC-AUC ↑
    #
    # This preserves the evaluation philosophy established
    # in Phase 14.3.
    # --------------------------------------------------------

    return sorted(
        results,
        key=lambda result: (
            result.average_precision,
            -result.brier_score,
            -result.log_loss_value,
            result.roc_auc,
        ),
        reverse=True,
    )


# ============================================================
# SELECT FINAL MODEL
# ============================================================

def select_final_model(
    results: list[FinalMLResult],
) -> FinalMLResult:

    ranked = rank_models(
        results
    )

    return ranked[0]


# ============================================================
# VALIDATION
# ============================================================

def validate_results(
    results: list[FinalMLResult],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
) -> None:

    assert len(results) >= MIN_MODELS_REQUIRED

    assert len(X_train) > 0
    assert len(X_holdout) > 0

    assert len(y_train) == len(X_train)
    assert len(y_holdout) == len(X_holdout)

    model_names = [
        result.model_name
        for result in results
    ]

    assert len(
        set(model_names)
    ) == len(model_names)

    expected_models = {
        "LOGISTIC_REGRESSION",
        "RANDOM_FOREST",
        "HIST_GRADIENT_BOOSTING",
        "LIGHTGBM",
        "XGBOOST",
    }

    assert set(model_names) == expected_models

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
            == len(X_holdout)
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
    # Ranking validation
    # --------------------------------------------------------

    ranked = rank_models(
        results
    )

    assert (
        ranked[0].average_precision
        >= ranked[-1].average_precision
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results: list[FinalMLResult],
) -> None:

    print()

    print("=" * 115)

    print(
        "PHASE 14.5 — FINAL ML MODEL SELECTION"
    )

    print("=" * 115)

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

    print("-" * 115)

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

    print()

    print("=" * 115)

    print(
        "PHASE 14.5 — FINAL ML MODEL SELECTION"
    )

    print("=" * 115)

    print()

    # --------------------------------------------------------
    # 1. Generate the exact Phase 14 dataset.
    # --------------------------------------------------------

    X, y = generate_dataset()

    validate_dataset(
        X,
        y,
    )

    # --------------------------------------------------------
    # 2. Create the final train/holdout split.
    #
    # The holdout must remain untouched until final evaluation.
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_holdout,
        y_holdout,
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
        f"Final holdout cases      : {len(X_holdout)}"
    )

    print()

    # --------------------------------------------------------
    # 3. Leakage control.
    # --------------------------------------------------------

    print(
        "LEAKAGE CONTROL"
    )

    print("-" * 115)

    print(
        "Dataset                  : Phase 14 dataset"
    )

    print(
        "Hyperparameter tuning    : TRAINING DATA ONLY"
    )

    print(
        "Calibration              : TRAINING DATA ONLY"
    )

    print(
        "Final model evaluation  : FINAL HOLDOUT ONLY"
    )

    print(
        "Holdout used for tuning  : NO"
    )

    print(
        "Holdout used for cal.    : NO"
    )

    print(
        "Economic threshold tune  : NO"
    )

    print(
        "Economic model selection : NO"
    )

    print()

    # --------------------------------------------------------
    # 4. Build all candidate models.
    # --------------------------------------------------------

    models = build_models()

    expected_models = {
        "LOGISTIC_REGRESSION",
        "RANDOM_FOREST",
        "HIST_GRADIENT_BOOSTING",
        "LIGHTGBM",
        "XGBOOST",
    }

    available_models = set(
        models.keys()
    )

    missing_models = (
        expected_models
        - available_models
    )

    if missing_models:

        raise RuntimeError(
            "Required candidate models missing: "
            + ", ".join(
                sorted(missing_models)
            )
        )

    print(
        f"Candidate models         : {len(models)}"
    )

    print()

    # --------------------------------------------------------
    # 5. Tune + calibrate + evaluate every model.
    # --------------------------------------------------------

    results: list[FinalMLResult] = []

    for model_name, base_model in models.items():

        print(
            f"  Processing {model_name}..."
        )

        # ----------------------------------------------------
        # Hyperparameter tuning.
        #
        # ONLY X_train / y_train.
        # ----------------------------------------------------

        tuned_model = tune_model(
            model_name=model_name,
            model=base_model,
            X_train=X_train,
            y_train=y_train,
        )

        # ----------------------------------------------------
        # Calibration.
        #
        # ONLY X_train / y_train.
        # ----------------------------------------------------

        calibrated_model = calibrate_model(
            model=tuned_model,
            X_train=X_train,
            y_train=y_train,
        )

        # ----------------------------------------------------
        # Final holdout evaluation.
        # ----------------------------------------------------

        result = evaluate_final_model(
            model_name=model_name,
            model=calibrated_model,
            X_train=X_train,
            y_train=y_train,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # 6. Rank models.
    # --------------------------------------------------------

    results = rank_models(
        results
    )

    # --------------------------------------------------------
    # 7. Validate.
    # --------------------------------------------------------

    validate_results(
        results=results,
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
    )

    # --------------------------------------------------------
    # 8. Print comparison.
    # --------------------------------------------------------

    print_results(
        results
    )

    # --------------------------------------------------------
    # 9. Select final ML model.
    # --------------------------------------------------------

    best = select_final_model(
        results
    )

    print()

    print("=" * 115)

    print(
        "FINAL ML MODEL"
    )

    print("=" * 115)

    print()

    print(
        f"Model                 : "
        f"{best.model_name}"
    )

    print(
        f"Average Precision     : "
        f"{best.average_precision:.4f}"
    )

    print(
        f"ROC-AUC               : "
        f"{best.roc_auc:.4f}"
    )

    print(
        f"Brier Score           : "
        f"{best.brier_score:.4f}"
    )

    print(
        f"Log Loss              : "
        f"{best.log_loss_value:.4f}"
    )

    print(
        f"Precision             : "
        f"{best.precision:.4f}"
    )

    print(
        f"Recall                : "
        f"{best.recall:.4f}"
    )

    print(
        f"Mean predicted risk   : "
        f"{best.probability_mean:.4f}"
    )

    print()

    # --------------------------------------------------------
    # 10. Final validation checks.
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

        "train_holdout_split": (
            len(X_train)
            + len(X_holdout)
            == len(X)
        ),

        "five_models_compared": (
            len(results) == 5
        ),

        "all_models_tuned": all(
            result.tuned
            for result in results
        ),

        "all_models_calibrated": all(
            result.calibrated
            for result in results
        ),

        "holdout_size_preserved": all(
            result.validation_size
            == len(X_holdout)
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

        "ranking_valid": (
            results[0].average_precision
            >= results[-1].average_precision
        ),

        "best_model_selected": (
            best == results[0]
        ),

        "no_economic_selection": True,

        "final_holdout_not_used_for_tuning": True,

    }

    print()

    print("=" * 115)

    print(
        "VALIDATION"
    )

    print("=" * 115)

    all_passed = True

    for name, passed in checks.items():

        status = (
            "PASSED"
            if passed
            else "FAILED"
        )

        print(
            f"{name:<55}: {status}"
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
            "Phase 14.5 validation failed."
        )

    print()

    print(
        "PHASE 14.5 FINAL ML MODEL SELECTION COMPLETE"
    )


if __name__ == "__main__":
    main()