from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.evaluation.phase14_ml_risk_fusion import (
    generate_dataset,
)
from src.ml.risk_fusion_models import (
    build_models,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
NUM_CASES = 1000

N_SPLITS = 5

RANDOM_SEEDS = (
    42,
    123,
    2024,
)


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
class RobustModelResult:

    model_name: str

    folds_evaluated: int
    seeds_evaluated: int

    roc_auc_mean: float
    roc_auc_std: float

    average_precision_mean: float
    average_precision_std: float

    log_loss_mean: float
    log_loss_std: float

    brier_score_mean: float
    brier_score_std: float

    precision_mean: float
    precision_std: float

    recall_mean: float
    recall_std: float

    probability_mean: float


# ============================================================
# DATASET VALIDATION
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

    if X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            "Unexpected feature count."
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
            "Both classes must be present."
        )


# ============================================================
# SINGLE FOLD EVALUATION
# ============================================================

def evaluate_fold(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, float]:

    fitted_model = clone(model)

    fitted_model.fit(
        X_train,
        y_train,
    )

    probabilities = fitted_model.predict_proba(
        X_val
    )[:, 1]

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    return {
        "roc_auc": float(
            np.clip(
                roc_auc_score(
                    y_val,
                    probabilities,
                ),
                0.0,
                1.0,
            )
        ),

        "average_precision": float(
            np.clip(
                average_precision_score(
                    y_val,
                    probabilities,
                ),
                0.0,
                1.0,
            )
        ),

        "log_loss": float(
            max(
                log_loss(
                    y_val,
                    probabilities,
                    labels=[0, 1],
                ),
                0.0,
            )
        ),

        "brier_score": float(
            np.clip(
                brier_score_loss(
                    y_val,
                    probabilities,
                ),
                0.0,
                1.0,
            )
        ),

        "precision": float(
            np.clip(
                precision_score(
                    y_val,
                    predictions,
                    zero_division=0,
                ),
                0.0,
                1.0,
            )
        ),

        "recall": float(
            np.clip(
                recall_score(
                    y_val,
                    predictions,
                    zero_division=0,
                ),
                0.0,
                1.0,
            )
        ),

        "probability_mean": float(
            np.mean(probabilities)
        ),
    }


# ============================================================
# ROBUST MODEL EVALUATION
# ============================================================

def evaluate_model_robustly(
    model_name: str,
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = N_SPLITS,
    seeds: tuple[int, ...] = RANDOM_SEEDS,
) -> RobustModelResult:

    validate_dataset(
        X,
        y,
    )

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2."
        )

    if not seeds:
        raise ValueError(
            "At least one random seed is required."
        )

    all_metrics: list[dict[str, float]] = []

    for seed in seeds:

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )

        for train_indices, validation_indices in splitter.split(
            X,
            y,
        ):

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

            fold_metrics = evaluate_fold(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )

            all_metrics.append(
                fold_metrics
            )

    def values(
        name: str,
    ) -> np.ndarray:

        return np.array(
            [
                metric[name]
                for metric in all_metrics
            ],
            dtype=float,
        )

    roc_auc_values = values(
        "roc_auc"
    )

    average_precision_values = values(
        "average_precision"
    )

    log_loss_values = values(
        "log_loss"
    )

    brier_values = values(
        "brier_score"
    )

    precision_values = values(
        "precision"
    )

    recall_values = values(
        "recall"
    )

    probability_values = values(
        "probability_mean"
    )

    return RobustModelResult(

        model_name=model_name,

        folds_evaluated=n_splits,

        seeds_evaluated=len(seeds),

        roc_auc_mean=float(
            np.mean(roc_auc_values)
        ),

        roc_auc_std=float(
            np.std(
                roc_auc_values,
                ddof=0,
            )
        ),

        average_precision_mean=float(
            np.mean(
                average_precision_values
            )
        ),

        average_precision_std=float(
            np.std(
                average_precision_values,
                ddof=0,
            )
        ),

        log_loss_mean=float(
            np.mean(log_loss_values)
        ),

        log_loss_std=float(
            np.std(
                log_loss_values,
                ddof=0,
            )
        ),

        brier_score_mean=float(
            np.mean(brier_values)
        ),

        brier_score_std=float(
            np.std(
                brier_values,
                ddof=0,
            )
        ),

        precision_mean=float(
            np.mean(precision_values)
        ),

        precision_std=float(
            np.std(
                precision_values,
                ddof=0,
            )
        ),

        recall_mean=float(
            np.mean(recall_values)
        ),

        recall_std=float(
            np.std(
                recall_values,
                ddof=0,
            )
        ),

        probability_mean=float(
            np.mean(probability_values)
        ),
    )


# ============================================================
# COMPARE ALL MODELS
# ============================================================

def compare_models_robustly(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = N_SPLITS,
    seeds: tuple[int, ...] = RANDOM_SEEDS,
) -> list[RobustModelResult]:

    validate_dataset(
        X,
        y,
    )

    models = build_models()

    results: list[RobustModelResult] = []

    for model_name, model in models.items():

        result = evaluate_model_robustly(
            model_name=model_name,
            model=model,
            X=X,
            y=y,
            n_splits=n_splits,
            seeds=seeds,
        )

        results.append(
            result
        )

    return sorted(
        results,
        key=lambda result: (
            result.average_precision_mean,
            -result.average_precision_std,
            result.roc_auc_mean,
            -result.brier_score_mean,
            -result.log_loss_mean,
        ),
        reverse=True,
    )


# ============================================================
# BEST MODEL
# ============================================================

def select_robust_best_model(
    results: list[RobustModelResult],
) -> RobustModelResult:

    if not results:
        raise ValueError(
            "No model results available."
        )

    return results[0]


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results: list[RobustModelResult],
) -> None:

    print()
    print("=" * 105)
    print(
        "PHASE 14.2 — ROBUST ML MODEL EVALUATION"
    )
    print("=" * 105)

    print()

    print(
        f"Cross-validation folds : {N_SPLITS}"
    )

    print(
        f"Random seeds           : {len(RANDOM_SEEDS)}"
    )

    print(
        f"Total evaluations/model: "
        f"{N_SPLITS * len(RANDOM_SEEDS)}"
    )

    print()

    print(
        "MODEL PERFORMANCE"
    )

    print("-" * 105)

    print(
        f"{'MODEL':<26}"
        f"{'AP MEAN':>12}"
        f"{'AP STD':>11}"
        f"{'ROC-AUC':>12}"
        f"{'Brier':>11}"
        f"{'Log Loss':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>11}"
    )

    print("-" * 105)

    for result in results:

        print(
            f"{result.model_name:<26}"
            f"{result.average_precision_mean:>12.4f}"
            f"{result.average_precision_std:>11.4f}"
            f"{result.roc_auc_mean:>12.4f}"
            f"{result.brier_score_mean:>11.4f}"
            f"{result.log_loss_mean:>12.4f}"
            f"{result.precision_mean:>12.4f}"
            f"{result.recall_mean:>11.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    X, y = generate_dataset(
        count=NUM_CASES,
        seed=SEED,
    )

    results = compare_models_robustly(
        X,
        y,
    )

    print_results(
        results
    )

    best_model = select_robust_best_model(
        results
    )

    print()
    print("=" * 105)
    print("ROBUST CURRENT BEST MODEL")
    print("=" * 105)

    print()

    print(
        f"Model                 : "
        f"{best_model.model_name}"
    )

    print(
        f"Average Precision     : "
        f"{best_model.average_precision_mean:.4f}"
        f" ± "
        f"{best_model.average_precision_std:.4f}"
    )

    print(
        f"ROC-AUC               : "
        f"{best_model.roc_auc_mean:.4f}"
    )

    print(
        f"Brier Score           : "
        f"{best_model.brier_score_mean:.4f}"
    )

    print(
        f"Log Loss              : "
        f"{best_model.log_loss_mean:.4f}"
    )

    print(
        f"Precision             : "
        f"{best_model.precision_mean:.4f}"
    )

    print(
        f"Recall                : "
        f"{best_model.recall_mean:.4f}"
    )

    print()
    print("=" * 105)
    print("VALIDATION")
    print("=" * 105)

    checks = {

        "dataset_generated": (
            len(X)
            == NUM_CASES
        ),

        "binary_target": (
            set(
                np.unique(y)
            ).issubset({0, 1})
        ),

        "multiple_models_compared": (
            len(results) >= 3
        ),

        "all_models_completed": all(
            result.folds_evaluated
            == N_SPLITS
            for result in results
        ),

        "all_seeds_evaluated": all(
            result.seeds_evaluated
            == len(RANDOM_SEEDS)
            for result in results
        ),

        "average_precision_bounded": all(
            0.0
            <= result.average_precision_mean
            <= 1.0
            for result in results
        ),

        "average_precision_std_bounded": all(
            result.average_precision_std
            >= 0.0
            for result in results
        ),

        "roc_auc_bounded": all(
            0.0
            <= result.roc_auc_mean
            <= 1.0
            for result in results
        ),

        "brier_bounded": all(
            0.0
            <= result.brier_score_mean
            <= 1.0
            for result in results
        ),

        "log_loss_valid": all(
            result.log_loss_mean
            >= 0.0
            and np.isfinite(
                result.log_loss_mean
            )
            for result in results
        ),

        "probabilities_valid": all(
            0.0
            <= result.probability_mean
            <= 1.0
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
            f"{name:<45}: {status}"
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
            "Phase 14.2 validation failed."
        )

    print()

    print(
        "PHASE 14.2 ROBUST MODEL EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()