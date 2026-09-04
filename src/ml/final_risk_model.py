from __future__ import annotations

from typing import Any

import numpy as np

from sklearn.calibration import CalibratedClassifierCV
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
# FEATURE CONTRACT
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
# HYPERPARAMETER GRIDS
# ============================================================

def get_parameter_grids() -> dict[str, dict[str, list[Any]]]:

    return {

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
# VALIDATION
# ============================================================

def validate_dataset(
    X: np.ndarray,
    y: np.ndarray,
) -> None:

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")

    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array.")

    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")

    if y.ndim != 1:
        raise ValueError("y must be one-dimensional.")

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
# TUNING
# ============================================================

def tune_model(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:

    parameter_grid = get_parameter_grids().get(
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
# CALIBRATION
# ============================================================

def calibrate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:

    calibration_cv = StratifiedKFold(
        n_splits=INNER_CV_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=CALIBRATION_METHOD,
        cv=calibration_cv,
        n_jobs=-1,
    )

    calibrated.fit(
        X_train,
        y_train,
    )

    return calibrated


# ============================================================
# BUILD FINAL MODEL
# ============================================================

def build_final_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:

    validate_dataset(
        X_train,
        y_train,
    )

    models = build_models()

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    base_model = models[model_name]

    tuned_model = tune_model(
        model_name=model_name,
        model=base_model,
        X_train=X_train,
        y_train=y_train,
    )

    calibrated_model = calibrate_model(
        model=tuned_model,
        X_train=X_train,
        y_train=y_train,
    )

    return calibrated_model


# ============================================================
# PREDICT RISK
# ============================================================

def predict_risk(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:

    if not isinstance(X, np.ndarray):
        raise TypeError(
            "X must be a numpy array."
        )

    if X.ndim != 2:
        raise ValueError(
            "X must be two-dimensional."
        )

    if X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            "Unexpected feature count."
        )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Model produced invalid probabilities."
        )

    return np.clip(
        probabilities,
        0.0,
        1.0,
    )