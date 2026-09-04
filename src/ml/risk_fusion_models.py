from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42


# ============================================================
# NUMERICAL SAFETY
# ============================================================

def _bounded_metric(
    value: float,
) -> float:
    """
    Keep a metric inside its mathematically valid [0, 1]
    interval.

    Floating-point arithmetic can occasionally produce
    values such as:

        1.0000000000000002

    even when the true metric is exactly 1.0.

    This is numerical protection only. It does not change
    meaningful metric values.
    """

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(
            "Metric value must be finite."
        )

    return max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class ModelResult:

    model_name: str

    roc_auc: float
    average_precision: float
    log_loss_value: float
    brier_score: float

    precision: float
    recall: float

    train_size: int
    validation_size: int

    probability_mean: float


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_feature_matrix(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Validate the feature matrix and binary target.
    """

    if not isinstance(
        X,
        np.ndarray,
    ):
        raise TypeError(
            "X must be a numpy array."
        )

    if not isinstance(
        y,
        np.ndarray,
    ):
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

    if len(X) == 0:
        raise ValueError(
            "Feature matrix cannot be empty."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "X contains NaN or infinite values."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            "y contains NaN or infinite values."
        )

    unique_labels = set(
        np.unique(y).tolist()
    )

    if not unique_labels.issubset(
        {0, 1}
    ):
        raise ValueError(
            "y must contain only binary labels 0 and 1."
        )

    if len(unique_labels) < 2:
        raise ValueError(
            "y must contain both classes 0 and 1."
        )


# ============================================================
# MODEL FACTORY
# ============================================================

def build_models() -> dict[str, Any]:
    """
    Build the candidate supervised ML models.

    Phase 14 compares multiple model families instead of
    assuming that one algorithm is automatically superior.

    Candidates:

        1. Logistic Regression
        2. Random Forest
        3. HistGradientBoosting
        4. LightGBM
        5. XGBoost

    LightGBM and XGBoost are optional dependencies.
    """

    models: dict[str, Any] = {

        # ----------------------------------------------------
        # 1. LOGISTIC REGRESSION
        # ----------------------------------------------------

        "LOGISTIC_REGRESSION": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # ----------------------------------------------------
        # 2. RANDOM FOREST
        # ----------------------------------------------------

        "RANDOM_FOREST": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        # ----------------------------------------------------
        # 3. HISTOGRAM GRADIENT BOOSTING
        # ----------------------------------------------------

        "HIST_GRADIENT_BOOSTING": (
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            )
        ),
    }

    # --------------------------------------------------------
    # 4. LIGHTGBM
    # --------------------------------------------------------

    if LGBMClassifier is not None:

        models["LIGHTGBM"] = LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

    # --------------------------------------------------------
    # 5. XGBOOST
    # --------------------------------------------------------

    if XGBClassifier is not None:

        models["XGBOOST"] = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    return models


# ============================================================
# PROBABILITY VALIDATION
# ============================================================

def _validate_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Validate and safely clip predicted probabilities.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.ndim != 1:
        raise ValueError(
            "Predicted probabilities must be one-dimensional."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Model produced NaN or infinite probabilities."
        )

    return np.clip(
        probabilities,
        0.0,
        1.0,
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> ModelResult:
    """
    Train and evaluate one candidate ML model.
    """

    validate_feature_matrix(
        X_train,
        y_train,
    )

    validate_feature_matrix(
        X_val,
        y_val,
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    probabilities = _validate_probabilities(
        probabilities
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc = _bounded_metric(
        roc_auc_score(
            y_val,
            probabilities,
        )
    )

    # --------------------------------------------------------
    # Average Precision
    # --------------------------------------------------------

    average_precision = _bounded_metric(
        average_precision_score(
            y_val,
            probabilities,
        )
    )

    # --------------------------------------------------------
    # Log Loss
    # --------------------------------------------------------

    log_loss_value = float(
        log_loss(
            y_val,
            probabilities,
            labels=[0, 1],
        )
    )

    if not np.isfinite(
        log_loss_value
    ):
        raise ValueError(
            "Log loss must be finite."
        )

    # --------------------------------------------------------
    # Brier Score
    # --------------------------------------------------------

    brier_score = _bounded_metric(
        brier_score_loss(
            y_val,
            probabilities,
        )
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = _bounded_metric(
        precision_score(
            y_val,
            predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = _bounded_metric(
        recall_score(
            y_val,
            predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Mean predicted probability
    # --------------------------------------------------------

    probability_mean = _bounded_metric(
        np.mean(
            probabilities
        )
    )

    return ModelResult(

        model_name=model_name,

        roc_auc=roc_auc,

        average_precision=average_precision,

        log_loss_value=log_loss_value,

        brier_score=brier_score,

        precision=precision,

        recall=recall,

        train_size=len(X_train),

        validation_size=len(X_val),

        probability_mean=probability_mean,
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> list[ModelResult]:
    """
    Train and evaluate all available candidate models.

    Every model receives exactly the same train/validation
    split to make the comparison fair.
    """

    models = build_models()

    results: list[ModelResult] = []

    for model_name, model in models.items():

        result = evaluate_model(
            model_name=model_name,
            model=model,
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
    # --------------------------------------------------------
    #
    # Primary:
    #     Average Precision
    #
    # Secondary:
    #     ROC-AUC
    #
    # Tertiary:
    #     Brier score
    #
    # Final:
    #     Log loss
    #
    # Higher AP/AUC is better.
    # Lower Brier/log-loss is better.
    # --------------------------------------------------------

    return sorted(
        results,
        key=lambda result: (
            result.average_precision,
            result.roc_auc,
            -result.brier_score,
            -result.log_loss_value,
        ),
        reverse=True,
    )


# ============================================================
# BEST MODEL
# ============================================================

def select_best_model(
    results: list[ModelResult],
) -> ModelResult:
    """
    Select the highest-ranked candidate model.
    """

    if not results:
        raise ValueError(
            "No model results available."
        )

    return results[0]