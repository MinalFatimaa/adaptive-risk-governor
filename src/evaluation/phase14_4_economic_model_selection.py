#Phase 14.4 — REMOVED
#Reason:
#Economic model selection was conceptually coupled to an
#incorrect threshold/action mapping and therefore was removed
#from final ML model selection.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

SEED = 42

NUM_CASES = 1000
NUM_FEATURES = 12

DEVELOPMENT_FRACTION = 0.70
INNER_VALIDATION_FRACTION = 0.25

CALIBRATION_CV = 3

THRESHOLDS = np.round(
    np.arange(
        0.05,
        0.951,
        0.05,
    ),
    2,
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
# ECONOMIC CASE
# ============================================================

@dataclass(frozen=True)
class EconomicCase:
    case_id: str
    requested_amount: float
    is_legitimate: bool
    fraud_loss_if_allowed: float
    friction_cost_if_verified: float
    review_cost_if_escalated: float
    recovery_if_escalated: float = 0.0


# ============================================================
# ECONOMIC RESULT
# ============================================================

@dataclass(frozen=True)
class EconomicResult:
    model_name: str
    threshold: float

    total_net_benefit: float
    loss_prevention_rate: float
    false_intervention_rate: float

    fraud_loss_remaining: float
    friction_cost: float
    review_cost: float

    legitimate_refunds_preserved: float

    fraud_cases: int
    legitimate_cases: int

    average_precision: float
    roc_auc: float
    brier_score: float
    log_loss_value: float


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(
    *,
    count: int = NUM_CASES,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)

    X = rng.uniform(
        0.0,
        1.0,
        size=(count, NUM_FEATURES),
    )

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

    noise = rng.normal(
        loc=0.0,
        scale=0.12,
        size=count,
    )

    latent_score += noise

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

    y = (
        rng.random(count)
        < probability
    ).astype(int)

    if np.all(y == 0):
        y[0] = 1

    if np.all(y == 1):
        y[0] = 0

    return X, y


# ============================================================
# ECONOMIC CASE GENERATION
# ============================================================

def generate_economic_cases(
    y: np.ndarray,
    *,
    seed: int = SEED,
) -> list[EconomicCase]:

    rng = np.random.default_rng(seed + 1000)

    cases: list[EconomicCase] = []

    for index, label in enumerate(y):

        amount = round(
            float(
                rng.uniform(
                    500.0,
                    8000.0,
                )
            ),
            2,
        )

        is_legitimate = bool(label == 0)

        fraud_loss = (
            0.0
            if is_legitimate
            else amount
        )

        cases.append(
            EconomicCase(
                case_id=f"CASE_{index:05d}",
                requested_amount=amount,
                is_legitimate=is_legitimate,
                fraud_loss_if_allowed=fraud_loss,
                friction_cost_if_verified=25.0,
                review_cost_if_escalated=100.0,
                recovery_if_escalated=fraud_loss,
            )
        )

    return cases


# ============================================================
# MODEL FACTORY
# ============================================================

def build_base_models() -> dict[str, Any]:

    models: dict[str, Any] = {

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
                        random_state=SEED,
                    ),
                ),
            ]
        ),

        "RANDOM_FOREST": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        ),

        "HIST_GRADIENT_BOOSTING": (
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=SEED,
            )
        ),
    }

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
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )

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
            random_state=SEED,
            n_jobs=-1,
        )

    return models


# ============================================================
# CALIBRATION
# ============================================================

def build_calibrated_model(
    model: Any,
) -> CalibratedClassifierCV:

    return CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv=CALIBRATION_CV,
    )


# ============================================================
# DATA SPLITTING
# ============================================================

def split_development_holdout(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = SEED,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    rng = np.random.default_rng(seed)

    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]

    rng.shuffle(positive)
    rng.shuffle(negative)

    positive_holdout = max(
        1,
        int(
            len(positive)
            * (1.0 - DEVELOPMENT_FRACTION)
        ),
    )

    negative_holdout = max(
        1,
        int(
            len(negative)
            * (1.0 - DEVELOPMENT_FRACTION)
        ),
    )

    holdout_indices = np.concatenate(
        [
            positive[:positive_holdout],
            negative[:negative_holdout],
        ]
    )

    holdout_set = set(
        holdout_indices.tolist()
    )

    development_indices = np.array(
        [
            index
            for index in range(len(X))
            if index not in holdout_set
        ]
    )

    rng.shuffle(
        development_indices
    )

    rng.shuffle(
        holdout_indices
    )

    return (
        X[development_indices],
        y[development_indices],
        X[holdout_indices],
        y[holdout_indices],
    )


def split_inner_validation(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = SEED,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    rng = np.random.default_rng(seed)

    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]

    rng.shuffle(positive)
    rng.shuffle(negative)

    positive_count = max(
        1,
        int(
            len(positive)
            * INNER_VALIDATION_FRACTION
        ),
    )

    negative_count = max(
        1,
        int(
            len(negative)
            * INNER_VALIDATION_FRACTION
        ),
    )

    validation_indices = np.concatenate(
        [
            positive[:positive_count],
            negative[:negative_count],
        ]
    )

    validation_set = set(
        validation_indices.tolist()
    )

    train_indices = np.array(
        [
            index
            for index in range(len(X))
            if index not in validation_set
        ]
    )

    rng.shuffle(
        train_indices
    )

    rng.shuffle(
        validation_indices
    )

    return (
        X[train_indices],
        y[train_indices],
        X[validation_indices],
        y[validation_indices],
    )


# ============================================================
# GOVERNOR ACTION FROM PROBABILITY
# ============================================================

def probability_to_action(
    probability: float,
    threshold: float,
) -> str:

    if probability < threshold:

        return "ALLOW_AGENT_A_DECISION"

    if probability < min(
        threshold + 0.20,
        0.95,
    ):

        return "REQUEST_ADDITIONAL_EVIDENCE"

    return "ESCALATE_TO_HUMAN_REVIEW"


# ============================================================
# ECONOMIC EVALUATION
# ============================================================

def evaluate_economics(
    cases: list[EconomicCase],
    probabilities: np.ndarray,
    *,
    threshold: float,
    model_name: str,
    y_true: np.ndarray,
) -> EconomicResult:

    if len(cases) != len(probabilities):
        raise ValueError(
            "Cases and probabilities must have "
            "the same length."
        )

    total_net_cost = 0.0

    fraud_exposure = 0.0
    loss_prevented = 0.0
    remaining_loss = 0.0

    friction_cost = 0.0
    review_cost = 0.0

    legitimate_refunds = 0.0

    false_interventions = 0
    legitimate_cases = 0
    fraud_cases = 0

    for case, probability in zip(
        cases,
        probabilities,
    ):

        action = probability_to_action(
            float(probability),
            threshold,
        )

        amount = max(
            float(case.requested_amount),
            0.0,
        )

        fraud_loss = max(
            float(case.fraud_loss_if_allowed),
            0.0,
        )

        if case.is_legitimate:

            legitimate_cases += 1

            legitimate_refunds += amount

            if action != "ALLOW_AGENT_A_DECISION":

                false_interventions += 1

            if action == "REQUEST_ADDITIONAL_EVIDENCE":

                friction_cost += max(
                    case.friction_cost_if_verified,
                    0.0,
                )

            elif action == "ESCALATE_TO_HUMAN_REVIEW":

                review_cost += max(
                    case.review_cost_if_escalated,
                    0.0,
                )

        else:

            fraud_cases += 1

            fraud_exposure += fraud_loss

            if action == "ALLOW_AGENT_A_DECISION":

                remaining_loss += fraud_loss

            elif action == "REQUEST_ADDITIONAL_EVIDENCE":

                friction_cost += max(
                    case.friction_cost_if_verified,
                    0.0,
                )

                loss_prevented += fraud_loss

            elif action == "ESCALATE_TO_HUMAN_REVIEW":

                review_cost += max(
                    case.review_cost_if_escalated,
                    0.0,
                )

                recovery = min(
                    max(
                        case.recovery_if_escalated,
                        0.0,
                    ),
                    fraud_loss,
                )

                loss_prevented += recovery
                remaining_loss += (
                    fraud_loss - recovery
                )

    total_net_cost = (
        remaining_loss
        + friction_cost
        + review_cost
    )

    total_net_benefit = (
        fraud_exposure
        - total_net_cost
    )

    loss_prevention_rate = (
        loss_prevented / fraud_exposure
        if fraud_exposure > 0.0
        else 0.0
    )

    false_intervention_rate = (
        false_interventions / legitimate_cases
        if legitimate_cases > 0
        else 0.0
    )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    return EconomicResult(
        model_name=model_name,
        threshold=threshold,
        total_net_benefit=float(
            total_net_benefit
        ),
        loss_prevention_rate=float(
            np.clip(
                loss_prevention_rate,
                0.0,
                1.0,
            )
        ),
        false_intervention_rate=float(
            np.clip(
                false_intervention_rate,
                0.0,
                1.0,
            )
        ),
        fraud_loss_remaining=float(
            max(
                remaining_loss,
                0.0,
            )
        ),
        friction_cost=float(
            max(
                friction_cost,
                0.0,
            )
        ),
        review_cost=float(
            max(
                review_cost,
                0.0,
            )
        ),
        legitimate_refunds_preserved=float(
            max(
                legitimate_refunds,
                0.0,
            )
        ),
        fraud_cases=fraud_cases,
        legitimate_cases=legitimate_cases,
        average_precision=float(
            np.clip(
                average_precision_score(
                    y_true,
                    probabilities,
                ),
                0.0,
                1.0,
            )
        ),
        roc_auc=float(
            np.clip(
                roc_auc_score(
                    y_true,
                    probabilities,
                ),
                0.0,
                1.0,
            )
        ),
        brier_score=float(
            np.clip(
                brier_score_loss(
                    y_true,
                    probabilities,
                ),
                0.0,
                1.0,
            )
        ),
        log_loss_value=float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),
    )


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def select_economic_threshold(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_inner_val: np.ndarray,
    y_inner_val: np.ndarray,
    cases_inner_val: list[EconomicCase],
) -> EconomicResult:

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_inner_val
    )[:, 1]

    candidates: list[EconomicResult] = []

    for threshold in THRESHOLDS:

        result = evaluate_economics(
            cases_inner_val,
            probabilities,
            threshold=float(threshold),
            model_name=model_name,
            y_true=y_inner_val,
        )

        candidates.append(result)

    return max(
        candidates,
        key=lambda result: (
            result.total_net_benefit,
            result.loss_prevention_rate,
            -result.false_intervention_rate,
        ),
    )


# ============================================================
# FINAL MODEL EVALUATION
# ============================================================

def evaluate_final_model(
    model_name: str,
    model: Any,
    X_development: np.ndarray,
    y_development: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    cases_holdout: list[EconomicCase],
    threshold: float,
) -> EconomicResult:

    model.fit(
        X_development,
        y_development,
    )

    probabilities = model.predict_proba(
        X_holdout
    )[:, 1]

    return evaluate_economics(
        cases_holdout,
        probabilities,
        threshold=threshold,
        model_name=model_name,
        y_true=y_holdout,
    )


# ============================================================
# MODEL SELECTION
# ============================================================

def run_model_selection(
    X: np.ndarray,
    y: np.ndarray,
) -> list[EconomicResult]:

    (
        X_development,
        y_development,
        X_holdout,
        y_holdout,
    ) = split_development_holdout(
        X,
        y,
    )

    (
        X_inner_train,
        y_inner_train,
        X_inner_val,
        y_inner_val,
    ) = split_inner_validation(
        X_development,
        y_development,
        seed=SEED + 1,
    )

    all_cases = generate_economic_cases(
        y,
    )

    development_indices = set(
        range(
            len(X_development)
        )
    )

    # Reconstruct indices using deterministic
    # development split.
    rng = np.random.default_rng(SEED)

    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]

    rng.shuffle(positive)
    rng.shuffle(negative)

    positive_holdout = max(
        1,
        int(
            len(positive)
            * (1.0 - DEVELOPMENT_FRACTION)
        ),
    )

    negative_holdout = max(
        1,
        int(
            len(negative)
            * (1.0 - DEVELOPMENT_FRACTION)
        ),
    )

    holdout_indices = np.concatenate(
        [
            positive[:positive_holdout],
            negative[:negative_holdout],
        ]
    )

    holdout_set = set(
        holdout_indices.tolist()
    )

    development_indices_array = [
        index
        for index in range(len(X))
        if index not in holdout_set
    ]

    development_indices_array = np.array(
        development_indices_array
    )

    # Inner split reconstruction.
    rng = np.random.default_rng(
        SEED + 1
    )

    positive_inner = development_indices_array[
        y[development_indices_array] == 1
    ]

    negative_inner = development_indices_array[
        y[development_indices_array] == 0
    ]

    rng.shuffle(positive_inner)
    rng.shuffle(negative_inner)

    positive_inner_count = max(
        1,
        int(
            len(positive_inner)
            * INNER_VALIDATION_FRACTION
        ),
    )

    negative_inner_count = max(
        1,
        int(
            len(negative_inner)
            * INNER_VALIDATION_FRACTION
        ),
    )

    inner_val_indices = np.concatenate(
        [
            positive_inner[
                :positive_inner_count
            ],
            negative_inner[
                :negative_inner_count
            ],
        ]
    )

    inner_val_set = set(
        inner_val_indices.tolist()
    )

    inner_train_indices = np.array(
        [
            index
            for index in development_indices_array
            if index not in inner_val_set
        ]
    )

    inner_val_cases = [
        all_cases[index]
        for index in inner_val_indices
    ]

    models = build_base_models()

    final_results: list[EconomicResult] = []

    print()
    print("=" * 96)
    print("PHASE 14.4 — ECONOMIC MODEL SELECTION")
    print("=" * 96)

    print()
    print(
        f"Total cases                  : {len(X)}"
    )

    print(
        f"Development cases            : "
        f"{len(X_development)}"
    )

    print(
        f"Final holdout cases          : "
        f"{len(X_holdout)}"
    )

    print()
    print("LEAKAGE CONTROL")
    print("-" * 96)

    print(
        "Model fitting                : DEVELOPMENT DATA ONLY"
    )

    print(
        "Economic threshold tuning    : INNER VALIDATION ONLY"
    )

    print(
        "Final holdout used for tune  : NO"
    )

    print(
        "Final holdout used for model : NO"
    )

    print()

    selected_thresholds: dict[
        str,
        float,
    ] = {}

    for model_name, base_model in models.items():

        print(
            f"  Optimizing {model_name}..."
        )

        calibrated_model = (
            build_calibrated_model(
                base_model
            )
        )

        inner_result = select_economic_threshold(
            model_name=model_name,
            model=calibrated_model,
            X_train=X_inner_train,
            y_train=y_inner_train,
            X_inner_val=X_inner_val,
            y_inner_val=y_inner_val,
            cases_inner_val=inner_val_cases,
        )

        selected_thresholds[
            model_name
        ] = inner_result.threshold

    print()
    print("=" * 96)
    print("FINAL HOLDOUT ECONOMIC EVALUATION")
    print("=" * 96)

    print()
    print(
        f"{'MODEL':<28}"
        f"{'THRESHOLD':>11}"
        f"{'NET BENEFIT':>16}"
        f"{'LOSS PREV.':>13}"
        f"{'FALSE INT.':>13}"
        f"{'REMAINING':>15}"
    )

    print("-" * 96)

    for model_name, base_model in models.items():

        calibrated_model = (
            build_calibrated_model(
                base_model
            )
        )

        threshold = selected_thresholds[
            model_name
        ]

        result = evaluate_final_model(
            model_name=model_name,
            model=calibrated_model,
            X_development=X_development,
            y_development=y_development,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
            cases_holdout=[
                all_cases[index]
                for index in holdout_indices
            ],
            threshold=threshold,
        )

        final_results.append(
            result
        )

        print(
            f"{result.model_name:<28}"
            f"{result.threshold:>11.2f}"
            f"₹ {result.total_net_benefit:>13,.2f}"
            f"{result.loss_prevention_rate * 100:>11.2f}%"
            f"{result.false_intervention_rate * 100:>11.2f}%"
            f"₹ {result.fraud_loss_remaining:>12,.2f}"
        )

    return sorted(
        final_results,
        key=lambda result: (
            result.total_net_benefit,
            result.loss_prevention_rate,
            -result.false_intervention_rate,
        ),
        reverse=True,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_results(
    results: list[EconomicResult],
) -> None:

    assert len(results) >= 3

    for result in results:

        assert (
            0.0
            <= result.loss_prevention_rate
            <= 1.0
        )

        assert (
            0.0
            <= result.false_intervention_rate
            <= 1.0
        )

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
            result.fraud_loss_remaining
            >= 0.0
        )

        assert (
            result.friction_cost
            >= 0.0
        )

        assert (
            result.review_cost
            >= 0.0
        )

        assert (
            0.05
            <= result.threshold
            <= 0.95
        )

        assert np.isfinite(
            result.total_net_benefit
        )

    assert (
        results[0].total_net_benefit
        >= results[-1].total_net_benefit
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    X, y = generate_dataset()

    results = run_model_selection(
        X,
        y,
    )

    validate_results(
        results
    )

    best = results[0]

    print()
    print("=" * 96)
    print("ECONOMICALLY SELECTED MODEL")
    print("=" * 96)

    print()

    print(
        f"Model                         : "
        f"{best.model_name}"
    )

    print(
        f"Economic threshold            : "
        f"{best.threshold:.2f}"
    )

    print(
        f"Net economic benefit           : "
        f"₹ {best.total_net_benefit:,.2f}"
    )

    print(
        f"Loss prevention rate           : "
        f"{best.loss_prevention_rate * 100:.2f}%"
    )

    print(
        f"False intervention rate        : "
        f"{best.false_intervention_rate * 100:.2f}%"
    )

    print(
        f"Remaining fraud loss           : "
        f"₹ {best.fraud_loss_remaining:,.2f}"
    )

    print(
        f"Friction cost                  : "
        f"₹ {best.friction_cost:,.2f}"
    )

    print(
        f"Human review cost              : "
        f"₹ {best.review_cost:,.2f}"
    )

    print()
    print("=" * 96)
    print("VALIDATION")
    print("=" * 96)

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

        "all_models_have_threshold": all(
            0.05
            <= result.threshold
            <= 0.95
            for result in results
        ),

        "economic_metrics_bounded": all(
            0.0
            <= result.loss_prevention_rate
            <= 1.0
            and
            0.0
            <= result.false_intervention_rate
            <= 1.0
            for result in results
        ),

        "remaining_loss_non_negative": all(
            result.fraud_loss_remaining
            >= 0.0
            for result in results
        ),

        "financial_metrics_finite": all(
            np.isfinite(
                result.total_net_benefit
            )
            and np.isfinite(
                result.log_loss_value
            )
            for result in results
        ),

        "best_model_selected": (
            best == results[0]
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
            "Phase 14.4 validation failed."
        )

    print()
    print(
        "PHASE 14.4 ECONOMIC MODEL SELECTION COMPLETE"
    )


if __name__ == "__main__":
    main()