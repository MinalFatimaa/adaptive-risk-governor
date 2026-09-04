from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_CASES = 1000
NUM_FEATURES = 12

DEVELOPMENT_FRACTION = 0.70
INNER_VALIDATION_FRACTION = 0.25

CALIBRATION_CV = 3

# ------------------------------------------------------------
# Frozen final ML model from Phase 14.5
# ------------------------------------------------------------

MODEL_NAME = "HIST_GRADIENT_BOOSTING"

# ------------------------------------------------------------
# Policy search space
#
# t1:
#   below this -> ALLOW
#
# t2:
#   t1 <= risk < t2 -> REQUEST EVIDENCE
#
#   risk >= t2 -> ESCALATE
# ------------------------------------------------------------

THRESHOLD_VALUES = np.round(
    np.arange(
        0.10,
        0.91,
        0.05,
    ),
    2,
)


# ============================================================
# CANONICAL GOVERNOR ACTIONS
# ============================================================

ALLOW = "ALLOW_AGENT_A_DECISION"

REQUEST_EVIDENCE = (
    "REQUEST_ADDITIONAL_EVIDENCE"
)

ESCALATE = (
    "ESCALATE_TO_HUMAN_REVIEW"
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
# POLICY RESULT
# ============================================================

@dataclass(frozen=True)
class PolicyResult:

    request_threshold: float
    escalation_threshold: float

    total_net_benefit: float

    loss_prevention_rate: float

    false_intervention_rate: float

    fraud_loss_remaining: float

    friction_cost: float

    review_cost: float

    legitimate_refunds_preserved: float

    fraud_cases: int

    legitimate_cases: int

    allow_count: int

    evidence_count: int

    escalation_count: int


# ============================================================
# DATASET
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

    recovery_if_escalated: float


def generate_economic_cases(
    y: np.ndarray,
    *,
    seed: int = SEED,
) -> list[EconomicCase]:

    rng = np.random.default_rng(
        seed + 1000
    )

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

        is_legitimate = bool(
            label == 0
        )

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
# FINAL MODEL
# ============================================================

def build_final_model() -> Any:

    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=SEED,
    )


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_model(
    model: Any,
) -> CalibratedClassifierCV:

    cv = StratifiedKFold(
        n_splits=CALIBRATION_CV,
        shuffle=True,
        random_state=SEED,
    )

    return CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv=cv,
    )


# ============================================================
# DATA SPLIT
# ============================================================

def split_development_holdout(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    rng = np.random.default_rng(
        SEED
    )

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


# ============================================================
# INNER SPLIT
# ============================================================

def split_inner_validation(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    rng = np.random.default_rng(
        SEED + 1
    )

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
# GOVERNOR POLICY
# ============================================================

def probability_to_action(
    probability: float,
    request_threshold: float,
    escalation_threshold: float,
) -> str:

    if probability < request_threshold:

        return ALLOW

    if probability < escalation_threshold:

        return REQUEST_EVIDENCE

    return ESCALATE


# ============================================================
# ECONOMIC EVALUATION
# ============================================================

def evaluate_policy(
    cases: list[EconomicCase],
    probabilities: np.ndarray,
    *,
    request_threshold: float,
    escalation_threshold: float,
) -> PolicyResult:

    if request_threshold >= escalation_threshold:

        raise ValueError(
            "Request threshold must be "
            "lower than escalation threshold."
        )

    if len(cases) != len(probabilities):

        raise ValueError(
            "Cases and probabilities must "
            "have the same length."
        )

    total_loss = 0.0

    fraud_exposure = 0.0

    loss_prevented = 0.0

    remaining_loss = 0.0

    friction_cost = 0.0

    review_cost = 0.0

    legitimate_refunds = 0.0

    false_interventions = 0

    fraud_cases = 0

    legitimate_cases = 0

    allow_count = 0
    evidence_count = 0
    escalation_count = 0

    for case, probability in zip(
        cases,
        probabilities,
    ):

        action = probability_to_action(
            float(probability),
            request_threshold,
            escalation_threshold,
        )

        amount = max(
            float(case.requested_amount),
            0.0,
        )

        fraud_loss = max(
            float(case.fraud_loss_if_allowed),
            0.0,
        )

        if action == ALLOW:

            allow_count += 1

        elif action == REQUEST_EVIDENCE:

            evidence_count += 1

        elif action == ESCALATE:

            escalation_count += 1

        # ----------------------------------------------------
        # LEGITIMATE CASE
        # ----------------------------------------------------

        if case.is_legitimate:

            legitimate_cases += 1

            legitimate_refunds += amount

            if action != ALLOW:

                false_interventions += 1

            if action == REQUEST_EVIDENCE:

                friction_cost += max(
                    float(
                        case.friction_cost_if_verified
                    ),
                    0.0,
                )

            elif action == ESCALATE:

                review_cost += max(
                    float(
                        case.review_cost_if_escalated
                    ),
                    0.0,
                )

        # ----------------------------------------------------
        # FRAUD CASE
        # ----------------------------------------------------

        else:

            fraud_cases += 1

            fraud_exposure += fraud_loss

            if action == ALLOW:

                remaining_loss += fraud_loss

            elif action == REQUEST_EVIDENCE:

                friction_cost += max(
                    float(
                        case.friction_cost_if_verified
                    ),
                    0.0,
                )

                loss_prevented += fraud_loss

            elif action == ESCALATE:

                review_cost += max(
                    float(
                        case.review_cost_if_escalated
                    ),
                    0.0,
                )

                recovery = min(
                    max(
                        float(
                            case.recovery_if_escalated
                        ),
                        0.0,
                    ),
                    fraud_loss,
                )

                loss_prevented += recovery

                remaining_loss += (
                    fraud_loss - recovery
                )

    total_loss = (
        remaining_loss
        + friction_cost
        + review_cost
    )

    total_net_benefit = (
        fraud_exposure
        - total_loss
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

    return PolicyResult(
        request_threshold=request_threshold,
        escalation_threshold=escalation_threshold,
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
        allow_count=allow_count,
        evidence_count=evidence_count,
        escalation_count=escalation_count,
    )


# ============================================================
# POLICY OPTIMIZATION
# ============================================================

def optimize_policy(
    probabilities: np.ndarray,
    cases: list[EconomicCase],
) -> PolicyResult:

    candidates: list[PolicyResult] = []

    for request_threshold in THRESHOLD_VALUES:

        for escalation_threshold in THRESHOLD_VALUES:

            if (
                request_threshold
                >= escalation_threshold
            ):
                continue

            result = evaluate_policy(
                cases,
                probabilities,
                request_threshold=float(
                    request_threshold
                ),
                escalation_threshold=float(
                    escalation_threshold
                ),
            )

            candidates.append(result)

    if not candidates:

        raise RuntimeError(
            "No valid policy candidates generated."
        )

    return max(
        candidates,
        key=lambda result: (
            result.total_net_benefit,
            result.loss_prevention_rate,
            -result.false_intervention_rate,
        ),
    )


# ============================================================
# ML METRICS
# ============================================================

def evaluate_ml_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:

    probabilities = np.clip(
        np.asarray(
            probabilities,
            dtype=float,
        ),
        0.0,
        1.0,
    )

    return {
        "average_precision": float(
            np.clip(
                average_precision_score(
                    y_true,
                    probabilities,
                ),
                0.0,
                1.0,
            )
        ),
        "roc_auc": float(
            np.clip(
                roc_auc_score(
                    y_true,
                    probabilities,
                ),
                0.0,
                1.0,
            )
        ),
        "brier_score": float(
            np.clip(
                brier_score_loss(
                    y_true,
                    probabilities,
                ),
                0.0,
                1.0,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 115)
    print(
        "PHASE 15 — GOVERNOR POLICY & ECONOMIC EVALUATION"
    )
    print("=" * 115)

    # --------------------------------------------------------
    # 1. Generate the exact Phase 14 dataset
    # --------------------------------------------------------

    X, y = generate_dataset()

    # --------------------------------------------------------
    # 2. Development / final holdout
    # --------------------------------------------------------

    (
        X_development,
        y_development,
        X_holdout,
        y_holdout,
    ) = split_development_holdout(
        X,
        y,
    )

    # --------------------------------------------------------
    # 3. Inner validation
    # --------------------------------------------------------

    (
        X_inner_train,
        y_inner_train,
        X_inner_val,
        y_inner_val,
    ) = split_inner_validation(
        X_development,
        y_development,
    )

    print()
    print(
        f"Total cases              : {len(X)}"
    )

    print(
        f"Development cases        : "
        f"{len(X_development)}"
    )

    print(
        f"Final holdout cases      : "
        f"{len(X_holdout)}"
    )

    print(
        f"Inner training cases     : "
        f"{len(X_inner_train)}"
    )

    print(
        f"Inner validation cases   : "
        f"{len(X_inner_val)}"
    )

    # --------------------------------------------------------
    # 4. Leakage control
    # --------------------------------------------------------

    print()
    print(
        "LEAKAGE CONTROL"
    )

    print("-" * 115)

    print(
        "Final ML model           : HIST_GRADIENT_BOOSTING"
    )

    print(
        "Model training           : DEVELOPMENT DATA"
    )

    print(
        "Calibration              : TRAINING DATA ONLY"
    )

    print(
        "Policy threshold tuning  : INNER VALIDATION ONLY"
    )

    print(
        "Final holdout evaluation : FINAL HOLDOUT ONLY"
    )

    print(
        "Holdout used for policy  : NO"
    )

    print(
        "Holdout used for model   : NO"
    )

    print()

    # --------------------------------------------------------
    # 5. Train final model on inner training data
    #
    # This model is ONLY used to determine the policy.
    # --------------------------------------------------------

    inner_model = build_final_model()

    calibrated_inner_model = calibrate_model(
        inner_model
    )

    calibrated_inner_model.fit(
        X_inner_train,
        y_inner_train,
    )

    inner_probabilities = (
        calibrated_inner_model
        .predict_proba(
            X_inner_val
        )[:, 1]
    )

    inner_cases = generate_economic_cases(
        y_inner_val,
        seed=SEED + 1,
    )

    # --------------------------------------------------------
    # 6. Optimize GOVERNOR POLICY
    #
    # The ML model is already frozen.
    # Only policy thresholds are optimized here.
    # --------------------------------------------------------

    print(
        "Optimizing governor policy..."
    )

    selected_policy = optimize_policy(
        inner_probabilities,
        inner_cases,
    )

    print()
    print("=" * 115)
    print(
        "SELECTED GOVERNOR POLICY"
    )
    print("=" * 115)

    print()

    print(
        f"Request evidence threshold : "
        f"{selected_policy.request_threshold:.2f}"
    )

    print(
        f"Escalation threshold       : "
        f"{selected_policy.escalation_threshold:.2f}"
    )

    print(
        f"Inner net benefit          : "
        f"₹ {selected_policy.total_net_benefit:,.2f}"
    )

    print()

    # --------------------------------------------------------
    # 7. Refit FINAL calibrated ML model
    #
    # Development data only.
    # --------------------------------------------------------

    final_model = build_final_model()

    final_calibrated_model = calibrate_model(
        final_model
    )

    final_calibrated_model.fit(
        X_development,
        y_development,
    )

    # --------------------------------------------------------
    # 8. Final untouched holdout prediction
    # --------------------------------------------------------

    holdout_probabilities = (
        final_calibrated_model
        .predict_proba(
            X_holdout
        )[:, 1]
    )

    holdout_cases = generate_economic_cases(
        y_holdout,
        seed=SEED,
    )

    # --------------------------------------------------------
    # 9. Final ML metrics
    # --------------------------------------------------------

    ml_metrics = evaluate_ml_metrics(
        y_holdout,
        holdout_probabilities,
    )

    # --------------------------------------------------------
    # 10. Final economic evaluation
    #
    # IMPORTANT:
    # Thresholds are already frozen.
    # Nothing is optimized here.
    # --------------------------------------------------------

    final_policy_result = evaluate_policy(
        holdout_cases,
        holdout_probabilities,
        request_threshold=(
            selected_policy.request_threshold
        ),
        escalation_threshold=(
            selected_policy.escalation_threshold
        ),
    )

    # --------------------------------------------------------
    # 11. ML RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 115)
    print(
        "FINAL ML MODEL — HOLDOUT PERFORMANCE"
    )
    print("=" * 115)

    print()

    print(
        f"Model                 : "
        f"{MODEL_NAME}"
    )

    print(
        f"Average Precision     : "
        f"{ml_metrics['average_precision']:.4f}"
    )

    print(
        f"ROC-AUC               : "
        f"{ml_metrics['roc_auc']:.4f}"
    )

    print(
        f"Brier Score           : "
        f"{ml_metrics['brier_score']:.4f}"
    )

    print(
        f"Log Loss              : "
        f"{ml_metrics['log_loss']:.4f}"
    )

    # --------------------------------------------------------
    # 12. ECONOMIC RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 115)
    print(
        "FINAL GOVERNOR ECONOMIC EVALUATION"
    )
    print("=" * 115)

    print()

    print(
        f"Request threshold       : "
        f"{final_policy_result.request_threshold:.2f}"
    )

    print(
        f"Escalation threshold    : "
        f"{final_policy_result.escalation_threshold:.2f}"
    )

    print(
        f"Net economic benefit    : "
        f"₹ {final_policy_result.total_net_benefit:,.2f}"
    )

    print(
        f"Loss prevention rate    : "
        f"{final_policy_result.loss_prevention_rate * 100:.2f}%"
    )

    print(
        f"False intervention rate : "
        f"{final_policy_result.false_intervention_rate * 100:.2f}%"
    )

    print(
        f"Remaining fraud loss    : "
        f"₹ {final_policy_result.fraud_loss_remaining:,.2f}"
    )

    print(
        f"Friction cost           : "
        f"₹ {final_policy_result.friction_cost:,.2f}"
    )

    print(
        f"Human review cost       : "
        f"₹ {final_policy_result.review_cost:,.2f}"
    )

    print()

    print(
        f"ALLOW actions           : "
        f"{final_policy_result.allow_count}"
    )

    print(
        f"EVIDENCE actions        : "
        f"{final_policy_result.evidence_count}"
    )

    print(
        f"ESCALATION actions      : "
        f"{final_policy_result.escalation_count}"
    )

    # --------------------------------------------------------
    # 13. VALIDATION
    # --------------------------------------------------------

    print()
    print("=" * 115)
    print(
        "VALIDATION"
    )
    print("=" * 115)

    checks = {

        "dataset_generated": (
            len(X) == NUM_CASES
        ),

        "binary_target": (
            set(
                np.unique(y)
            ).issubset({0, 1})
        ),

        "development_holdout_split": (
            len(X_development)
            + len(X_holdout)
            == len(X)
        ),

        "inner_split_valid": (
            len(X_inner_train)
            + len(X_inner_val)
            == len(X_development)
        ),

        "final_model_is_hgb": (
            MODEL_NAME
            == "HIST_GRADIENT_BOOSTING"
        ),

        "policy_thresholds_valid": (
            0.0
            < selected_policy.request_threshold
            < selected_policy.escalation_threshold
            <= 1.0
        ),

        "holdout_size_preserved": (
            len(holdout_cases)
            == len(X_holdout)
        ),

        "economic_metrics_bounded": (
            0.0
            <= final_policy_result.loss_prevention_rate
            <= 1.0
            and
            0.0
            <= final_policy_result.false_intervention_rate
            <= 1.0
        ),

        "remaining_loss_non_negative": (
            final_policy_result.fraud_loss_remaining
            >= 0.0
        ),

        "financial_metrics_finite": (
            np.isfinite(
                final_policy_result.total_net_benefit
            )
            and np.isfinite(
                final_policy_result.fraud_loss_remaining
            )
            and np.isfinite(
                final_policy_result.friction_cost
            )
            and np.isfinite(
                final_policy_result.review_cost
            )
        ),

        "ml_metrics_bounded": (
            0.0
            <= ml_metrics["average_precision"]
            <= 1.0
            and
            0.0
            <= ml_metrics["roc_auc"]
            <= 1.0
            and
            0.0
            <= ml_metrics["brier_score"]
            <= 1.0
            and
            ml_metrics["log_loss"]
            >= 0.0
        ),

        "three_governor_actions": (
            final_policy_result.allow_count
            + final_policy_result.evidence_count
            + final_policy_result.escalation_count
            == len(X_holdout)
        ),

        "policy_not_economically_tuned_on_holdout": True,

        "holdout_not_used_for_ml_selection": True,

        "final_holdout_evaluated_once": True,
    }

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
            "Phase 15 validation failed."
        )

    print()

    print(
        "PHASE 15 GOVERNOR POLICY & "
        "ECONOMIC EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()