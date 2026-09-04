from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
)


# ============================================================
# OBSERVATION
# ============================================================

@dataclass
class PolicyObservation:
    """
    One observation available to Agent B about Agent A.

    IMPORTANT:
    This object contains only information observable to the
    customer/adaptive agent.

    Ground truth and Governor information must NEVER be stored
    here.
    """

    claim_type: str

    requested_amount: float

    evidence_available: bool

    evidence_count: int

    previous_refund_count: int = 0

    previous_refund_amount: float = 0.0

    account_age_days: int = 0

    previous_support_decisions: int = 0

    previous_support_approvals: int = 0

    previous_support_evidence_requests: int = 0

    previous_support_escalations: int = 0

    time_since_previous_request_hours: float = 999.0

    # Agent A's response is the TARGET, not an input feature.
    support_decision: Optional[str] = None


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

class PolicyFeatureExtractor:

    CLAIM_TYPES = [
        "SHORTAGE_CLAIM",
        "WRONG_ITEM_CLAIM",
        "NON_DELIVERY_CLAIM",
        "SUBSTITUTED_RETURN",
    ]

    DECISIONS = [
        "APPROVE",
        "REQUEST_EVIDENCE",
        "ESCALATE",
        "REJECT",
    ]

    def transform(
        self,
        observations: list[PolicyObservation],
    ) -> pd.DataFrame:

        rows = []

        for obs in observations:

            row = {
                "requested_amount":
                    obs.requested_amount,

                "evidence_available":
                    int(obs.evidence_available),

                "evidence_count":
                    obs.evidence_count,

                "previous_refund_count":
                    obs.previous_refund_count,

                "previous_refund_amount":
                    obs.previous_refund_amount,

                "account_age_days":
                    obs.account_age_days,

                "previous_support_decisions":
                    obs.previous_support_decisions,

                "previous_support_approvals":
                    obs.previous_support_approvals,

                "previous_support_evidence_requests":
                    obs.previous_support_evidence_requests,

                "previous_support_escalations":
                    obs.previous_support_escalations,

                "time_since_previous_request_hours":
                    obs.time_since_previous_request_hours,

                "claim_type":
                    obs.claim_type,
            }

            rows.append(row)

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        # ----------------------------------------------------
        # One-hot encode claim type.
        # ----------------------------------------------------

        df = pd.get_dummies(
            df,
            columns=["claim_type"],
            prefix="claim",
        )

        # ----------------------------------------------------
        # Ensure stable feature columns.
        # ----------------------------------------------------

        for claim_type in self.CLAIM_TYPES:

            column = f"claim_{claim_type}"

            if column not in df.columns:
                df[column] = 0

        return df


# ============================================================
# POLICY LEARNER
# ============================================================

class SupportPolicyLearner:

    def __init__(
        self,
        random_state: int = 42,
    ):

        self.random_state = random_state

        self.feature_extractor = (
            PolicyFeatureExtractor()
        )

        self.model = (
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=15,
                random_state=random_state,
            )
        )

        self.label_encoder = LabelEncoder()

        self.is_fitted = False

        self.feature_columns: list[str] = []

    # ========================================================
    # TRAIN
    # ========================================================

    def fit(
        self,
        observations: list[PolicyObservation],
    ) -> "SupportPolicyLearner":

        if not observations:
            raise ValueError(
                "Cannot train policy learner "
                "with zero observations."
            )

        missing_targets = [
            obs
            for obs in observations
            if obs.support_decision is None
        ]

        if missing_targets:
            raise ValueError(
                "Every training observation must "
                "contain support_decision."
            )

        X = self.feature_extractor.transform(
            observations
        )

        y = [
            obs.support_decision
            for obs in observations
        ]

        self.feature_columns = list(X.columns)

        y_encoded = (
            self.label_encoder.fit_transform(y)
        )

        # ----------------------------------------------------
        # Need at least two classes.
        # ----------------------------------------------------

        if len(
            self.label_encoder.classes_
        ) < 2:

            raise ValueError(
                "Policy learner requires at least "
                "two different Agent A decisions."
            )

        self.model.fit(
            X,
            y_encoded,
        )

        self.is_fitted = True

        return self

    # ========================================================
    # INTERNAL FEATURE ALIGNMENT
    # ========================================================

    def _prepare_features(
        self,
        observations: list[PolicyObservation],
    ) -> pd.DataFrame:

        X = self.feature_extractor.transform(
            observations
        )

        # Add missing columns.
        for column in self.feature_columns:

            if column not in X.columns:
                X[column] = 0

        # Remove unexpected columns.
        X = X[
            self.feature_columns
        ]

        return X

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        observation: PolicyObservation,
    ) -> str:

        if not self.is_fitted:
            raise RuntimeError(
                "Policy learner has not been fitted."
            )

        X = self._prepare_features(
            [observation]
        )

        prediction = self.model.predict(X)[0]

        return str(
            self.label_encoder.inverse_transform(
                [prediction]
            )[0]
        )

    # ========================================================
    # PREDICT PROBABILITIES
    # ========================================================

    def predict_proba(
        self,
        observation: PolicyObservation,
    ) -> dict[str, float]:

        if not self.is_fitted:
            raise RuntimeError(
                "Policy learner has not been fitted."
            )

        X = self._prepare_features(
            [observation]
        )

        probabilities = (
            self.model.predict_proba(X)[0]
        )

        classes = (
            self.label_encoder.inverse_transform(
                range(
                    len(
                        self.label_encoder.classes_
                    )
                )
            )
        )

        return {
            str(label): float(probability)
            for label, probability
            in zip(
                classes,
                probabilities,
            )
        }

    # ========================================================
    # EVALUATE
    # ========================================================

    def evaluate(
        self,
        observations: list[PolicyObservation],
    ) -> dict:

        if not self.is_fitted:
            raise RuntimeError(
                "Policy learner has not been fitted."
            )

        X = self._prepare_features(
            observations
        )

        y_true = self.label_encoder.transform(
            [
                obs.support_decision
                for obs in observations
            ]
        )

        y_pred = self.model.predict(X)

        probabilities = (
            self.model.predict_proba(X)
        )

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        result = {
            "accuracy": float(accuracy),

            "log_loss": float(
                log_loss(
                    y_true,
                    probabilities,
                    labels=list(
                        range(
                            len(
                                self.label_encoder.classes_
                            )
                        )
                    ),
                )
            ),

            "classification_report":
                classification_report(
                    y_true,
                    y_pred,
                    target_names=(
                        self.label_encoder.classes_
                    ),
                    zero_division=0,
                ),
        }

        return result