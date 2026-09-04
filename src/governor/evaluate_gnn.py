from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..environment.population_config import (
    DEFAULT_POPULATION_CONFIG,
)
from ..environment.world_generator import (
    create_world,
)
from .interaction_graph import (
    build_interaction_graph,
)
from .gnn_risk_governor import (
    GNNRiskGovernor,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "policy"
    / "gnn_training_dataset.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "evaluation"
    / "gnn_risk_governor.pt"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TEST_FRACTION = 0.15

THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed() -> None:

    np.random.seed(
        RANDOM_SEED
    )

    torch.manual_seed(
        RANDOM_SEED
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            "GNN training dataset was not found:\n"
            f"{DATASET_FILE}"
        )

    df = pd.read_csv(
        DATASET_FILE
    )

    if df.empty:

        raise ValueError(
            "GNN training dataset is empty."
        )

    required_columns = {
        "customer_id",
        "customer_node_index",
        "is_abusive",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    return df


# ============================================================
# CUSTOMER-LEVEL TEST SPLIT
# ============================================================

def get_test_customers(
    df: pd.DataFrame,
) -> set[str]:

    customers = sorted(
        df["customer_id"]
        .unique()
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    customers = np.array(
        customers
    )

    rng.shuffle(
        customers
    )

    test_size = int(
        len(customers)
        * TEST_FRACTION
    )

    test_customers = set(
        customers[:test_size]
    )

    return test_customers


# ============================================================
# BUILD WORLD + GRAPH
# ============================================================

def build_graph():

    world = create_world(
        config=DEFAULT_POPULATION_CONFIG,
        seed=RANDOM_SEED,
    )

    graph = build_interaction_graph(
        world
    )

    return world, graph


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    input_dim: int,
) -> GNNRiskGovernor:

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            "Trained GNN model was not found:\n"
            f"{MODEL_FILE}"
        )

    model = GNNRiskGovernor(
        input_dim=input_dim,
        hidden_dim=32,
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
    )

    # --------------------------------------------------------
    # Support both:
    #
    # 1. Raw state_dict
    # 2. Dictionary containing model_state_dict
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# CUSTOMER RISK PREDICTIONS
# ============================================================

@torch.no_grad()
def predict_test_customers(
    model: GNNRiskGovernor,
    graph,
    df: pd.DataFrame,
    test_customers: set[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:

    logits = model.forward(
        graph.data
    )

    probabilities = torch.sigmoid(
        logits
    ).cpu().numpy()

    y_true = []
    y_score = []
    customer_ids = []

    for customer_id in sorted(
        test_customers
    ):

        rows = df[
            df["customer_id"]
            == customer_id
        ]

        if rows.empty:

            continue

        node_index = int(
            rows.iloc[0][
                "customer_node_index"
            ]
        )

        label = int(
            rows.iloc[0][
                "is_abusive"
            ]
        )

        score = float(
            probabilities[node_index]
        )

        y_true.append(
            label
        )

        y_score.append(
            score
        )

        customer_ids.append(
            customer_id
        )

    return (
        np.asarray(y_true),
        np.asarray(y_score),
        customer_ids,
    )


# ============================================================
# MAIN METRICS
# ============================================================

def evaluate_main_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> None:

    roc_auc = roc_auc_score(
        y_true,
        y_score,
    )

    average_precision = (
        average_precision_score(
            y_true,
            y_score,
        )
    )

    print()
    print("=" * 80)
    print("GNN HELD-OUT TEST EVALUATION")
    print("=" * 80)

    print()
    print("TEST DATA")
    print("-" * 80)

    print(
        f"Test customers       : "
        f"{len(y_true)}"
    )

    print(
        f"Abusive              : "
        f"{int(y_true.sum())}"
    )

    print(
        f"Legitimate           : "
        f"{int((y_true == 0).sum())}"
    )

    print()
    print("RANKING METRICS")
    print("-" * 80)

    print(
        f"ROC-AUC              : "
        f"{roc_auc:.4f}"
    )

    print(
        f"Average Precision    : "
        f"{average_precision:.4f}"
    )


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> None:

    print()
    print("THRESHOLD ANALYSIS")
    print("-" * 80)

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'TP':<8}"
        f"{'FP':<8}"
        f"{'FN':<8}"
        f"{'TN':<8}"
    )

    print("-" * 80)

    for threshold in THRESHOLDS:

        predictions = (
            y_score
            >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        tn, fp, fn, tp = (
            confusion_matrix(
                y_true,
                predictions,
                labels=[0, 1],
            ).ravel()
        )

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
            f"{tp:<8}"
            f"{fp:<8}"
            f"{fn:<8}"
            f"{tn:<8}"
        )


# ============================================================
# RISK DISTRIBUTION
# ============================================================

def evaluate_risk_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> None:

    abusive_scores = y_score[
        y_true == 1
    ]

    legitimate_scores = y_score[
        y_true == 0
    ]

    print()
    print("RISK SCORE DISTRIBUTION")
    print("-" * 80)

    print(
        f"Abusive mean         : "
        f"{abusive_scores.mean():.4f}"
    )

    print(
        f"Abusive median       : "
        f"{np.median(abusive_scores):.4f}"
    )

    print(
        f"Legitimate mean      : "
        f"{legitimate_scores.mean():.4f}"
    )

    print(
        f"Legitimate median    : "
        f"{np.median(legitimate_scores):.4f}"
    )


# ============================================================
# MODEL SANITY CHECK
# ============================================================

def sanity_check_scores(
    y_score: np.ndarray,
) -> None:

    if not np.all(
        np.isfinite(y_score)
    ):

        raise AssertionError(
            "GNN produced NaN or infinite "
            "risk scores."
        )

    if np.any(
        y_score < 0.0
    ) or np.any(
        y_score > 1.0
    ):

        raise AssertionError(
            "GNN risk scores must be "
            "between 0 and 1."
        )

    print()
    print(
        "Risk score validity     : PASSED"
    )


# ============================================================
# INFORMATION BOUNDARY
# ============================================================

def information_boundary_check(
    df: pd.DataFrame,
) -> None:

    forbidden = {
        "population",
        "behavior_type",
        "abuse_family",
        "counterparty_type",
        "objective",
        "current_strategy",
        "strategy_beliefs",
        "private_memory",
    }

    violations = (
        forbidden
        & set(df.columns)
    )

    if violations:

        raise AssertionError(
            "Information boundary violation: "
            f"{sorted(violations)}"
        )

    print(
        "Information boundary     : PASSED"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    set_seed()

    print()
    print("=" * 80)
    print("GNN RISK GOVERNOR EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = load_dataset()

    print()
    print("DATASET")
    print("-" * 80)

    print(
        f"Samples              : "
        f"{len(df)}"
    )

    print(
        f"Unique customers     : "
        f"{df['customer_id'].nunique()}"
    )

    information_boundary_check(
        df
    )

    # --------------------------------------------------------
    # Customer split
    # --------------------------------------------------------

    test_customers = (
        get_test_customers(
            df
        )
    )

    print()
    print("CUSTOMER TEST SPLIT")
    print("-" * 80)

    print(
        f"Test customers       : "
        f"{len(test_customers)}"
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    world, graph = build_graph()

    print()
    print("GRAPH")
    print("-" * 80)

    print(
        f"Graph nodes          : "
        f"{graph.data.num_nodes}"
    )

    print(
        f"Graph edges          : "
        f"{graph.data.num_edges}"
    )

    print(
        f"Node features        : "
        f"{tuple(graph.data.x.shape)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(
        input_dim=graph.data.x.shape[1]
    )

    print()
    print(
        "Model loading         : PASSED"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    (
        y_true,
        y_score,
        customer_ids,
    ) = predict_test_customers(
        model=model,
        graph=graph,
        df=df,
        test_customers=test_customers,
    )

    if len(y_true) == 0:

        raise RuntimeError(
            "No test predictions were generated."
        )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    sanity_check_scores(
        y_score
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    evaluate_main_metrics(
        y_true=y_true,
        y_score=y_score,
    )

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    evaluate_thresholds(
        y_true=y_true,
        y_score=y_score,
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    evaluate_risk_distribution(
        y_true=y_true,
        y_score=y_score,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("GNN EVALUATION: PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()