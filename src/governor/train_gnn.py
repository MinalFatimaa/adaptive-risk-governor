from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
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
    create_gnn_risk_governor,
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

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "evaluation"
)

MODEL_FILE = (
    MODEL_DIR
    / "gnn_risk_governor.pt"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

HIDDEN_DIM = 32

LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4

EPOCHS = 200

EARLY_STOPPING_PATIENCE = 30


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(
    seed: int,
) -> None:

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            "GNN training dataset not found:\n"
            f"{DATASET_FILE}"
        )

    df = pd.read_csv(
        DATASET_FILE
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
            "Dataset is missing required "
            f"columns: {sorted(missing)}"
        )

    if df.empty:

        raise ValueError(
            "GNN training dataset is empty."
        )

    return df


# ============================================================
# CUSTOMER-LEVEL LABELS
# ============================================================

def build_customer_labels(
    df: pd.DataFrame,
) -> dict[str, int]:

    labels: dict[str, int] = {}

    grouped = df.groupby(
        "customer_id"
    )

    for customer_id, group in grouped:

        unique_labels = (
            group["is_abusive"]
            .astype(int)
            .unique()
        )

        if len(unique_labels) != 1:

            raise ValueError(
                "A customer has inconsistent "
                "abuse labels: "
                f"{customer_id} -> "
                f"{unique_labels.tolist()}"
            )

        labels[
            customer_id
        ] = int(
            unique_labels[0]
        )

    return labels


# ============================================================
# CUSTOMER → NODE INDEX
# ============================================================

def build_customer_node_mapping(
    df: pd.DataFrame,
) -> dict[str, int]:

    mapping: dict[str, int] = {}

    grouped = (
        df[
            [
                "customer_id",
                "customer_node_index",
            ]
        ]
        .drop_duplicates()
    )

    for _, row in grouped.iterrows():

        customer_id = row[
            "customer_id"
        ]

        node_index = int(
            row[
                "customer_node_index"
            ]
        )

        if customer_id in mapping:

            if mapping[
                customer_id
            ] != node_index:

                raise ValueError(
                    "Customer maps to multiple "
                    "graph nodes: "
                    f"{customer_id}"
                )

        mapping[
            customer_id
        ] = node_index

    return mapping


# ============================================================
# CUSTOMER SPLIT
# ============================================================

def split_customers(
    labels: dict[str, int],
    seed: int,
) -> tuple[
    list[str],
    list[str],
    list[str],
]:

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    customers = list(
        labels.keys()
    )

    permutation = torch.randperm(
        len(customers),
        generator=generator,
    ).tolist()

    shuffled = [
        customers[index]
        for index in permutation
    ]

    n = len(
        shuffled
    )

    n_train = int(
        n * TRAIN_RATIO
    )

    n_validation = int(
        n * VALIDATION_RATIO
    )

    train_customers = shuffled[
        :n_train
    ]

    validation_customers = shuffled[
        n_train:
        n_train + n_validation
    ]

    test_customers = shuffled[
        n_train + n_validation:
    ]

    if not train_customers:

        raise ValueError(
            "Training split is empty."
        )

    if not validation_customers:

        raise ValueError(
            "Validation split is empty."
        )

    if not test_customers:

        raise ValueError(
            "Test split is empty."
        )

    return (
        train_customers,
        validation_customers,
        test_customers,
    )


# ============================================================
# CLASS WEIGHT
# ============================================================

def calculate_pos_weight(
    labels: dict[str, int],
    customer_ids: list[str],
) -> torch.Tensor:

    positives = sum(
        labels[cid]
        for cid in customer_ids
    )

    negatives = (
        len(customer_ids)
        - positives
    )

    if positives == 0:

        raise ValueError(
            "Training split contains no "
            "positive samples."
        )

    if negatives == 0:

        raise ValueError(
            "Training split contains no "
            "negative samples."
        )

    weight = (
        negatives
        / positives
    )

    return torch.tensor(
        [float(weight)],
        dtype=torch.float32,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:

    y_true = (
        labels
        .detach()
        .cpu()
        .numpy()
    )

    y_prob = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    metrics: dict[str, float] = {}

    # --------------------------------------------------------
    # Ranking metrics
    # --------------------------------------------------------

    if len(
        set(
            y_true.tolist()
        )
    ) > 1:

        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        )

        metrics["average_precision"] = float(
            average_precision_score(
                y_true,
                y_prob,
            )
        )

    else:

        metrics["roc_auc"] = float(
            "nan"
        )

        metrics["average_precision"] = float(
            "nan"
        )

    # --------------------------------------------------------
    # Threshold metrics
    # --------------------------------------------------------

    metrics["precision"] = float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    metrics["recall"] = float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    metrics["f1"] = float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    return metrics


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_nodes(
    model,
    data,
    node_indices: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[
    float,
    dict[str, float],
]:

    model.eval()

    logits = model(
        data
    )

    node_logits = logits[
        node_indices
    ]

    if node_logits.ndim == 2:

        if node_logits.shape[1] != 1:

            raise ValueError(
                "Unexpected model output shape:\n"
                f"{tuple(node_logits.shape)}"
            )

        node_logits = node_logits[
            :,
            0,
        ]

    elif node_logits.ndim != 1:

        raise ValueError(
            "Unexpected model output dimensions:\n"
            f"{tuple(node_logits.shape)}"
        )

    probabilities = torch.sigmoid(
        node_logits
    )

    metrics = calculate_metrics(
        probabilities=probabilities,
        labels=labels,
    )

    return (
        float(
            metrics["average_precision"]
        ),
        metrics,
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main() -> None:

    print()
    print("=" * 80)
    print("GNN RISK GOVERNOR TRAINING")
    print("=" * 80)

    set_seed(
        SEED
    )

    # ========================================================
    # DATASET
    # ========================================================

    df = load_dataset()

    customer_labels = (
        build_customer_labels(
            df
        )
    )

    customer_nodes = (
        build_customer_node_mapping(
            df
        )
    )

    print()
    print("DATASET")
    print("-" * 80)

    print(
        f"Episode samples       : "
        f"{len(df)}"
    )

    print(
        f"Unique customers      : "
        f"{len(customer_labels)}"
    )

    # ========================================================
    # CUSTOMER/NODE MAPPING
    # ========================================================

    if set(
        customer_labels
    ) != set(
        customer_nodes
    ):

        raise ValueError(
            "Customer label mapping and "
            "node mapping contain different "
            "customer IDs."
        )

    # ========================================================
    # WORLD
    # ========================================================

    world = create_world(
        config=DEFAULT_POPULATION_CONFIG,
        seed=SEED,
    )

    graph = build_interaction_graph(
        world
    )

    data = graph.data

    print()
    print("GRAPH")
    print("-" * 80)

    print(
        f"Graph nodes           : "
        f"{data.num_nodes}"
    )

    print(
        f"Graph edges           : "
        f"{data.num_edges}"
    )

    print(
        f"Node features         : "
        f"{tuple(data.x.shape)}"
    )

    # ========================================================
    # GRAPH VALIDATION
    # ========================================================

    if data.x is None:

        raise ValueError(
            "Graph contains no node features."
        )

    if data.edge_index is None:

        raise ValueError(
            "Graph contains no edge_index."
        )

    if data.x.ndim != 2:

        raise ValueError(
            "Graph node features must be 2-dimensional."
        )

    input_dim = int(
        data.x.shape[1]
    )

    if input_dim <= 0:

        raise ValueError(
            "Graph input dimension must be positive."
        )

    # ========================================================
    # CUSTOMER → NODE VALIDATION
    # ========================================================

    for (
        customer_id,
        node_index,
    ) in customer_nodes.items():

        if node_index < 0:

            raise ValueError(
                "Negative customer node index."
            )

        if node_index >= data.num_nodes:

            raise ValueError(
                "Customer node index is outside "
                "the graph: "
                f"{customer_id} -> {node_index}"
            )

        if graph.node_ids[
            node_index
        ] != customer_id:

            raise ValueError(
                "Customer node mapping mismatch: "
                f"{customer_id} -> {node_index} "
                f"-> {graph.node_ids[node_index]}"
            )

        if graph.node_types[
            node_index
        ] != "CUSTOMER":

            raise ValueError(
                "Mapped node is not a CUSTOMER: "
                f"{customer_id}"
            )

    print()
    print(
        "Customer → graph-node mapping: PASSED"
    )

    # ========================================================
    # CUSTOMER SPLIT
    # ========================================================

    (
        train_customers,
        validation_customers,
        test_customers,
    ) = split_customers(
        labels=customer_labels,
        seed=SEED,
    )

    print()
    print("CUSTOMER SPLIT")
    print("-" * 80)

    print(
        f"Train customers       : "
        f"{len(train_customers)}"
    )

    print(
        f"Validation customers  : "
        f"{len(validation_customers)}"
    )

    print(
        f"Test customers        : "
        f"{len(test_customers)}"
    )

    # ========================================================
    # NODE INDICES
    # ========================================================

    train_nodes = torch.tensor(
        [
            customer_nodes[cid]
            for cid in train_customers
        ],
        dtype=torch.long,
    )

    validation_nodes = torch.tensor(
        [
            customer_nodes[cid]
            for cid in validation_customers
        ],
        dtype=torch.long,
    )

    test_nodes = torch.tensor(
        [
            customer_nodes[cid]
            for cid in test_customers
        ],
        dtype=torch.long,
    )

    # ========================================================
    # LABELS
    # ========================================================

    train_labels = torch.tensor(
        [
            customer_labels[cid]
            for cid in train_customers
        ],
        dtype=torch.float32,
    )

    validation_labels = torch.tensor(
        [
            customer_labels[cid]
            for cid in validation_customers
        ],
        dtype=torch.float32,
    )

    test_labels = torch.tensor(
        [
            customer_labels[cid]
            for cid in test_customers
        ],
        dtype=torch.float32,
    )

    print()
    print("LABEL DISTRIBUTION")
    print("-" * 80)

    print(
        f"Train abusive        : "
        f"{int(train_labels.sum())}"
    )

    print(
        f"Train legitimate     : "
        f"{int((1 - train_labels).sum())}"
    )

    print(
        f"Validation abusive   : "
        f"{int(validation_labels.sum())}"
    )

    print(
        f"Validation legitimate: "
        f"{int((1 - validation_labels).sum())}"
    )

    print(
        f"Test abusive         : "
        f"{int(test_labels.sum())}"
    )

    print(
        f"Test legitimate      : "
        f"{int((1 - test_labels).sum())}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = create_gnn_risk_governor(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
    )

    print()
    print("MODEL")
    print("-" * 80)

    print(
        "Architecture         : GraphSAGE"
    )

    print(
        f"Input dimension      : "
        f"{input_dim}"
    )

    print(
        f"Hidden dimension     : "
        f"{HIDDEN_DIM}"
    )

    print()
    print(model)

    # ========================================================
    # CLASS IMBALANCE
    # ========================================================

    pos_weight = calculate_pos_weight(
        labels=customer_labels,
        customer_ids=train_customers,
    )

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print()
    print("TRAINING CONFIGURATION")
    print("-" * 80)

    print(
        f"Positive class weight : "
        f"{pos_weight.item():.4f}"
    )

    print(
        f"Learning rate         : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay          : "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Maximum epochs        : "
        f"{EPOCHS}"
    )

    print(
        f"Early stopping        : "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    # ========================================================
    # TRAINING
    # ========================================================

    best_validation_ap = float(
        "-inf"
    )

    best_state = None

    patience_counter = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        # ----------------------------------------------------
        # Training forward pass
        # ----------------------------------------------------

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            data
        )

        train_logits = logits[
            train_nodes
        ]

        if train_logits.ndim == 2:

            train_logits = train_logits[
                :,
                0,
            ]

        loss = criterion(
            train_logits,
            train_labels,
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Re-run the model after optimizer.step().
        #
        # The previous implementation used the old `logits`
        # tensor for validation. That tensor was generated
        # before the weights were updated.
        # ----------------------------------------------------

        (
            validation_ap,
            validation_metrics,
        ) = evaluate_nodes(
            model=model,
            data=data,
            node_indices=validation_nodes,
            labels=validation_labels,
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if (
            validation_ap
            > best_validation_ap
        ):

            best_validation_ap = (
                validation_ap
            )

            best_state = {
                key: value.detach().clone()
                for (
                    key,
                    value,
                )
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss.item():.4f} | "
                f"Val AP {validation_ap:.4f} | "
                f"Val ROC-AUC "
                f"{validation_metrics['roc_auc']:.4f}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    if best_state is None:

        raise RuntimeError(
            "Training did not produce "
            "a valid model state."
        )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    model.eval()

    with torch.no_grad():

        logits = model(
            data
        )

        test_logits = logits[
            test_nodes
        ]

        if test_logits.ndim == 2:

            if test_logits.shape[1] != 1:

                raise ValueError(
                    "Unexpected test output shape:\n"
                    f"{tuple(test_logits.shape)}"
                )

            test_logits = test_logits[
                :,
                0,
            ]

        test_probabilities = (
            torch.sigmoid(
                test_logits
            )
        )

    test_metrics = calculate_metrics(
        probabilities=test_probabilities,
        labels=test_labels,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)

    print(
        f"ROC-AUC              : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"Average Precision    : "
        f"{test_metrics['average_precision']:.4f}"
    )

    print(
        f"Precision             : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall                : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1                   : "
        f"{test_metrics['f1']:.4f}"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        "model_state_dict":
            model.state_dict(),

        # ----------------------------------------------------
        # Architecture metadata
        # ----------------------------------------------------

        "model_class":
            "GNNRiskModel",

        "architecture":
            "GraphSAGE",

        "input_dim":
            input_dim,

        "hidden_dim":
            HIDDEN_DIM,

        "risk_head_hidden_dim":
            16,

        # ----------------------------------------------------
        # Training metadata
        # ----------------------------------------------------

        "seed":
            SEED,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "epochs":
            EPOCHS,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "best_validation_ap":
            best_validation_ap,

        # ----------------------------------------------------
        # Graph metadata
        # ----------------------------------------------------

        "graph_num_nodes":
            int(data.num_nodes),

        "graph_num_edges":
            int(data.num_edges),

        "graph_feature_dim":
            input_dim,

        # ----------------------------------------------------
        # Test metrics
        # ----------------------------------------------------

        "test_metrics":
            test_metrics,
    }

    torch.save(
        checkpoint,
        MODEL_FILE,
    )

    # ========================================================
    # VERIFY SAVED CHECKPOINT
    # ========================================================

    print()
    print("VERIFYING SAVED CHECKPOINT")
    print("-" * 80)

    saved_checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        saved_checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Saved checkpoint is invalid."
        )

    saved_model = create_gnn_risk_governor(
        input_dim=int(
            saved_checkpoint[
                "input_dim"
            ]
        ),
        hidden_dim=int(
            saved_checkpoint[
                "hidden_dim"
            ]
        ),
    )

    saved_model.load_state_dict(
        saved_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    saved_model.eval()

    print(
        "Checkpoint architecture : PASSED"
    )

    print(
        "Checkpoint state_dict   : PASSED"
    )

    print(
        "Checkpoint reload       : PASSED"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("MODEL SAVED")
    print("-" * 80)

    print(
        f"Output file : {MODEL_FILE}"
    )

    print()
    print(
        "GNN training: PASSED"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()