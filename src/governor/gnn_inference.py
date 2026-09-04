from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "evaluation"
    / "gnn_risk_governor.pt"
)


# ============================================================
# GNN INFERENCE
# ============================================================

class GNNRiskInference:
    """
    Loads the trained GNN Risk Governor and performs inference.

    Responsibilities:

        - load checkpoint
        - reconstruct exact training architecture
        - validate graph dimensions
        - produce node risk probabilities

    Does NOT:

        - make Governor decisions
        - apply policy thresholds
        - access ground truth
        - access adaptive-agent attributes
        - modify Governor state
    """

    def __init__(
        self,
        model_file: Path = MODEL_FILE,
    ) -> None:

        self.model_file = Path(
            model_file
        )

        # ====================================================
        # 1. Validate model path
        # ====================================================

        if not self.model_file.exists():

            raise FileNotFoundError(
                "Trained GNN model was not found:\n"
                f"{self.model_file}"
            )

        if not self.model_file.is_file():

            raise FileNotFoundError(
                "GNN model path does not point "
                "to a file:\n"
                f"{self.model_file}"
            )

        # ====================================================
        # 2. Load checkpoint
        # ====================================================

        checkpoint = torch.load(
            self.model_file,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(
            checkpoint,
            dict,
        ):

            raise ValueError(
                "Invalid GNN checkpoint format. "
                "Expected a dictionary."
            )

        required_keys = {
            "model_state_dict",
            "input_dim",
            "hidden_dim",
        }

        missing = (
            required_keys
            - set(checkpoint.keys())
        )

        if missing:

            raise ValueError(
                "GNN checkpoint is missing "
                "required fields: "
                f"{sorted(missing)}"
            )

        # ====================================================
        # 3. Read architecture metadata
        # ====================================================

        try:

            self.input_dim = int(
                checkpoint["input_dim"]
            )

            self.hidden_dim = int(
                checkpoint["hidden_dim"]
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "Invalid GNN architecture metadata."
            ) from exc

        if self.input_dim <= 0:

            raise ValueError(
                "Checkpoint input_dim must be positive."
            )

        if self.hidden_dim <= 0:

            raise ValueError(
                "Checkpoint hidden_dim must be positive."
            )

        # ====================================================
        # 4. Lazy import of canonical model
        # ====================================================

        from .gnn_risk_governor import (
            create_gnn_risk_governor,
        )

        # ====================================================
        # 5. Reconstruct EXACT architecture
        # ====================================================

        self.model = create_gnn_risk_governor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
        )

        # ====================================================
        # 6. Validate state dict
        # ====================================================

        state_dict = checkpoint[
            "model_state_dict"
        ]

        if not isinstance(
            state_dict,
            dict,
        ):

            raise ValueError(
                "Invalid model_state_dict "
                "in checkpoint."
            )

        # ----------------------------------------------------
        # We deliberately do not manually compare PyG's
        # internal state-dict names.
        #
        # The model itself is the source of truth.
        # load_state_dict(strict=True) performs the exact
        # compatibility check.
        # ----------------------------------------------------

        try:

            self.model.load_state_dict(
                state_dict,
                strict=True,
            )

        except RuntimeError as exc:

            raise RuntimeError(
                "The saved GNN checkpoint cannot be loaded "
                "into the canonical GNNRiskModel.\n\n"
                "This usually means the checkpoint was "
                "created by an older/different architecture.\n\n"
                f"Checkpoint: {self.model_file}\n"
                f"Input dimension: {self.input_dim}\n"
                f"Hidden dimension: {self.hidden_dim}\n\n"
                "Delete the old checkpoint and run "
                "train_gnn.py again."
            ) from exc

        # ====================================================
        # 7. Evaluation mode
        # ====================================================

        self.model.eval()

        self.model.to(
            torch.device("cpu")
        )

        # ====================================================
        # 8. Metadata
        # ====================================================

        self.checkpoint = checkpoint

    # ========================================================
    # GRAPH VALIDATION
    # ========================================================

    def _validate_graph(
        self,
        data: Any,
    ) -> None:

        if data is None:

            raise ValueError(
                "Graph data cannot be None."
            )

        if not hasattr(data, "x"):

            raise ValueError(
                "Graph data does not contain "
                "node features."
            )

        if not hasattr(data, "edge_index"):

            raise ValueError(
                "Graph data does not contain "
                "edge_index."
            )

        if data.x is None:

            raise ValueError(
                "Graph node features cannot be None."
            )

        if data.edge_index is None:

            raise ValueError(
                "Graph edge_index cannot be None."
            )

        if data.x.ndim != 2:

            raise ValueError(
                "Graph node features must be "
                "2-dimensional."
            )

        if data.x.shape[0] == 0:

            raise ValueError(
                "Graph contains no nodes."
            )

        if data.x.shape[1] != self.input_dim:

            raise ValueError(
                "Graph feature dimension does not "
                "match the trained GNN.\n"
                f"Expected: {self.input_dim}\n"
                f"Received: {data.x.shape[1]}"
            )

        if data.x.dtype not in (
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ):

            raise ValueError(
                "Graph node features must "
                "contain floating-point values."
            )

        if data.edge_index.ndim != 2:

            raise ValueError(
                "edge_index must be 2-dimensional."
            )

        if data.edge_index.shape[0] != 2:

            raise ValueError(
                "edge_index must have shape "
                "[2, num_edges]."
            )

    # ========================================================
    # SINGLE CUSTOMER INFERENCE
    # ========================================================

    @torch.no_grad()
    def predict_customer_risk(
        self,
        data: Any,
        customer_node_index: int,
    ) -> float:

        self._validate_graph(
            data
        )

        if not isinstance(
            customer_node_index,
            int,
        ):

            raise TypeError(
                "Customer node index must be "
                "an integer."
            )

        if customer_node_index < 0:

            raise ValueError(
                "Customer node index cannot "
                "be negative."
            )

        if customer_node_index >= data.num_nodes:

            raise ValueError(
                "Customer node index is outside "
                "the graph.\n"
                f"Index: {customer_node_index}\n"
                f"Nodes: {data.num_nodes}"
            )

        self.model.eval()

        logits = self.model(
            data
        )

        # ----------------------------------------------------
        # Normalize output shape
        # ----------------------------------------------------

        if logits.ndim == 2:

            if logits.shape[1] != 1:

                raise ValueError(
                    "Unexpected GNN output shape:\n"
                    f"{tuple(logits.shape)}"
                )

            logits = logits[:, 0]

        elif logits.ndim != 1:

            raise ValueError(
                "Unexpected GNN output dimensions:\n"
                f"{tuple(logits.shape)}"
            )

        customer_logit = logits[
            customer_node_index
        ]

        probability = torch.sigmoid(
            customer_logit
        )

        return float(
            probability.item()
        )

    # ========================================================
    # BATCH INFERENCE
    # ========================================================

    @torch.no_grad()
    def predict_nodes(
        self,
        data: Any,
        node_indices: list[int],
    ) -> list[float]:

        if not node_indices:
            return []

        self._validate_graph(
            data
        )

        for node_index in node_indices:

            if not isinstance(
                node_index,
                int,
            ):

                raise TypeError(
                    "Every node index must "
                    "be an integer."
                )

            if node_index < 0:

                raise ValueError(
                    "Node index cannot be negative."
                )

            if node_index >= data.num_nodes:

                raise ValueError(
                    "Node index is outside "
                    "the graph."
                )

        self.model.eval()

        logits = self.model(
            data
        )

        if logits.ndim == 2:

            if logits.shape[1] != 1:

                raise ValueError(
                    "Unexpected GNN output shape:\n"
                    f"{tuple(logits.shape)}"
                )

            logits = logits[:, 0]

        elif logits.ndim != 1:

            raise ValueError(
                "Unexpected GNN output dimensions:\n"
                f"{tuple(logits.shape)}"
            )

        probabilities = torch.sigmoid(
            logits
        )

        return [
            float(
                probabilities[
                    node_index
                ].item()
            )
            for node_index in node_indices
        ]

    # ========================================================
    # FULL GRAPH INFERENCE
    # ========================================================

    @torch.no_grad()
    def predict_all(
        self,
        data: Any,
    ) -> list[float]:

        self._validate_graph(
            data
        )

        self.model.eval()

        logits = self.model(
            data
        )

        if logits.ndim == 2:

            if logits.shape[1] != 1:

                raise ValueError(
                    "Unexpected GNN output shape:\n"
                    f"{tuple(logits.shape)}"
                )

            logits = logits[:, 0]

        elif logits.ndim != 1:

            raise ValueError(
                "Unexpected GNN output dimensions:\n"
                f"{tuple(logits.shape)}"
            )

        probabilities = torch.sigmoid(
            logits
        )

        return [
            float(value)
            for value in probabilities.cpu().tolist()
        ]


# ============================================================
# MODULE SELF-CHECK
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("GNN INFERENCE MODULE")
    print("=" * 70)

    print(
        f"Model file     : {MODEL_FILE}"
    )

    try:

        inference = GNNRiskInference()

        print()
        print(
            "Checkpoint loaded successfully."
        )

        print(
            f"Input dimension : "
            f"{inference.input_dim}"
        )

        print(
            f"Hidden dimension: "
            f"{inference.hidden_dim}"
        )

        print()
        print("Model architecture:")
        print(
            inference.model
        )

        print()
        print(
            "State dictionary: PASSED"
        )

        print(
            "GNN inference initialization: PASSED"
        )

    except Exception as exc:

        print()
        print(
            "GNN inference self-check FAILED."
        )

        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise

    print("=" * 70)