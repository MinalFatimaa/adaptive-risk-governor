from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


# ============================================================
# GNN RISK MODEL
# ============================================================

class GNNRiskModel(nn.Module):
    """
    GraphSAGE model used by the Adaptive Risk Governor.

    Architecture:

        Node Features
             |
             v
        GraphSAGE
        input_dim -> hidden_dim
             |
           ReLU
             |
             v
        GraphSAGE
        hidden_dim -> hidden_dim
             |
           ReLU
             |
             v
        Risk Head
        Linear -> ReLU -> Linear
             |
             v
        Risk Logit

    The model returns ONE LOGIT per graph node.

    IMPORTANT
    ---------
    The architecture must remain identical between:

        training
        checkpoint loading
        inference
        Governor integration
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ) -> None:

        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        # ----------------------------------------------------
        # GraphSAGE Layer 1
        # ----------------------------------------------------

        self.conv1 = SAGEConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            aggr="mean",
        )

        # ----------------------------------------------------
        # GraphSAGE Layer 2
        # ----------------------------------------------------

        self.conv2 = SAGEConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            aggr="mean",
        )

        # ----------------------------------------------------
        # Risk Prediction Head
        # ----------------------------------------------------

        self.risk_head = nn.Sequential(

            nn.Linear(
                hidden_dim,
                16,
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                1,
            ),
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        data: Any,
    ) -> torch.Tensor:

        if data is None:

            raise ValueError(
                "Graph data cannot be None."
            )

        if not hasattr(
            data,
            "x",
        ):

            raise ValueError(
                "Graph data must contain "
                "node features 'x'."
            )

        if not hasattr(
            data,
            "edge_index",
        ):

            raise ValueError(
                "Graph data must contain "
                "'edge_index'."
            )

        x = data.x

        edge_index = data.edge_index

        if x is None:

            raise ValueError(
                "Graph node features cannot be None."
            )

        if edge_index is None:

            raise ValueError(
                "Graph edge_index cannot be None."
            )

        # ----------------------------------------------------
        # First GraphSAGE layer
        # ----------------------------------------------------

        x = self.conv1(
            x,
            edge_index,
        )

        x = torch.relu(
            x
        )

        # ----------------------------------------------------
        # Second GraphSAGE layer
        # ----------------------------------------------------

        x = self.conv2(
            x,
            edge_index,
        )

        x = torch.relu(
            x
        )

        # ----------------------------------------------------
        # Risk head
        # ----------------------------------------------------

        logits = self.risk_head(
            x
        )

        # ----------------------------------------------------
        # Return [num_nodes]
        # ----------------------------------------------------

        return logits.squeeze(
            -1
        )


# ============================================================
# MODEL FACTORY
# ============================================================

def create_gnn_risk_model(
    input_dim: int,
    hidden_dim: int,
) -> GNNRiskModel:
    """
    Single canonical factory for the GNN model.

    Every component of the system must use this factory.
    """

    return GNNRiskModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    )


# ============================================================
# MODULE SELF-CHECK
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("GNN MODEL SELF-CHECK")
    print("=" * 70)

    INPUT_DIM = 12
    HIDDEN_DIM = 32

    model = create_gnn_risk_model(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
    )

    print()
    print("MODEL")
    print("-" * 70)

    print(model)

    print()
    print(
        f"Input dimension : {INPUT_DIM}"
    )

    print(
        f"Hidden dimension: {HIDDEN_DIM}"
    )

    # --------------------------------------------------------
    # Verify state dictionary
    # --------------------------------------------------------

    print()
    print("STATE DICTIONARY")
    print("-" * 70)

    for key in model.state_dict().keys():

        print(
            f"  {key}"
        )

    print()
    print("=" * 70)
    print("GNN MODEL SELF-CHECK PASSED")
    print("=" * 70)