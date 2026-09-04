from __future__ import annotations

import csv
from pathlib import Path

from ..environment.population_config import (
    DEFAULT_POPULATION_CONFIG,
)
from ..environment.world_generator import (
    create_world,
)
from .interaction_graph import (
    build_interaction_graph,
    customer_node_index,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_TRAJECTORIES_FILE = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "policy"
    / "policy_trajectories.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "policy"
    / "gnn_training_dataset.csv"
)


# ============================================================
# OUTPUT COLUMNS
# ============================================================

OUTPUT_COLUMNS = [
    "episode_id",
    "customer_id",
    "sequence_number",

    # Graph information
    "customer_node_index",

    # Observable customer features
    "order_count",
    "refund_count",
    "refund_amount_total",
    "chargeback_count",
    "account_age_days",

    # Current interaction features
    "requested_amount",
    "evidence_available",
    "evidence_count",

    # Historical policy behavior
    "previous_refund_count",
    "previous_refund_amount",
    "previous_support_decisions",
    "previous_support_approvals",
    "previous_support_evidence_requests",
    "previous_support_escalations",
    "previous_support_denials",
    "time_since_previous_request_hours",

    # Current support decision
    "support_decision",

    # Training target
    "is_abusive",
]


# ============================================================
# HELPERS
# ============================================================

def parse_bool(value: str) -> int:

    value = str(value).strip().lower()

    if value in {
        "true",
        "1",
        "yes",
    }:
        return 1

    return 0


def parse_int(
    value: str,
) -> int:

    try:
        return int(float(value))
    except (
        TypeError,
        ValueError,
    ):
        return 0


def parse_float(
    value: str,
) -> float:

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return 0.0


# ============================================================
# LOAD POLICY TRAJECTORIES
# ============================================================

def load_policy_trajectories() -> list[dict]:

    if not POLICY_TRAJECTORIES_FILE.exists():

        raise FileNotFoundError(
            "Policy trajectory dataset was not found:\n"
            f"{POLICY_TRAJECTORIES_FILE}"
        )

    with POLICY_TRAJECTORIES_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    if not rows:

        raise ValueError(
            "policy_trajectories.csv is empty."
        )

    return rows


# ============================================================
# BUILD TRAINING DATASET
# ============================================================

def build_dataset() -> list[dict]:

    # --------------------------------------------------------
    # Load trajectory data
    # --------------------------------------------------------

    trajectories = (
        load_policy_trajectories()
    )

    # --------------------------------------------------------
    # Recreate deterministic world
    #
    # This gives us:
    #
    #   observable customer state
    #   ground truth
    #   graph structure
    #
    # Ground truth is used ONLY to create the training label.
    # It is NOT included as an input feature.
    # --------------------------------------------------------

    world = create_world(
        config=DEFAULT_POPULATION_CONFIG,
        seed=42,
    )

    graph = build_interaction_graph(
        world
    )

    dataset = []

    # --------------------------------------------------------
    # Process every policy interaction
    # --------------------------------------------------------

    for trajectory in trajectories:

        customer_id = trajectory[
            "customer_id"
        ]

        if customer_id not in world.customers:

            continue

        customer = world.customers[
            customer_id
        ]

        truth = world.ground_truth[
            customer_id
        ]

        node_index = customer_node_index(
            graph=graph,
            customer_id=customer_id,
        )

        row = {

            # ------------------------------------------------
            # Episode information
            # ------------------------------------------------

            "episode_id": trajectory[
                "episode_id"
            ],

            "customer_id": customer_id,

            "sequence_number": parse_int(
                trajectory[
                    "sequence_number"
                ]
            ),

            "customer_node_index": node_index,

            # ------------------------------------------------
            # Observable customer features
            # ------------------------------------------------

            "order_count": customer.order_count,

            "refund_count": customer.refund_count,

            "refund_amount_total": (
                customer.refund_amount_total
            ),

            "chargeback_count": (
                customer.chargeback_count
            ),

            "account_age_days": (
                customer.account_age_days
            ),

            # ------------------------------------------------
            # Current interaction
            # ------------------------------------------------

            "requested_amount": parse_float(
                trajectory[
                    "requested_amount"
                ]
            ),

            "evidence_available": parse_bool(
                trajectory[
                    "evidence_available"
                ]
            ),

            "evidence_count": parse_int(
                trajectory[
                    "evidence_count"
                ]
            ),

            # ------------------------------------------------
            # Historical behavior
            # ------------------------------------------------

            "previous_refund_count": parse_int(
                trajectory[
                    "previous_refund_count"
                ]
            ),

            "previous_refund_amount": parse_float(
                trajectory[
                    "previous_refund_amount"
                ]
            ),

            "previous_support_decisions": parse_int(
                trajectory[
                    "previous_support_decisions"
                ]
            ),

            "previous_support_approvals": parse_int(
                trajectory[
                    "previous_support_approvals"
                ]
            ),

            "previous_support_evidence_requests": (
                parse_int(
                    trajectory[
                        "previous_support_evidence_requests"
                    ]
                )
            ),

            "previous_support_escalations": parse_int(
                trajectory[
                    "previous_support_escalations"
                ]
            ),

            "previous_support_denials": parse_int(
                trajectory[
                    "previous_support_denials"
                ]
            ),

            "time_since_previous_request_hours": (
                parse_float(
                    trajectory[
                        "time_since_previous_request_hours"
                    ]
                )
            ),

            # ------------------------------------------------
            # Support decision
            # ------------------------------------------------

            "support_decision": trajectory[
                "support_decision"
            ],

            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------
            #
            # IMPORTANT:
            #
            # This is the ONLY place where ground truth
            # enters the training dataset.
            #
            # The GNN does not receive this value as an
            # input feature.
            #

            "is_abusive": int(
                truth.is_abusive
            ),
        }

        dataset.append(row)

    return dataset


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    rows: list[dict],
) -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_COLUMNS,
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(
    rows: list[dict],
) -> None:

    if not rows:

        raise ValueError(
            "Generated GNN dataset contains "
            "zero rows."
        )

    # --------------------------------------------------------
    # Verify label distribution
    # --------------------------------------------------------

    abusive = sum(
        row["is_abusive"]
        for row in rows
    )

    legitimate = (
        len(rows)
        - abusive
    )

    print()
    print("DATASET VALIDATION")
    print("-" * 80)

    print(
        f"Total samples       : {len(rows)}"
    )

    print(
        f"Abusive samples     : {abusive}"
    )

    print(
        f"Legitimate samples  : {legitimate}"
    )

    # --------------------------------------------------------
    # Verify forbidden fields are NOT present as features
    # --------------------------------------------------------

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

    actual_columns = set(
        OUTPUT_COLUMNS
    )

    violations = (
        forbidden
        & actual_columns
    )

    if violations:

        raise AssertionError(
            "Information boundary violation. "
            f"Forbidden fields found: {violations}"
        )

    print()
    print(
        "Information boundary : PASSED"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 80)
    print("GNN TRAINING DATASET BUILDER")
    print("=" * 80)

    rows = build_dataset()

    save_dataset(
        rows
    )

    validate_dataset(
        rows
    )

    print()
    print("DATASET CREATED")
    print("-" * 80)

    print(
        f"Output file : {OUTPUT_FILE}"
    )

    print(
        f"Samples     : {len(rows)}"
    )

    print()
    print(
        "GNN dataset construction: PASSED"
    )


if __name__ == "__main__":
    main()