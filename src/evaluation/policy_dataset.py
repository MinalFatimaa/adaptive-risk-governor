from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from random import Random

import pandas as pd

from ..agents.policy_learner import (
    PolicyObservation,
)

from ..agents.support_agent import (
    SupportAgent,
)

from ..environment.world_generator import (
    create_world,
)

from ..schemas.support import (
    RefundRequest,
)


# ============================================================
# CONFIGURATION
# ============================================================

CLAIM_TYPES = [
    "SHORTAGE_CLAIM",
    "WRONG_ITEM_CLAIM",
    "NON_DELIVERY_CLAIM",
    "SUBSTITUTED_RETURN_CLAIM",
]


EVIDENCE_OPTIONS = {
    "SHORTAGE_CLAIM": [
        [],
        ["delivery_photo"],
    ],

    "WRONG_ITEM_CLAIM": [
        [],
        ["package_photo"],
    ],

    "NON_DELIVERY_CLAIM": [
        [],
        ["delivery_evidence"],
    ],

    "SUBSTITUTED_RETURN_CLAIM": [
        [],
        ["return_receipt"],
    ],
}


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DEFAULT_N_CUSTOMERS = 1000

DEFAULT_INTERACTIONS_PER_CUSTOMER = 10

DEFAULT_SEED = 42

BASE_TIMESTAMP = datetime(
    2026,
    1,
    1,
    9,
    0,
    0,
)


# ============================================================
# REQUEST GENERATOR
# ============================================================

def build_refund_request(
    customer,
    order,
    claim_type: str,
    requested_amount: float,
    evidence: list[str],
    timestamp: datetime,
    request_number: int,
) -> RefundRequest:

    return RefundRequest(
        request_id=(
            f"POLICY_REQUEST_{request_number:07d}"
        ),

        customer_id=customer.customer_id,

        order_id=order.order_id,

        claim_type=claim_type,

        claim_text=(
            f"Customer reports a "
            f"{claim_type.lower()}."
        ),

        requested_amount=(
            requested_amount
        ),

        submitted_at=timestamp,

        evidence_available=list(
            evidence
        ),
    )


# ============================================================
# AMOUNT GENERATOR
# ============================================================

def generate_request_amount(
    order_amount: float,
    rng: Random,
    policy,
) -> float:

    amount_bucket = rng.choice(
        [
            "LOW",
            "MID",
            "EVIDENCE_THRESHOLD",
            "HIGH",
            "RANDOM",
        ]
    )

    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    if amount_bucket == "LOW":

        upper = min(
            policy.max_auto_refund_amount,
            order_amount,
        )

        requested_amount = rng.uniform(
            100,
            max(100, upper),
        )

    # --------------------------------------------------------
    # MID
    # --------------------------------------------------------

    elif amount_bucket == "MID":

        lower = min(
            policy.max_auto_refund_amount,
            order_amount,
        )

        upper = min(
            policy.evidence_required_above,
            order_amount,
        )

        if upper <= lower:

            requested_amount = rng.uniform(
                100,
                order_amount,
            )

        else:

            requested_amount = rng.uniform(
                lower,
                upper,
            )

    # --------------------------------------------------------
    # AROUND EVIDENCE THRESHOLD
    # --------------------------------------------------------

    elif amount_bucket == "EVIDENCE_THRESHOLD":

        lower = max(
            1,
            policy.evidence_required_above
            - 500,
        )

        upper = min(
            policy.evidence_required_above
            + 500,
            order_amount,
        )

        if upper <= lower:

            requested_amount = rng.uniform(
                100,
                order_amount,
            )

        else:

            requested_amount = rng.uniform(
                lower,
                upper,
            )

    # --------------------------------------------------------
    # HIGH VALUE
    # --------------------------------------------------------

    elif amount_bucket == "HIGH":

        lower = min(
            policy.human_review_threshold,
            order_amount,
        )

        if order_amount <= lower:

            requested_amount = order_amount

        else:

            requested_amount = rng.uniform(
                lower,
                order_amount,
            )

    # --------------------------------------------------------
    # RANDOM
    # --------------------------------------------------------

    else:

        requested_amount = rng.uniform(
            100,
            order_amount,
        )

    return round(
        min(
            requested_amount,
            order_amount,
        ),
        2,
    )


# ============================================================
# OBSERVATION BUILDER
# ============================================================

def build_policy_observation(
    customer,
    claim_type: str,
    requested_amount: float,
    evidence: list[str],
    support_decision,
    previous_support_decisions: int,
    previous_support_approvals: int,
    previous_support_evidence_requests: int,
    previous_support_escalations: int,
    time_since_previous_request_hours: float,
) -> PolicyObservation:

    return PolicyObservation(

        claim_type=claim_type,

        requested_amount=(
            requested_amount
        ),

        evidence_available=(
            len(evidence) > 0
        ),

        evidence_count=(
            len(evidence)
        ),

        previous_refund_count=(
            customer.refund_count
        ),

        previous_refund_amount=(
            customer.refund_amount_total
        ),

        account_age_days=(
            customer.account_age_days
        ),

        previous_support_decisions=(
            previous_support_decisions
        ),

        previous_support_approvals=(
            previous_support_approvals
        ),

        previous_support_evidence_requests=(
            previous_support_evidence_requests
        ),

        previous_support_escalations=(
            previous_support_escalations
        ),

        time_since_previous_request_hours=(
            time_since_previous_request_hours
        ),

        support_decision=(
            support_decision.decision
        ),
    )


# ============================================================
# CUSTOMER SUPPORT HISTORY
# ============================================================

def create_support_history() -> dict:

    return {
        "decisions": 0,

        "approvals": 0,

        "evidence_requests": 0,

        "escalations": 0,

        "denials": 0,

        "last_timestamp": None,

        "previous_claim_types": [],

        "previous_decisions": [],
    }


# ============================================================
# UPDATE SUPPORT HISTORY
# ============================================================

def update_support_history(
    history: dict,
    decision,
    claim_type: str,
    timestamp: datetime,
) -> None:

    history["decisions"] += 1

    if decision.decision == "APPROVE":

        history["approvals"] += 1

    elif decision.decision == "REQUEST_EVIDENCE":

        history[
            "evidence_requests"
        ] += 1

    elif decision.decision == "ESCALATE":

        history["escalations"] += 1

    elif decision.decision == "DENY":

        history["denials"] += 1

    history[
        "previous_claim_types"
    ].append(
        claim_type
    )

    history[
        "previous_decisions"
    ].append(
        decision.decision
    )

    history[
        "last_timestamp"
    ] = timestamp


# ============================================================
# TRAJECTORY GENERATOR
# ============================================================

def generate_customer_trajectory(
    customer,
    customer_orders,
    support_agent,
    rng: Random,
    episode_id: str,
    n_interactions: int,
    global_request_counter: int,
    base_timestamp: datetime,
) -> tuple[list[PolicyObservation], list[dict], int]:

    observations = []

    metadata_rows = []

    history = create_support_history()

    # --------------------------------------------------------
    # Every customer gets their own chronological trajectory.
    # --------------------------------------------------------

    for sequence_number in range(
        1,
        n_interactions + 1,
    ):

        # ----------------------------------------------------
        # Select an order.
        # ----------------------------------------------------

        order = rng.choice(
            customer_orders
        )

        # ----------------------------------------------------
        # Select claim type.
        #
        # We intentionally keep all claim families available
        # to every customer. This prevents the model from
        # simply learning:
        #
        # customer X = claim Y
        # ----------------------------------------------------

        claim_type = rng.choice(
            CLAIM_TYPES
        )

        # ----------------------------------------------------
        # Evidence.
        #
        # Most interactions start without evidence, but some
        # provide evidence.
        # ----------------------------------------------------

        evidence_options = (
            EVIDENCE_OPTIONS[
                claim_type
            ]
        )

        evidence = list(
            rng.choice(
                evidence_options
            )
        )

        # ----------------------------------------------------
        # Amount.
        # ----------------------------------------------------

        policy = (
            support_agent
            .merchant
            .refund_policy
        )

        requested_amount = (
            generate_request_amount(
                order_amount=(
                    order.order_amount
                ),
                rng=rng,
                policy=policy,
            )
        )

        # ----------------------------------------------------
        # Timestamp.
        #
        # We create a genuine chronological sequence.
        # ----------------------------------------------------

        timestamp = (
            base_timestamp
            + timedelta(
                days=rng.randint(
                    0,
                    29,
                ),
                hours=rng.randint(
                    0,
                    23,
                ),
                minutes=rng.randint(
                    0,
                    59,
                ),
            )
        )

        # Make sure timestamps are chronological.
        if history["last_timestamp"] is not None:

            minimum_timestamp = (
                history["last_timestamp"]
                + timedelta(
                    minutes=rng.randint(
                        30,
                        1440,
                    )
                )
            )

            if timestamp <= minimum_timestamp:

                timestamp = (
                    minimum_timestamp
                )

        # ----------------------------------------------------
        # Time since previous interaction.
        # ----------------------------------------------------

        if (
            history["last_timestamp"]
            is None
        ):

            time_since_previous = 999.0

        else:

            time_since_previous = (
                timestamp
                - history["last_timestamp"]
            ).total_seconds() / 3600.0

        # ----------------------------------------------------
        # Build request.
        # ----------------------------------------------------

        request = build_refund_request(
            customer=customer,

            order=order,

            claim_type=claim_type,

            requested_amount=(
                requested_amount
            ),

            evidence=evidence,

            timestamp=timestamp,

            request_number=(
                global_request_counter
            ),
        )

        # ----------------------------------------------------
        # Agent A makes the actual decision.
        # ----------------------------------------------------

        decision = support_agent.decide(
            customer=customer,

            order=order,

            request=request,
        )

        # ----------------------------------------------------
        # Build ML observation.
        #
        # IMPORTANT:
        #
        # No ground truth.
        # No counterparty type.
        # No abuse label.
        # No Agent B information.
        # ----------------------------------------------------

        observation = (
            build_policy_observation(
                customer=customer,

                claim_type=claim_type,

                requested_amount=(
                    requested_amount
                ),

                evidence=evidence,

                support_decision=decision,

                previous_support_decisions=(
                    history[
                        "decisions"
                    ]
                ),

                previous_support_approvals=(
                    history[
                        "approvals"
                    ]
                ),

                previous_support_evidence_requests=(
                    history[
                        "evidence_requests"
                    ]
                ),

                previous_support_escalations=(
                    history[
                        "escalations"
                    ]
                ),

                time_since_previous_request_hours=(
                    time_since_previous
                ),
            )
        )

        observations.append(
            observation
        )

        # ----------------------------------------------------
        # Additional trajectory metadata.
        #
        # This is NOT passed to Agent B as hidden information.
        # It exists so we can evaluate sequences correctly.
        # ----------------------------------------------------

        metadata_rows.append(
            {
                "episode_id": episode_id,

                "customer_id": (
                    customer.customer_id
                ),

                "sequence_number": (
                    sequence_number
                ),

                "timestamp": timestamp,

                "request_id": (
                    request.request_id
                ),

                "order_id": (
                    order.order_id
                ),

                "claim_type": (
                    claim_type
                ),

                "claim_text": (
                    request.claim_text
                ),

                "requested_amount": (
                    requested_amount
                ),

                "evidence_available": (
                    len(evidence) > 0
                ),

                "evidence_count": (
                    len(evidence)
                ),

                "evidence_types": (
                    "|".join(evidence)
                ),

                "support_decision": (
                    decision.decision
                ),

                "reason_code": (
                    decision.reason_code
                ),

                "approved_amount": (
                    decision.approved_amount
                ),

                "requires_followup": (
                    decision.requires_followup
                ),

                "previous_support_decisions": (
                    history[
                        "decisions"
                    ]
                ),

                "previous_support_approvals": (
                    history[
                        "approvals"
                    ]
                ),

                "previous_support_evidence_requests": (
                    history[
                        "evidence_requests"
                    ]
                ),

                "previous_support_escalations": (
                    history[
                        "escalations"
                    ]
                ),

                "previous_support_denials": (
                    history[
                        "denials"
                    ]
                ),

                "time_since_previous_request_hours": (
                    time_since_previous
                ),
            }
        )

        # ----------------------------------------------------
        # Update history AFTER observing A's response.
        # ----------------------------------------------------

        update_support_history(
            history=history,

            decision=decision,

            claim_type=claim_type,

            timestamp=timestamp,
        )

        global_request_counter += 1

    return (
        observations,
        metadata_rows,
        global_request_counter,
    )


# ============================================================
# FULL DATASET GENERATOR
# ============================================================

def generate_policy_dataset(
    n_customers: int = DEFAULT_N_CUSTOMERS,
    interactions_per_customer: int = (
        DEFAULT_INTERACTIONS_PER_CUSTOMER
    ),
    seed: int = DEFAULT_SEED,
) -> tuple[
    list[PolicyObservation],
    list[dict],
]:

    rng = Random(seed)

    # --------------------------------------------------------
    # WORLD
    #
    # The current world generator already uses the configured
    # 1000-customer population.
    # --------------------------------------------------------

    world = create_world(
        seed=seed,
    )

    support_agent = SupportAgent(
        merchant=world.merchant
    )

    customers = list(
        world.customers.values()
    )

    # --------------------------------------------------------
    # Restrict to requested number of customers.
    # --------------------------------------------------------

    customers = customers[
        :n_customers
    ]

    orders_by_customer = {}

    for order in world.orders.values():

        orders_by_customer.setdefault(
            order.customer_id,
            [],
        ).append(order)

    all_observations = []

    all_metadata = []

    request_counter = 1

    # --------------------------------------------------------
    # Generate one trajectory per customer.
    # --------------------------------------------------------

    for customer_index, customer in enumerate(
        customers
    ):

        customer_orders = (
            orders_by_customer.get(
                customer.customer_id,
                [],
            )
        )

        if not customer_orders:

            continue

        episode_id = (
            f"EPISODE_{customer_index + 1:05d}"
        )

        (
            observations,
            metadata_rows,
            request_counter,
        ) = generate_customer_trajectory(

            customer=customer,

            customer_orders=customer_orders,

            support_agent=support_agent,

            rng=rng,

            episode_id=episode_id,

            n_interactions=(
                interactions_per_customer
            ),

            global_request_counter=(
                request_counter
            ),

            base_timestamp=(
                BASE_TIMESTAMP
            ),
        )

        all_observations.extend(
            observations
        )

        all_metadata.extend(
            metadata_rows
        )

    return (
        all_observations,
        all_metadata,
    )


# ============================================================
# SAVE DATASET
# ============================================================

def save_policy_dataset(
    observations: list[PolicyObservation],
    metadata_rows: list[dict],
    output_path: str,
) -> pd.DataFrame:

    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Convert PolicyObservation objects.
    # --------------------------------------------------------

    observation_rows = [
        asdict(observation)
        for observation
        in observations
    ]

    observation_df = pd.DataFrame(
        observation_rows
    )

    # --------------------------------------------------------
    # Convert trajectory metadata.
    # --------------------------------------------------------

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    # --------------------------------------------------------
    # Merge.
    #
    # Both dataframes are generated in exactly the same
    # chronological order.
    # --------------------------------------------------------

    if len(observation_df) != len(
        metadata_df
    ):

        raise ValueError(
            "Observation and metadata "
            "counts do not match."
        )

    # Avoid duplicated columns.
    metadata_only_columns = [
        column
        for column in metadata_df.columns
        if column
        not in observation_df.columns
    ]

    df = pd.concat(
        [
            observation_df,
            metadata_df[
                metadata_only_columns
            ],
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Safety checks.
    # --------------------------------------------------------

    if df["episode_id"].isna().any():

        raise ValueError(
            "Missing episode_id detected."
        )

    if df["sequence_number"].isna().any():

        raise ValueError(
            "Missing sequence_number detected."
        )

    # --------------------------------------------------------
    # Ensure chronological ordering.
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "episode_id",
            "sequence_number",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    df.to_csv(
        path,
        index=False,
    )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_trajectory_dataset(
    df: pd.DataFrame,
    expected_customers: int,
    expected_interactions_per_customer: int,
) -> None:

    # --------------------------------------------------------
    # Basic row count.
    # --------------------------------------------------------

    expected_rows = (
        expected_customers
        * expected_interactions_per_customer
    )

    if len(df) != expected_rows:

        raise ValueError(
            f"Expected {expected_rows} rows "
            f"but generated {len(df)}."
        )

    # --------------------------------------------------------
    # Episode count.
    # --------------------------------------------------------

    episode_count = (
        df["episode_id"]
        .nunique()
    )

    if episode_count != expected_customers:

        raise ValueError(
            f"Expected {expected_customers} "
            f"episodes but found "
            f"{episode_count}."
        )

    # --------------------------------------------------------
    # Every trajectory should contain exactly N interactions.
    # --------------------------------------------------------

    trajectory_sizes = (
        df.groupby(
            "episode_id"
        )["sequence_number"]
        .count()
    )

    if not (
        trajectory_sizes
        == expected_interactions_per_customer
    ).all():

        raise ValueError(
            "Not every episode contains "
            "the expected number of interactions."
        )

    # --------------------------------------------------------
    # Sequence numbers should be 1..N.
    # --------------------------------------------------------

    for _, group in df.groupby(
        "episode_id"
    ):

        expected_sequence = list(
            range(
                1,
                expected_interactions_per_customer
                + 1,
            )
        )

        actual_sequence = (
            group[
                "sequence_number"
            ]
            .tolist()
        )

        if actual_sequence != expected_sequence:

            raise ValueError(
                "Invalid sequence detected "
                f"for episode."
            )

    # --------------------------------------------------------
    # No hidden ground truth leakage.
    # --------------------------------------------------------

    forbidden_columns = {
        "is_abusive",
        "population",
        "counterparty_type",
        "abuse_family",
        "ground_truth",
        "customer_private",
    }

    leaked = (
        forbidden_columns
        .intersection(
            set(df.columns)
        )
    )

    if leaked:

        raise ValueError(
            "Ground-truth leakage detected: "
            f"{sorted(leaked)}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "GENERATING AGENT A TRAJECTORY DATASET"
    )

    print("=" * 60)

    n_customers = 1000

    interactions_per_customer = 10

    observations, metadata_rows = (
        generate_policy_dataset(

            n_customers=n_customers,

            interactions_per_customer=(
                interactions_per_customer
            ),

            seed=42,
        )
    )

    output_path = (
        "data/generated/policy/"
        "policy_trajectories.csv"
    )

    df = save_policy_dataset(
        observations=observations,

        metadata_rows=metadata_rows,

        output_path=output_path,
    )

    validate_trajectory_dataset(
        df=df,

        expected_customers=n_customers,

        expected_interactions_per_customer=(
            interactions_per_customer
        ),
    )

    print()

    print(
        f"Generated observations: {len(df)}"
    )

    print(
        f"Customers / episodes: "
        f"{df['episode_id'].nunique()}"
    )

    print(
        f"Interactions per customer: "
        f"{interactions_per_customer}"
    )

    print(
        f"Saved to: {output_path}"
    )

    print()

    print(
        "Decision distribution:"
    )

    print(
        df[
            "support_decision"
        ].value_counts()
    )

    print()

    print(
        "Claim distribution:"
    )

    print(
        df[
            "claim_type"
        ].value_counts()
    )

    print()

    print(
        "Trajectory size distribution:"
    )

    print(
        df.groupby(
            "episode_id"
        ).size().value_counts()
    )

    print()

    print(
        "First customer trajectory:"
    )

    first_episode = (
        df["episode_id"]
        .iloc[0]
    )

    print(
        df[
            df["episode_id"]
            == first_episode
        ][
            [
                "episode_id",
                "sequence_number",
                "claim_type",
                "requested_amount",
                "evidence_available",
                "support_decision",
                "reason_code",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Validation: PASSED"
    )