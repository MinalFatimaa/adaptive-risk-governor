from collections import Counter

from src.environment.world_generator import create_world
from src.environment.population_config import (
    DEFAULT_POPULATION_CONFIG,
)

from src.environment.strategic_simulator import (
    run_strategic_simulation,
)


# ============================================================
# CONFIGURATION
# ============================================================

config = DEFAULT_POPULATION_CONFIG


# ============================================================
# CREATE WORLD
# ============================================================

world = create_world(
    config=config,
)

print("=" * 70)
print("PHASE 8 — STRATEGIC A ↔ B SIMULATION")
print("=" * 70)

print(
    f"Customers       : {len(world.customers)}"
)

print(
    f"Orders          : {len(world.orders)}"
)

print(
    f"Existing refunds: {len(world.refunds)}"
)

print(
    f"Graph nodes     : {len(world.graph_nodes)}"
)

print(
    f"Graph edges     : {len(world.graph_edges)}"
)


# ============================================================
# RUN STRATEGIC SIMULATION
# ============================================================

episodes = run_strategic_simulation(
    world=world,

    n_episodes=100,

    interactions_per_episode=20,

    seed=42,
)


# ============================================================
# BASIC SUMMARY
# ============================================================

print()
print("=" * 70)
print("SIMULATION SUMMARY")
print("=" * 70)

print(
    f"Episodes generated      : {len(episodes)}"
)

print(
    "Interactions generated  : "
    f"{sum(e.length for e in episodes)}"
)

print(
    "Expected interactions   : "
    f"{100 * 20}"
)


# ============================================================
# DECISION DISTRIBUTION
# ============================================================

decisions = Counter()

for episode in episodes:

    for interaction in episode.interactions:

        decisions[
            interaction.support_decision
        ] += 1


print()
print("=" * 70)
print("AGENT A DECISION DISTRIBUTION")
print("=" * 70)

for decision, count in decisions.items():

    percentage = (
        count
        / sum(decisions.values())
        * 100
    )

    print(
        f"{decision:20s}: "
        f"{count:5d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# CLAIM DISTRIBUTION
# ============================================================

claims = Counter()

for episode in episodes:

    for interaction in episode.interactions:

        claims[
            interaction.claim_type
        ] += 1


print()
print("=" * 70)
print("CLAIM DISTRIBUTION")
print("=" * 70)

for claim, count in claims.items():

    print(
        f"{claim:30s}: {count:5d}"
    )


# ============================================================
# PREDICTION ACCURACY
# ============================================================

accuracies = [
    episode.prediction_accuracy()
    for episode in episodes
]

mean_accuracy = (
    sum(accuracies)
    / len(accuracies)
)


print()
print("=" * 70)
print("AGENT B POLICY PREDICTION")
print("=" * 70)

print(
    f"Mean prediction accuracy: "
    f"{mean_accuracy:.3f}"
)


# ============================================================
# STRATEGIC IMPROVEMENT
# ============================================================

initial_success = [
    episode.initial_expected_success
    for episode in episodes
]

final_success = [
    episode.final_expected_success
    for episode in episodes
]

mean_initial = (
    sum(initial_success)
    / len(initial_success)
)

mean_final = (
    sum(final_success)
    / len(final_success)
)

delta = (
    mean_final
    - mean_initial
)


print()
print("=" * 70)
print("STRATEGIC IMPROVEMENT")
print("=" * 70)

print(
    f"Initial expected success: "
    f"{mean_initial:.3f}"
)

print(
    f"Final expected success  : "
    f"{mean_final:.3f}"
)

print(
    f"Change                  : "
    f"{delta:+.3f}"
)


# ============================================================
# FIRST EPISODE
# ============================================================

first = episodes[0]

print()
print("=" * 70)
print("FIRST STRATEGIC EPISODE")
print("=" * 70)

print(
    f"Episode : {first.episode_id}"
)

print(
    f"Customer: {first.customer_id}"
)


print()
print("INITIAL AGENT B BELIEFS")
print("-" * 70)

for key, value in (
    first.initial_policy_beliefs.items()
):

    print(
        f"{key:25s}: {value:.3f}"
    )


print()
print("=" * 70)
print("STRATEGIC TRAJECTORY")
print("=" * 70)

for interaction in first.interactions:

    prediction = (
        interaction.predicted_decision
        if interaction.predicted_decision
        is not None
        else "N/A"
    )

    correctness = (
        "✓"
        if interaction.prediction_correct
        else "✗"
        if interaction.prediction_correct
        is not None
        else "-"
    )

    print(
        f"{interaction.sequence_number:02d} | "
        f"{interaction.claim_type:30s} | "
        f"₹{interaction.requested_amount:8.2f} | "
        f"evidence={bool(interaction.evidence_available):5} | "
        f"pred={prediction:18s} | "
        f"A={interaction.support_decision:18s} | "
        f"{correctness}"
    )


print()
print("FINAL AGENT B BELIEFS")
print("-" * 70)

for key, value in (
    first.final_policy_beliefs.items()
):

    print(
        f"{key:25s}: {value:.3f}"
    )


# ============================================================
# LEARNING CHECK
# ============================================================

print()
print("=" * 70)
print("AGENT B LEARNING CHECK")
print("=" * 70)

all_keys = set(
    first.initial_policy_beliefs
) | set(
    first.final_policy_beliefs
)

for key in all_keys:

    initial = (
        first.initial_policy_beliefs
        .get(key, 0.0)
    )

    final = (
        first.final_policy_beliefs
        .get(key, 0.0)
    )

    print(
        f"{key:25s}: "
        f"{initial:.3f} → "
        f"{final:.3f} "
        f"(Δ {final - initial:+.3f})"
    )


# ============================================================
# VALIDATION
# ============================================================

assert len(episodes) == 100

assert all(
    episode.length == 20
    for episode in episodes
)

assert (
    sum(e.length for e in episodes)
    == 2000
)


print()
print("=" * 70)
print("VALIDATION")
print("=" * 70)

print(
    "Episode structure : PASSED"
)

print(
    "Interaction count : PASSED"
)

print(
    "Strategic learning: "
    + (
        "DETECTED"
        if delta != 0
        else "NO CHANGE"
    )
)

print()
print("Validation: PASSED")