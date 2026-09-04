from collections import Counter

from src.environment.population_config import (
    DEFAULT_POPULATION_CONFIG,
)
from src.environment.world_generator import create_world
from src.environment.simulator import run_adaptive_simulation


# ============================================================
# CONFIGURATION
# ============================================================

config = DEFAULT_POPULATION_CONFIG

N_EPISODES = 100
INTERACTIONS_PER_EPISODE = 10
SEED = config.seed


# ============================================================
# CREATE WORLD
# ============================================================

world = create_world(
    config=config,
)


print("=" * 60)
print("PHASE 7 — ADAPTIVE A ↔ B SIMULATION")
print("=" * 60)

print(
    f"Customers: {len(world.customers)}"
)

print(
    f"Orders: {len(world.orders)}"
)

print(
    f"Existing refunds: {len(world.refunds)}"
)

print(
    f"Graph nodes: {len(world.graph_nodes)}"
)

print(
    f"Graph edges: {len(world.graph_edges)}"
)


# ============================================================
# RUN ADAPTIVE SIMULATION
# ============================================================

episodes = run_adaptive_simulation(
    world=world,
    n_episodes=N_EPISODES,
    interactions_per_episode=INTERACTIONS_PER_EPISODE,
    seed=SEED,
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print()
print("=" * 60)
print("SIMULATION SUMMARY")
print("=" * 60)

print(
    f"Episodes generated: {len(episodes)}"
)

total_interactions = sum(
    episode.length
    for episode in episodes
)

print(
    f"Interactions generated: {total_interactions}"
)

expected_interactions = (
    N_EPISODES *
    INTERACTIONS_PER_EPISODE
)

print(
    f"Expected interactions: {expected_interactions}"
)


assert len(episodes) == N_EPISODES

assert (
    total_interactions
    == expected_interactions
)


# ============================================================
# AGENT A DECISION DISTRIBUTION
# ============================================================

decisions = Counter()

for episode in episodes:

    for interaction in episode.interactions:

        decisions[
            interaction.support_decision
        ] += 1


print()
print("=" * 60)
print("AGENT A DECISION DISTRIBUTION")
print("=" * 60)

for decision, count in decisions.most_common():

    percentage = (
        count /
        total_interactions
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
print("=" * 60)
print("CLAIM DISTRIBUTION")
print("=" * 60)

for claim_type, count in claims.most_common():

    print(
        f"{claim_type:30s}: "
        f"{count:5d}"
    )


# ============================================================
# INSPECT FIRST EPISODE
# ============================================================

first = episodes[0]


print()
print("=" * 60)
print("FIRST ADAPTIVE EPISODE")
print("=" * 60)

print(
    f"Episode: {first.episode_id}"
)

print(
    f"Customer: {first.customer_id}"
)


# ============================================================
# INITIAL B POLICY
# ============================================================

print()
print("INITIAL AGENT B POLICY BELIEFS")
print("-" * 60)

for key, value in (
    first.initial_policy_beliefs.items()
):

    print(
        f"{key:25s}: "
        f"{value:.3f}"
    )


# ============================================================
# INTERACTION TRAJECTORY
# ============================================================

print()
print("=" * 60)
print("AGENT A ↔ AGENT B TRAJECTORY")
print("=" * 60)

for interaction in first.interactions:

    print(
        f"{interaction.sequence_number:02d} | "
        f"{interaction.claim_type:30s} | "
        f"₹{interaction.requested_amount:8.2f} | "
        f"evidence="
        f"{bool(interaction.evidence_available):5} | "
        f"A="
        f"{interaction.support_decision:18s}"
    )


# ============================================================
# FINAL B POLICY
# ============================================================

print()
print("=" * 60)
print("FINAL AGENT B POLICY BELIEFS")
print("=" * 60)

for key, value in (
    first.final_policy_beliefs.items()
):

    print(
        f"{key:25s}: "
        f"{value:.3f}"
    )


# ============================================================
# LEARNING CHECK
# ============================================================

print()
print("=" * 60)
print("AGENT B LEARNING CHECK")
print("=" * 60)

policy_changed = False

for key in first.initial_policy_beliefs:

    initial = (
        first.initial_policy_beliefs[key]
    )

    final = (
        first.final_policy_beliefs[key]
    )

    delta = final - initial

    if abs(delta) > 1e-9:
        policy_changed = True

    print(
        f"{key:25s}: "
        f"{initial:.3f} → "
        f"{final:.3f} "
        f"(Δ {delta:+.3f})"
    )


# ============================================================
# OBSERVATION COUNT
# ============================================================

print()
print(
    f"Agent B observations: "
    f"{sum(len(e.interactions) for e in episodes)}"
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 60)
print("VALIDATION")
print("=" * 60)

assert len(episodes) == N_EPISODES

assert all(
    episode.length
    == INTERACTIONS_PER_EPISODE
    for episode in episodes
)

assert total_interactions > 0

assert len(decisions) > 0

assert len(claims) > 0

print(
    "Simulation structure: PASSED"
)

if policy_changed:

    print(
        "Agent B adaptation: DETECTED"
    )

else:

    print(
        "Agent B adaptation: NOT DETECTED"
    )

print()
print("Validation: PASSED")