from src.environment.world_generator import create_world
from src.environment.population_config import PopulationConfig


def make_test_config():

    return PopulationConfig(
        n_customers=20,

        human_legitimate=14,
        human_abusive=2,

        adaptive_legitimate=3,
        adaptive_abusive=1,

        n_support_agents=2,

        target_orders=100,

        simulation_days=30,

        n_devices=16,
        n_addresses=17,
        n_payment_instruments=18,

        min_refund_episodes=10,
        max_refund_episodes=30,

        seed=42,
    )


def test_world_generation():

    config = make_test_config()

    world = create_world(
        config=config,
        seed=42,
    )

    assert world.merchant.merchant_id == (
        "MERCHANT_001"
    )

    assert len(world.customers) == 20

    assert len(world.orders) == 100

    assert len(world.customer_private) == 20

    assert len(world.ground_truth) == 20

    assert len(world.graph_nodes) > 0

    assert len(world.graph_edges) > 0


def test_ground_truth_is_separate_from_customer_observation():

    config = make_test_config()

    world = create_world(
        config=config,
        seed=42,
    )

    customer_id = "CUSTOMER_0001"

    truth = world.ground_truth[
        customer_id
    ]

    private_state = world.customer_private[
        customer_id
    ]

    assert truth.counterparty_type in {
        "HUMAN",
        "ADAPTIVE_AGENT",
    }

    assert isinstance(
        truth.is_abusive,
        bool,
    )

    assert private_state.objective is not None


def test_population_composition():

    config = make_test_config()

    world = create_world(
        config=config,
        seed=42,
    )

    from collections import Counter

    composition = Counter(
        (
            truth.counterparty_type,
            truth.population,
        )
        for truth in world.ground_truth.values()
    )

    assert composition[
        ("HUMAN", "LEGITIMATE")
    ] == 14

    assert composition[
        ("HUMAN", "ABUSIVE")
    ] == 2

    assert composition[
        ("ADAPTIVE_AGENT", "LEGITIMATE")
    ] == 3

    assert composition[
        ("ADAPTIVE_AGENT", "ABUSIVE")
    ] == 1


def test_world_is_reproducible():

    config = make_test_config()

    world_a = create_world(
        config=config,
        seed=42,
    )

    world_b = create_world(
        config=config,
        seed=42,
    )

    assert list(
        world_a.customers.keys()
    ) == list(
        world_b.customers.keys()
    )

    assert list(
        world_a.orders.keys()
    ) == list(
        world_b.orders.keys()
    )

    assert len(world_a.refunds) == (
        len(world_b.refunds)
    )

    assert len(world_a.graph_nodes) == (
        len(world_b.graph_nodes)
    )

    assert len(world_a.graph_edges) == (
        len(world_b.graph_edges)
    )

def test_abusive_customers_have_documented_mechanisms():

    config = make_test_config()

    world = create_world(
        config=config,
        seed=42,
    )

    for truth in world.ground_truth.values():

        if truth.is_abusive:

            assert truth.abuse_family in {
                "shortage_claim",
                "wrong_item_claim",
                "non_delivery_claim",
                "substituted_return",
            }