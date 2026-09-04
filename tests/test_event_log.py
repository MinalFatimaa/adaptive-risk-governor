from src.environment.world_generator import create_world
from src.environment.population_config import PopulationConfig


def make_event_test_config():
    return PopulationConfig(
        n_customers=5,

        human_legitimate=3,
        human_abusive=1,

        adaptive_legitimate=1,
        adaptive_abusive=0,

        n_support_agents=1,

        target_orders=10,

        simulation_days=30,

        n_devices=5,
        n_addresses=5,
        n_payment_instruments=5,

        min_refund_episodes=2,
        max_refund_episodes=5,

        seed=42,
    )


def test_event_recording():

    config = make_event_test_config()

    world = create_world(
        config=config,
        seed=42,
    )

    # The event log should initially exist and be empty.
    assert world.events == []


def test_event_retrieval():

    config = make_event_test_config()

    world = create_world(
        config=config,
        seed=42,
    )

    # The event collection should be available
    # for later interaction/event simulation.
    assert isinstance(world.events, list)