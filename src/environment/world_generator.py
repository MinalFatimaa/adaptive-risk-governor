from datetime import datetime, timedelta
from random import Random

from ..schemas.customer import (
    CustomerObservableState,
    CustomerPrivateState,
    CustomerGroundTruth,
)

from ..schemas.merchant import (
    MerchantState,
    MerchantPolicy,
    MerchantEconomicState,
)

from ..schemas.order import OrderState
from ..schemas.refund import RefundState

from ..schemas.graph import (
    GraphNode,
    GraphEdge,
)

from .world import EnvironmentState
from .population_config import (
    PopulationConfig,
    DEFAULT_POPULATION_CONFIG,
    CustomerType,
    BehaviorClass,
    AbuseMechanism,
)


# ============================================================
# CONSTANTS
# ============================================================

PRODUCT_CATEGORIES = [
    "ELECTRONICS",
    "FASHION",
    "HOME",
    "BEAUTY",
    "GROCERY",
]

SUPPORT_AGENT_PROFILES = [
    "BALANCED",
    "SATISFACTION_ORIENTED",
    "CONSERVATIVE",
    "EVIDENCE_HEAVY",
    "FAST_RESOLUTION",
    "BALANCED",
    "ESCALATION_HEAVY",
    "EVIDENCE_HEAVY",
]


# ============================================================
# DETERMINISTIC ID GENERATOR
# ============================================================

class IdGenerator:
    """
    Deterministic ID generator.

    We deliberately do NOT use uuid4() here because the same
    random seed should reproduce the same world exactly.
    """

    def __init__(self):
        self.counters = {}

    def make(self, prefix: str) -> str:

        self.counters[prefix] = (
            self.counters.get(prefix, 0) + 1
        )

        return f"{prefix}_{self.counters[prefix]:06d}"


# ============================================================
# MERCHANT
# ============================================================

def create_merchant() -> MerchantState:

    policy = MerchantPolicy(
        refund_window_days=30,
        max_auto_refund_amount=5000,
        evidence_required_above=2000,
        human_review_threshold=5000,
    )

    economic_state = MerchantEconomicState()

    return MerchantState(
        merchant_id="MERCHANT_001",
        name="Synthetic Merchant",
        currency="INR",
        refund_policy=policy,
        economic_state=economic_state,
    )


# ============================================================
# SUPPORT AGENTS
# ============================================================

def create_support_agents(
    config: PopulationConfig,
) -> list[dict]:

    agents = []

    for i in range(config.n_support_agents):

        profile = SUPPORT_AGENT_PROFILES[
            i % len(SUPPORT_AGENT_PROFILES)
        ]

        agents.append(
            {
                "support_agent_id": f"SUPPORT_{i + 1:03d}",
                "profile": profile,
            }
        )

    return agents


# ============================================================
# CUSTOMER
# ============================================================

def create_customer(
    customer_id: str,
    customer_type: CustomerType,
    behavior_class: BehaviorClass,
    abuse_mechanism: AbuseMechanism | None,
    account_age_days: int,
    device_id: str,
    address_id: str,
    payment_id: str,
    id_generator: IdGenerator,
    rng: Random,
    created_at: datetime,
) -> tuple[
    CustomerObservableState,
    CustomerPrivateState,
    CustomerGroundTruth,
]:

    is_abusive = (
        behavior_class == BehaviorClass.ABUSIVE
    )

    # --------------------------------------------------------
    # CUSTOMER ACTIVITY
    # --------------------------------------------------------

    if is_abusive:

        order_count = rng.randint(4, 12)

    else:

        order_count = rng.randint(2, 10)

    completed_order_count = max(
        1,
        order_count - rng.randint(0, 1),
    )

    # Historical refund activity is deliberately noisy.
    #
    # We do NOT make:
    #
    #     refund_count > X => abusive
    #
    # because that would make the Governor's job trivial.

    if is_abusive:

        refund_count = rng.randint(1, 4)

        refund_amount_total = round(
            rng.uniform(1000, 8000),
            2,
        )

    else:

        refund_count = rng.choices(
            [0, 1, 2],
            weights=[0.65, 0.30, 0.05],
            k=1,
        )[0]

        refund_amount_total = round(
            rng.uniform(0, 3000),
            2,
        )

    # Chargebacks are retained as an observable historical field,
    # but are NOT our current loss class.
    chargeback_count = 0

    # --------------------------------------------------------
    # OBSERVABLE STATE
    # --------------------------------------------------------

    observable = CustomerObservableState(
        customer_id=customer_id,

        account_id=id_generator.make(
            "ACCOUNT"
        ),

        account_age_days=account_age_days,

        order_count=order_count,

        completed_order_count=completed_order_count,

        refund_count=refund_count,

        refund_amount_total=refund_amount_total,

        chargeback_count=chargeback_count,

        current_order_ids=[],

        device_ids=[device_id],

        address_ids=[address_id],

        payment_ids=[payment_id],

        created_at=created_at,
    )

    # --------------------------------------------------------
    # PRIVATE STATE
    # --------------------------------------------------------

    if customer_type == CustomerType.ADAPTIVE_AGENT:

        if is_abusive:

            objective = (
                "maximize_illegitimate_refund_value"
            )

            current_strategy = (
                abuse_mechanism.value
                if abuse_mechanism
                else "REFUND_ABUSE"
            )

        else:

            objective = (
                "maximize_successful_legitimate_resolution"
            )

            current_strategy = (
                "STANDARD_DISPUTE"
            )

    else:

        if is_abusive:

            objective = "obtain_refund"

            current_strategy = (
                abuse_mechanism.value
                if abuse_mechanism
                else "REFUND_REQUEST"
            )

        else:

            objective = "resolve_issue"

            current_strategy = (
                "STANDARD_DISPUTE"
            )

    private_state = CustomerPrivateState(
        customer_id=customer_id,

        objective=objective,

        current_strategy=current_strategy,

        strategy_beliefs={},

        successful_strategies=[],

        failed_strategies=[],

        observed_support_responses=[],

        observed_outcomes=[],

        private_memory=[],
    )

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    ground_truth = CustomerGroundTruth(
        customer_id=customer_id,

        population=(
            behavior_class.value.upper()
        ),

        behavior_type=(
            "REFUND_ABUSE"
            if is_abusive
            else "NORMAL_CUSTOMER"
        ),

        abuse_family=(
            abuse_mechanism.value
            if abuse_mechanism
            else None
        ),

        is_abusive=is_abusive,

        counterparty_type=(
            customer_type.value.upper()
        ),
    )

    return (
        observable,
        private_state,
        ground_truth,
    )


# ============================================================
# ORDER
# ============================================================

def create_order(
    customer: CustomerObservableState,
    order_id: str,
    rng: Random,
    current_time: datetime,
) -> OrderState:

    order_amount = round(
        rng.uniform(500, 10000),
        2,
    )

    category = rng.choice(
        PRODUCT_CATEGORIES
    )

    order_age_days = rng.randint(
        1,
        20,
    )

    delivery_age_days = rng.randint(
        0,
        min(order_age_days, 5),
    )

    order_timestamp = (
        current_time
        - timedelta(days=order_age_days)
    )

    delivery_timestamp = (
        current_time
        - timedelta(days=delivery_age_days)
    )

    return OrderState(

        order_id=order_id,

        customer_id=customer.customer_id,

        order_timestamp=order_timestamp,

        delivery_timestamp=delivery_timestamp,

        order_amount=order_amount,

        product_category=category,

        quantity=rng.randint(1, 3),

        delivery_status="DELIVERED",

        payment_id=customer.payment_ids[0],

        shipping_address_id=customer.address_ids[0],

        device_id=customer.device_ids[0],
    )


# ============================================================
# HISTORICAL REFUNDS
# ============================================================

def create_historical_refunds(
    customer: CustomerObservableState,
    orders: list[OrderState],
    behavior_class: BehaviorClass,
    abuse_mechanism: AbuseMechanism | None,
    id_generator: IdGenerator,
    rng: Random,
) -> list[RefundState]:

    refunds = []

    is_abusive = (
        behavior_class == BehaviorClass.ABUSIVE
    )

    if is_abusive:

        refund_probability = 0.35

    else:

        refund_probability = 0.10

    for order in orders:

        if rng.random() > refund_probability:
            continue

        amount = min(
            order.order_amount,
            round(
                rng.uniform(
                    500,
                    order.order_amount,
                ),
                2,
            ),
        )

        refund = RefundState(

            refund_id=id_generator.make(
                "REFUND"
            ),

            refund_request_id=id_generator.make(
                "REQUEST"
            ),

            order_id=order.order_id,

            customer_id=customer.customer_id,

            requested_amount=amount,

            approved_amount=amount,

            status="COMPLETED",

            requested_at=(
                order.order_timestamp
                + timedelta(days=1)
            ),

            approved_at=(
                order.order_timestamp
                + timedelta(days=1)
            ),

            executed_at=(
                order.order_timestamp
                + timedelta(days=1)
            ),

            decision_source="HISTORICAL",
        )

        refunds.append(refund)

    return refunds


# ============================================================
# RELATIONSHIP GRAPH
# ============================================================

def create_customer_graph(
    customers: list[CustomerObservableState],
    orders: dict[str, OrderState],
    refunds: dict[str, RefundState],
    id_generator: IdGenerator,
    current_time: datetime,
) -> tuple[
    list[GraphNode],
    list[GraphEdge],
]:

    nodes = []
    edges = []

    # --------------------------------------------------------
    # Deduplicate nodes
    # --------------------------------------------------------

    seen_nodes = set()

    def add_node(
        node_id: str,
        node_type: str,
    ):

        if node_id in seen_nodes:
            return

        nodes.append(
            GraphNode(
                node_id=node_id,
                node_type=node_type,
            )
        )

        seen_nodes.add(node_id)

    # --------------------------------------------------------
    # CUSTOMER + RESOURCE RELATIONSHIPS
    # --------------------------------------------------------

    for customer in customers:

        add_node(
            customer.customer_id,
            "CUSTOMER",
        )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        for device_id in customer.device_ids:

            add_node(
                device_id,
                "DEVICE",
            )

            edges.append(
                GraphEdge(
                    edge_id=id_generator.make(
                        "EDGE"
                    ),
                    source_id=customer.customer_id,
                    target_id=device_id,
                    edge_type="USES_DEVICE",
                    first_seen=customer.created_at,
                    last_seen=current_time,
                    weight=1.0,
                )
            )

        # ----------------------------------------------------
        # Address
        # ----------------------------------------------------

        for address_id in customer.address_ids:

            add_node(
                address_id,
                "ADDRESS",
            )

            edges.append(
                GraphEdge(
                    edge_id=id_generator.make(
                        "EDGE"
                    ),
                    source_id=customer.customer_id,
                    target_id=address_id,
                    edge_type="USES_ADDRESS",
                    first_seen=customer.created_at,
                    last_seen=current_time,
                    weight=1.0,
                )
            )

        # ----------------------------------------------------
        # Payment
        # ----------------------------------------------------

        for payment_id in customer.payment_ids:

            add_node(
                payment_id,
                "PAYMENT",
            )

            edges.append(
                GraphEdge(
                    edge_id=id_generator.make(
                        "EDGE"
                    ),
                    source_id=customer.customer_id,
                    target_id=payment_id,
                    edge_type="USES_PAYMENT",
                    first_seen=customer.created_at,
                    last_seen=current_time,
                    weight=1.0,
                )
            )

        # ----------------------------------------------------
        # Customer → Order
        # ----------------------------------------------------

        for order_id in customer.current_order_ids:

            order = orders.get(
                order_id
            )

            if order is None:
                continue

            add_node(
                order.order_id,
                "ORDER",
            )

            edges.append(
                GraphEdge(
                    edge_id=id_generator.make(
                        "EDGE"
                    ),
                    source_id=customer.customer_id,
                    target_id=order.order_id,
                    edge_type="PLACED_ORDER",
                    first_seen=order.order_timestamp,
                    last_seen=(
                        order.delivery_timestamp
                        or current_time
                    ),
                    weight=1.0,
                )
            )

    # --------------------------------------------------------
    # ORDER → REFUND
    # --------------------------------------------------------

    for refund in refunds.values():

        # ----------------------------------------------------
        # Make sure the order exists.
        # ----------------------------------------------------

        order = orders.get(
            refund.order_id
        )

        if order is None:
            continue

        # ----------------------------------------------------
        # Add refund node
        # ----------------------------------------------------

        add_node(
            refund.refund_id,
            "REFUND",
        )

        # ----------------------------------------------------
        # Order → Refund
        # ----------------------------------------------------

        edges.append(
            GraphEdge(
                edge_id=id_generator.make(
                    "EDGE"
                ),
                source_id=refund.order_id,
                target_id=refund.refund_id,
                edge_type="HAS_REFUND",
                first_seen=refund.requested_at,
                last_seen=(
                    refund.executed_at
                    or current_time
                ),
                weight=1.0,
            )
        )

        # ----------------------------------------------------
        # Customer → Refund
        #
        # This provides a direct customer interaction path
        # in addition to:
        #
        # CUSTOMER → ORDER → REFUND
        # ----------------------------------------------------

        if refund.customer_id in {
            customer.customer_id
            for customer in customers
        }:

            edges.append(
                GraphEdge(
                    edge_id=id_generator.make(
                        "EDGE"
                    ),
                    source_id=refund.customer_id,
                    target_id=refund.refund_id,
                    edge_type="REQUESTED_REFUND",
                    first_seen=refund.requested_at,
                    last_seen=(
                        refund.executed_at
                        or current_time
                    ),
                    weight=1.0,
                )
            )

    return nodes, edges

# ============================================================
# SHARED INFRASTRUCTURE ASSIGNMENT
# ============================================================

def create_shared_resources(
    config: PopulationConfig,
    id_generator: IdGenerator,
) -> tuple[
    list[str],
    list[str],
    list[str],
]:

    devices = [
        id_generator.make("DEVICE")
        for _ in range(config.n_devices)
    ]

    addresses = [
        id_generator.make("ADDRESS")
        for _ in range(config.n_addresses)
    ]

    payments = [
        id_generator.make("PAYMENT")
        for _ in range(
            config.n_payment_instruments
        )
    ]

    return (
        devices,
        addresses,
        payments,
    )


# ============================================================
# RESOURCE SELECTION
# ============================================================

def choose_resource(
    resources: list[str],
    customer_index: int,
    rng: Random,
) -> str:

    # Mostly unique assignment, with occasional sharing.
    #
    # Sharing is intentionally NOT equivalent to fraud.

    if customer_index < len(resources):

        if rng.random() < 0.82:

            return resources[
                customer_index
            ]

    return rng.choice(resources)


# ============================================================
# CUSTOMER PROFILE ASSIGNMENT
# ============================================================

def build_customer_profiles(
    config: PopulationConfig,
) -> list[
    tuple[
        CustomerType,
        BehaviorClass,
        AbuseMechanism | None,
    ]
]:

    profiles = []

    mechanisms = list(
        config.abuse_mechanisms
    )

    # --------------------------------------------------------
    # HUMAN LEGITIMATE
    # --------------------------------------------------------

    profiles.extend(
        [
            (
                CustomerType.HUMAN,
                BehaviorClass.LEGITIMATE,
                None,
            )
            for _ in range(
                config.human_legitimate
            )
        ]
    )

    # --------------------------------------------------------
    # HUMAN ABUSIVE
    # --------------------------------------------------------

    for i in range(
        config.human_abusive
    ):

        mechanism = mechanisms[
            i % len(mechanisms)
        ]

        profiles.append(
            (
                CustomerType.HUMAN,
                BehaviorClass.ABUSIVE,
                mechanism,
            )
        )

    # --------------------------------------------------------
    # ADAPTIVE LEGITIMATE
    # --------------------------------------------------------

    profiles.extend(
        [
            (
                CustomerType.ADAPTIVE_AGENT,
                BehaviorClass.LEGITIMATE,
                None,
            )
            for _ in range(
                config.adaptive_legitimate
            )
        ]
    )

    # --------------------------------------------------------
    # ADAPTIVE ABUSIVE
    # --------------------------------------------------------

    for i in range(
        config.adaptive_abusive
    ):

        mechanism = mechanisms[
            i % len(mechanisms)
        ]

        profiles.append(
            (
                CustomerType.ADAPTIVE_AGENT,
                BehaviorClass.ABUSIVE,
                mechanism,
            )
        )

    return profiles


# ============================================================
# WORLD GENERATOR
# ============================================================

def create_world(
    config: PopulationConfig | None = None,
    seed: int | None = None,
) -> EnvironmentState:

    if config is None:

        config = DEFAULT_POPULATION_CONFIG

    config.validate()

    actual_seed = (
        config.seed
        if seed is None
        else seed
    )

    rng = Random(actual_seed)

    id_generator = IdGenerator()

    # --------------------------------------------------------
    # SIMULATION CLOCK
    # --------------------------------------------------------

    current_time = datetime(
        2026,
        8,
        24,
        12,
        0,
        0,
    )

    # --------------------------------------------------------
    # MERCHANT
    # --------------------------------------------------------

    merchant = create_merchant()

    # --------------------------------------------------------
    # SUPPORT AGENTS
    # --------------------------------------------------------

    # Currently created as metadata.
    #
    # Support Agent schemas/events will be introduced in the
    # next phase.

    support_agents = create_support_agents(
        config
    )

    # Prevent unused-variable warnings and make the intent
    # explicit. Support agents will enter EnvironmentState
    # once their dedicated schema exists.
    _ = support_agents

    # --------------------------------------------------------
    # SHARED RESOURCES
    # --------------------------------------------------------

    (
        devices,
        addresses,
        payments,
    ) = create_shared_resources(
        config,
        id_generator,
    )

    # --------------------------------------------------------
    # CUSTOMER PROFILES
    # --------------------------------------------------------

    profiles = build_customer_profiles(
        config
    )

    # Shuffle profiles so population classes aren't ordered
    # by customer ID.
    rng.shuffle(profiles)

    # --------------------------------------------------------
    # STATE CONTAINERS
    # --------------------------------------------------------

    customers = {}

    customer_private = {}

    ground_truth = {}

    orders = {}

    refunds = {}

    customer_objects = []

    # --------------------------------------------------------
    # CUSTOMERS
    # --------------------------------------------------------

    for i, (
        customer_type,
        behavior_class,
        abuse_mechanism,
    ) in enumerate(profiles):

        customer_id = (
            f"CUSTOMER_{i + 1:04d}"
        )

        account_age_days = rng.randint(
            30,
            1500,
        )

        created_at = (
            current_time
            - timedelta(
                days=account_age_days
            )
        )

        device_id = choose_resource(
            devices,
            i,
            rng,
        )

        address_id = choose_resource(
            addresses,
            i,
            rng,
        )

        payment_id = choose_resource(
            payments,
            i,
            rng,
        )

        (
            observable,
            private_state,
            truth,
        ) = create_customer(

            customer_id=customer_id,

            customer_type=customer_type,

            behavior_class=behavior_class,

            abuse_mechanism=abuse_mechanism,

            account_age_days=account_age_days,

            device_id=device_id,

            address_id=address_id,

            payment_id=payment_id,

            id_generator=id_generator,

            rng=rng,

            created_at=created_at,
        )

        customers[
            customer_id
        ] = observable

        customer_private[
            customer_id
        ] = private_state

        ground_truth[
            customer_id
        ] = truth

        customer_objects.append(
            observable
        )

    # --------------------------------------------------------
    # ORDERS
    # --------------------------------------------------------

    #
    # We target approximately config.target_orders.
    #
    # Customers receive a variable number of orders instead
    # of exactly the same number.
    #

    remaining_orders = (
        config.target_orders
    )

    for i, customer in enumerate(
        customer_objects
    ):

        customers_remaining = (
            len(customer_objects) - i
        )

        if customers_remaining <= 1:

            n_orders = max(
                1,
                remaining_orders,
            )

        else:

            average_remaining = (
                remaining_orders
                / customers_remaining
            )

            lower = max(
                1,
                int(average_remaining * 0.4),
            )

            upper = max(
                lower,
                int(average_remaining * 1.6),
            )

            n_orders = rng.randint(
                lower,
                upper,
            )

        # Prevent exceeding the target.
        n_orders = min(
            n_orders,
            remaining_orders
            - (customers_remaining - 1),
        )

        for _ in range(n_orders):

            order_id = (
                id_generator.make(
                    "ORDER"
                )
            )

            order = create_order(
                customer=customer,
                order_id=order_id,
                rng=rng,
                current_time=current_time,
            )

            orders[
                order_id
            ] = order

            customer.current_order_ids.append(
                order_id
            )

        remaining_orders -= n_orders

    # --------------------------------------------------------
    # UPDATE OBSERVABLE ORDER COUNTS
    # --------------------------------------------------------

    for customer in customer_objects:

        customer.order_count = len(
            customer.current_order_ids
        )

        customer.completed_order_count = (
            customer.order_count
        )

    # --------------------------------------------------------
    # HISTORICAL REFUNDS
    # --------------------------------------------------------

    for customer in customer_objects:

        customer_orders = [
            orders[order_id]
            for order_id
            in customer.current_order_ids
        ]

        truth = ground_truth[
            customer.customer_id
        ]

        behavior_class = (
            BehaviorClass.ABUSIVE
            if truth.is_abusive
            else BehaviorClass.LEGITIMATE
        )

        abuse_mechanism = None

        if truth.abuse_family:

            abuse_mechanism = (
                AbuseMechanism(
                    truth.abuse_family
                )
            )

        historical_refunds = (
            create_historical_refunds(

                customer=customer,

                orders=customer_orders,

                behavior_class=behavior_class,

                abuse_mechanism=abuse_mechanism,

                id_generator=id_generator,

                rng=rng,
            )
        )

        for refund in historical_refunds:

            refunds[
                refund.refund_id
            ] = refund

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------
    graph_nodes, graph_edges = create_customer_graph(
    customers=customer_objects,
    orders=orders,
    refunds=refunds,
    id_generator=id_generator,
    current_time=current_time,)

    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------

    return EnvironmentState(

        merchant=merchant,

        customers=customers,

        customer_private=customer_private,

        ground_truth=ground_truth,

        orders=orders,

        refunds=refunds,

        events=[],

        graph_nodes=graph_nodes,

        graph_edges=graph_edges,
    )