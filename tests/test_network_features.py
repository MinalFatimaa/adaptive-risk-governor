from datetime import datetime, timedelta

from src.governor.network_features import (
    NetworkFeatureBuilder,
    NetworkObservation,
)


def test_shared_ip_alone_is_not_high_risk():

    now = datetime.now()

    observations = [

        NetworkObservation(
            customer_id="CUSTOMER_001",
            ip_address="10.0.0.1",
            device_id="DEVICE_001",
            payment_method_id="PAY_001",
            shipping_address_id="ADDR_001",
            timestamp=now,
            claim_type="SHORTAGE_CLAIM",
            requested_amount=1000,
            support_decision="APPROVE",
        ),

        NetworkObservation(
            customer_id="CUSTOMER_002",
            ip_address="10.0.0.1",
            device_id="DEVICE_002",
            payment_method_id="PAY_002",
            shipping_address_id="ADDR_002",
            timestamp=now,
            claim_type="WRONG_ITEM_CLAIM",
            requested_amount=1200,
            support_decision="APPROVE",
        ),
    ]

    builder = NetworkFeatureBuilder()

    features = builder.build(
        customer_id="CUSTOMER_001",
        observations=observations,
        current_ip="10.0.0.1",
        current_device_id="DEVICE_001",
        current_payment_method_id="PAY_001",
        current_shipping_address_id="ADDR_001",
        current_claim_type="SHORTAGE_CLAIM",
        current_amount=1000,
        now=now,
    )

    assert (
        features[
            "shared_ip_account_count"
        ] == 1
    )

    assert (
        features[
            "shared_device_account_count"
        ] == 0
    )

    assert (
        features[
            "shared_payment_account_count"
        ] == 0
    )


def test_multiple_shared_identifiers_increase_link_strength():

    now = datetime.now()

    observations = [

        NetworkObservation(
            customer_id="CUSTOMER_001",
            ip_address="10.0.0.1",
            device_id="DEVICE_001",
            payment_method_id="PAY_001",
            shipping_address_id="ADDR_001",
            timestamp=now,
            claim_type="SHORTAGE_CLAIM",
            requested_amount=2000,
            support_decision="DENY",
        ),

        NetworkObservation(
            customer_id="CUSTOMER_002",
            ip_address="10.0.0.1",
            device_id="DEVICE_001",
            payment_method_id="PAY_001",
            shipping_address_id="ADDR_001",
            timestamp=now,
            claim_type="SHORTAGE_CLAIM",
            requested_amount=3000,
            support_decision="DENY",
        ),
    ]

    builder = NetworkFeatureBuilder()

    features = builder.build(
        customer_id="CUSTOMER_001",
        observations=observations,
        current_ip="10.0.0.1",
        current_device_id="DEVICE_001",
        current_payment_method_id="PAY_001",
        current_shipping_address_id="ADDR_001",
        current_claim_type="SHORTAGE_CLAIM",
        current_amount=2000,
        now=now,
    )

    assert (
        features[
            "shared_ip_account_count"
        ] == 1
    )

    assert (
        features[
            "shared_device_account_count"
        ] == 1
    )

    assert (
        features[
            "shared_payment_account_count"
        ] == 1
    )

    assert (
        features[
            "shared_address_account_count"
        ] == 1
    )

    assert (
        features[
            "identity_link_strength"
        ] == 1.0
    )


def test_velocity_features():

    now = datetime.now()

    observations = []

    for i in range(5):

        observations.append(
            NetworkObservation(
                customer_id="CUSTOMER_001",
                ip_address="10.0.0.1",
                device_id="DEVICE_001",
                timestamp=(
                    now
                    - timedelta(
                        hours=i
                    )
                ),
                claim_type="SHORTAGE_CLAIM",
                requested_amount=1000,
            )
        )

    builder = NetworkFeatureBuilder()

    features = builder.build(
        customer_id="CUSTOMER_001",
        observations=observations,
        current_ip="10.0.0.1",
        current_device_id="DEVICE_001",
        current_claim_type="SHORTAGE_CLAIM",
        current_amount=1000,
        now=now,
    )

    assert (
        features["requests_24h"] == 5
    )

    assert (
        features["requests_7d"] == 5
    )

    assert (
        features["refund_amount_24h"] == 5000
    )


def test_network_behavior_score_is_bounded():

    now = datetime.now()

    observations = []

    for i in range(20):

        observations.append(
            NetworkObservation(
                customer_id=f"CUSTOMER_{i}",
                ip_address="10.0.0.1",
                device_id="DEVICE_001",
                payment_method_id="PAY_001",
                shipping_address_id="ADDR_001",
                timestamp=now,
                claim_type="SHORTAGE_CLAIM",
                requested_amount=10000,
                support_decision="APPROVE",
            )
        )

    builder = NetworkFeatureBuilder()

    features = builder.build(
        customer_id="CUSTOMER_001",
        observations=observations,
        current_ip="10.0.0.1",
        current_device_id="DEVICE_001",
        current_payment_method_id="PAY_001",
        current_shipping_address_id="ADDR_001",
        current_claim_type="SHORTAGE_CLAIM",
        current_amount=10000,
        now=now,
    )

    assert (
        0.0
        <= features[
            "network_behavior_score"
        ]
        <= 1.0
    )