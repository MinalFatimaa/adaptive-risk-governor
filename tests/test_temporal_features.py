from datetime import datetime, timedelta

from src.governor.temporal_features import (
    TemporalFeatureBuilder,
    TemporalObservation,
)


def test_empty_observations_return_zero_features():

    builder = TemporalFeatureBuilder()

    features = builder.build([])

    assert (
        features[
            "temporal_request_count"
        ]
        == 0.0
    )

    assert (
        features[
            "temporal_abnormality_score"
        ]
        == 0.0
    )


def test_recent_request_count():

    now = datetime(
        2026,
        8,
        26,
        12,
        0,
        0,
    )

    observations = [

        TemporalObservation(
            customer_id="C1",
            timestamp=(
                now
                - timedelta(hours=1)
            ),
            claim_type="SHORTAGE_CLAIM",
            requested_amount=1000,
        ),

        TemporalObservation(
            customer_id="C1",
            timestamp=(
                now
                - timedelta(hours=12)
            ),
            claim_type="WRONG_ITEM_CLAIM",
            requested_amount=1200,
        ),

        TemporalObservation(
            customer_id="C1",
            timestamp=(
                now
                - timedelta(days=3)
            ),
            claim_type="NON_DELIVERY_CLAIM",
            requested_amount=1500,
        ),
    ]

    builder = TemporalFeatureBuilder()

    features = builder.build(
        observations,
        now=now,
    )

    assert (
        features[
            "requests_last_24h"
        ]
        == 2
    )

    assert (
        features[
            "requests_last_7d"
        ]
        == 3
    )


def test_burst_behavior_detected():

    now = datetime(
        2026,
        8,
        26,
        12,
        0,
        0,
    )

    observations = []

    for i in range(6):

        observations.append(
            TemporalObservation(
                customer_id="C1",
                timestamp=(
                    now
                    - timedelta(
                        minutes=i * 10
                    )
                ),
                claim_type="SHORTAGE_CLAIM",
                requested_amount=1000,
            )
        )

    builder = TemporalFeatureBuilder()

    features = builder.build(
        observations,
        now=now,
    )

    assert (
        features[
            "request_burst_score"
        ]
        > 0.80
    )


def test_claim_switching_detected():

    now = datetime(
        2026,
        8,
        26,
        12,
        0,
        0,
    )

    claims = [
        "SHORTAGE_CLAIM",
        "WRONG_ITEM_CLAIM",
        "NON_DELIVERY_CLAIM",
        "SUBSTITUTED_RETURN_CLAIM",
    ]

    observations = [

        TemporalObservation(
            customer_id="C1",
            timestamp=(
                now
                - timedelta(
                    minutes=30 - i * 10
                )
            ),
            claim_type=claim,
            requested_amount=1000,
        )

        for i, claim in enumerate(claims)
    ]

    builder = TemporalFeatureBuilder()

    features = builder.build(
        observations,
        now=now,
    )

    assert (
        features[
            "claim_switch_rate"
        ]
        > 0.90
    )


def test_amount_acceleration_detected():

    now = datetime(
        2026,
        8,
        26,
        12,
        0,
        0,
    )

    amounts = [
        1000,
        1100,
        1200,
        2500,
        4000,
        5000,
    ]

    observations = [

        TemporalObservation(
            customer_id="C1",
            timestamp=(
                now
                - timedelta(
                    hours=6 - i
                )
            ),
            claim_type="SHORTAGE_CLAIM",
            requested_amount=amount,
        )

        for i, amount in enumerate(amounts)
    ]

    builder = TemporalFeatureBuilder()

    features = builder.build(
        observations,
        now=now,
    )

    assert (
        features[
            "amount_acceleration_score"
        ]
        > 0.20
    )


def test_temporal_abnormality_is_bounded():

    now = datetime(
        2026,
        8,
        26,
        12,
        0,
        0,
    )

    observations = [

        TemporalObservation(
            customer_id="C1",
            timestamp=(
                now
                - timedelta(
                    minutes=i
                )
            ),
            claim_type="SHORTAGE_CLAIM",
            requested_amount=10000,
        )

        for i in range(20)
    ]

    builder = TemporalFeatureBuilder()

    features = builder.build(
        observations,
        now=now,
    )

    assert (
        0.0
        <= features[
            "temporal_abnormality_score"
        ]
        <= 1.0
    )