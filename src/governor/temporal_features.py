from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


# ============================================================
# TEMPORAL OBSERVATION
# ============================================================

@dataclass
class TemporalObservation:

    customer_id: str

    timestamp: datetime

    claim_type: str

    requested_amount: float

    support_decision: str | None = None


# ============================================================
# TEMPORAL FEATURE BUILDER
# ============================================================

class TemporalFeatureBuilder:
    """
    Builds time/sequence-based behavioral features.

    This layer answers questions such as:

        - Is the customer suddenly submitting many claims?
        - Are refund requests arriving in bursts?
        - Is the customer repeatedly changing claim types?
        - Is refund value increasing rapidly?
        - Is the behavior concentrated in a short time window?

    It does not decide whether behavior is fraudulent.
    """

    def build(
        self,
        observations: list[TemporalObservation],
        *,
        now: datetime | None = None,
    ) -> dict[str, float]:

        if not observations:

            return {
                "temporal_request_count": 0.0,
                "requests_last_24h": 0.0,
                "requests_last_7d": 0.0,
                "refund_amount_last_24h": 0.0,
                "refund_amount_last_7d": 0.0,
                "request_burst_score": 0.0,
                "claim_switch_rate": 0.0,
                "amount_acceleration_score": 0.0,
                "temporal_abnormality_score": 0.0,
            }

        now = now or max(
            observation.timestamp
            for observation in observations
        )

        ordered = sorted(
            observations,
            key=lambda x: x.timestamp,
        )

        requests_24h = self._count_recent(
            ordered,
            now,
            hours=24,
        )

        requests_7d = self._count_recent(
            ordered,
            now,
            hours=24 * 7,
        )

        amount_24h = self._amount_recent(
            ordered,
            now,
            hours=24,
        )

        amount_7d = self._amount_recent(
            ordered,
            now,
            hours=24 * 7,
        )

        burst_score = self._burst_score(
            ordered
        )

        claim_switch_rate = (
            self._claim_switch_rate(
                ordered
            )
        )

        acceleration_score = (
            self._amount_acceleration(
                ordered
            )
        )

        temporal_abnormality = (
            self._temporal_abnormality(
                requests_24h=requests_24h,
                request_burst_score=burst_score,
                claim_switch_rate=claim_switch_rate,
                amount_acceleration_score=(
                    acceleration_score
                ),
            )
        )

        return {
            "temporal_request_count": float(
                len(ordered)
            ),
            "requests_last_24h": float(
                requests_24h
            ),
            "requests_last_7d": float(
                requests_7d
            ),
            "refund_amount_last_24h": float(
                amount_24h
            ),
            "refund_amount_last_7d": float(
                amount_7d
            ),
            "request_burst_score": float(
                burst_score
            ),
            "claim_switch_rate": float(
                claim_switch_rate
            ),
            "amount_acceleration_score": float(
                acceleration_score
            ),
            "temporal_abnormality_score": float(
                temporal_abnormality
            ),
        }

    # ========================================================
    # RECENT COUNT
    # ========================================================

    @staticmethod
    def _count_recent(
        observations: list[TemporalObservation],
        now: datetime,
        hours: int,
    ) -> int:

        seconds = hours * 60 * 60

        return sum(
            1
            for observation in observations
            if 0
            <= (
                now - observation.timestamp
            ).total_seconds()
            <= seconds
        )

    # ========================================================
    # RECENT AMOUNT
    # ========================================================

    @staticmethod
    def _amount_recent(
        observations: list[TemporalObservation],
        now: datetime,
        hours: int,
    ) -> float:

        seconds = hours * 60 * 60

        return sum(
            max(
                float(
                    observation.requested_amount
                ),
                0.0,
            )
            for observation in observations
            if 0
            <= (
                now - observation.timestamp
            ).total_seconds()
            <= seconds
        )

    # ========================================================
    # BURST SCORE
    # ========================================================

    @staticmethod
    def _burst_score(
        observations: list[TemporalObservation],
    ) -> float:

        if len(observations) < 2:
            return 0.0

        ordered = sorted(
            observations,
            key=lambda x: x.timestamp,
        )

        intervals = []

        for previous, current in zip(
            ordered,
            ordered[1:],
        ):

            interval_minutes = (
                current.timestamp
                - previous.timestamp
            ).total_seconds() / 60.0

            intervals.append(
                max(
                    interval_minutes,
                    0.0,
                )
            )

        if not intervals:
            return 0.0

        short_intervals = sum(
            1
            for interval in intervals
            if interval <= 30
        )

        score = (
            short_intervals
            / len(intervals)
        )

        return max(
            0.0,
            min(
                score,
                1.0,
            ),
        )

    # ========================================================
    # CLAIM SWITCH RATE
    # ========================================================

    @staticmethod
    def _claim_switch_rate(
        observations: list[TemporalObservation],
    ) -> float:

        if len(observations) < 2:
            return 0.0

        ordered = sorted(
            observations,
            key=lambda x: x.timestamp,
        )

        switches = 0

        transitions = len(
            ordered
        ) - 1

        for previous, current in zip(
            ordered,
            ordered[1:],
        ):

            if (
                previous.claim_type
                != current.claim_type
            ):

                switches += 1

        return (
            switches / transitions
        )

    # ========================================================
    # AMOUNT ACCELERATION
    # ========================================================

    @staticmethod
    def _amount_acceleration(
        observations: list[TemporalObservation],
    ) -> float:

        if len(observations) < 3:
            return 0.0

        ordered = sorted(
            observations,
            key=lambda x: x.timestamp,
        )

        amounts = [
            max(
                float(
                    observation.requested_amount
                ),
                0.0,
            )
            for observation in ordered
        ]

        first_half = amounts[
            : len(amounts) // 2
        ]

        second_half = amounts[
            len(amounts) // 2 :
        ]

        if not first_half or not second_half:
            return 0.0

        first_average = (
            sum(first_half)
            / len(first_half)
        )

        second_average = (
            sum(second_half)
            / len(second_half)
        )

        if first_average <= 0:
            return 0.0

        ratio = (
            second_average
            / first_average
        )

        # No acceleration when later requests are
        # approximately the same or smaller.

        if ratio <= 1.0:
            return 0.0

        score = min(
            (ratio - 1.0) / 2.0,
            1.0,
        )

        return score

    # ========================================================
    # COMBINED TEMPORAL ABNORMALITY
    # ========================================================

    @staticmethod
    def _temporal_abnormality(
        *,
        requests_24h: int,
        request_burst_score: float,
        claim_switch_rate: float,
        amount_acceleration_score: float,
    ) -> float:

        velocity_score = min(
            requests_24h / 5.0,
            1.0,
        )

        score = (
            0.35 * velocity_score
            + 0.30 * request_burst_score
            + 0.20 * claim_switch_rate
            + 0.15 * amount_acceleration_score
        )

        return max(
            0.0,
            min(
                score,
                1.0,
            ),
        )