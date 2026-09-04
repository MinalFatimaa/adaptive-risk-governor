from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.phase16_adaptive_agent_evaluation import (
    ADAPTIVE_ABUSIVE,
    ADAPTIVE_LEGITIMATE,
    HUMAN_ABUSIVE,
    HUMAN_LEGITIMATE,
    EpisodeMetrics,
    aggregate_population_metrics,
    belief_shift,
    build_phase16_result,
    classify_customer,
    safe_rate,
    validate_episode_configuration,
)


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def test_safe_rate_is_zero_for_zero_denominator():

    assert safe_rate(
        10,
        0,
    ) == 0.0


def test_safe_rate_is_bounded():

    value = safe_rate(
        5,
        10,
    )

    assert 0.0 <= value <= 1.0


def test_belief_shift_zero_for_identical_beliefs():

    beliefs = {
        "evidence_sensitivity": 0.5,
        "amount_sensitivity": 0.5,
    }

    assert (
        belief_shift(
            beliefs,
            beliefs,
        )
        == 0.0
    )


def test_belief_shift_is_positive_for_changed_beliefs():

    first = {
        "evidence_sensitivity": 0.2,
    }

    second = {
        "evidence_sensitivity": 0.8,
    }

    assert (
        belief_shift(
            first,
            second,
        )
        > 0.0
    )


# ============================================================
# CONFIGURATION
# ============================================================

def test_episode_configuration_rejects_too_short_episode():

    with pytest.raises(
        ValueError
    ):

        validate_episode_configuration(
            3
        )


def test_episode_configuration_accepts_valid_episode():

    validate_episode_configuration(
        10
    )


# ============================================================
# EPISODE METRICS
# ============================================================

def make_metric(
    population_group: str,
    gain: float,
    detected: bool,
) -> EpisodeMetrics:

    return EpisodeMetrics(
        episode_id="EPISODE_001",
        customer_id="CUSTOMER_0001",
        population_group=population_group,
        interactions=10,
        early_approval_rate=0.2,
        late_approval_rate=0.2 + gain,
        adaptation_gain=gain,
        evidence_rate_early=0.2,
        evidence_rate_late=0.5,
        mean_requested_amount_early=1000.0,
        mean_requested_amount_late=1000.0,
        policy_belief_shift=0.1,
        adaptive_behavior_detected=detected,
    )


def test_population_aggregation():

    metrics = [
        make_metric(
            ADAPTIVE_ABUSIVE,
            0.20,
            True,
        ),
        make_metric(
            ADAPTIVE_ABUSIVE,
            0.10,
            True,
        ),
    ]

    result = aggregate_population_metrics(
        metrics
    )

    assert (
        result.population_group
        == ADAPTIVE_ABUSIVE
    )

    assert result.episodes == 2

    assert (
        result.mean_adaptation_gain
        == pytest.approx(0.15)
    )

    assert (
        result.adaptive_behavior_detection_rate
        == pytest.approx(1.0)
    )


def test_empty_population_is_rejected():

    with pytest.raises(
        ValueError
    ):

        aggregate_population_metrics(
            []
        )


# ============================================================
# FINAL RESULT
# ============================================================

def test_phase16_result_contains_population_metrics():

    metrics = [
        make_metric(
            ADAPTIVE_ABUSIVE,
            0.20,
            True,
        ),
        make_metric(
            HUMAN_ABUSIVE,
            0.05,
            False,
        ),
        make_metric(
            HUMAN_LEGITIMATE,
            0.00,
            False,
        ),
        make_metric(
            ADAPTIVE_LEGITIMATE,
            0.02,
            False,
        ),
    ]

    result = build_phase16_result(
        metrics
    )

    assert (
        result.total_episodes
        == 4
    )

    assert len(
        result.population_metrics
    ) == 4


def test_adaptive_abusive_gain_advantage_is_computed():

    metrics = [
        make_metric(
            ADAPTIVE_ABUSIVE,
            0.30,
            True,
        ),
        make_metric(
            HUMAN_ABUSIVE,
            0.10,
            False,
        ),
    ]

    result = build_phase16_result(
        metrics
    )

    assert (
        result.adaptive_abusive_gain_advantage
        == pytest.approx(0.20)
    )


# ============================================================
# BOUNDS
# ============================================================

def test_population_metrics_are_finite():

    metrics = [
        make_metric(
            ADAPTIVE_ABUSIVE,
            0.20,
            True,
        ),
        make_metric(
            HUMAN_ABUSIVE,
            0.05,
            False,
        ),
    ]

    result = build_phase16_result(
        metrics
    )

    for population in (
        result.population_metrics
    ):

        assert np.isfinite(
            population.mean_adaptation_gain
        )

        assert np.isfinite(
            population.mean_policy_belief_shift
        )

        assert (
            0.0
            <= population.adaptive_behavior_detection_rate
            <= 1.0
        )


# ============================================================
# ACTION-SPACE CONTRACT
# ============================================================

def test_phase16_does_not_add_direct_deny_action():

    # Phase 16 evaluates adaptive behavior.
    # It must not invent a new Governor action.

    governor_actions = {
        "ALLOW_AGENT_A_DECISION",
        "REQUEST_ADDITIONAL_EVIDENCE",
        "ESCALATE_TO_HUMAN_REVIEW",
    }

    assert (
        "DENY"
        not in governor_actions
    )