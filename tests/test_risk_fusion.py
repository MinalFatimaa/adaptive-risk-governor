from types import SimpleNamespace

from src.governor.risk_fusion import (
    RiskFusionEngine,
)


# ============================================================
# TEST DATA
# ============================================================

NORMAL_NETWORK = {

    "ip_reuse_score": 0.10,

    "device_reuse_score": 0.08,

    "payment_reuse_score": 0.05,

    "address_reuse_score": 0.20,

    "refund_velocity_score": 0.10,

    "claim_similarity_score": 0.15,

    "network_abnormality_score": 0.05,
}


COORDINATED_NETWORK = {

    "ip_reuse_score": 0.92,

    "device_reuse_score": 0.94,

    "payment_reuse_score": 0.86,

    "address_reuse_score": 0.82,

    "refund_velocity_score": 0.91,

    "claim_similarity_score": 0.88,

    "network_abnormality_score": 0.93,
}


# ============================================================
# GOVERNOR FACTORY
# ============================================================

def fake_governor(
    risk_score=0.20,
    strategic_score=0.20,
):

    return SimpleNamespace(

        risk_score=risk_score,

        risk_level="LOW",

        action=(
            "ALLOW_AGENT_A_DECISION"
        ),

        reason_codes=[],

        features={
            "strategic_adaptation_score": (
                strategic_score
            )
        },
    )


# ============================================================
# NETWORK SCORE
# ============================================================

def test_normal_network_has_low_risk():

    engine = RiskFusionEngine()

    score = (
        engine.calculate_network_risk(
            NORMAL_NETWORK
        )
    )

    assert 0.0 <= score <= 1.0

    assert score < 0.30


def test_coordinated_network_has_high_risk():

    engine = RiskFusionEngine()

    score = (
        engine.calculate_network_risk(
            COORDINATED_NETWORK
        )
    )

    assert 0.0 <= score <= 1.0

    assert score > 0.70


# ============================================================
# NORMAL CUSTOMER
# ============================================================

def test_normal_customer_is_not_high_risk():

    engine = RiskFusionEngine()

    decision = engine.fuse(

        governor_decision=(
            fake_governor(
                risk_score=0.20,
                strategic_score=0.10,
            )
        ),

        network_features=(
            NORMAL_NETWORK
        ),
    )

    assert decision.risk_level != "HIGH"

    assert (
        decision.fused_risk_score
        < 0.70
    )


# ============================================================
# COORDINATED CUSTOMER
# ============================================================

def test_coordinated_customer_becomes_high_risk():

    engine = RiskFusionEngine()

    decision = engine.fuse(

        governor_decision=(
            fake_governor(
                risk_score=0.20,
                strategic_score=0.10,
            )
        ),

        network_features=(
            COORDINATED_NETWORK
        ),
    )

    assert (
        decision.network_risk
        > 0.70
    )

    assert (
        decision.fused_risk_score
        > 0.70
    )

    assert (
        decision.risk_level
        == "HIGH"
    )

    assert (
        decision.action
        == "ESCALATE_TO_HUMAN_REVIEW"
    )


# ============================================================
# STRATEGIC + NETWORK
# ============================================================

def test_strategic_and_network_risk_combine():

    engine = RiskFusionEngine()

    decision = engine.fuse(

        governor_decision=(
            fake_governor(
                risk_score=0.50,
                strategic_score=0.90,
            )
        ),

        network_features=(
            COORDINATED_NETWORK
        ),
    )

    assert (
        decision.strategic_risk
        == 0.90
    )

    assert (
        decision.network_risk
        > 0.70
    )

    assert (
        decision.fused_risk_score
        > 0.70
    )


# ============================================================
# REASONS
# ============================================================

def test_network_reasons_are_generated():

    engine = RiskFusionEngine()

    decision = engine.fuse(

        governor_decision=(
            fake_governor()
        ),

        network_features=(
            COORDINATED_NETWORK
        ),
    )

    assert (
        "HIGH_IP_REUSE"
        in decision.reason_codes
    )

    assert (
        "HIGH_DEVICE_REUSE"
        in decision.reason_codes
    )

    assert (
        "HIGH_REFUND_VELOCITY"
        in decision.reason_codes
    )


# ============================================================
# MISSING FEATURES
# ============================================================

def test_missing_network_features_are_safe():

    engine = RiskFusionEngine()

    decision = engine.fuse(

        governor_decision=(
            fake_governor()
        ),

        network_features={},
    )

    assert (
        decision.network_risk
        == 0.0
    )

    assert (
        decision.risk_level
        == "LOW"
    )


# ============================================================
# BOUNDS
# ============================================================

def test_network_score_is_bounded():

    engine = RiskFusionEngine()

    features = {

        "ip_reuse_score": 10.0,

        "device_reuse_score": 10.0,

        "payment_reuse_score": 10.0,

        "address_reuse_score": 10.0,

        "refund_velocity_score": 10.0,

        "claim_similarity_score": 10.0,

        "network_abnormality_score": 10.0,
    }

    score = (
        engine.calculate_network_risk(
            features
        )
    )

    assert score == 1.0


# ============================================================
# FALSE POSITIVE
# ============================================================

def test_shared_household_does_not_automatically_mean_fraud():

    engine = RiskFusionEngine()

    household_network = {

        "ip_reuse_score": 0.80,

        "device_reuse_score": 0.10,

        "payment_reuse_score": 0.05,

        "address_reuse_score": 0.90,

        "refund_velocity_score": 0.05,

        "claim_similarity_score": 0.05,

        "network_abnormality_score": 0.05,
    }

    decision = engine.fuse(

        governor_decision=(
            fake_governor(
                risk_score=0.10,
                strategic_score=0.05,
            )
        ),

        network_features=(
            household_network
        ),
    )

    assert (
        decision.risk_level
        != "HIGH"
    )