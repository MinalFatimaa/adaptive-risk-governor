from __future__ import annotations

"""
PHASE 18 UI ADAPTER

This module intentionally contains NO risk logic.

The canonical implementation remains:

    src.evaluation.phase18_gnn_governor_evaluation

The UI calls:

    evaluate_phase18(seed=...)

and receives the real Phase18Result.

This file exists only to keep Streamlit presentation code separate
from the evaluation module.
"""

import sys
from pathlib import Path
from typing import Any


# ============================================================================
# REPOSITORY PATH
# ============================================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CANONICAL PHASE 18 IMPORT
# ============================================================================

from src.evaluation.phase18_gnn_governor_evaluation import (  # noqa: E402
    Phase18Result,
    evaluate_phase18,
)


# ============================================================================
# PUBLIC UI API
# ============================================================================

def run_live_evaluation(
    *,
    seed: int = 42,
) -> Phase18Result:
    """
    Run the real Phase 18 evaluation.

    No calculations are performed here.

    No metrics are reconstructed here.

    No console output is parsed here.

    The canonical Phase 18 evaluator is the source of truth.
    """

    return evaluate_phase18(
        seed=int(seed),
    )


# ============================================================================
# PRESENTATION SERIALISATION
# ============================================================================

def result_to_dict(
    result: Phase18Result,
) -> dict[str, Any]:
    """
    Convert the real Phase18Result into a UI/debug-friendly dictionary.

    This performs no calculations.
    """

    return {
        "total_episodes": result.total_episodes,

        "adaptive_abusive_baseline_approval": (
            result.adaptive_abusive_baseline_approval
        ),

        "adaptive_abusive_governed_approval": (
            result.adaptive_abusive_governed_approval
        ),

        "human_abusive_baseline_approval": (
            result.human_abusive_baseline_approval
        ),

        "human_abusive_governed_approval": (
            result.human_abusive_governed_approval
        ),

        "adaptive_abusive_baseline_value": (
            result.adaptive_abusive_baseline_value
        ),

        "adaptive_abusive_governed_value": (
            result.adaptive_abusive_governed_value
        ),

        "human_abusive_baseline_value": (
            result.human_abusive_baseline_value
        ),

        "human_abusive_governed_value": (
            result.human_abusive_governed_value
        ),

        "adaptive_abusive_prevented_value": (
            result.adaptive_abusive_prevented_value
        ),

        "human_abusive_prevented_value": (
            result.human_abusive_prevented_value
        ),

        "adaptive_abusive_intervention_rate": (
            result.adaptive_abusive_intervention_rate
        ),

        "adaptive_legitimate_intervention_rate": (
            result.adaptive_legitimate_intervention_rate
        ),

        "human_abusive_intervention_rate": (
            result.human_abusive_intervention_rate
        ),

        "human_legitimate_intervention_rate": (
            result.human_legitimate_intervention_rate
        ),

        "adaptive_governor_advantage": (
            result.adaptive_governor_advantage
        ),

        "adaptive_vs_human_intervention_gap": (
            result.adaptive_vs_human_intervention_gap
        ),

        "adaptive_policy_change": (
            result.adaptive_policy_change
        ),

        "population_metrics": [
            {
                "population_group": metric.population_group,
                "episodes": metric.episodes,
                "baseline_approval_rate": (
                    metric.baseline_approval_rate
                ),
                "governed_approval_rate": (
                    metric.governed_approval_rate
                ),
                "approval_reduction": (
                    metric.approval_reduction
                ),
                "baseline_approved_value": (
                    metric.baseline_approved_value
                ),
                "governed_approved_value": (
                    metric.governed_approved_value
                ),
                "prevented_approval_value": (
                    metric.prevented_approval_value
                ),
                "intervention_rate": (
                    metric.intervention_rate
                ),
                "evidence_request_rate": (
                    metric.evidence_request_rate
                ),
                "human_review_rate": (
                    metric.human_review_rate
                ),
                "mean_gnn_risk": (
                    metric.mean_gnn_risk
                ),
                "maximum_gnn_risk": (
                    metric.maximum_gnn_risk
                ),
                "mean_fused_risk": (
                    metric.mean_fused_risk
                ),
                "maximum_fused_risk": (
                    metric.maximum_fused_risk
                ),
                "mean_policy_change": (
                    metric.mean_policy_change
                ),
                "mean_detection_latency": (
                    metric.mean_detection_latency
                ),
            }
            for metric in result.population_metrics
        ],
    }