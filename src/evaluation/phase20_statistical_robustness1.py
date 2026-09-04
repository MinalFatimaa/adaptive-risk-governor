"""
Phase 20 — Statistical Robustness Evaluation
=============================================

Purpose
-------
Evaluate the existing Phase 18 GNN Governor closed-loop experiment
across multiple deterministic seeds.

IMPORTANT DESIGN PRINCIPLES
---------------------------
1. Phase 18 is NOT modified.
2. Phase 18 console output is NEVER parsed.
3. Metrics are NEVER fabricated.
4. Phase 20 uses the Python objects returned by Phase 18.
5. build_phase18_result() is treated as Phase 18's public summary
   contract.
6. Metrics not exposed by Phase 18 are represented as NaN / N/A.
7. Approval reduction is derived only when both baseline and governed
   approval are genuinely available:

       reduction = baseline_approval - governed_approval

8. Legitimate populations are evaluated primarily through the
   intervention-rate signal that Phase 18 actually exposes.
9. Missing GNN/fusion population metrics remain unavailable rather
   than being converted to zero.
10. A missing metric is not equivalent to a failed experiment.

Run
---
    python -m src.evaluation.phase20_statistical_robustness
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import random
import re
import statistics
from dataclasses import dataclass
from typing import Any


# ============================================================================
# CONFIGURATION
# ============================================================================

PHASE18_MODULE = (
    "src.evaluation.phase18_gnn_governor_evaluation"
)

# Seeds used for statistical robustness.
SEEDS: tuple[int, ...] = (
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
)

# Number of expected population groups.
POPULATIONS: tuple[str, ...] = (
    "ADAPTIVE_ABUSIVE",
    "ADAPTIVE_LEGITIMATE",
    "HUMAN_ABUSIVE",
    "HUMAN_LEGITIMATE",
)

ABUSIVE_POPULATIONS: tuple[str, ...] = (
    "ADAPTIVE_ABUSIVE",
    "HUMAN_ABUSIVE",
)

LEGITIMATE_POPULATIONS: tuple[str, ...] = (
    "ADAPTIVE_LEGITIMATE",
    "HUMAN_LEGITIMATE",
)

# Population aliases are used only for locating fields that Phase 18
# actually exposes. They are NOT used to infer missing metrics.
POPULATION_ALIASES: dict[str, tuple[str, ...]] = {
    "ADAPTIVE_ABUSIVE": (
        "adaptive_abusive",
        "adaptiveabusive",
        "adaptive_abuse",
        "adaptiveabuse",
    ),
    "ADAPTIVE_LEGITIMATE": (
        "adaptive_legitimate",
        "adaptivelegitimate",
        "adaptive_legit",
        "adaptivelegit",
    ),
    "HUMAN_ABUSIVE": (
        "human_abusive",
        "humanabusive",
        "human_abuse",
        "humanabuse",
    ),
    "HUMAN_LEGITIMATE": (
        "human_legitimate",
        "humanlegitimate",
        "human_legit",
        "humanlegit",
    ),
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PopulationMetrics:
    """
    Metrics that Phase 20 can legitimately obtain for one population.

    NaN means:
        Phase 18 did not expose this metric through its returned
        Python result object.

    It does NOT mean zero.
    """

    population: str

    baseline_approval: float = math.nan
    governed_approval: float = math.nan
    reduction: float = math.nan

    intervention_rate: float = math.nan

    baseline_value: float = math.nan
    governed_value: float = math.nan
    prevented_value: float = math.nan

    gnn_risk: float = math.nan
    fused_risk: float = math.nan


@dataclass
class SeedMetrics:
    """All Phase 20 metrics recovered from one Phase 18 seed."""

    seed: int

    populations: dict[str, PopulationMetrics]

    adaptive_policy_change: float = math.nan
    adaptive_governor_advantage: float = math.nan
    adaptive_vs_human_intervention_gap: float = math.nan

    human_abusive_prevented_value: float = math.nan

    expected_episodes_per_group: int = 0
    expected_total_episodes: int = 0


@dataclass
class SummaryStats:
    """Descriptive statistics for one metric across seeds."""

    metric: str
    population: str | None

    n: int
    mean: float
    std: float
    minimum: float
    maximum: float
    cv: float


# ============================================================================
# BASIC HELPERS
# ============================================================================

def _normalise_name(value: Any) -> str:
    """
    Normalize names for tolerant matching.

    Example:
        adaptive_abusive_baseline_approval
        ->
        adaptiveabusivebaselineapproval
    """

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


def _is_scalar(value: Any) -> bool:
    """Return whether a value is a terminal scalar."""

    return (
        value is None
        or isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
                bytes,
            ),
        )
    )


def _as_float(value: Any) -> float:
    """
    Convert a scalar numeric value to float.

    Raises a clear error for non-numeric values.
    """

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    raise TypeError(
        f"Expected numeric scalar, got "
        f"{type(value).__name__}: {value!r}"
    )


def _finite_or_nan(value: Any) -> float:
    """
    Convert numeric values to float.

    Non-finite values become NaN.
    """

    try:
        result = _as_float(value)
    except (TypeError, ValueError):
        return math.nan

    if not math.isfinite(result):
        return math.nan

    return result


def _is_finite_number(value: float) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


# ============================================================================
# PYTHON OBJECT INSPECTION
# ============================================================================

def _object_items(obj: Any):
    """
    Yield named fields from dictionaries and ordinary Python objects.

    This deliberately does not inspect arbitrary Python internals.
    """

    if isinstance(obj, dict):

        for key, value in obj.items():
            yield str(key), value

        return

    if dataclasses.is_dataclass(obj):

        for field in dataclasses.fields(obj):

            try:
                value = getattr(
                    obj,
                    field.name,
                )
            except Exception:
                continue

            yield field.name, value

        return

    if hasattr(obj, "__dict__"):

        try:
            values = vars(obj)
        except TypeError:
            return

        for key, value in values.items():

            if str(key).startswith("__"):
                continue

            yield str(key), value


def _flatten_scalars(
    obj: Any,
    *,
    path: str = "root",
    visited: set[int] | None = None,
    max_depth: int = 12,
) -> list[tuple[str, Any]]:
    """
    Flatten scalar leaves from the returned Phase 18 Python object.

    Containers are traversed, but only scalar leaves are returned.

    This function does NOT calculate any metrics.
    """

    if visited is None:
        visited = set()

    if max_depth < 0:
        return []

    if _is_scalar(obj):
        return [(path, obj)]

    object_id = id(obj)

    if object_id in visited:
        return []

    visited.add(object_id)

    output: list[tuple[str, Any]] = []

    if isinstance(obj, (list, tuple)):

        for index, value in enumerate(obj):

            output.extend(
                _flatten_scalars(
                    value,
                    path=f"{path}[{index}]",
                    visited=visited,
                    max_depth=max_depth - 1,
                )
            )

        return output

    for name, value in _object_items(obj):

        child_path = f"{path}.{name}"

        output.extend(
            _flatten_scalars(
                value,
                path=child_path,
                visited=visited,
                max_depth=max_depth - 1,
            )
        )

    return output


# ============================================================================
# FIELD RESOLUTION
# ============================================================================

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "baseline_approval": (
        "baseline_approval",
        "baseline_approval_rate",
        "base_approval",
        "baselineapproval",
        "baseapproval",
    ),
    "governed_approval": (
        "governed_approval",
        "governed_approval_rate",
        "gov_approval",
        "governedapproval",
        "govapproval",
    ),
    "intervention_rate": (
        "intervention_rate",
        "interventionrate",
        "intervention",
    ),
    "baseline_value": (
        "baseline_value",
        "baselinevalue",
        "base_value",
        "basevalue",
    ),
    "governed_value": (
        "governed_value",
        "governedvalue",
        "gov_value",
        "govvalue",
    ),
    "prevented_value": (
        "prevented_value",
        "preventedvalue",
        "value_prevented",
        "valueprevented",
    ),
    "gnn_risk": (
        "gnn_risk",
        "gnnrisk",
        "gnn_score",
        "gnnscore",
    ),
    "fused_risk": (
        "fused_risk",
        "fusedrisk",
        "fusion_risk",
        "fusionrisk",
        "risk_fusion",
        "riskfusion",
    ),
}


def _field_matches_population(
    path: str,
    population: str,
) -> bool:
    """Check whether a path explicitly identifies a population."""

    normalized_path = _normalise_name(path)

    aliases = POPULATION_ALIASES[
        population
    ]

    return any(
        _normalise_name(alias) in normalized_path
        for alias in aliases
    )


def _field_matches_metric(
    path: str,
    metric: str,
) -> bool:
    """Check whether a path explicitly identifies a metric."""

    normalized_path = _normalise_name(path)

    aliases = METRIC_ALIASES[
        metric
    ]

    return any(
        _normalise_name(alias) in normalized_path
        for alias in aliases
    )


def _candidate_score(
    path: str,
    population: str,
    metric: str,
) -> int:
    """
    Score a matching field.

    Exact normalized population + metric combinations are preferred
    over looser substring matches.
    """

    normalized_path = _normalise_name(path)

    score = 0

    population_aliases = POPULATION_ALIASES[
        population
    ]

    metric_aliases = METRIC_ALIASES[
        metric
    ]

    for alias in population_aliases:

        normalized_alias = _normalise_name(
            alias
        )

        if normalized_path == normalized_alias:
            score += 100

        elif normalized_alias in normalized_path:
            score += 30

    for alias in metric_aliases:

        normalized_alias = _normalise_name(
            alias
        )

        if normalized_alias in normalized_path:
            score += 20

    # Prefer shallower paths.
    score -= path.count(".")
    score -= path.count("[")

    return score


def _resolve_population_metric(
    flattened: list[tuple[str, Any]],
    population: str,
    metric: str,
) -> float:
    """
    Resolve a metric only if Phase 18 actually exposed it.

    Returns NaN when unavailable.

    IMPORTANT:
    No default zero is used.
    """

    candidates: list[
        tuple[int, str, float]
    ] = []

    for path, value in flattened:

        if not _field_matches_population(
            path,
            population,
        ):
            continue

        if not _field_matches_metric(
            path,
            metric,
        ):
            continue

        if not isinstance(
            value,
            (int, float),
        ):
            continue

        numeric_value = _finite_or_nan(
            value
        )

        if math.isnan(numeric_value):
            continue

        candidates.append(
            (
                _candidate_score(
                    path,
                    population,
                    metric,
                ),
                path,
                numeric_value,
            )
        )

    if not candidates:
        return math.nan

    candidates.sort(
        key=lambda item: (
            -item[0],
            len(item[1]),
        )
    )

    return candidates[0][2]


# ============================================================================
# GLOBAL FIELD RESOLUTION
# ============================================================================

GLOBAL_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "adaptive_policy_change": (
        "adaptive_policy_change",
        "adaptivepolicychange",
        "adaptive_attacker_policy_change",
        "adaptiveattackerpolicychange",
        "attacker_policy_change",
        "attackerpolicychange",
    ),
    "adaptive_governor_advantage": (
        "adaptive_governor_advantage",
        "adaptivegovernoradvantage",
        "governor_advantage",
        "governoradvantage",
    ),
    "adaptive_vs_human_intervention_gap": (
        "adaptive_vs_human_intervention_gap",
        "adaptivevshumaninterventiongap",
        "adaptive_human_intervention_gap",
        "adaptivehumaninterventiongap",
        "intervention_gap",
        "interventiongap",
    ),
    "human_abusive_prevented_value": (
        "human_abusive_prevented_value",
        "humanabusivepreventedvalue",
        "human_abuse_prevented_value",
        "humanabusepreventedvalue",
    ),
}


def _resolve_global_metric(
    flattened: list[tuple[str, Any]],
    metric: str,
) -> float:
    """Resolve a globally exposed Phase 18 metric."""

    aliases = GLOBAL_METRIC_ALIASES[
        metric
    ]

    candidates: list[
        tuple[int, str, float]
    ] = []

    for path, value in flattened:

        if not isinstance(
            value,
            (int, float),
        ):
            continue

        normalized_path = _normalise_name(
            path
        )

        matched_alias = None

        for alias in aliases:

            normalized_alias = _normalise_name(
                alias
            )

            if normalized_alias in normalized_path:
                matched_alias = normalized_alias
                break

        if matched_alias is None:
            continue

        numeric_value = _finite_or_nan(
            value
        )

        if math.isnan(numeric_value):
            continue

        score = 20

        if normalized_path == matched_alias:
            score += 100

        score -= path.count(".")
        score -= path.count("[")

        candidates.append(
            (
                score,
                path,
                numeric_value,
            )
        )

    if not candidates:
        return math.nan

    candidates.sort(
        key=lambda item: (
            -item[0],
            len(item[1]),
        )
    )

    return candidates[0][2]


# ============================================================================
# PHASE 18 IMPORT
# ============================================================================

def _load_phase18():
    """
    Import Phase 18 without executing its main() function.
    """

    try:

        return importlib.import_module(
            PHASE18_MODULE
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not import Phase 18 module:\n"
            f"  {PHASE18_MODULE}\n\n"
            "Phase 20 cannot continue because the "
            "existing Phase 18 implementation could "
            "not be imported."
        ) from exc


# ============================================================================
# PHASE 18 EXECUTION
# ============================================================================

def _call_phase18_simulation(
    phase18: Any,
    world: Any,
    seed: int,
) -> Any:
    """
    Call the existing Phase 18 simulation directly.

    No console output is captured or parsed.
    """

    return phase18.run_phase18_simulation(
        world=world,
        episodes_per_population=(
            phase18.EPISODES_PER_GROUP
        ),
        interactions_per_episode=(
            phase18.INTERACTIONS_PER_EPISODE
        ),
        seed=seed,
    )


def _run_phase18_for_seed(
    phase18: Any,
    seed: int,
) -> SeedMetrics:
    """
    Execute the complete Phase 18 Python pipeline for one seed.

    Execution path:

        create_world()
            ↓
        build_interaction_graph()
            ↓
        run_phase18_simulation()
            ↓
        build_phase18_result()
            ↓
        Phase 20 extraction
    """

    print()
    print("=" * 110)
    print(
        f"RUNNING PHASE 18 — SEED {seed}"
    )
    print("=" * 110)

    # --------------------------------------------------------
    # Deterministic Python-level random state
    #
    # Phase 18 also receives seed explicitly.
    # These calls do not modify Phase 18 logic.
    # --------------------------------------------------------

    random.seed(seed)

    try:

        import numpy as np

        np.random.seed(seed)

    except Exception:
        pass

    # --------------------------------------------------------
    # World
    # --------------------------------------------------------

    print(
        "Creating deterministic mixed-population world..."
    )

    world = phase18.create_world(
        config=phase18.DEFAULT_POPULATION_CONFIG,
        seed=seed,
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    print(
        "Building interaction graph..."
    )

    graph = phase18.build_interaction_graph(
        world
    )

    if graph is None:

        raise RuntimeError(
            "Phase 18 returned None for the interaction graph."
        )

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    print(
        "Building and evaluating GNN Governor "
        "closed-loop pipeline..."
    )

    metrics = _call_phase18_simulation(
        phase18=phase18,
        world=world,
        seed=seed,
    )

    # --------------------------------------------------------
    # Public Phase 18 summary
    # --------------------------------------------------------

    result = phase18.build_phase18_result(
        metrics
    )

    # --------------------------------------------------------
    # Flatten only the returned Python objects.
    #
    # We inspect both metrics and result because:
    #
    #   metrics = lower-level simulation return
    #   result  = Phase 18 public summary
    #
    # Neither console output nor printed reports are used.
    # --------------------------------------------------------

    result_flat = _flatten_scalars(
        result,
        path="result",
    )

    metrics_flat = _flatten_scalars(
        metrics,
        path="metrics",
    )

    # The result is intentionally first because it is the public
    # summary contract. Metrics is retained only as a secondary
    # Python-level source in case a scalar is exposed there.
    flattened = (
        result_flat
        + metrics_flat
    )

    # --------------------------------------------------------
    # Population metrics
    # --------------------------------------------------------

    population_metrics: dict[
        str,
        PopulationMetrics,
    ] = {}

    for population in POPULATIONS:

        baseline_approval = (
            _resolve_population_metric(
                flattened,
                population,
                "baseline_approval",
            )
        )

        governed_approval = (
            _resolve_population_metric(
                flattened,
                population,
                "governed_approval",
            )
        )

        intervention_rate = (
            _resolve_population_metric(
                flattened,
                population,
                "intervention_rate",
            )
        )

        baseline_value = (
            _resolve_population_metric(
                flattened,
                population,
                "baseline_value",
            )
        )

        governed_value = (
            _resolve_population_metric(
                flattened,
                population,
                "governed_value",
            )
        )

        prevented_value = (
            _resolve_population_metric(
                flattened,
                population,
                "prevented_value",
            )
        )

        gnn_risk = (
            _resolve_population_metric(
                flattened,
                population,
                "gnn_risk",
            )
        )

        fused_risk = (
            _resolve_population_metric(
                flattened,
                population,
                "fused_risk",
            )
        )

        # ----------------------------------------------------
        # Reduction is DERIVED ONLY when both source metrics
        # genuinely exist.
        # ----------------------------------------------------

        if (
            _is_finite_number(
                baseline_approval
            )
            and _is_finite_number(
                governed_approval
            )
        ):

            reduction = (
                baseline_approval
                - governed_approval
            )

        else:

            reduction = math.nan

        population_metrics[
            population
        ] = PopulationMetrics(
            population=population,
            baseline_approval=(
                baseline_approval
            ),
            governed_approval=(
                governed_approval
            ),
            reduction=reduction,
            intervention_rate=(
                intervention_rate
            ),
            baseline_value=(
                baseline_value
            ),
            governed_value=(
                governed_value
            ),
            prevented_value=(
                prevented_value
            ),
            gnn_risk=gnn_risk,
            fused_risk=fused_risk,
        )

    # --------------------------------------------------------
    # Global metrics
    # --------------------------------------------------------

    adaptive_policy_change = (
        _resolve_global_metric(
            flattened,
            "adaptive_policy_change",
        )
    )

    adaptive_governor_advantage = (
        _resolve_global_metric(
            flattened,
            "adaptive_governor_advantage",
        )
    )

    adaptive_vs_human_intervention_gap = (
        _resolve_global_metric(
            flattened,
            "adaptive_vs_human_intervention_gap",
        )
    )

    human_abusive_prevented_value = (
        _resolve_global_metric(
            flattened,
            "human_abusive_prevented_value",
        )
    )

    # --------------------------------------------------------
    # Experiment configuration
    # --------------------------------------------------------

    episodes_per_group = int(
        getattr(
            phase18,
            "EPISODES_PER_GROUP",
            0,
        )
    )

    expected_total_episodes = (
        episodes_per_group
        * len(POPULATIONS)
    )

    return SeedMetrics(
        seed=seed,
        populations=population_metrics,
        adaptive_policy_change=(
            adaptive_policy_change
        ),
        adaptive_governor_advantage=(
            adaptive_governor_advantage
        ),
        adaptive_vs_human_intervention_gap=(
            adaptive_vs_human_intervention_gap
        ),
        human_abusive_prevented_value=(
            human_abusive_prevented_value
        ),
        expected_episodes_per_group=(
            episodes_per_group
        ),
        expected_total_episodes=(
            expected_total_episodes
        ),
    )


# ============================================================================
# STATISTICAL HELPERS
# ============================================================================

def _finite_values(
    values: list[float],
) -> list[float]:
    """Return only finite numeric values."""

    return [
        float(value)
        for value in values
        if _is_finite_number(value)
    ]


def _coefficient_of_variation(
    mean: float,
    std: float,
) -> float:
    """
    Calculate CV.

    CV is undefined when mean == 0, so return NaN.
    """

    if not (
        _is_finite_number(mean)
        and _is_finite_number(std)
    ):
        return math.nan

    if mean == 0.0:
        return math.nan

    return abs(std / mean)


def summarize_metric(
    values: list[float],
    *,
    metric: str,
    population: str | None = None,
) -> SummaryStats:

    finite = _finite_values(
        values
    )

    if not finite:

        return SummaryStats(
            metric=metric,
            population=population,
            n=0,
            mean=math.nan,
            std=math.nan,
            minimum=math.nan,
            maximum=math.nan,
            cv=math.nan,
        )

    mean = statistics.fmean(
        finite
    )

    if len(finite) >= 2:

        std = statistics.stdev(
            finite
        )

    else:

        std = 0.0

    return SummaryStats(
        metric=metric,
        population=population,
        n=len(finite),
        mean=mean,
        std=std,
        minimum=min(finite),
        maximum=max(finite),
        cv=_coefficient_of_variation(
            mean,
            std,
        ),
    )


def _format_number(
    value: float,
    digits: int = 4,
) -> str:

    if not _is_finite_number(value):
        return "N/A"

    return f"{value:.{digits}f}"


def _format_percent(
    value: float,
    digits: int = 2,
) -> str:

    if not _is_finite_number(value):
        return "N/A"

    return (
        f"{value * 100:.{digits}f}%"
    )


def _format_currency(
    value: float,
    digits: int = 2,
) -> str:

    if not _is_finite_number(value):
        return "N/A"

    return (
        f"₹{value:,.{digits}f}"
    )


# ============================================================================
# SUMMARY EXTRACTION
# ============================================================================

def _population_metric_values(
    results: list[SeedMetrics],
    population: str,
    metric: str,
) -> list[float]:

    output: list[float] = []

    for result in results:

        population_result = (
            result.populations[
                population
            ]
        )

        output.append(
            getattr(
                population_result,
                metric,
            )
        )

    return output


def _global_metric_values(
    results: list[SeedMetrics],
    metric: str,
) -> list[float]:

    return [
        getattr(
            result,
            metric,
        )
        for result in results
    ]


# ============================================================================
# VALIDATION
# ============================================================================

def _validate_rate(
    value: float,
    *,
    metric: str,
    population: str,
    seed: int,
) -> list[str]:

    errors: list[str] = []

    if math.isnan(value):
        return errors

    if not math.isfinite(value):

        errors.append(
            f"seed={seed}, "
            f"{population}.{metric} is non-finite"
        )

        return errors

    if value < 0.0 or value > 1.0:

        errors.append(
            f"seed={seed}, "
            f"{population}.{metric}={value:.6f} "
            f"is outside [0, 1]"
        )

    return errors


def validate_seed_result(
    result: SeedMetrics,
) -> list[str]:

    errors: list[str] = []

    # --------------------------------------------------------
    # Population rate validation
    # --------------------------------------------------------

    for population in POPULATIONS:

        metrics = result.populations[
            population
        ]

        errors.extend(
            _validate_rate(
                metrics.baseline_approval,
                metric="baseline_approval",
                population=population,
                seed=result.seed,
            )
        )

        errors.extend(
            _validate_rate(
                metrics.governed_approval,
                metric="governed_approval",
                population=population,
                seed=result.seed,
            )
        )

        errors.extend(
            _validate_rate(
                metrics.intervention_rate,
                metric="intervention_rate",
                population=population,
                seed=result.seed,
            )
        )

        # ----------------------------------------------------
        # Reduction is a difference of two [0,1] rates.
        #
        # Therefore the mathematically valid range is [-1,1].
        # ----------------------------------------------------

        if not math.isnan(
            metrics.reduction
        ):

            if (
                not math.isfinite(
                    metrics.reduction
                )
                or metrics.reduction < -1.0
                or metrics.reduction > 1.0
            ):

                errors.append(
                    f"seed={result.seed}, "
                    f"{population}.reduction="
                    f"{metrics.reduction:.6f} "
                    f"is outside [-1, 1]"
                )

        # ----------------------------------------------------
        # Value metrics only need to be finite when exposed.
        # We deliberately do not impose non-negative assumptions
        # on them because the simulator defines their semantics.
        # ----------------------------------------------------

        for metric_name in (
            "baseline_value",
            "governed_value",
            "prevented_value",
            "gnn_risk",
            "fused_risk",
        ):

            value = getattr(
                metrics,
                metric_name,
            )

            if (
                not math.isnan(value)
                and not math.isfinite(value)
            ):

                errors.append(
                    f"seed={result.seed}, "
                    f"{population}.{metric_name} "
                    f"is non-finite"
                )

    # --------------------------------------------------------
    # Global metrics
    # --------------------------------------------------------

    for metric_name in (
        "adaptive_policy_change",
        "adaptive_governor_advantage",
        "adaptive_vs_human_intervention_gap",
        "human_abusive_prevented_value",
    ):

        value = getattr(
            result,
            metric_name,
        )

        if (
            not math.isnan(value)
            and not math.isfinite(value)
        ):

            errors.append(
                f"seed={result.seed}, "
                f"{metric_name} is non-finite"
            )

    return errors


# ============================================================================
# REPORTING
# ============================================================================

def print_seed_summary(
    result: SeedMetrics,
) -> None:

    print()
    print(
        f"SEED {result.seed} — EXPOSED METRICS"
    )
    print("-" * 110)

    for population in POPULATIONS:

        metrics = result.populations[
            population
        ]

        print()
        print(
            f"{population}"
        )

        print(
            "  baseline approval : "
            f"{_format_percent(metrics.baseline_approval)}"
        )

        print(
            "  governed approval : "
            f"{_format_percent(metrics.governed_approval)}"
        )

        print(
            "  approval reduction : "
            f"{_format_percent(metrics.reduction)}"
        )

        print(
            "  intervention rate : "
            f"{_format_percent(metrics.intervention_rate)}"
        )

        print(
            "  baseline value    : "
            f"{_format_currency(metrics.baseline_value)}"
        )

        print(
            "  governed value    : "
            f"{_format_currency(metrics.governed_value)}"
        )

        print(
            "  prevented value   : "
            f"{_format_currency(metrics.prevented_value)}"
        )

        print(
            "  GNN risk          : "
            f"{_format_number(metrics.gnn_risk)}"
        )

        print(
            "  fused risk        : "
            f"{_format_number(metrics.fused_risk)}"
        )

    print()
    print(
        "Global adaptive policy change      : "
        f"{_format_number(result.adaptive_policy_change)}"
    )

    print(
        "Adaptive Governor advantage        : "
        f"{_format_number(result.adaptive_governor_advantage)}"
    )

    print(
        "Adaptive-vs-human intervention gap : "
        f"{_format_number(result.adaptive_vs_human_intervention_gap)}"
    )

    print(
        "Human abusive prevented value      : "
        f"{_format_currency(result.human_abusive_prevented_value)}"
    )


def print_statistical_table(
    results: list[SeedMetrics],
) -> None:

    print()
    print("=" * 110)
    print(
        "PHASE 20 — CROSS-SEED STATISTICAL SUMMARY"
    )
    print("=" * 110)

    # --------------------------------------------------------
    # Population metrics
    # --------------------------------------------------------

    for population in POPULATIONS:

        print()
        print(
            f"[{population}]"
        )

        print(
            f"{'Metric':<25}"
            f"{'N':>6}"
            f"{'Mean':>14}"
            f"{'Std':>14}"
            f"{'Min':>14}"
            f"{'Max':>14}"
            f"{'CV':>12}"
        )

        print(
            "-" * 99
        )

        for metric in (
            "baseline_approval",
            "governed_approval",
            "reduction",
            "intervention_rate",
            "baseline_value",
            "governed_value",
            "prevented_value",
            "gnn_risk",
            "fused_risk",
        ):

            values = _population_metric_values(
                results,
                population,
                metric,
            )

            summary = summarize_metric(
                values,
                metric=metric,
                population=population,
            )

            print(
                f"{metric:<25}"
                f"{summary.n:>6}"
                f"{_format_number(summary.mean):>14}"
                f"{_format_number(summary.std):>14}"
                f"{_format_number(summary.minimum):>14}"
                f"{_format_number(summary.maximum):>14}"
                f"{_format_number(summary.cv):>12}"
            )

    # --------------------------------------------------------
    # Global metrics
    # --------------------------------------------------------

    print()
    print(
        "[GLOBAL METRICS]"
    )

    print(
        f"{'Metric':<40}"
        f"{'N':>6}"
        f"{'Mean':>14}"
        f"{'Std':>14}"
        f"{'Min':>14}"
        f"{'Max':>14}"
        f"{'CV':>12}"
    )

    print(
        "-" * 110
    )

    for metric in (
        "adaptive_policy_change",
        "adaptive_governor_advantage",
        "adaptive_vs_human_intervention_gap",
        "human_abusive_prevented_value",
    ):

        values = _global_metric_values(
            results,
            metric,
        )

        summary = summarize_metric(
            values,
            metric=metric,
        )

        print(
            f"{metric:<40}"
            f"{summary.n:>6}"
            f"{_format_number(summary.mean):>14}"
            f"{_format_number(summary.std):>14}"
            f"{_format_number(summary.minimum):>14}"
            f"{_format_number(summary.maximum):>14}"
            f"{_format_number(summary.cv):>12}"
        )


# ============================================================================
# ROBUSTNESS CONCLUSIONS
# ============================================================================

def _mean_metric(
    results: list[SeedMetrics],
    population: str,
    metric: str,
) -> float:

    values = _population_metric_values(
        results,
        population,
        metric,
    )

    finite = _finite_values(
        values
    )

    if not finite:
        return math.nan

    return statistics.fmean(
        finite
    )


def _mean_global_metric(
    results: list[SeedMetrics],
    metric: str,
) -> float:

    values = _global_metric_values(
        results,
        metric,
    )

    finite = _finite_values(
        values
    )

    if not finite:
        return math.nan

    return statistics.fmean(
        finite
    )


def print_conclusions(
    results: list[SeedMetrics],
) -> bool:
    """
    Print the substantive Phase 20 conclusions.

    Returns True when the statistical evaluation itself is valid
    and completed. It does NOT require every optional metric to
    exist.
    """

    print()
    print("=" * 110)
    print(
        "PHASE 20 — ROBUSTNESS CONCLUSIONS"
    )
    print("=" * 110)

    # --------------------------------------------------------
    # Abusive approval suppression
    # --------------------------------------------------------

    print()
    print(
        "1. ABUSIVE APPROVAL SUPPRESSION"
    )
    print("-" * 110)

    abusive_reduction_means: list[float] = []

    for population in ABUSIVE_POPULATIONS:

        mean_reduction = _mean_metric(
            results,
            population,
            "reduction",
        )

        if math.isnan(
            mean_reduction
        ):

            print(
                f"{population}: NOT EVALUABLE "
                "(baseline/governed approval unavailable)"
            )

        else:

            abusive_reduction_means.append(
                mean_reduction
            )

            print(
                f"{population}: mean approval reduction = "
                f"{_format_percent(mean_reduction)}"
            )

    if abusive_reduction_means:

        if all(
            value > 0.0
            for value in abusive_reduction_means
        ):

            print(
                "Conclusion: abusive approval was "
                "suppressed consistently in the evaluated "
                "abusive populations."
            )

        else:

            print(
                "Conclusion: abusive approval suppression "
                "was not positive for every evaluated abusive "
                "population."
            )

    else:

        print(
            "Conclusion: abusive approval suppression "
            "could not be evaluated."
        )

    # --------------------------------------------------------
    # Legitimate intervention
    # --------------------------------------------------------

    print()
    print(
        "2. LEGITIMATE TRAFFIC — FALSE INTERVENTION"
    )
    print("-" * 110)

    legitimate_intervention_means: list[
        float
    ] = []

    for population in LEGITIMATE_POPULATIONS:

        mean_intervention = _mean_metric(
            results,
            population,
            "intervention_rate",
        )

        if math.isnan(
            mean_intervention
        ):

            print(
                f"{population}: NOT EVALUABLE "
                "(intervention rate unavailable)"
            )

        else:

            legitimate_intervention_means.append(
                mean_intervention
            )

            print(
                f"{population}: mean intervention rate = "
                f"{_format_percent(mean_intervention)}"
            )

    if legitimate_intervention_means:

        if all(
            value == 0.0
            for value in legitimate_intervention_means
        ):

            print(
                "Conclusion: no legitimate-population "
                "interventions were observed across the "
                "evaluated seeds."
            )

        else:

            print(
                "Conclusion: legitimate traffic experienced "
                "some Governor intervention. This should be "
                "reported as a false-intervention signal, not "
                "silently treated as zero."
            )

    else:

        print(
            "Conclusion: legitimate false-intervention "
            "behavior could not be evaluated."
        )

    # --------------------------------------------------------
    # Adaptive attacker policy change
    # --------------------------------------------------------

    print()
    print(
        "3. ADAPTIVE ATTACKER POLICY CHANGE"
    )
    print("-" * 110)

    policy_change = _mean_global_metric(
        results,
        "adaptive_policy_change",
    )

    if math.isnan(
        policy_change
    ):

        print(
            "Adaptive policy change: NOT AVAILABLE"
        )

    else:

        print(
            "Mean adaptive policy change: "
            f"{_format_number(policy_change)}"
        )

        if policy_change > 0.0:

            print(
                "Conclusion: the adaptive attacker changed "
                "policy across the evaluated experiment."
            )

        else:

            print(
                "Conclusion: no positive adaptive policy "
                "change was observed."
            )

    # --------------------------------------------------------
    # Adaptive Governor advantage
    # --------------------------------------------------------

    print()
    print(
        "4. ADAPTIVE GOVERNOR ADVANTAGE"
    )
    print("-" * 110)

    governor_advantage = _mean_global_metric(
        results,
        "adaptive_governor_advantage",
    )

    if math.isnan(
        governor_advantage
    ):

        print(
            "Adaptive Governor advantage: NOT AVAILABLE"
        )

    else:

        print(
            "Mean Adaptive Governor advantage: "
            f"{_format_number(governor_advantage)}"
        )

        if governor_advantage > 0.0:

            print(
                "Conclusion: adaptive Governor advantage "
                "was positive on average."
            )

        elif governor_advantage < 0.0:

            print(
                "Conclusion: adaptive Governor advantage "
                "was negative on average."
            )

        else:

            print(
                "Conclusion: adaptive Governor advantage "
                "was approximately neutral on average."
            )

    # --------------------------------------------------------
    # Adaptive vs human intervention gap
    # --------------------------------------------------------

    print()
    print(
        "5. ADAPTIVE-vs-HUMAN INTERVENTION GAP"
    )
    print("-" * 110)

    intervention_gap = _mean_global_metric(
        results,
        "adaptive_vs_human_intervention_gap",
    )

    if math.isnan(
        intervention_gap
    ):

        print(
            "Adaptive-vs-human intervention gap: "
            "NOT AVAILABLE"
        )

    else:

        print(
            "Mean adaptive-vs-human intervention gap: "
            f"{_format_number(intervention_gap)}"
        )

        if intervention_gap > 0.0:

            print(
                "Conclusion: adaptive traffic experienced "
                "higher intervention on average."
            )

        elif intervention_gap < 0.0:

            print(
                "Conclusion: adaptive traffic experienced "
                "lower intervention on average."
            )

        else:

            print(
                "Conclusion: adaptive and human intervention "
                "rates were equal on average."
            )

    # --------------------------------------------------------
    # Missing-data disclosure
    # --------------------------------------------------------

    print()
    print(
        "6. METRIC AVAILABILITY"
    )
    print("-" * 110)

    legitimate_approval_available = 0

    legitimate_approval_total = (
        len(results)
        * len(LEGITIMATE_POPULATIONS)
    )

    for result in results:

        for population in LEGITIMATE_POPULATIONS:

            metrics = result.populations[
                population
            ]

            if (
                _is_finite_number(
                    metrics.baseline_approval
                )
                and _is_finite_number(
                    metrics.governed_approval
                )
            ):

                legitimate_approval_available += 1

    print(
        "Legitimate baseline/governed approval pairs: "
        f"{legitimate_approval_available}/"
        f"{legitimate_approval_total}"
    )

    if (
        legitimate_approval_available
        < legitimate_approval_total
    ):

        print(
            "Disclosure: Phase 18 does not expose "
            "baseline/governed approval for all legitimate "
            "populations. Phase 20 therefore does not "
            "calculate or infer those values."
        )

    gnn_available = 0
    fused_available = 0
    total_population_seed_pairs = (
        len(results)
        * len(POPULATIONS)
    )

    for result in results:

        for population in POPULATIONS:

            metrics = result.populations[
                population
            ]

            if _is_finite_number(
                metrics.gnn_risk
            ):
                gnn_available += 1

            if _is_finite_number(
                metrics.fused_risk
            ):
                fused_available += 1

    print(
        "Population-level GNN risk observations: "
        f"{gnn_available}/"
        f"{total_population_seed_pairs}"
    )

    print(
        "Population-level fused-risk observations: "
        f"{fused_available}/"
        f"{total_population_seed_pairs}"
    )

    print()
    print(
        "Missing metrics are reported as N/A rather than "
        "being replaced with zero."
    )

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    print()
    print("=" * 110)

    print(
        "PHASE 20 STATISTICAL EVALUATION: "
        "COMPLETED"
    )

    print(
        "The evaluation used direct Python-level Phase 18 "
        "outputs and did not parse console output."
    )

    print(
        "Unavailable metrics were not fabricated."
    )

    print("=" * 110)

    return True


# ============================================================================
# FULL VALIDATION
# ============================================================================

def validate_all_results(
    results: list[SeedMetrics],
) -> None:

    print()
    print("=" * 110)
    print(
        "VALIDATING PHASE 20 SEED RESULTS"
    )
    print("=" * 110)

    all_errors: list[str] = []

    for result in results:

        errors = validate_seed_result(
            result
        )

        if errors:

            all_errors.extend(
                errors
            )

        else:

            print(
                f"Seed {result.seed}: PASS"
            )

    if all_errors:

        print()
        print(
            "VALIDATION ERRORS:"
        )

        for error in all_errors:

            print(
                f"  - {error}"
            )

        raise RuntimeError(
            "Phase 20 statistical validation failed."
        )

    print()
    print(
        "All available metrics passed "
        "range/finite-value validation."
    )

    print(
        "Unavailable metrics were intentionally "
        "excluded from validation."
    )


# ============================================================================
# DATA AVAILABILITY AUDIT
# ============================================================================

def print_data_contract_audit(
    results: list[SeedMetrics],
) -> None:

    print()
    print("=" * 110)
    print(
        "PHASE 20 — DATA CONTRACT AUDIT"
    )
    print("=" * 110)

    for population in POPULATIONS:

        print()
        print(
            population
        )
        print("-" * 60)

        metric_names = (
            "baseline_approval",
            "governed_approval",
            "reduction",
            "intervention_rate",
            "baseline_value",
            "governed_value",
            "prevented_value",
            "gnn_risk",
            "fused_risk",
        )

        for metric_name in metric_names:

            available = 0

            for result in results:

                value = getattr(
                    result.populations[
                        population
                    ],
                    metric_name,
                )

                if _is_finite_number(
                    value
                ):

                    available += 1

            print(
                f"{metric_name:<25}"
                f"{available:>4}/"
                f"{len(results):<4}"
                f"{'AVAILABLE' if available else 'N/A'}"
            )

    print()
    print(
        "This audit describes what Phase 18 actually exposes "
        "to Phase 20. It is not an attempt to fill missing data."
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print()
    print("=" * 110)
    print(
        "PHASE 20 — STATISTICAL ROBUSTNESS EVALUATION"
    )
    print("=" * 110)

    print()
    print(
        "Phase 20 will evaluate the existing Phase 18 "
        "Python pipeline across deterministic seeds."
    )

    print(
        "No Phase 18 source logic will be modified."
    )

    print(
        "No console output will be parsed."
    )

    print(
        "Unavailable metrics will remain N/A."
    )

    print()
    print(
        f"Seeds: {SEEDS}"
    )

    # --------------------------------------------------------
    # Import Phase 18
    # --------------------------------------------------------

    phase18 = _load_phase18()

    # --------------------------------------------------------
    # Expected experiment configuration
    # --------------------------------------------------------

    episodes_per_group = int(
        getattr(
            phase18,
            "EPISODES_PER_GROUP",
            0,
        )
    )

    interactions_per_episode = int(
        getattr(
            phase18,
            "INTERACTIONS_PER_EPISODE",
            0,
        )
    )

    print()
    print(
        "Phase 18 configuration:"
    )

    print(
        f"  episodes per population : "
        f"{episodes_per_group}"
    )

    print(
        f"  interactions per episode: "
        f"{interactions_per_episode}"
    )

    print(
        f"  populations             : "
        f"{len(POPULATIONS)}"
    )

    print(
        f"  expected episodes/seed  : "
        f"{episodes_per_group * len(POPULATIONS)}"
    )

    # --------------------------------------------------------
    # Run every seed
    # --------------------------------------------------------

    results: list[SeedMetrics] = []

    for seed in SEEDS:

        try:

            result = _run_phase18_for_seed(
                phase18=phase18,
                seed=seed,
            )

            results.append(
                result
            )

            print_seed_summary(
                result
            )

        except Exception as exc:

            raise RuntimeError(
                f"\n"
                f"Phase 20 stopped at seed {seed}.\n\n"
                f"Phase 18 could not be evaluated "
                f"through its direct Python pipeline.\n\n"
                f"Execution path:\n"
                f"  create_world()\n"
                f"  build_interaction_graph()\n"
                f"  run_phase18_simulation()\n"
                f"  build_phase18_result()\n\n"
                f"Original error:\n"
                f"{exc}"
            ) from exc

    # --------------------------------------------------------
    # Ensure all requested seeds completed
    # --------------------------------------------------------

    completed_seeds = [
        result.seed
        for result in results
    ]

    if completed_seeds != list(SEEDS):

        raise RuntimeError(
            "Phase 20 did not complete all requested seeds.\n"
            f"Expected: {list(SEEDS)}\n"
            f"Completed: {completed_seeds}"
        )

    # --------------------------------------------------------
    # Validate available metrics
    # --------------------------------------------------------

    validate_all_results(
        results
    )

    # --------------------------------------------------------
    # Contract audit
    # --------------------------------------------------------

    print_data_contract_audit(
        results
    )

    # --------------------------------------------------------
    # Statistical summary
    # --------------------------------------------------------

    print_statistical_table(
        results
    )

    # --------------------------------------------------------
    # Conclusions
    # --------------------------------------------------------

    completed = print_conclusions(
        results
    )

    if not completed:

        raise RuntimeError(
            "Phase 20 did not complete its statistical evaluation."
        )

    print()
    print("=" * 110)
    print(
        "PHASE 20 COMPLETE"
    )
    print("=" * 110)

    print()
    print(
        f"Successfully evaluated "
        f"{len(results)} deterministic seeds:"
    )

    print(
        f"  {completed_seeds}"
    )

    print()
    print(
        "The report distinguishes between:"
    )

    print(
        "  - genuinely returned metrics,"
    )

    print(
        "  - metrics legitimately derived from returned values,"
    )

    print(
        "  - and metrics that Phase 18 does not expose."
    )

    print()
    print(
        "No console-output parsing or metric fabrication "
        "was performed."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()