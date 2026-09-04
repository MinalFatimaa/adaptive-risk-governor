Yes. **Phase 20 is genuinely complete and the run is valid.** More importantly, the final GNN integration tests also passed:

* `test_end_to_end_gnn_governor_pipeline.py` → **PASS**
* `test_gnn_governor_integration.py` → **PASS**
* Phase 20 → **10/10 seeds PASS**
* Phase 18 source → **not modified**
* No console-output parsing
* No fabricated metrics

At this point, **I would stop adding evaluation phases.** The engineering/evaluation loop has reached a defensible endpoint.

There is one thing I would do next: **turn the Phase 20 results into permanent machine-readable artifacts**, so your judges don't have to run a 7-minute experiment or read terminal output to see your findings.

### Next step: create the results artifact

Create:

```text
results/
├── phase18_results.json
├── phase20_results.json
└── README.md
```

But I don't want to invent values by manually copying the terminal output into JSON. The correct approach is to make Phase 20 **persist the exact Python-level results it already computes**.

So the next code change should be **only Phase 20**, and it should append JSON serialization to the existing validated `results` objects.

Replace your current `src/evaluation/phase20_statistical_robustness.py` with this version. It preserves the evaluation logic and adds structured result persistence; it does **not** change Phase 18 or the experiment itself.

```python
"""
Phase 20 — Statistical Robustness Evaluation
=============================================

Purpose
-------
Evaluate the existing Phase 18 GNN Governor closed-loop experiment
across multiple deterministic seeds and persist the resulting
Python-level metrics as a machine-readable JSON artifact.

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
   approval are genuinely available.
8. Legitimate populations are evaluated primarily through the
   intervention-rate signal that Phase 18 actually exposes.
9. Missing GNN/fusion population metrics remain unavailable rather
   than being converted to zero.
10. A missing metric is not equivalent to a failed experiment.
11. The persisted JSON artifact contains exactly the metrics recovered
    from the Python-level Phase 18 result objects.

Run
---
    python -m src.evaluation.phase20_statistical_robustness

Output
------
    results/phase20_results.json
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ============================================================================
# CONFIGURATION
# ============================================================================

PHASE18_MODULE = (
    "src.evaluation.phase18_gnn_governor_evaluation"
)

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

RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "phase20_results.json"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PopulationMetrics:
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
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


def _is_scalar(value: Any) -> bool:
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
    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    raise TypeError(
        f"Expected numeric scalar, got "
        f"{type(value).__name__}: {value!r}"
    )


def _finite_or_nan(value: Any) -> float:
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

    normalized_path = _normalise_name(path)

    score = 0

    population_aliases = POPULATION_ALIASES[
        population
    ]

    metric_aliases = METRIC_ALIASES[
        metric
    ]

    for alias in population_aliases:
        normalized_alias = _normalise_name(alias)

        if normalized_path == normalized_alias:
            score += 100

        elif normalized_alias in normalized_path:
            score += 30

    for alias in metric_aliases:
        normalized_alias = _normalise_name(alias)

        if normalized_alias in normalized_path:
            score += 20

    score -= path.count(".")
    score -= path.count("[")

    return score


def _resolve_population_metric(
    flattened: list[tuple[str, Any]],
    population: str,
    metric: str,
) -> float:

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

        numeric_value = _finite_or_nan(value)

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

        normalized_path = _normalise_name(path)

        matched_alias = None

        for alias in aliases:
            normalized_alias = _normalise_name(alias)

            if normalized_alias in normalized_path:
                matched_alias = normalized_alias
                break

        if matched_alias is None:
            continue

        numeric_value = _finite_or_nan(value)

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

    print()
    print("=" * 110)
    print(
        f"RUNNING PHASE 18 — SEED {seed}"
    )
    print("=" * 110)

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)

    except Exception:
        pass

    print(
        "Creating deterministic mixed-population world..."
    )

    world = phase18.create_world(
        config=phase18.DEFAULT_POPULATION_CONFIG,
        seed=seed,
    )

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

    print(
        "Building and evaluating GNN Governor "
        "closed-loop pipeline..."
    )

    metrics = _call_phase18_simulation(
        phase18=phase18,
        world=world,
        seed=seed,
    )

    result = phase18.build_phase18_result(
        metrics
    )

    result_flat = _flatten_scalars(
        result,
        path="result",
    )

    metrics_flat = _flatten_scalars(
        metrics,
        path="metrics",
    )

    flattened = (
        result_flat
        + metrics_flat
    )

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
            baseline_approval=baseline_approval,
            governed_approval=governed_approval,
            reduction=reduction,
            intervention_rate=intervention_rate,
            baseline_value=baseline_value,
            governed_value=governed_value,
            prevented_value=prevented_value,
            gnn_risk=gnn_risk,
            fused_risk=fused_risk,
        )

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
        adaptive_policy_change=adaptive_policy_change,
        adaptive_governor_advantage=adaptive_governor_advantage,
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

    return [
        float(value)
        for value in values
        if _is_finite_number(value)
    ]


def _coefficient_of_variation(
    mean: float,
    std: float,
) -> float:

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

    finite = _finite_values(values)

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
            population
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

    finite = _finite_values(values)

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

    finite = _finite_values(values)

    if not finite:
        return math.nan

    return statistics.fmean(
        finite
    )


def print_conclusions(
    results: list[SeedMetrics],
) -> bool:

    print()
    print("=" * 110)
    print(
        "PHASE 20 — ROBUSTNESS CONCLUSIONS"
    )
    print("=" * 110)

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

        if math.isnan(mean_reduction):

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

        if math.isnan(mean_intervention):

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

    print()
    print(
        "3. ADAPTIVE ATTACKER POLICY CHANGE"
    )
    print("-" * 110)

    policy_change = _mean_global_metric(
        results,
        "adaptive_policy_change",
    )

    if math.isnan(policy_change):

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

    print()
    print(
        "4. ADAPTIVE GOVERNOR ADVANTAGE"
    )
    print("-" * 110)

    governor_advantage = _mean_global_metric(
        results,
        "adaptive_governor_advantage",
    )

    if math.isnan(governor_advantage):

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
                "Conclusion: adaptive
```
