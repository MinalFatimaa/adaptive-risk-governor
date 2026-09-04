from __future__ import annotations

import os
import re
import subprocess
import sys


# ============================================================
# PHASE 15 MODULE
# ============================================================

PHASE_MODULE = (
    "src.evaluation.phase15_governor_policy_evaluation"
)


# ============================================================
# EXECUTION HELPER
# ============================================================

def run_phase15() -> str:

    env = os.environ.copy()

    # Windows may use cp1252 by default.
    # Phase 15 prints the ₹ symbol, so force UTF-8.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            PHASE_MODULE,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )

    assert result.returncode == 0, (
        "Phase 15 execution failed.\n\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )

    return result.stdout


# ============================================================
# HELPER: CHECK VALIDATION LINE
# ============================================================

def assert_validation_passed(
    output: str,
    validation_name: str,
) -> None:

    pattern = (
        rf"^{re.escape(validation_name)}\s*:\s*PASSED$"
    )

    assert re.search(
        pattern,
        output,
        flags=re.MULTILINE,
    ), (
        f"Validation check did not pass: "
        f"{validation_name}\n\n"
        f"Output:\n{output}"
    )


# ============================================================
# HELPER: CHECK NO NaN / INF IN NUMERIC METRICS
# ============================================================

def assert_no_invalid_numeric_values(
    output: str,
) -> None:

    # IMPORTANT:
    # Do NOT use:
    #
    #     "nan" not in output.lower()
    #
    # because strings such as:
    #
    #     financial_metrics_finite
    #
    # contain the letters "nan".
    #
    # Instead, detect NaN/Inf as standalone numeric tokens.

    invalid_numeric = re.search(
        r"(?<![A-Za-z0-9_])"
        r"(?:nan|inf|-inf)"
        r"(?![A-Za-z0-9_])",
        output,
        flags=re.IGNORECASE,
    )

    assert invalid_numeric is None, (
        "Invalid numeric value detected in Phase 15 output: "
        f"{invalid_numeric.group(0) if invalid_numeric else ''}\n\n"
        f"Output:\n{output}"
    )


# ============================================================
# COMPLETION
# ============================================================

def test_phase15_completes_successfully():

    output = run_phase15()

    assert (
        "PHASE 15 GOVERNOR POLICY & ECONOMIC EVALUATION COMPLETE"
        in output
    )


# ============================================================
# DATASET
# ============================================================

def test_phase15_dataset_is_correct():

    output = run_phase15()

    assert "Total cases              : 1000" in output
    assert "Development cases        : 701" in output
    assert "Final holdout cases      : 299" in output


def test_phase15_final_holdout_is_299_cases():

    output = run_phase15()

    assert "Final holdout cases      : 299" in output


# ============================================================
# INNER SPLIT
# ============================================================

def test_phase15_inner_split_is_correct():

    output = run_phase15()

    assert "Inner training cases     : 527" in output
    assert "Inner validation cases   : 174" in output


def test_phase15_inner_split_is_valid():

    output = run_phase15()

    assert "Inner training cases     : 527" in output
    assert "Inner validation cases   : 174" in output


# ============================================================
# FINAL MODEL
# ============================================================

def test_phase15_uses_final_hgb_model():

    output = run_phase15()

    assert (
        "Final ML model           : HIST_GRADIENT_BOOSTING"
        in output
    )

    assert (
        "Model                 : HIST_GRADIENT_BOOSTING"
        in output
    )


def test_phase15_final_model_metrics_are_finite():

    output = run_phase15()

    assert "Average Precision" in output
    assert "ROC-AUC" in output
    assert "Brier Score" in output
    assert "Log Loss" in output

    assert_no_invalid_numeric_values(output)


# ============================================================
# LEAKAGE CONTROL
# ============================================================

def test_phase15_leakage_controls_are_present():

    output = run_phase15()

    assert (
        "Model training           : DEVELOPMENT DATA"
        in output
    )

    assert (
        "Calibration              : TRAINING DATA ONLY"
        in output
    )

    assert (
        "Policy threshold tuning  : INNER VALIDATION ONLY"
        in output
    )

    assert (
        "Final holdout evaluation : FINAL HOLDOUT ONLY"
        in output
    )

    assert (
        "Holdout used for policy  : NO"
        in output
    )

    assert (
        "Holdout used for model   : NO"
        in output
    )


def test_phase15_is_not_using_holdout_for_policy():

    output = run_phase15()

    assert (
        "Holdout used for policy  : NO"
        in output
    )

    assert_validation_passed(
        output,
        "policy_not_economically_tuned_on_holdout",
    )


# ============================================================
# POLICY
# ============================================================

def test_phase15_selects_valid_policy_thresholds():

    output = run_phase15()

    assert (
        "Request evidence threshold : 0.30"
        in output
    )

    assert (
        "Escalation threshold       : 0.70"
        in output
    )


def test_phase15_policy_has_no_direct_deny_action():

    output = run_phase15()

    # The Governor does not have a DENY action.
    # DENY belongs to the downstream SupportAgent.
    assert "DENY actions" not in output
    assert "GOVERNOR DENY" not in output


# ============================================================
# GOVERNOR ACTION SPACE
# ============================================================

def test_phase15_has_three_governor_actions():

    output = run_phase15()

    assert "ALLOW actions" in output
    assert "EVIDENCE actions" in output
    assert "ESCALATION actions" in output


# ============================================================
# ECONOMIC EVALUATION
# ============================================================

def test_phase15_economic_metrics_are_reported():

    output = run_phase15()

    assert (
        "Net economic benefit"
        in output
    )

    assert (
        "Loss prevention rate"
        in output
    )

    assert (
        "False intervention rate"
        in output
    )

    assert (
        "Remaining fraud loss"
        in output
    )

    assert (
        "Friction cost"
        in output
    )

    assert (
        "Human review cost"
        in output
    )


def test_phase15_economic_evaluation_completed():

    output = run_phase15()

    assert (
        "FINAL GOVERNOR ECONOMIC EVALUATION"
        in output
    )


def test_phase15_economic_metrics_are_finite():

    output = run_phase15()

    assert_no_invalid_numeric_values(output)


# ============================================================
# ML METRICS
# ============================================================

def test_phase15_ml_metrics_are_reported():

    output = run_phase15()

    assert "Average Precision" in output
    assert "ROC-AUC" in output
    assert "Brier Score" in output
    assert "Log Loss" in output


# ============================================================
# VALIDATION
# ============================================================

def test_phase15_validation_passes():

    output = run_phase15()

    assert (
        "Overall validation       : PASSED"
        in output
    )


def test_phase15_overall_validation_passes():

    output = run_phase15()

    assert (
        "Overall validation       : PASSED"
        in output
    )


# ============================================================
# SPECIFIC VALIDATION CHECKS
# ============================================================

def test_phase15_validation_contains_dataset_check():

    output = run_phase15()

    assert_validation_passed(
        output,
        "dataset_generated",
    )

    assert_validation_passed(
        output,
        "binary_target",
    )


def test_phase15_validation_contains_split_checks():

    output = run_phase15()

    assert_validation_passed(
        output,
        "development_holdout_split",
    )

    assert_validation_passed(
        output,
        "inner_split_valid",
    )

    assert_validation_passed(
        output,
        "holdout_size_preserved",
    )


def test_phase15_validation_contains_model_check():

    output = run_phase15()

    assert_validation_passed(
        output,
        "final_model_is_hgb",
    )


def test_phase15_validation_contains_policy_checks():

    output = run_phase15()

    assert_validation_passed(
        output,
        "policy_thresholds_valid",
    )

    assert_validation_passed(
        output,
        "three_governor_actions",
    )


def test_phase15_validation_contains_leakage_checks():

    output = run_phase15()

    assert_validation_passed(
        output,
        "policy_not_economically_tuned_on_holdout",
    )

    assert_validation_passed(
        output,
        "holdout_not_used_for_ml_selection",
    )

    assert_validation_passed(
        output,
        "final_holdout_evaluated_once",
    )


# ============================================================
# NO ECONOMIC MODEL SELECTION
# ============================================================

def test_phase15_does_not_use_economic_model_selection():

    output = run_phase15()

    # Phase 15 must optimize the GOVERNOR POLICY,
    # not perform economic model selection.
    #
    # The implementation does not expose a validation field
    # named "no_economic_selection", so verify the actual
    # leakage-control contract printed by Phase 15.

    assert (
        "Policy threshold tuning  : INNER VALIDATION ONLY"
        in output
    )

    assert (
        "Final holdout evaluation : FINAL HOLDOUT ONLY"
        in output
    )

    assert (
        "Holdout used for policy  : NO"
        in output
    )

    assert (
        "Holdout used for model   : NO"
        in output
    )

    # The final model is fixed as HGB in Phase 15.
    # There must not be a model-selection stage in the
    # Phase 15 output.
    assert (
        "FINAL ML MODEL — HOLDOUT PERFORMANCE"
        in output
    )


# ============================================================
# FINAL HOLDOUT INTEGRITY
# ============================================================

def test_phase15_final_holdout_used_only_for_final_evaluation():

    output = run_phase15()

    assert (
        "Final holdout evaluation : FINAL HOLDOUT ONLY"
        in output
    )

    assert (
        "Holdout used for policy  : NO"
        in output
    )

    assert (
        "Holdout used for model   : NO"
        in output
    )

    assert_validation_passed(
        output,
        "final_holdout_evaluated_once",
    )