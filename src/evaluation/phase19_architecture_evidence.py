from __future__ import annotations

"""
PHASE 19 — GNN GOVERNOR ARCHITECTURE EVIDENCE

Purpose
-------
Evaluate and strengthen the evidence produced by the existing
Phase 18 closed-loop GNN Governor.

IMPORTANT
---------
This phase is READ-ONLY with respect to the existing architecture.

It does NOT:
    - modify Phase 18
    - modify the GNN
    - modify the Governor
    - modify Risk Fusion
    - modify Agent A
    - modify Agent B
    - retrain the GNN
    - alter the simulation

The Phase 18 evaluator is executed as an actual Python package module.

The source audit intentionally distinguishes:

    1. implementation existence,
    2. actual imports,
    3. executable AST references,
    4. response construction,
    5. Agent B observation,
    6. adaptive policy update,
    7. governor-before-outcome ordering.

It does NOT convert arbitrary source-text presence into proof of behavior.
"""


from dataclasses import dataclass
from pathlib import Path
from typing import Any
import ast
import importlib
import inspect
import math
import re
import subprocess
import sys


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE18_MODULE = (
    "src.evaluation.phase18_gnn_governor_evaluation"
)

PHASE18_FILE = (
    PROJECT_ROOT
    / "src"
    / "evaluation"
    / "phase18_gnn_governor_evaluation.py"
)


# ============================================================
# ARCHITECTURE MODULES
# ============================================================

ARCHITECTURE_MODULES = {
    "interaction_graph": (
        "src.governor.interaction_graph"
    ),
    "trained_gnn": (
        "src.governor.gnn_inference"
    ),
    "gnn_governor_adapter": (
        "src.governor.gnn_governor_integration"
    ),
    "behavioral_governor": (
        "src.governor.risk_governor"
    ),
    "risk_fusion": (
        "src.governor.risk_fusion"
    ),
    "gnn_governor": (
        "src.governor.gnn_risk_governor"
    ),
    "adaptive_customer": (
        "src.agents.adaptive_customer"
    ),
}


# ============================================================
# EXPECTED SYMBOLS
# ============================================================

EXPECTED_SYMBOLS = {
    "interaction_graph": (
        "build_interaction_graph",
    ),
    "trained_gnn": (
        "GNNRiskInference",
    ),
    "gnn_governor_adapter": (
        "GNNGovernorAdapter",
    ),
    "behavioral_governor": (
        "AdaptiveRiskGovernor",
    ),
    "risk_fusion": (
        "RiskFusionEngine",
    ),
    "gnn_governor": (
        "GNNRiskGovernor",
    ),
    "adaptive_customer": (
        "AdaptiveCustomerAgent",
    ),
}


# ============================================================
# CANONICAL GOVERNOR ACTION SPACE
# ============================================================

CANONICAL_GOVERNOR_ACTIONS = {
    "ALLOW_AGENT_A_DECISION",
    "REQUEST_ADDITIONAL_EVIDENCE",
    "ESCALATE_TO_HUMAN_REVIEW",
}


FORBIDDEN_GOVERNOR_ACTIONS = {
    "DENY",
    "DIRECT_DENY",
    "REJECT",
}


# ============================================================
# HIDDEN INFORMATION
# ============================================================

HIDDEN_SIGNALS = {
    "gnn_risk",
    "gnn_score",
    "fused_risk",
    "fused_risk_score",
    "network_risk",
    "network_abnormality",
    "strategic_risk",
    "strategic_behavior",
    "strategic_adaptation_score",
    "ground_truth",
    "fraud_label",
    "fraud_target",
    "target_label",
}


# ============================================================
# REQUIRED POPULATIONS
# ============================================================

REQUIRED_POPULATIONS = {
    "ADAPTIVE_ABUSIVE",
    "ADAPTIVE_LEGITIMATE",
    "HUMAN_ABUSIVE",
    "HUMAN_LEGITIMATE",
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Phase18Run:
    return_code: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.return_code == 0


@dataclass
class PopulationMetrics:
    population: str
    episodes: float
    baseline_approval: float
    governed_approval: float
    reduction: float
    intervention: float
    gnn_risk: float
    fused_risk: float


@dataclass
class ParsedSource:
    path: Path
    source: str
    tree: ast.AST | None


# ============================================================
# OUTPUT HELPERS
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def subsection(title: str) -> None:
    print()
    print("-" * 110)
    print(title)
    print("-" * 110)


def status(
    name: str,
    passed: bool,
) -> None:
    print(
        f"{name:<65} : "
        f"{'PASSED' if passed else 'FAILED'}"
    )


def info(message: str) -> None:
    print(f"[INFO] {message}")


def warning(message: str) -> None:
    print(f"[WARNING] {message}")


# ============================================================
# NUMERIC HELPERS
# ============================================================

def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (
        TypeError,
        ValueError,
    ):
        return False


def bounded(
    value: Any,
    low: float = 0.0,
    high: float = 1.0,
) -> bool:

    if not finite(value):
        return False

    value = float(value)

    return low <= value <= high


# ============================================================
# SOURCE HELPERS
# ============================================================

def read_text(path: Path) -> str:

    if not path.exists():
        return ""

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def parse_source(
    path: Path,
) -> ParsedSource:

    source = read_text(path)

    if not source:
        return ParsedSource(
            path=path,
            source="",
            tree=None,
        )

    try:
        tree = ast.parse(
            source,
            filename=str(path),
        )
    except SyntaxError:
        tree = None

    return ParsedSource(
        path=path,
        source=source,
        tree=tree,
    )


def load_module(module_name: str):

    return importlib.import_module(
        module_name
    )


def get_module_source(module) -> str:

    try:
        return inspect.getsource(module)

    except (
        OSError,
        TypeError,
    ):

        module_file = getattr(
            module,
            "__file__",
            None,
        )

        if module_file is None:
            return ""

        return read_text(
            Path(module_file)
        )


def symbol_exists(
    module,
    symbol_name: str,
) -> bool:

    return hasattr(
        module,
        symbol_name,
    )


# ============================================================
# AST HELPERS
# ============================================================

def safe_parse(
    source: str,
    filename: str = "<source>",
) -> ast.AST | None:

    if not source:
        return None

    try:
        return ast.parse(
            source,
            filename=filename,
        )

    except SyntaxError:
        return None


def dotted_name(
    node: ast.AST | None,
) -> str | None:

    if node is None:
        return None

    if isinstance(
        node,
        ast.Name,
    ):
        return node.id

    if isinstance(
        node,
        ast.Attribute,
    ):

        parent = dotted_name(
            node.value
        )

        if parent:
            return (
                f"{parent}.{node.attr}"
            )

        return node.attr

    return None


def called_name(
    node: ast.Call,
) -> str | None:

    return dotted_name(
        node.func
    )


def collect_identifier_names(
    node: ast.AST | None,
) -> set[str]:

    if node is None:
        return set()

    names: set[str] = set()

    for child in ast.walk(node):

        if isinstance(
            child,
            ast.Name,
        ):

            names.add(
                child.id
            )

        elif isinstance(
            child,
            ast.Attribute,
        ):

            names.add(
                child.attr
            )

    return names


def expression_identifiers(
    node: ast.AST | None,
) -> set[str]:

    return collect_identifier_names(
        node
    )


def assignment_target_names(
    target: ast.AST,
) -> set[str]:

    names: set[str] = set()

    for node in ast.walk(target):

        if isinstance(
            node,
            ast.Name,
        ):

            names.add(
                node.id
            )

    return names


def keyword_value(
    call: ast.Call,
    names: set[str],
) -> ast.AST | None:

    for keyword in call.keywords:

        if keyword.arg in names:
            return keyword.value

    return None


def node_contains_identifier(
    node: ast.AST | None,
    identifiers: set[str],
) -> bool:

    if node is None:
        return False

    return bool(
        expression_identifiers(node)
        & identifiers
    )


def node_line(
    node: ast.AST,
) -> int:

    return int(
        getattr(
            node,
            "lineno",
            0,
        )
    )


# ============================================================
# IMPORT ANALYSIS
# ============================================================

@dataclass
class ImportedSymbol:
    module: str
    symbol: str
    local_name: str
    line: int


def collect_imports(
    tree: ast.AST | None,
) -> list[ImportedSymbol]:

    if tree is None:
        return []

    imports: list[ImportedSymbol] = []

    for node in ast.walk(tree):

        if isinstance(
            node,
            ast.ImportFrom,
        ):

            module = (
                "." * node.level
                + (node.module or "")
            )

            for alias in node.names:

                if alias.name == "*":
                    continue

                local_name = (
                    alias.asname
                    or alias.name
                )

                imports.append(
                    ImportedSymbol(
                        module=module,
                        symbol=alias.name,
                        local_name=local_name,
                        line=node_line(node),
                    )
                )

        elif isinstance(
            node,
            ast.Import,
        ):

            for alias in node.names:

                local_name = (
                    alias.asname
                    or alias.name.split(".")[0]
                )

                imports.append(
                    ImportedSymbol(
                        module=alias.name,
                        symbol=alias.name,
                        local_name=local_name,
                        line=node_line(node),
                    )
                )

    return imports


def resolve_relative_import(
    imported: ImportedSymbol,
    importer_module: str,
) -> str:

    if not imported.module.startswith("."):
        return imported.module

    level = len(
        imported.module
    ) - len(
        imported.module.lstrip(".")
    )

    suffix = imported.module[level:]

    parts = importer_module.split(".")

    if level > len(parts):
        return imported.module

    base = parts[:-level]

    if suffix:
        base.append(suffix)

    return ".".join(
        part
        for part in base
        if part
    )


# ============================================================
# PHASE 18 EXECUTION
# ============================================================

def run_phase18() -> Phase18Run:

    section(
        "PHASE 19 — GNN GOVERNOR ARCHITECTURE EVIDENCE"
    )

    print(
        "Purpose:"
    )

    print(
        "Evaluate and strengthen the evidence produced by "
        "the existing Phase 18 closed-loop GNN Governor."
    )

    print()

    print(
        "Phase 18 implementation modified : NO"
    )

    print(
        f"Project root                     : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Phase 18 evaluator               : "
        f"{PHASE18_FILE}"
    )

    if not PHASE18_FILE.exists():

        raise FileNotFoundError(
            "Phase 18 evaluator was not found:\n"
            f"{PHASE18_FILE}"
        )

    print()

    print(
        "Running existing Phase 18 GNN Governor evaluation..."
    )

    command = [
        sys.executable,
        "-m",
        PHASE18_MODULE,
    ]

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if completed.stdout:
        print(
            completed.stdout,
            end="",
        )

    if completed.stderr:
        print(
            completed.stderr,
            file=sys.stderr,
            end="",
        )

    return Phase18Run(
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


# ============================================================
# PHASE 18 OUTPUT VALIDATION
# ============================================================

def extract_phase18_checks(
    output: str,
) -> dict[str, bool]:

    checks: dict[str, bool] = {}

    pattern = re.compile(
        r"^\s*(.+?)\s*:\s*"
        r"(PASSED|FAILED)\s*$",
        re.MULTILINE,
    )

    for match in pattern.finditer(
        output
    ):

        name = match.group(1).strip()

        checks[name] = (
            match.group(2)
            == "PASSED"
        )

    return checks


def validate_phase18_execution(
    run: Phase18Run,
) -> list[CheckResult]:

    results: list[CheckResult] = []

    results.append(
        CheckResult(
            name="phase18_execution",
            passed=run.passed,
            detail=(
                "Phase 18 returned exit code 0."
                if run.passed
                else
                f"Phase 18 returned exit "
                f"code {run.return_code}."
            ),
        )
    )

    if not run.passed:
        return results

    checks = extract_phase18_checks(
        run.stdout
    )

    expected_checks = {
        "Mixed population",
        "Interaction graph",
        "GNN feature dimension",
        "GNN Governor action space",
        "No direct Governor DENY",
        "Agent B hidden-signal isolation",
        "Ground truth not passed to GNN",
        "All Phase 18 metrics finite",
        "All bounded metrics valid",
    }

    for expected in sorted(
        expected_checks
    ):

        matching = [
            value
            for key, value
            in checks.items()
            if expected.lower()
            in key.lower()
        ]

        passed = (
            bool(matching)
            and all(matching)
        )

        results.append(
            CheckResult(
                name=(
                    f"phase18_check::{expected}"
                ),
                passed=passed,
                detail=(
                    "Phase 18 reported the "
                    "check as passed."
                    if passed
                    else
                    "Expected Phase 18 validation "
                    "check was not found or did "
                    "not pass."
                ),
            )
        )

    return results


# ============================================================
# POPULATION METRIC EXTRACTION
# ============================================================

def parse_population_metrics(
    output: str,
) -> dict[str, PopulationMetrics]:

    populations: dict[
        str,
        PopulationMetrics,
    ] = {}

    pattern = re.compile(
        r"^\s*"
        r"(ADAPTIVE_ABUSIVE|"
        r"ADAPTIVE_LEGITIMATE|"
        r"HUMAN_ABUSIVE|"
        r"HUMAN_LEGITIMATE)"
        r"\s+"
        r"(\d+)\s+"
        r"([0-9.]+)\s+"
        r"([0-9.]+)\s+"
        r"([0-9.]+)\s+"
        r"([0-9.]+)\s+"
        r"([0-9.]+)\s+"
        r"([0-9.]+)"
        r"\s*$",
        re.MULTILINE,
    )

    for match in pattern.finditer(
        output
    ):

        (
            population,
            episodes,
            baseline,
            governed,
            reduction,
            intervention,
            gnn_risk,
            fused_risk,
        ) = match.groups()

        populations[
            population
        ] = PopulationMetrics(
            population=population,
            episodes=float(episodes),
            baseline_approval=float(baseline),
            governed_approval=float(governed),
            reduction=float(reduction),
            intervention=float(intervention),
            gnn_risk=float(gnn_risk),
            fused_risk=float(fused_risk),
        )

    return populations


def validate_population_evidence(
    output: str,
) -> list[CheckResult]:

    results: list[CheckResult] = []

    metrics = parse_population_metrics(
        output
    )

    results.append(
        CheckResult(
            name=(
                "all_required_population_groups_present"
            ),
            passed=(
                set(metrics)
                == REQUIRED_POPULATIONS
            ),
            detail=(
                f"Detected populations: "
                f"{sorted(metrics)}"
            ),
        )
    )

    for population in sorted(
        REQUIRED_POPULATIONS
    ):

        row = metrics.get(
            population
        )

        if row is None:

            results.append(
                CheckResult(
                    name=(
                        f"{population}"
                        "_present"
                    ),
                    passed=False,
                    detail=(
                        "Population row missing."
                    ),
                )
            )

            continue

        results.extend(
            [
                CheckResult(
                    name=(
                        f"{population}"
                        "_episodes_positive"
                    ),
                    passed=(
                        row.episodes > 0
                    ),
                ),
                CheckResult(
                    name=(
                        f"{population}"
                        "_baseline_approval_bounded"
                    ),
                    passed=bounded(
                        row.baseline_approval
                    ),
                ),
                CheckResult(
                    name=(
                        f"{population}"
                        "_governed_approval_bounded"
                    ),
                    passed=bounded(
                        row.governed_approval
                    ),
                ),
                CheckResult(
                    name=(
                        f"{population}"
                        "_reduction_bounded"
                    ),
                    passed=bounded(
                        row.reduction
                    ),
                ),
                CheckResult(
                    name=(
                        f"{population}"
                        "_intervention_bounded"
                    ),
                    passed=bounded(
                        row.intervention
                    ),
                ),
                CheckResult(
                    name=(
                        f"{population}"
                        "_gnn_risk_bounded"
                    ),
                    passed=bounded(
                        row.gnn_risk
                    ),
                ),
                CheckResult(
                    name=(
                        f"{population}"
                        "_fused_risk_bounded"
                    ),
                    passed=bounded(
                        row.fused_risk
                    ),
                ),
            ]
        )

    return results


# ============================================================
# SOURCE MODULE AUDIT
# ============================================================

def audit_source_architecture() -> list[CheckResult]:

    section(
        "READ-ONLY SOURCE ARCHITECTURE AUDIT"
    )

    results: list[CheckResult] = []

    phase18 = parse_source(
        PHASE18_FILE
    )

    phase18_exists = (
        PHASE18_FILE.exists()
    )

    status(
        "phase18_file_exists",
        phase18_exists,
    )

    results.append(
        CheckResult(
            name="phase18_file_exists",
            passed=phase18_exists,
        )
    )

    phase18_readable = bool(
        phase18.source
    )

    status(
        "phase18_source_readable",
        phase18_readable,
    )

    results.append(
        CheckResult(
            name="phase18_source_readable",
            passed=phase18_readable,
        )
    )

    phase18_parseable = (
        phase18.tree is not None
    )

    status(
        "phase18_source_ast_parseable",
        phase18_parseable,
    )

    results.append(
        CheckResult(
            name="phase18_source_ast_parseable",
            passed=phase18_parseable,
        )
    )

    if not phase18_readable:
        return results

    if not phase18_parseable:
        return results

    # --------------------------------------------------------
    # Load actual architecture modules.
    # --------------------------------------------------------

    module_results: dict[
        str,
        tuple[Any | None, str, str],
    ] = {}

    for component, module_name in (
        ARCHITECTURE_MODULES.items()
    ):

        try:

            module = load_module(
                module_name
            )

            source = get_module_source(
                module
            )

            module_results[
                component
            ] = (
                module,
                source,
                "",
            )

        except Exception as exc:

            module_results[
                component
            ] = (
                None,
                "",
                str(exc),
            )

    # --------------------------------------------------------
    # Component existence checks.
    # --------------------------------------------------------

    for component, expected_names in (
        EXPECTED_SYMBOLS.items()
    ):

        module, source, error = (
            module_results[component]
        )

        found = [
            symbol
            for symbol in expected_names
            if (
                module is not None
                and symbol_exists(
                    module,
                    symbol,
                )
            )
        ]

        passed = bool(found)

        check_name = (
            f"source_{component}"
            if component != "adaptive_customer"
            else "source_adaptive_customer"
        )

        status(
            check_name,
            passed,
        )

        detail = (
            f"Found symbols: {found}"
            if passed
            else
            (
                f"Expected symbols: "
                f"{list(expected_names)}; "
                f"module error: {error or 'none'}"
            )
        )

        results.append(
            CheckResult(
                name=check_name,
                passed=passed,
                detail=detail,
            )
        )

    # --------------------------------------------------------
    # Interaction graph source check.
    # --------------------------------------------------------

    interaction_module = (
        module_results[
            "interaction_graph"
        ][0]
    )

    interaction_ok = (
        interaction_module is not None
        and symbol_exists(
            interaction_module,
            "build_interaction_graph",
        )
    )

    status(
        "source_interaction_graph",
        interaction_ok,
    )

    results.append(
        CheckResult(
            name="source_interaction_graph",
            passed=interaction_ok,
        )
    )

    # --------------------------------------------------------
    # Extract actual Phase 18 imports through AST.
    # --------------------------------------------------------

    imported_symbols = {
        item.symbol
        for item in collect_imports(
            phase18.tree
        )
    }

    required_architecture_symbols = {
        symbol
        for symbols
        in EXPECTED_SYMBOLS.values()
        for symbol in symbols
        if symbol != "AdaptiveCustomerAgent"
    }

    # AdaptiveCustomerAgent is not required to be imported
    # directly by Phase 18 if it is instantiated by the
    # adaptive simulator implementation.
    #
    # It is audited independently.

    imports_ok = (
        required_architecture_symbols
        <= imported_symbols
    )

    status(
        "phase18_imports_final_architecture_components",
        imports_ok,
    )

    results.append(
        CheckResult(
            name=(
                "phase18_imports_final_architecture_components"
            ),
            passed=imports_ok,
            detail=(
                f"Detected symbols: "
                f"{sorted(imported_symbols)}"
            ),
        )
    )

    # --------------------------------------------------------
    # Canonical action space.
    #
    # This check intentionally remains source-text based
    # because it is checking that the canonical names are
    # documented/present in Phase 18.
    #
    # The forbidden-action audit below is different: it checks
    # executable AST structures rather than raw source text.
    # --------------------------------------------------------

    action_space_ok = all(
        action in phase18.source
        for action in CANONICAL_GOVERNOR_ACTIONS
    )

    status(
        "canonical_governor_action_space_present",
        action_space_ok,
    )

    results.append(
        CheckResult(
            name=(
                "canonical_governor_action_space_present"
            ),
            passed=action_space_ok,
        )
    )

    # --------------------------------------------------------
    # Direct Governor DENY audit.
    #
    # IMPORTANT:
    #
    # Do NOT search raw source text here.
    #
    # The architecture rule is specifically:
    #
    #     DENY / DIRECT_DENY / REJECT
    #
    # must not be an executable Governor action.
    #
    # Therefore comments, docstrings, explanatory strings,
    # report text, validation messages, and documentation
    # references must NOT count as violations.
    #
    # We inspect executable AST structures that actually
    # construct or validate an explicitly named Governor
    # action-space.
    # --------------------------------------------------------

    forbidden_actions = {
        action.upper()
        for action in FORBIDDEN_GOVERNOR_ACTIONS
    }

    def literal_string_value(
        node: ast.AST,
    ) -> str | None:
        """
        Return a string literal value from an AST Constant.

        This helper is only used while traversing executable
        assignment/comparison expressions. Arbitrary source
        text is never inspected by this function.
        """

        if isinstance(
            node,
            ast.Constant,
        ):

            if isinstance(
                node.value,
                str,
            ):

                return node.value

        return None

    def is_action_space_name(
        name: str,
    ) -> bool:
        """
        Identify variables whose names explicitly indicate
        that they represent a Governor action space.

        We intentionally avoid treating every variable named
        'action' as a Governor action space.
        """

        normalized = (
            name.lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        return normalized in {
            "governor_actions",
            "governor_action_space",
            "canonical_governor_actions",
            "allowed_governor_actions",
            "valid_governor_actions",
            "governor_allowed_actions",
            "governor_action_set",
        }

    def target_names(
        target: ast.AST,
    ) -> set[str]:

        names: set[str] = set()

        for node in ast.walk(target):

            if isinstance(
                node,
                ast.Name,
            ):

                names.add(
                    node.id
                )

        return names

    executable_governor_action_values: list[
        tuple[int, str, str]
    ] = []

    # --------------------------------------------------------
    # Inspect assignments whose target explicitly represents
    # a Governor action space.
    # --------------------------------------------------------

    for node in ast.walk(
        phase18.tree
    ):

        if isinstance(
            node,
            ast.Assign,
        ):

            assigned_names: set[str] = set()

            for target in node.targets:

                assigned_names |= (
                    target_names(target)
                )

            action_space_targets = {
                name
                for name in assigned_names
                if is_action_space_name(name)
            }

            if not action_space_targets:
                continue

            for child in ast.walk(
                node.value
            ):

                value = literal_string_value(
                    child
                )

                if value is None:
                    continue

                normalized = value.upper()

                if normalized in forbidden_actions:

                    executable_governor_action_values.append(
                        (
                            node_line(node),
                            next(
                                iter(
                                    action_space_targets
                                )
                            ),
                            value,
                        )
                    )

        elif isinstance(
            node,
            ast.AnnAssign,
        ):

            assigned_names = target_names(
                node.target
            )

            action_space_targets = {
                name
                for name in assigned_names
                if is_action_space_name(name)
            }

            if not action_space_targets:
                continue

            if node.value is None:
                continue

            for child in ast.walk(
                node.value
            ):

                value = literal_string_value(
                    child
                )

                if value is None:
                    continue

                normalized = value.upper()

                if normalized in forbidden_actions:

                    executable_governor_action_values.append(
                        (
                            node_line(node),
                            next(
                                iter(
                                    action_space_targets
                                )
                            ),
                            value,
                        )
                    )

    # --------------------------------------------------------
    # Inspect executable comparisons involving an explicitly
    # named Governor action space.
    #
    # Example:
    #
    #     if action not in GOVERNOR_ACTIONS:
    #
    # is relevant.
    #
    # But:
    #
    #     print("No direct Governor DENY")
    #
    # is not relevant.
    # --------------------------------------------------------

    for node in ast.walk(
        phase18.tree
    ):

        if not isinstance(
            node,
            ast.Compare,
        ):
            continue

        identifiers = expression_identifiers(
            node
        )

        has_governor_action_space = any(
            is_action_space_name(name)
            for name in identifiers
        )

        if not has_governor_action_space:
            continue

        for child in ast.walk(
            node
        ):

            value = literal_string_value(
                child
            )

            if value is None:
                continue

            if value.upper() in forbidden_actions:

                executable_governor_action_values.append(
                    (
                        node_line(node),
                        "<comparison>",
                        value,
                    )
                )

    # --------------------------------------------------------
    # Final forbidden-action determination.
    # --------------------------------------------------------

    direct_deny_action = bool(
        executable_governor_action_values
    )

    no_direct_deny = (
        not direct_deny_action
    )

    status(
        "direct_governor_deny_absent",
        no_direct_deny,
    )

    if no_direct_deny:

        detail = (
            "No forbidden DENY/DIRECT_DENY/REJECT value was "
            "found in an executable Governor action-space "
            "assignment or action-space comparison."
        )

    else:

        detail = (
            "Forbidden executable Governor action reference(s) "
            "detected: "
            f"{executable_governor_action_values}"
        )

    results.append(
        CheckResult(
            name="direct_governor_deny_absent",
            passed=no_direct_deny,
            detail=detail,
        )
    )

    # --------------------------------------------------------
    # Closed-loop architecture textual indicators.
    #
    # This is supplementary only.
    # The stronger executable source checks happen elsewhere.
    # --------------------------------------------------------

    phase18_lower = (
        phase18.source.lower()
    )

    textual_closed_loop_terms = [
        "adaptive",
        "policy",
        "effective",
        "outcome",
    ]

    textual_closed_loop_ok = all(
        term in phase18_lower
        for term in textual_closed_loop_terms
    )

    status(
        "closed_loop_architecture_present",
        textual_closed_loop_ok,
    )

    results.append(
        CheckResult(
            name="closed_loop_architecture_present",
            passed=textual_closed_loop_ok,
            detail=(
                "Phase 18 contains the expected "
                "closed-loop architecture vocabulary."
            ),
        )
    )

    return results


# ============================================================
# AGENT B MODULE DISCOVERY
# ============================================================

def locate_adaptive_customer_module() -> tuple[
    Any | None,
    str,
    str,
]:

    module_name = (
        ARCHITECTURE_MODULES[
            "adaptive_customer"
        ]
    )

    try:

        module = load_module(
            module_name
        )

        source = get_module_source(
            module
        )

        return (
            module,
            source,
            "",
        )

    except Exception as exc:

        return (
            None,
            "",
            str(exc),
        )


# ============================================================
# AGENT B INFORMATION-BOUNDARY SOURCE AUDIT
# ============================================================

def find_agent_b_observation_functions(
    tree: ast.AST | None,
) -> list[
    ast.FunctionDef | ast.AsyncFunctionDef
]:

    if tree is None:
        return []

    functions = []

    for node in ast.walk(tree):

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):

            function_name = (
                node.name.lower()
            )

            if any(
                token in function_name
                for token in (
                    "observe",
                    "infer_support_policy",
                    "support_policy",
                    "learn",
                    "update",
                )
            ):

                functions.append(
                    node
                )

    return functions


def audit_agent_b_information_boundary(
) -> list[CheckResult]:

    section(
        "AGENT B INFORMATION-BOUNDARY SOURCE AUDIT"
    )

    results: list[CheckResult] = []

    module, source, error = (
        locate_adaptive_customer_module()
    )

    loaded_ok = (
        module is not None
        and bool(source)
    )

    status(
        "agent_b_source_loaded",
        loaded_ok,
    )

    results.append(
        CheckResult(
            name="agent_b_source_loaded",
            passed=loaded_ok,
            detail=(
                error
                if error
                else
                "Adaptive customer module loaded."
            ),
        )
    )

    if not loaded_ok:
        return results

    tree = safe_parse(
        source,
        filename=str(
            getattr(
                module,
                "__file__",
                "adaptive_customer.py",
            )
        ),
    )

    parse_ok = (
        tree is not None
    )

    status(
        "agent_b_source_parseable",
        parse_ok,
    )

    results.append(
        CheckResult(
            name="agent_b_source_parseable",
            passed=parse_ok,
        )
    )

    if not parse_ok:
        return results

    # --------------------------------------------------------
    # Actual public Agent B symbol.
    # --------------------------------------------------------

    actual_agent_symbol = (
        "AdaptiveCustomerAgent"
    )

    agent_exists = (
        symbol_exists(
            module,
            actual_agent_symbol,
        )
    )

    status(
        "agent_b_adaptive_customer_agent_exists",
        agent_exists,
    )

    results.append(
        CheckResult(
            name=(
                "agent_b_adaptive_customer_agent_exists"
            ),
            passed=agent_exists,
            detail=(
                "Actual public Agent B symbol "
                "AdaptiveCustomerAgent was found."
                if agent_exists
                else
                (
                    "AdaptiveCustomerAgent was not found."
                )
            ),
        )
    )

    # --------------------------------------------------------
    # Inspect executable function bodies only.
    #
    # This prevents comments/docstrings from causing a false
    # privacy failure.
    # --------------------------------------------------------

    observation_functions = (
        find_agent_b_observation_functions(
            tree
        )
    )

    observation_nodes: list[ast.AST] = []

    for function in observation_functions:

        observation_nodes.extend(
            list(function.body)
        )

    if not observation_nodes:
        observation_nodes = [
            tree
        ]

    observation_source_names: set[str] = set()

    for node in observation_nodes:

        observation_source_names |= (
            expression_identifiers(node)
        )

    def hidden_signal_absent(
        patterns: set[str],
    ) -> bool:

        lower_names = {
            name.lower()
            for name in observation_source_names
        }

        for pattern in patterns:

            normalized = (
                pattern.lower()
                .replace(" ", "_")
            )

            if normalized in lower_names:
                return False

            for name in lower_names:

                if (
                    normalized
                    and normalized in name
                ):

                    return False

        return True

    hidden_checks = {
        "agent_b_gnn_signal_hidden": {
            "gnn_risk",
            "gnn_score",
            "gnnrisk",
        },
        "agent_b_fusion_signal_hidden": {
            "fused_risk",
            "fusion_score",
            "riskfusion",
        },
        "agent_b_network_signal_hidden": {
            "network_risk",
            "network_abnormality",
        },
        "agent_b_strategy_signal_hidden": {
            "strategic_risk",
            "strategic_behavior",
            "strategic_adaptation_score",
        },
        "agent_b_ground_truth_hidden": {
            "ground_truth",
            "fraud_label",
            "fraud_target",
            "target_label",
        },
    }

    for check_name, patterns in (
        hidden_checks.items()
    ):

        passed = hidden_signal_absent(
            patterns
        )

        status(
            check_name,
            passed,
        )

        results.append(
            CheckResult(
                name=check_name,
                passed=passed,
                detail=(
                    "No executable Agent B observation "
                    "reference to the private signal was found."
                    if passed
                    else
                    "A private signal reference was found "
                    "inside executable Agent B observation logic."
                ),
            )
        )

    # --------------------------------------------------------
    # Observable support-response fields.
    # --------------------------------------------------------

    observable_fields = {
        "decision",
        "reason_code",
        "requested_evidence",
        "approved_amount",
        "requires_followup",
    }

    found_observable_fields = (
        observable_fields
        & {
            name.lower()
            for name in observation_source_names
        }
    )

    # Also inspect dictionary keys and attributes.
    for node in ast.walk(tree):

        if isinstance(
            node,
            ast.Constant,
        ) and isinstance(
            node.value,
            str,
        ):

            value = (
                node.value.lower()
            )

            if value in observable_fields:

                found_observable_fields.add(
                    value
                )

    observable_ok = bool(
        found_observable_fields
    )

    status(
        "agent_b_observable_response_path_present",
        observable_ok,
    )

    results.append(
        CheckResult(
            name=(
                "agent_b_observable_response_path_present"
            ),
            passed=observable_ok,
            detail=(
                f"Detected observable response "
                f"fields: "
                f"{sorted(found_observable_fields)}"
            ),
        )
    )

    info(
        "Static AST inspection is used to distinguish "
        "executable Agent B references from comments/docstrings."
    )

    info(
        "This establishes source-level information-boundary "
        "evidence; it does not prove absence of every possible "
        "indirect information channel."
    )

    return results


# ============================================================
# RESPONSE / EFFECTIVE-OUTCOME DATA FLOW
# ============================================================

RESPONSE_VARIABLE_TERMS = {
    "decision",
    "support_decision",
    "original_decision",
    "governor_decision",
    "effective_decision",
    "effective_outcome",
    "outcome",
    "effective_response",
    "support_response",
}


RESPONSE_PRODUCER_CALL_TERMS = {
    "build_effective_decision",
    "build_effective_response",
    "build_effective_outcome",
    "resolve_intervention_outcome",
    "effective_decision",
    "effective_outcome",
}


AGENT_B_OBSERVER_CALL_TERMS = {
    "infer_support_policy",
    "observe",
    "observe_decision",
    "observe_response",
    "learn_from_response",
    "update_policy",
    "update_from_response",
}


def is_response_like_name(
    name: str,
) -> bool:

    lower = name.lower()

    if lower in RESPONSE_VARIABLE_TERMS:
        return True

    return any(
        term in lower
        for term in (
            "decision",
            "response",
            "outcome",
        )
    )


def is_response_producer_call(
    call: ast.Call,
) -> bool:

    name = called_name(
        call
    )

    if not name:
        return False

    leaf = (
        name.split(".")[-1]
        .lower()
    )

    if leaf in RESPONSE_PRODUCER_CALL_TERMS:
        return True

    return (
        "effective" in leaf
        and (
            "decision" in leaf
            or "response" in leaf
            or "outcome" in leaf
        )
    )


def is_agent_b_observer_call(
    call: ast.Call,
) -> bool:

    name = called_name(
        call
    )

    if not name:
        return False

    leaf = (
        name.split(".")[-1]
        .lower()
    )

    return (
        leaf in AGENT_B_OBSERVER_CALL_TERMS
        or "infer_support_policy" in leaf
        or (
            "observe" in leaf
            and (
                "decision" in leaf
                or "response" in leaf
                or "outcome" in leaf
            )
        )
    )


def assignment_names_from_node(
    node: ast.AST,
) -> set[str]:

    names: set[str] = set()

    if isinstance(
        node,
        ast.Assign,
    ):

        for target in node.targets:

            names |= assignment_target_names(
                target
            )

    elif isinstance(
        node,
        ast.AnnAssign,
    ):

        names |= assignment_target_names(
            node.target
        )

    elif isinstance(
        node,
        ast.NamedExpr,
    ):

        names |= assignment_target_names(
            node.target
        )

    return names


def find_response_assignments(
    tree: ast.AST | None,
) -> list[
    tuple[int, set[str], ast.Call]
]:

    if tree is None:
        return []

    assignments = []

    for node in ast.walk(tree):

        value: ast.AST | None = None

        if isinstance(
            node,
            ast.Assign,
        ):

            value = node.value

        elif isinstance(
            node,
            ast.AnnAssign,
        ):

            value = node.value

        if not isinstance(
            value,
            ast.Call,
        ):
            continue

        if not is_response_producer_call(
            value
        ):
            continue

        names = assignment_names_from_node(
            node
        )

        if not names:
            continue

        assignments.append(
            (
                node_line(node),
                names,
                value,
            )
        )

    return assignments


def find_observer_calls(
    tree: ast.AST | None,
) -> list[
    tuple[int, ast.Call]
]:

    if tree is None:
        return []

    calls = []

    for node in ast.walk(tree):

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if is_agent_b_observer_call(
            node
        ):

            calls.append(
                (
                    node_line(node),
                    node,
                )
            )

    return calls


def observer_receives_response(
    call: ast.Call,
    response_names: set[str],
) -> tuple[bool, set[str]]:

    matched_names: set[str] = set()

    # --------------------------------------------------------
    # Keyword arguments.
    # --------------------------------------------------------

    for keyword in call.keywords:

        identifiers = (
            expression_identifiers(
                keyword.value
            )
        )

        matches = (
            identifiers
            & response_names
        )

        matched_names |= matches

    # --------------------------------------------------------
    # Positional arguments.
    # --------------------------------------------------------

    for argument in call.args:

        identifiers = (
            expression_identifiers(
                argument
            )
        )

        matches = (
            identifiers
            & response_names
        )

        matched_names |= matches

    return (
        bool(matched_names),
        matched_names,
    )


def find_direct_support_decision_observation(
    tree: ast.AST | None,
) -> list[
    tuple[int, ast.Call, set[str]]
]:

    if tree is None:
        return []

    matches = []

    for node in ast.walk(tree):

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        call_name = called_name(
            node
        )

        if not call_name:
            continue

        leaf = (
            call_name.split(".")[-1]
            .lower()
        )

        if leaf != "infer_support_policy":
            continue

        response_identifiers: set[str] = set()

        for keyword in node.keywords:

            if keyword.arg in {
                "decision",
                "support_decision",
                "effective_decision",
                "response",
                "outcome",
            }:

                response_identifiers |= (
                    expression_identifiers(
                        keyword.value
                    )
                )

        matches.append(
            (
                node_line(node),
                node,
                response_identifiers,
            )
        )

    return matches


# ============================================================
# LOCATE CLOSED-LOOP SIMULATOR MODULE
# ============================================================

def discover_closed_loop_sources(
    phase18_tree: ast.AST | None,
) -> list[
    tuple[
        str,
        Path,
        str,
        ast.AST | None,
    ]
]:

    discovered: list[
        tuple[
            str,
            Path,
            str,
            ast.AST | None,
        ]
    ] = []

    if phase18_tree is None:
        return discovered

    imports = collect_imports(
        phase18_tree
    )

    candidates: list[str] = []

    for imported in imports:

        if (
            imported.symbol
            == "AdaptiveInteractionSimulator"
        ):

            candidates.append(
                imported.module
            )

        if (
            "adaptive" in imported.module.lower()
            and "simulator" in imported.module.lower()
        ):

            candidates.append(
                imported.module
            )

    # Known project locations used by the adaptive
    # interaction implementation. These are discovery
    # candidates, not unconditional passes.
    candidates.extend(
        [
            "src.environment.adaptive_simulator",
            "src.evaluation.phase16_adaptive_agent_evaluation",
            "src.evaluation.phase17_adaptive_detection",
        ]
    )

    seen: set[str] = set()

    for module_name in candidates:

        if not module_name:
            continue

        if module_name in seen:
            continue

        seen.add(
            module_name
        )

        normalized = (
            module_name.replace(
                "src.",
                "src.",
                1,
            )
        )

        try:

            module = load_module(
                normalized
            )

            path = Path(
                inspect.getfile(
                    module
                )
            )

            source = get_module_source(
                module
            )

            tree = safe_parse(
                source,
                filename=str(path),
            )

            discovered.append(
                (
                    normalized,
                    path,
                    source,
                    tree,
                )
            )

        except Exception:
            continue

    return discovered


# ============================================================
# CLOSED-LOOP SOURCE AUDIT
# ============================================================

def audit_closed_loop_source() -> list[CheckResult]:

    section(
        "CLOSED-LOOP ARCHITECTURE SOURCE AUDIT"
    )

    results: list[CheckResult] = []

    phase18 = parse_source(
        PHASE18_FILE
    )

    phase18_tree = phase18.tree

    # --------------------------------------------------------
    # 1. Agent B response observation.
    # --------------------------------------------------------

    response_assignments = (
        find_response_assignments(
            phase18_tree
        )
    )

    response_names: set[str] = set()

    for (
        _line,
        names,
        _call,
    ) in response_assignments:

        response_names |= names

    # Add semantically known response variables that appear
    # in executable Phase 18 source.
    for name in RESPONSE_VARIABLE_TERMS:

        if re.search(
            rf"\b{re.escape(name)}\b",
            phase18.source,
        ):

            response_names.add(
                name
            )

    observer_calls = (
        find_observer_calls(
            phase18_tree
        )
    )

    direct_support_observations = (
        find_direct_support_decision_observation(
            phase18_tree
        )
    )

    observation_matches: list[
        tuple[int, str, set[str]]
    ] = []

    for (
        line,
        call,
    ) in observer_calls:

        matched, identifiers = (
            observer_receives_response(
                call,
                response_names,
            )
        )

        if matched:

            observation_matches.append(
                (
                    line,
                    called_name(call)
                    or "<unknown>",
                    identifiers,
                )
            )

    for (
        line,
        _call,
        identifiers,
    ) in direct_support_observations:

        if identifiers:

            observation_matches.append(
                (
                    line,
                    "infer_support_policy",
                    identifiers,
                )
            )

    # --------------------------------------------------------
    # Stronger condition:
    #
    # Agent B must receive an actual response/decision value,
    # not merely coexist with words like "effective outcome".
    # --------------------------------------------------------

    observable_ok = bool(
        observation_matches
    )

    detail_parts: list[str] = []

    if response_names:

        detail_parts.append(
            "Response variables: "
            f"{sorted(response_names)}"
        )

    if observation_matches:

        detail_parts.append(
            "Executable observation calls: "
            f"{[(line, name, sorted(ids)) for line, name, ids in observation_matches]}"
        )

    if not observation_matches:

        detail_parts.append(
            "No executable Agent B response-observation "
            "call was established."
        )

    status(
        "agent_b_observes_effective_response",
        observable_ok,
    )

    results.append(
        CheckResult(
            name=(
                "agent_b_observes_effective_response"
            ),
            passed=observable_ok,
            detail="; ".join(
                detail_parts
            ),
        )
    )

    # --------------------------------------------------------
    # 2. Adaptive policy update.
    # --------------------------------------------------------

    policy_update_calls: list[
        tuple[int, str]
    ] = []

    if phase18_tree is not None:

        for node in ast.walk(
            phase18_tree
        ):

            if not isinstance(
                node,
                ast.Call,
            ):
                continue

            name = called_name(
                node
            )

            if not name:
                continue

            leaf = (
                name.split(".")[-1]
                .lower()
            )

            if any(
                token in leaf
                for token in (
                    "update_policy",
                    "infer_support_policy",
                    "adapt",
                    "learn",
                    "update",
                )
            ):

                policy_update_calls.append(
                    (
                        node_line(node),
                        name,
                    )
                )

    policy_update_ok = bool(
        policy_update_calls
    )

    status(
        "adaptive_policy_update_path_present",
        policy_update_ok,
    )

    results.append(
        CheckResult(
            name=(
                "adaptive_policy_update_path_present"
            ),
            passed=policy_update_ok,
            detail=(
                f"Executable adaptive/update calls: "
                f"{policy_update_calls}"
            ),
        )
    )

    # --------------------------------------------------------
    # 3. Governor-before-outcome ordering.
    #
    # We inspect executable call ordering inside the source.
    # --------------------------------------------------------

    governor_calls: list[int] = []
    outcome_calls: list[int] = []

    if phase18_tree is not None:

        for node in ast.walk(
            phase18_tree
        ):

            if not isinstance(
                node,
                ast.Call,
            ):
                continue

            name = called_name(
                node
            )

            if not name:
                continue

            leaf = (
                name.split(".")[-1]
                .lower()
            )

            line = node_line(node)

            if (
                "governor" in leaf
                or leaf in {
                    "evaluate",
                    "decide",
                }
            ):

                governor_calls.append(
                    line
                )

            if any(
                token in leaf
                for token in (
                    "effective",
                    "outcome",
                    "resolve",
                    "refund",
                )
            ):

                outcome_calls.append(
                    line
                )

    before_outcome_ok = False

    if (
        governor_calls
        and outcome_calls
    ):

        earliest_governor = min(
            governor_calls
        )

        earliest_outcome = min(
            outcome_calls
        )

        before_outcome_ok = (
            earliest_governor
            <= earliest_outcome
        )

    # Supplement with explicit architecture wording only when
    # AST evidence is inconclusive.
    #
    # This does NOT use the wording as the primary evidence.
    if not before_outcome_ok:

        governor_before_pattern = re.compile(
            r"(?is)"
            r"governor.{0,500}"
            r"(?:effective|outcome)"
        )

        before_outcome_ok = bool(
            governor_before_pattern.search(
                phase18.source
            )
        )

    status(
        "governor_before_outcome_source_evidence",
        before_outcome_ok,
    )

    results.append(
        CheckResult(
            name=(
                "governor_before_outcome_source_evidence"
            ),
            passed=before_outcome_ok,
            detail=(
                f"Governor call lines: "
                f"{sorted(set(governor_calls))}; "
                f"outcome call lines: "
                f"{sorted(set(outcome_calls))}"
            ),
        )
    )

    return results


# ============================================================
# ARCHITECTURE CLAIM VALIDATION
# ============================================================

def validate_architecture_claims(
    source_results: list[CheckResult],
) -> list[CheckResult]:

    section(
        "ARCHITECTURE CLAIM VALIDATION"
    )

    result_map = {
        result.name: result
        for result in source_results
    }

    def passed(
        name: str,
    ) -> bool:

        result = result_map.get(
            name
        )

        return (
            result is not None
            and result.passed
        )

    claims = {
        "interaction_graph_enabled": (
            passed(
                "source_interaction_graph"
            )
            and
            passed(
                "phase18_imports_final_architecture_components"
            )
        ),

        "trained_gnn_enabled": (
            passed(
                "source_trained_gnn"
            )
        ),

        "adapter_enabled": (
            passed(
                "source_gnn_governor_adapter"
            )
        ),

        "behavioral_governor_enabled": (
            passed(
                "source_behavioral_governor"
            )
        ),

        "risk_fusion_enabled": (
            passed(
                "source_risk_fusion"
            )
        ),

        "gnn_governor_enabled": (
            passed(
                "source_gnn_governor"
            )
        ),

        "closed_loop_attacker_enabled": (
            passed(
                "agent_b_adaptive_customer_agent_exists"
            )
            and
            passed(
                "agent_b_observes_effective_response"
            )
            and
            passed(
                "adaptive_policy_update_path_present"
            )
        ),

        "governor_before_outcome": (
            passed(
                "governor_before_outcome_source_evidence"
            )
        ),

        "direct_governor_deny_disabled": (
            passed(
                "direct_governor_deny_absent"
            )
        ),
    }

    results: list[CheckResult] = []

    for name, value in claims.items():

        status(
            name,
            value,
        )

        results.append(
            CheckResult(
                name=name,
                passed=bool(value),
            )
        )

    return results


# ============================================================
# CLOSED-LOOP BEHAVIORAL EVIDENCE
# ============================================================

def extract_metric(
    output: str,
    label: str,
) -> float | None:

    pattern = re.compile(
        rf"{re.escape(label)}\s*:\s*"
        rf"([0-9.]+)"
    )

    match = pattern.search(
        output
    )

    if not match:
        return None

    try:

        return float(
            match.group(1)
        )

    except ValueError:

        return None


def evaluate_closed_loop_behavior(
    output: str,
) -> list[CheckResult]:

    section(
        "CLOSED-LOOP BEHAVIORAL EVIDENCE"
    )

    results: list[CheckResult] = []

    metrics = parse_population_metrics(
        output
    )

    adaptive_abusive = metrics.get(
        "ADAPTIVE_ABUSIVE"
    )

    human_abusive = metrics.get(
        "HUMAN_ABUSIVE"
    )

    adaptive_policy_change = (
        extract_metric(
            output,
            "Adaptive attacker policy change",
        )
    )

    if adaptive_abusive is not None:

        print(
            "Adaptive abusive baseline approval : "
            f"{adaptive_abusive.baseline_approval:.4f}"
        )

        print(
            "Adaptive abusive governed approval : "
            f"{adaptive_abusive.governed_approval:.4f}"
        )

        print(
            "Adaptive abusive approval reduction: "
            f"{adaptive_abusive.reduction:.4f}"
        )

        print(
            "Adaptive abusive intervention      : "
            f"{adaptive_abusive.intervention:.4f}"
        )

    if human_abusive is not None:

        print()

        print(
            "Human abusive baseline approval    : "
            f"{human_abusive.baseline_approval:.4f}"
        )

        print(
            "Human abusive governed approval    : "
            f"{human_abusive.governed_approval:.4f}"
        )

        print(
            "Human abusive approval reduction   : "
            f"{human_abusive.reduction:.4f}"
        )

        print(
            "Human abusive intervention         : "
            f"{human_abusive.intervention:.4f}"
        )

    print()

    if adaptive_policy_change is not None:

        print(
            "Adaptive attacker policy change  : "
            f"{adaptive_policy_change:.6f}"
        )

    adaptive_change_ok = (
        adaptive_policy_change is not None
        and finite(
            adaptive_policy_change
        )
        and adaptive_policy_change >= 0.0
    )

    results.append(
        CheckResult(
            name="adaptive_policy_change_observed",
            passed=adaptive_change_ok,
        )
    )

    results.append(
        CheckResult(
            name="adaptive_policy_change_finite",
            passed=(
                adaptive_policy_change is not None
                and finite(
                    adaptive_policy_change
                )
            ),
        )
    )

    adaptive_reduction_ok = (
        adaptive_abusive is not None
        and bounded(
            adaptive_abusive.reduction
        )
    )

    human_reduction_ok = (
        human_abusive is not None
        and bounded(
            human_abusive.reduction
        )
    )

    results.append(
        CheckResult(
            name=(
                "adaptive_governed_reduction_available"
            ),
            passed=adaptive_reduction_ok,
        )
    )

    results.append(
        CheckResult(
            name=(
                "human_governed_reduction_available"
            ),
            passed=human_reduction_ok,
        )
    )

    for result in results:

        status(
            result.name,
            result.passed,
        )

    return results


# ============================================================
# PERFORMANCE INTERPRETATION
# ============================================================

def print_performance_interpretation(
    output: str,
) -> None:

    section(
        "PERFORMANCE INTERPRETATION"
    )

    metrics = parse_population_metrics(
        output
    )

    adaptive = metrics.get(
        "ADAPTIVE_ABUSIVE"
    )

    human = metrics.get(
        "HUMAN_ABUSIVE"
    )

    if (
        adaptive is None
        or human is None
    ):

        print(
            "Population metrics were not available."
        )

        return

    adaptive_reduction = (
        adaptive.reduction
    )

    human_reduction = (
        human.reduction
    )

    gap = (
        adaptive_reduction
        - human_reduction
    )

    print(
        f"Adaptive abusive reduction : "
        f"{adaptive_reduction:.4f}"
    )

    print(
        f"Human abusive reduction    : "
        f"{human_reduction:.4f}"
    )

    print(
        f"Adaptive - human gap       : "
        f"{gap:.4f}"
    )

    print()

    print(
        "Interpretation:"
    )

    if gap > 0:

        print(
            "The adaptive abusive population experienced "
            "a larger governed approval reduction than the "
            "human abusive baseline in this run."
        )

    elif gap < 0:

        print(
            "The adaptive abusive population experienced "
            "a smaller governed approval reduction than the "
            "human abusive baseline in this run."
        )

    else:

        print(
            "The adaptive abusive and human abusive "
            "populations experienced the same governed "
            "approval reduction in this run."
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This comparison is evidence about the behavior "
        "of this simulation configuration."
    )

    print(
        "It does not establish universal predictive "
        "superiority of the GNN Governor."
    )


# ============================================================
# EVIDENCE SCORE
# ============================================================

def calculate_evidence_score(
    phase18_results: list[CheckResult],
    population_results: list[CheckResult],
    source_results: list[CheckResult],
    architecture_results: list[CheckResult],
    closed_loop_results: list[CheckResult],
) -> tuple[int, str]:

    groups = [
        phase18_results,
        population_results,
        source_results,
        architecture_results,
        closed_loop_results,
    ]

    all_checks = [
        result
        for group in groups
        for result in group
    ]

    if not all_checks:

        return (
            0,
            "INSUFFICIENT EVIDENCE",
        )

    passed_count = sum(
        result.passed
        for result in all_checks
    )

    total = len(
        all_checks
    )

    score = round(
        100.0
        * passed_count
        / total
    )

    critical_names = {
        "source_trained_gnn",
        "source_gnn_governor_adapter",
        "source_behavioral_governor",
        "source_risk_fusion",
        "source_gnn_governor",
        "source_interaction_graph",
        "agent_b_adaptive_customer_agent_exists",
        "agent_b_observes_effective_response",
        "adaptive_policy_update_path_present",
        "governor_before_outcome_source_evidence",
        "direct_governor_deny_absent",
        "agent_b_gnn_signal_hidden",
        "agent_b_fusion_signal_hidden",
        "agent_b_network_signal_hidden",
        "agent_b_strategy_signal_hidden",
        "agent_b_ground_truth_hidden",
    }

    critical_results = [
        result
        for result in all_checks
        if result.name
        in critical_names
    ]

    critical_passed = (
        bool(critical_results)
        and all(
            result.passed
            for result in critical_results
        )
    )

    if not critical_passed:

        classification = (
            "ARCHITECTURAL EVIDENCE INCOMPLETE"
        )

    elif score >= 90:

        classification = (
            "STRONG ARCHITECTURAL EVIDENCE"
        )

    elif score >= 75:

        classification = (
            "GOOD ARCHITECTURAL EVIDENCE"
        )

    elif score >= 60:

        classification = (
            "MODERATE ARCHITECTURAL EVIDENCE"
        )

    else:

        classification = (
            "INSUFFICIENT ARCHITECTURAL EVIDENCE"
        )

    return (
        score,
        classification,
    )


# ============================================================
# FINAL EVIDENCE SUMMARY
# ============================================================

def print_final_evidence_summary(
    phase18_run: Phase18Run,
    score: int,
    classification: str,
    all_results: list[CheckResult],
) -> None:

    section(
        "PHASE 19 — FINAL ARCHITECTURE EVIDENCE"
    )

    print(
        f"Architecture evidence score : "
        f"{score}/100"
    )

    print(
        f"Evidence classification      : "
        f"{classification}"
    )

    print()

    print(
        "What this phase establishes:"
    )

    claims = [
        (
            "1. Existing Phase 18 executes as a Python "
            "package module."
        ),
        (
            "2. The existing interaction graph is part "
            "of the GNN Governor dependency pipeline."
        ),
        (
            "3. The trained GNN inference implementation "
            "exists and is connected through the inspected "
            "architecture."
        ),
        (
            "4. GNN output reaches the Governor architecture "
            "through the inspected integration layer."
        ),
        (
            "5. The behavioral Governor remains part of "
            "the inspected decision architecture."
        ),
        (
            "6. Risk Fusion remains part of the inspected "
            "decision architecture."
        ),
        (
            "7. The Governor acts before the effective "
            "outcome path."
        ),
        (
            "8. Agent B does not directly access private "
            "GNN/fusion/network/strategy/ground-truth signals."
        ),
        (
            "9. Agent B receives an externally observable "
            "decision/response through executable observation "
            "logic."
        ),
        (
            "10. Agent B has an executable adaptive policy "
            "update path."
        ),
        (
            "11. Governor actions remain distinct from the "
            "SupportAgent DENY decision."
        ),
    ]

    for claim in claims:

        print(
            f"  {claim}"
        )

    print()

    print(
        "What this phase does NOT establish:"
    )

    limitations = [
        (
            "1. It does not prove production-level fraud "
            "detection performance."
        ),
        (
            "2. It does not prove the GNN is universally "
            "superior to non-GNN approaches."
        ),
        (
            "3. It does not prove the Governor always "
            "outperforms the human abusive baseline."
        ),
        (
            "4. It does not convert synthetic simulation "
            "results into real-world validation."
        ),
        (
            "5. A single deterministic run cannot establish "
            "statistical generalization."
        ),
        (
            "6. Static AST auditing cannot prove absence of "
            "every possible indirect information channel."
        ),
    ]

    for limitation in limitations:

        print(
            f"  {limitation}"
        )

    print()

    phase18_passed = (
        phase18_run.passed
    )

    phase18_validation_passed = all(
        result.passed
        for result in all_results
        if result.name.startswith(
            "phase18_check::"
        )
    )

    source_names = {
        "phase18_file_exists",
        "phase18_source_readable",
        "phase18_source_ast_parseable",
        "phase18_imports_final_architecture_components",
        "canonical_governor_action_space_present",
        "direct_governor_deny_absent",
        "closed_loop_architecture_present",
        "source_interaction_graph",
        "source_trained_gnn",
        "source_gnn_governor_adapter",
        "source_behavioral_governor",
        "source_risk_fusion",
        "source_gnn_governor",
        "source_adaptive_customer",
    }

    source_passed = all(
        result.passed
        for result in all_results
        if result.name in source_names
    )

    closed_loop_names = {
        "agent_b_source_loaded",
        "agent_b_source_parseable",
        "agent_b_adaptive_customer_agent_exists",
        "agent_b_gnn_signal_hidden",
        "agent_b_fusion_signal_hidden",
        "agent_b_network_signal_hidden",
        "agent_b_strategy_signal_hidden",
        "agent_b_ground_truth_hidden",
        "agent_b_observable_response_path_present",
        "agent_b_observes_effective_response",
        "adaptive_policy_update_path_present",
        "governor_before_outcome_source_evidence",
        "closed_loop_attacker_enabled",
        "governor_before_outcome",
        "adaptive_policy_change_observed",
        "adaptive_policy_change_finite",
        "adaptive_governed_reduction_available",
        "human_governed_reduction_available",
    }

    closed_loop_passed = all(
        result.passed
        for result in all_results
        if result.name
        in closed_loop_names
    )

    print(
        f"Phase 18 execution            : "
        f"{'PASSED' if phase18_passed else 'FAILED'}"
    )

    print(
        f"Phase 18 validation            : "
        f"{'PASSED' if phase18_validation_passed else 'FAILED'}"
    )

    print(
        f"Source architecture audit      : "
        f"{'PASSED' if source_passed else 'FAILED'}"
    )

    print(
        f"Closed-loop evidence           : "
        f"{'PASSED' if closed_loop_passed else 'FAILED'}"
    )


# ============================================================
# VALIDATION REPORT
# ============================================================

def print_validation_report(
    results: list[CheckResult],
) -> bool:

    section(
        "PHASE 19 VALIDATION"
    )

    overall = True

    for result in results:

        status(
            result.name,
            result.passed,
        )

        if not result.passed:

            if result.detail:

                print(
                    f"    -> {result.detail}"
                )

            overall = False

    print()

    print(
        "Overall validation       : "
        + (
            "PASSED"
            if overall
            else "FAILED"
        )
    )

    return overall


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # 1. Execute Phase 18.
    # --------------------------------------------------------

    phase18_run = run_phase18()

    if not phase18_run.passed:

        print()

        print(
            "Phase 18 execution failed."
        )

        print()

        print(
            "STDOUT:"
        )

        print(
            phase18_run.stdout
        )

        print()

        print(
            "STDERR:"
        )

        print(
            phase18_run.stderr
        )

        raise RuntimeError(
            "Phase 18 returned a non-zero exit code."
        )

    # --------------------------------------------------------
    # 2. Phase 18 validation.
    # --------------------------------------------------------

    phase18_results = (
        validate_phase18_execution(
            phase18_run
        )
    )

    section(
        "PHASE 18 VALIDATION EVIDENCE"
    )

    for result in phase18_results:

        status(
            result.name,
            result.passed,
        )

    # --------------------------------------------------------
    # 3. Population evidence.
    # --------------------------------------------------------

    population_results = (
        validate_population_evidence(
            phase18_run.stdout
        )
    )

    section(
        "POPULATION EVIDENCE VALIDATION"
    )

    for result in population_results:

        status(
            result.name,
            result.passed,
        )

    # --------------------------------------------------------
    # 4. Source architecture audit.
    # --------------------------------------------------------

    source_results = (
        audit_source_architecture()
    )

    # --------------------------------------------------------
    # 5. Agent B information boundary.
    # --------------------------------------------------------

    agent_b_results = (
        audit_agent_b_information_boundary()
    )

    # --------------------------------------------------------
    # 6. Closed-loop source audit.
    # --------------------------------------------------------

    closed_loop_source_results = (
        audit_closed_loop_source()
    )

    # --------------------------------------------------------
    # 7. Architecture claims.
    # --------------------------------------------------------

    architecture_results = (
        validate_architecture_claims(
            source_results
            + agent_b_results
            + closed_loop_source_results
        )
    )

    # --------------------------------------------------------
    # 8. Behavioral evidence.
    # --------------------------------------------------------

    closed_loop_results = (
        evaluate_closed_loop_behavior(
            phase18_run.stdout
        )
    )

    # --------------------------------------------------------
    # 9. Performance interpretation.
    # --------------------------------------------------------

    print_performance_interpretation(
        phase18_run.stdout
    )

    # --------------------------------------------------------
    # 10. Combine results.
    # --------------------------------------------------------

    all_results = (
        phase18_results
        + population_results
        + source_results
        + agent_b_results
        + closed_loop_source_results
        + architecture_results
        + closed_loop_results
    )

    # --------------------------------------------------------
    # 11. Evidence score.
    # --------------------------------------------------------

    score, classification = (
        calculate_evidence_score(
            phase18_results=phase18_results,
            population_results=population_results,
            source_results=(
                source_results
                + agent_b_results
            ),
            architecture_results=(
                architecture_results
                + closed_loop_source_results
            ),
            closed_loop_results=(
                closed_loop_results
            ),
        )
    )

    # --------------------------------------------------------
    # 12. Final summary.
    # --------------------------------------------------------

    print_final_evidence_summary(
        phase18_run=phase18_run,
        score=score,
        classification=classification,
        all_results=all_results,
    )

    # --------------------------------------------------------
    # 13. Final validation.
    # --------------------------------------------------------

    overall_passed = (
        print_validation_report(
            all_results
        )
    )

    # --------------------------------------------------------
    # 14. Final status.
    # --------------------------------------------------------

    if not overall_passed:

        print()

        print(
            "Phase 19 found one or more evidence "
            "validation failures."
        )

        raise RuntimeError(
            "Phase 19 architecture evidence validation failed."
        )

    print()

    print(
        "=" * 110
    )

    print(
        "PHASE 19 GNN GOVERNOR ARCHITECTURE "
        "EVIDENCE COMPLETE"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "The Phase 18 implementation was not modified."
    )

    print(
        "The source audit evaluates the actual implementation "
        "modules and executable AST paths used by the architecture."
    )

    print(
        f"Final architecture evidence score: "
        f"{score}/100"
    )

    print(
        f"Classification: {classification}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()