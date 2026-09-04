from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


# ============================================================================
# PATH SETUP
# ============================================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


# ============================================================================
# LOCAL IMPORTS
# ============================================================================

from phase18_bridge import (  # noqa: E402
    result_to_dict,
    run_live_evaluation,
)

from story import (  # noqa: E402
    GOVERNOR_ACTIONS,
    get_population_info,
    get_story,
)


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Adaptive Risk Governor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# SESSION STATE
# ============================================================================

if "phase18_result" not in st.session_state:
    st.session_state.phase18_result = None

if "seed" not in st.session_state:
    st.session_state.seed = 42


# ============================================================================
# CSS
# ============================================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ---------------------------------------------------------
   Hero
--------------------------------------------------------- */

.hero {
    padding: 2.4rem 2.5rem;
    border-radius: 22px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 2rem;
}

.hero-eyebrow {
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 0.6rem;
}

.hero-title {
    font-size: 2.7rem;
    line-height: 1.15;
    font-weight: 800;
    margin-bottom: 0.8rem;
}

.hero-text {
    font-size: 1.08rem;
    line-height: 1.7;
    max-width: 900px;
    opacity: 0.78;
}

/* ---------------------------------------------------------
   Story cards
--------------------------------------------------------- */

.story-card {
    padding: 1.5rem 1.7rem;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.23);
    margin: 1rem 0;
}

.story-eyebrow {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    opacity: 0.58;
    margin-bottom: 0.45rem;
}

.story-title {
    font-size: 1.45rem;
    font-weight: 750;
    margin-bottom: 0.55rem;
}

.story-text {
    font-size: 1rem;
    line-height: 1.7;
}

/* ---------------------------------------------------------
   Flow
--------------------------------------------------------- */

.flow-box {
    padding: 1.25rem;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.22);
    text-align: center;
    min-height: 120px;
}

.flow-number {
    font-size: 0.75rem;
    opacity: 0.55;
    font-weight: 800;
}

.flow-title {
    font-size: 1rem;
    font-weight: 750;
    margin-top: 0.35rem;
}

.flow-description {
    font-size: 0.82rem;
    opacity: 0.68;
    margin-top: 0.3rem;
}

/* ---------------------------------------------------------
   Result cards
--------------------------------------------------------- */

.result-card {
    padding: 1.3rem;
    border-radius: 17px;
    border: 1px solid rgba(128,128,128,0.22);
    min-height: 130px;
}

.result-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.58;
    font-weight: 800;
}

.result-number {
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 0.35rem;
}

.result-description {
    font-size: 0.83rem;
    opacity: 0.67;
    margin-top: 0.25rem;
}

/* ---------------------------------------------------------
   Section heading
--------------------------------------------------------- */

.section-number {
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    opacity: 0.52;
}

.section-title {
    font-size: 1.75rem;
    font-weight: 780;
}

/* ---------------------------------------------------------
   Footer
--------------------------------------------------------- */

.footer {
    text-align: center;
    opacity: 0.5;
    font-size: 0.8rem;
    padding-top: 2rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.markdown("## 🛡️ Adaptive Risk Governor")

    st.caption(
        "Phase 18 closed-loop GNN Governor demonstration"
    )

    st.divider()

    st.markdown("### Experiment")

    seed = st.number_input(
        "Deterministic seed",
        min_value=0,
        max_value=999999,
        value=int(st.session_state.seed),
        step=1,
    )

    st.session_state.seed = int(seed)

    run_button = st.button(
        "▶ Run Phase 18",
        type="primary",
        use_container_width=True,
    )

    clear_button = st.button(
        "Clear results",
        use_container_width=True,
    )

    st.divider()

    st.caption(
        "The UI does not implement the Governor. "
        "It visualizes the canonical Phase 18 result."
    )


# ============================================================================
# CLEAR
# ============================================================================

if clear_button:

    st.session_state.phase18_result = None

    st.rerun()


# ============================================================================
# HERO
# ============================================================================

opening = get_story("opening")

st.markdown(
    f"""
<div class="hero">

<div class="hero-eyebrow">
{opening["eyebrow"]}
</div>

<div class="hero-title">
🛡️ {opening["title"]}
</div>

<div class="hero-text">
{opening["text"]}
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# STORY — BASELINE
# ============================================================================

baseline = get_story("baseline")

st.markdown(
    f"""
<div class="story-card">

<div class="story-eyebrow">
{baseline["eyebrow"]}
</div>

<div class="story-title">
{baseline["title"]}
</div>

<div class="story-text">
{baseline["text"]}
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# BASELINE FLOW
# ============================================================================

st.markdown("### The baseline loop")

flow_columns = st.columns(4)

baseline_flow = [
    (
        "01",
        "Agent B",
        "Generates a request",
    ),
    (
        "02",
        "Agent A",
        "Makes a support decision",
    ),
    (
        "03",
        "Observable outcome",
        "Becomes visible to Agent B",
    ),
    (
        "04",
        "Agent B learns",
        "Updates its policy",
    ),
]

for column, (number, title, description) in zip(
    flow_columns,
    baseline_flow,
):

    with column:

        st.markdown(
            f"""
            <div class="flow-box">

            <div class="flow-number">
            {number}
            </div>

            <div class="flow-title">
            {title}
            </div>

            <div class="flow-description">
            {description}
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# STORY — GOVERNED
# ============================================================================

governed = get_story("governed")

st.markdown(
    f"""
<div class="story-card">

<div class="story-eyebrow">
{governed["eyebrow"]}
</div>

<div class="story-title">
{governed["title"]}
</div>

<div class="story-text">
{governed["text"]}
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# GOVERNED FLOW
# ============================================================================

st.markdown("### The governed loop")

governed_columns = st.columns(6)

governed_flow = [
    (
        "01",
        "Agent B",
        "Request",
    ),
    (
        "02",
        "Agent A",
        "Decision",
    ),
    (
        "03",
        "GNN Governor",
        "Risk assessment",
    ),
    (
        "04",
        "Governor",
        "Allow / evidence / review",
    ),
    (
        "05",
        "Effective outcome",
        "What Agent B observes",
    ),
    (
        "06",
        "Agent B",
        "Adapts",
    ),
]

for column, (number, title, description) in zip(
    governed_columns,
    governed_flow,
):

    with column:

        st.markdown(
            f"""
            <div class="flow-box">

            <div class="flow-number">
            {number}
            </div>

            <div class="flow-title">
            {title}
            </div>

            <div class="flow-description">
            {description}
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# INFORMATION BOUNDARY
# ============================================================================

hidden = get_story("hidden_information")

st.markdown(
    f"""
<div class="story-card">

<div class="story-eyebrow">
{hidden["eyebrow"]}
</div>

<div class="story-title">
{hidden["title"]}
</div>

<div class="story-text">
{hidden["text"]}
</div>

</div>
""",
    unsafe_allow_html=True,
)


boundary_columns = st.columns(2)

with boundary_columns[0]:

    st.markdown("#### Agent B can observe")

    st.success("Effective outcome")
    st.success("Observable support response")
    st.success("Claim/request context")


with boundary_columns[1]:

    st.markdown("#### Agent B cannot observe")

    st.error("GNN risk")
    st.error("Fused risk")
    st.error("Network risk")
    st.error("Strategic risk")
    st.error("Ground truth")
    st.error("Governor identity")


# ============================================================================
# GOVERNOR ACTION SPACE
# ============================================================================

st.markdown("## Governor action space")

st.caption(
    "The Governor does not directly deny a request."
)

action_columns = st.columns(3)

for column, action in zip(
    action_columns,
    GOVERNOR_ACTIONS,
):

    with column:

        st.markdown(
            f"""
            <div class="result-card">

            <div class="result-label">
            Governor action
            </div>

            <div class="result-number">
            {action["title"]}
            </div>

            <div class="result-description">
            {action["name"]}<br><br>
            {action["description"]}
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# RUN EXPERIMENT
# ============================================================================

st.markdown("## Run the experiment")

st.markdown(
    """
The next step executes the **real Phase 18 evaluation**.

This is not a mock result and the UI does not calculate the metrics.
The canonical evaluator builds the world, constructs the interaction
graph, runs the GNN Governor, executes the adaptive closed loop, validates
the information boundary, and returns the `Phase18Result`.
"""
)

if run_button:

    with st.spinner(
        "Running the closed-loop Phase 18 evaluation..."
    ):

        try:

            result = run_live_evaluation(
                seed=st.session_state.seed
            )

            st.session_state.phase18_result = result

        except Exception as exc:

            st.error(
                "Phase 18 evaluation failed."
            )

            st.exception(exc)

            st.stop()


# ============================================================================
# WAITING STATE
# ============================================================================

if st.session_state.phase18_result is None:

    st.info(
        "Click **Run Phase 18** in the sidebar to start the live evaluation."
    )

    st.stop()


# ============================================================================
# RESULT
# ============================================================================

result = st.session_state.phase18_result


# ============================================================================
# STORY — ADAPTATION
# ============================================================================

adaptation = get_story("adaptation")

st.markdown(
    f"""
<div class="story-card">

<div class="story-eyebrow">
{adaptation["eyebrow"]}
</div>

<div class="story-title">
{adaptation["title"]}
</div>

<div class="story-text">
{adaptation["text"]}
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# EXECUTION SUMMARY
# ============================================================================

st.markdown("## Phase 18 execution")

summary_columns = st.columns(3)

with summary_columns[0]:

    st.metric(
        "Total episodes",
        result.total_episodes,
    )

with summary_columns[1]:

    st.metric(
        "Adaptive attacker policy change",
        f"{result.adaptive_policy_change:.4f}",
    )

with summary_columns[2]:

    st.metric(
        "Adaptive intervention rate",
        f"{result.adaptive_abusive_intervention_rate:.1%}",
    )


# ============================================================================
# MAIN RESULT
# ============================================================================

result_story = get_story("result")

st.markdown(
    f"""
<div class="story-card">

<div class="story-eyebrow">
{result_story["eyebrow"]}
</div>

<div class="story-title">
{result_story["title"]}
</div>

<div class="story-text">
{result_story["text"]}
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# ADAPTIVE ATTACKER — HERO RESULT
# ============================================================================

st.markdown("## The key result: adaptive abuse")

st.caption(
    "This is the population most important to the closed-loop experiment."
)

adaptive_columns = st.columns(4)

with adaptive_columns[0]:

    st.metric(
        "Baseline approval",
        f"{result.adaptive_abusive_baseline_approval:.1%}",
    )

with adaptive_columns[1]:

    st.metric(
        "Governed approval",
        f"{result.adaptive_abusive_governed_approval:.1%}",
    )

with adaptive_columns[2]:

    st.metric(
        "Approval reduction",
        (
            f"{max(0.0, result.adaptive_abusive_baseline_approval - result.adaptive_abusive_governed_approval):.1%}"
        ),
    )

with adaptive_columns[3]:

    st.metric(
        "Prevented value",
        f"₹{result.adaptive_abusive_prevented_value:,.0f}",
    )


# ============================================================================
# ADAPTIVE VS HUMAN
# ============================================================================

st.markdown("## Does it still work against an adaptive attacker?")

comparison_columns = st.columns(2)

with comparison_columns[0]:

    st.markdown("### Adaptive abusive")

    st.metric(
        "Baseline approval",
        f"{result.adaptive_abusive_baseline_approval:.1%}",
    )

    st.metric(
        "Governed approval",
        f"{result.adaptive_abusive_governed_approval:.1%}",
    )

    st.metric(
        "Intervention rate",
        f"{result.adaptive_abusive_intervention_rate:.1%}",
    )

with comparison_columns[1]:

    st.markdown("### Human abusive")

    st.metric(
        "Baseline approval",
        f"{result.human_abusive_baseline_approval:.1%}",
    )

    st.metric(
        "Governed approval",
        f"{result.human_abusive_governed_approval:.1%}",
    )

    st.metric(
        "Intervention rate",
        f"{result.human_abusive_intervention_rate:.1%}",
    )


# ============================================================================
# POPULATION RESULTS
# ============================================================================

st.markdown("## What happened across the population?")

population_metrics = {
    metric.population_group: metric
    for metric in result.population_metrics
}

for population_group in (
    "ADAPTIVE_ABUSIVE",
    "ADAPTIVE_LEGITIMATE",
    "HUMAN_ABUSIVE",
    "HUMAN_LEGITIMATE",
):

    metric = population_metrics.get(
        population_group
    )

    if metric is None:
        continue

    info = get_population_info(
        population_group
    )

    with st.expander(
        info["label"],
        expanded=(
            population_group
            == "ADAPTIVE_ABUSIVE"
        ),
    ):

        st.caption(
            info["description"]
        )

        columns = st.columns(6)

        with columns[0]:

            st.metric(
                "Episodes",
                metric.episodes,
            )

        with columns[1]:

            st.metric(
                "Baseline approval",
                f"{metric.baseline_approval_rate:.1%}",
            )

        with columns[2]:

            st.metric(
                "Governed approval",
                f"{metric.governed_approval_rate:.1%}",
            )

        with columns[3]:

            st.metric(
                "Approval reduction",
                f"{metric.approval_reduction:.1%}",
            )

        with columns[4]:

            st.metric(
                "Intervention rate",
                f"{metric.intervention_rate:.1%}",
            )

        with columns[5]:

            st.metric(
                "Mean fused risk",
                f"{metric.mean_fused_risk:.3f}",
            )


# ============================================================================
# VALUE PROTECTION
# ============================================================================

st.markdown("## Economic protection")

value_columns = st.columns(2)

with value_columns[0]:

    st.markdown("### Adaptive abusive")

    st.metric(
        "Baseline approved value",
        f"₹{result.adaptive_abusive_baseline_value:,.0f}",
    )

    st.metric(
        "Governed approved value",
        f"₹{result.adaptive_abusive_governed_value:,.0f}",
    )

    st.metric(
        "Prevented approval value",
        f"₹{result.adaptive_abusive_prevented_value:,.0f}",
    )

with value_columns[1]:

    st.markdown("### Human abusive")

    st.metric(
        "Baseline approved value",
        f"₹{result.human_abusive_baseline_value:,.0f}",
    )

    st.metric(
        "Governed approved value",
        f"₹{result.human_abusive_governed_value:,.0f}",
    )

    st.metric(
        "Prevented approval value",
        f"₹{result.human_abusive_prevented_value:,.0f}",
    )


# ============================================================================
# ADAPTATION RESULT
# ============================================================================

st.markdown("## The attacker was allowed to learn")

adapt_columns = st.columns(3)

with adapt_columns[0]:

    st.metric(
        "Policy change",
        f"{result.adaptive_policy_change:.4f}",
    )

with adapt_columns[1]:

    st.metric(
        "Governor advantage",
        f"{result.adaptive_governor_advantage:.4f}",
    )

with adapt_columns[2]:

    st.metric(
        "Adaptive vs human intervention gap",
        f"{result.adaptive_vs_human_intervention_gap:.1%}",
    )


if result.adaptive_policy_change > 0:

    st.success(
        "The adaptive attacker changed its inferred policy during the "
        "closed-loop experiment. The evaluation therefore did not simply "
        "freeze the adversary."
    )

else:

    st.info(
        "No measurable policy change was observed for the adaptive "
        "population in this deterministic run."
    )


# ============================================================================
# INTERPRETATION
# ============================================================================

st.markdown("## What this demonstrates")

adaptive_reduction = (
    result.adaptive_abusive_baseline_approval
    - result.adaptive_abusive_governed_approval
)

if adaptive_reduction > 0:

    st.success(
        "The Governor reduced approval for the adaptive abusive population."
    )

else:

    st.warning(
        "The Governor did not reduce adaptive-abusive approval in this run."
    )


if result.adaptive_abusive_intervention_rate > 0:

    st.success(
        "The Governor intervened during adaptive-abusive interactions."
    )

else:

    st.warning(
        "No Governor intervention occurred for the adaptive-abusive "
        "population in this run."
    )


# ============================================================================
# TECHNICAL TRACE
# ============================================================================

with st.expander(
    "Technical architecture trace"
):

    st.markdown(
        """
### Canonical Phase 18 pipeline

```text
Agent B
   ↓
Agent A
   ↓
Interaction Graph
   ↓
Trained GNN
   ↓
GNN → Governor Adapter
   ↓
Behavioral Governor
   ↓
Risk Fusion
   ↓
Final Governor
   ↓
Allow / Additional Evidence / Human Review
   ↓
Effective observable outcome
   ↓
Agent B updates policy
   ↓
Next interaction""")