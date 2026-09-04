from __future__ import annotations

"""
STORYLINE FOR THE ADAPTIVE RISK GOVERNOR DEMO

Presentation only.

This module does not:
    - calculate risk
    - make Governor decisions
    - generate synthetic metrics
    - execute Phase 18
    - modify the simulation
"""


# ============================================================================
# STORY
# ============================================================================

STORY = {
    "opening": {
        "eyebrow": "THE PROBLEM",
        "title": "What happens when the customer learns?",
        "text": (
            "Traditional abuse detection often assumes that the behaviour "
            "being detected stays relatively stable. But an adaptive "
            "attacker can observe what happens after each interaction and "
            "change its strategy."
        ),
    },

    "baseline": {
        "eyebrow": "WITHOUT THE GOVERNOR",
        "title": "The attacker learns from the outcome",
        "text": (
            "Agent B interacts with Agent A. Agent A produces a decision. "
            "The resulting outcome becomes observable to Agent B, which "
            "updates its policy before the next interaction."
        ),
    },

    "governed": {
        "eyebrow": "WITH THE GOVERNOR",
        "title": "The decision is checked before the outcome",
        "text": (
            "The GNN Risk Governor observes the interaction before the "
            "business outcome. It can allow the Support Agent decision, "
            "request additional evidence, or escalate the case for human "
            "review."
        ),
    },

    "hidden_information": {
        "eyebrow": "THE INFORMATION BOUNDARY",
        "title": "The attacker never sees the Governor's private signals",
        "text": (
            "Agent B learns only from the observable effective outcome. "
            "It does not receive the GNN risk, fused risk, network risk, "
            "strategic risk, ground truth, or the identity of the Governor."
        ),
    },

    "adaptation": {
        "eyebrow": "THE CLOSED LOOP",
        "title": "The attacker is allowed to adapt",
        "text": (
            "This is important: the experiment does not freeze the "
            "attacker. Agent B can change its inferred policy after seeing "
            "the observable result. The question is whether the Governor "
            "still protects the system under that adaptation."
        ),
    },

    "result": {
        "eyebrow": "THE RESULT",
        "title": "Does the Governor change the outcome?",
        "text": (
            "Phase 18 compares the ungovened baseline with the governed "
            "closed-loop system across human and adaptive populations."
        ),
    },
}


# ============================================================================
# POPULATION PRESENTATION
# ============================================================================

POPULATION_INFO = {
    "ADAPTIVE_ABUSIVE": {
        "label": "Adaptive Attacker",
        "description": (
            "The key adversarial population. The attacker can update its "
            "policy based on observable outcomes."
        ),
    },

    "ADAPTIVE_LEGITIMATE": {
        "label": "Adaptive Legitimate",
        "description": (
            "An adaptive customer whose objective is legitimate."
        ),
    },

    "HUMAN_ABUSIVE": {
        "label": "Human Abusive",
        "description": (
            "A non-adaptive abusive customer population."
        ),
    },

    "HUMAN_LEGITIMATE": {
        "label": "Human Legitimate",
        "description": (
            "A normal legitimate customer population."
        ),
    },
}


# ============================================================================
# GOVERNOR ACTIONS
# ============================================================================

GOVERNOR_ACTIONS = [
    {
        "name": "ALLOW_AGENT_A_DECISION",
        "title": "Allow",
        "description": (
            "The Support Agent's decision becomes the effective outcome."
        ),
    },

    {
        "name": "REQUEST_ADDITIONAL_EVIDENCE",
        "title": "Request Evidence",
        "description": (
            "The observable outcome becomes a request for additional "
            "evidence."
        ),
    },

    {
        "name": "ESCALATE_TO_HUMAN_REVIEW",
        "title": "Human Review",
        "description": (
            "The observable outcome becomes an escalation to human review."
        ),
    },
]


# ============================================================================
# HELPERS
# ============================================================================

def get_story(section: str) -> dict[str, str]:
    """Return a story section."""

    if section not in STORY:
        raise KeyError(
            f"Unknown story section: {section}"
        )

    return STORY[section]


def get_population_info(
    population_group: str,
) -> dict[str, str]:
    """Return presentation information for a population."""

    if population_group not in POPULATION_INFO:
        raise KeyError(
            f"Unknown population group: {population_group}"
        )

    return POPULATION_INFO[
        population_group
    ]