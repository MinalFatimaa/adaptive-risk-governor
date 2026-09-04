from dataclasses import dataclass, field
from random import Random


# ============================================================
# ACTION SPACE
# ============================================================

ACTIONS = (
    "SUBMIT_REQUEST",
    "SUBMIT_EVIDENCE",
    "ADJUST_AMOUNT",
    "RESPOND_TO_FOLLOWUP",
    "ACCEPT_DECISION",
    "ABANDON_CASE",
)


# ============================================================
# CLAIM TYPES
# ============================================================

CLAIM_TYPES = (
    "SHORTAGE_CLAIM",
    "WRONG_ITEM_CLAIM",
    "NON_DELIVERY_CLAIM",
    "SUBSTITUTED_RETURN",
)


# ============================================================
# EVIDENCE TYPES
# ============================================================

EVIDENCE_TYPES = (
    "delivery_photo",
    "package_photo",
    "return_tracking",
    "order_confirmation",
)


# ============================================================
# AGENT B STATE
# ============================================================

@dataclass
class AdaptiveCustomerState:

    customer_id: str

    objective: str

    abuse_mechanism: str | None = None

    # --------------------------------------------------------
    # WHAT DOES B BELIEVE WILL WORK?
    # --------------------------------------------------------

    strategy_beliefs: dict[str, float] = field(
        default_factory=lambda: {
            "DIRECT_REQUEST": 0.25,
            "EVIDENCE_FIRST": 0.25,
            "LOWER_AMOUNT": 0.25,
            "FOLLOWUP_RESPONSE": 0.25,
        }
    )

    # --------------------------------------------------------
    # ACTION HISTORY
    # --------------------------------------------------------

    successful_actions: list[str] = field(
        default_factory=list
    )

    failed_actions: list[str] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # OBSERVED AGENT A RESPONSES
    # --------------------------------------------------------

    support_observations: list[dict] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # OUTCOME HISTORY
    # --------------------------------------------------------

    outcome_history: list[dict] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # PRIVATE MEMORY
    # --------------------------------------------------------

    private_memory: list[str] = field(
        default_factory=list
    )

    # ========================================================
    # LEARNED MODEL OF AGENT A
    # ========================================================

    inferred_policy: dict[str, float] = field(
        default_factory=lambda: {
            "evidence_sensitivity": 0.50,
            "amount_sensitivity": 0.50,
            "high_value_escalation": 0.50,
            "followup_sensitivity": 0.50,
        }
    )

    # --------------------------------------------------------
    # OBSERVATIONS GROUPED BY CLAIM TYPE
    # --------------------------------------------------------

    observations_by_claim: dict[str, list[dict]] = field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # AMOUNT / EVIDENCE HISTORY
    # --------------------------------------------------------

    amount_outcomes: list[dict] = field(
        default_factory=list
    )

    evidence_outcomes: list[dict] = field(
        default_factory=list
    )


# ============================================================
# ADAPTIVE CUSTOMER AGENT
# ============================================================

class AdaptiveCustomerAgent:

    """
    Agent B.

    This agent represents the customer-side decision maker.

    Important architectural rule:

    Agent B does NOT receive:

        - MerchantPolicy
        - SupportAgent implementation
        - Governor state
        - ground truth
        - is_abusive
        - counterparty_type

    It learns from observable interactions with Agent A.
    """

    def __init__(
        self,
        state: AdaptiveCustomerState,
        seed: int = 42,
    ):

        self.state = state

        self.rng = Random(seed)

    # ========================================================
    # OBSERVE SUPPORT AGENT
    # ========================================================

    def observe_decision(
        self,
        decision,
    ) -> None:

        """
        Record the observable response from Agent A.

        This method intentionally does not inspect
        Agent A's internal state.
        """

        observation = {
            "decision": decision.decision,

            "reason_code": decision.reason_code,

            "requested_evidence": list(
                decision.requested_evidence
            ),

            "approved_amount": (
                decision.approved_amount
            ),

            "requires_followup": (
                decision.requires_followup
            ),
        }

        self.state.support_observations.append(
            observation
        )

    # ========================================================
    # INFER AGENT A POLICY
    # ========================================================

    def infer_support_policy(
        self,
        decision,
        claim_type: str,
        requested_amount: float,
        evidence_available: list[str],
    ) -> None:

        """
        Infer Agent A's behavior using ONLY observable
        interaction information.

        This is the important distinction between:

            learning Agent A's behavior

        and:

            reading Agent A's policy.
        """

        observation = {

            "claim_type": claim_type,

            "requested_amount": (
                requested_amount
            ),

            "evidence_available": list(
                evidence_available
            ),

            "decision": decision.decision,

            "reason_code": (
                decision.reason_code
            ),

            "requested_evidence": list(
                decision.requested_evidence
            ),

            "approved_amount": (
                decision.approved_amount
            ),

            "requires_followup": (
                decision.requires_followup
            ),
        }

        # ----------------------------------------------------
        # Store global observation
        # ----------------------------------------------------

        self.state.support_observations.append(
            observation
        )

        # ----------------------------------------------------
        # Store observation by claim type
        # ----------------------------------------------------

        self.state.observations_by_claim.setdefault(
            claim_type,
            [],
        ).append(
            observation
        )

        # ----------------------------------------------------
        # Store amount information
        # ----------------------------------------------------

        self.state.amount_outcomes.append(
            {
                "amount": requested_amount,
                "decision": decision.decision,
                "claim_type": claim_type,
            }
        )

        # ----------------------------------------------------
        # Store evidence information
        # ----------------------------------------------------

        self.state.evidence_outcomes.append(
            {
                "evidence_available": list(
                    evidence_available
                ),

                "requested_evidence": list(
                    decision.requested_evidence
                ),

                "decision": decision.decision,

                "claim_type": claim_type,
            }
        )

        # ----------------------------------------------------
        # Update inferred policy
        # ----------------------------------------------------

        self._update_policy_beliefs(
            observation
        )

    # ========================================================
    # POLICY BELIEF UPDATE
    # ========================================================

    def _update_policy_beliefs(
        self,
        observation: dict,
    ) -> None:

        decision = observation[
            "decision"
        ]

        amount = observation[
            "requested_amount"
        ]

        evidence_available = observation[
            "evidence_available"
        ]

        requested_evidence = observation[
            "requested_evidence"
        ]

        # ====================================================
        # EVIDENCE SENSITIVITY
        # ====================================================

        if decision == "REQUEST_EVIDENCE":

            if requested_evidence:

                self.state.inferred_policy[
                    "evidence_sensitivity"
                ] += 0.10

        elif decision == "APPROVE":

            if evidence_available:

                self.state.inferred_policy[
                    "evidence_sensitivity"
                ] -= 0.03

        # ====================================================
        # HIGH-VALUE ESCALATION
        # ====================================================

        if decision == "ESCALATE":

            if amount >= 5000:

                self.state.inferred_policy[
                    "high_value_escalation"
                ] += 0.12

        # ====================================================
        # AMOUNT SENSITIVITY
        # ====================================================

        if decision == "REQUEST_EVIDENCE":

            if amount > 2000:

                self.state.inferred_policy[
                    "amount_sensitivity"
                ] += 0.05

        elif decision == "APPROVE":

            if amount <= 2000:

                self.state.inferred_policy[
                    "amount_sensitivity"
                ] -= 0.02

        # ====================================================
        # FOLLOW-UP SENSITIVITY
        # ====================================================

        if observation[
            "requires_followup"
        ]:

            self.state.inferred_policy[
                "followup_sensitivity"
            ] += 0.08

        # ====================================================
        # KEEP VALUES BOUNDED
        # ====================================================

        self._bound_policy_values()

    # ========================================================
    # BOUND POLICY VALUES
    # ========================================================

    def _bound_policy_values(
        self,
    ) -> None:

        for key in self.state.inferred_policy:

            self.state.inferred_policy[
                key
            ] = min(
                1.0,
                max(
                    0.0,
                    self.state.inferred_policy[
                        key
                    ],
                ),
            )

    # ========================================================
    # UPDATE ACTION BELIEFS
    # ========================================================

    def update_beliefs(
        self,
        action: str,
        outcome: str,
        reward: float,
    ) -> None:

        """
        Update B's belief about which strategies
        are effective.
        """

        self.state.outcome_history.append(
            {
                "action": action,

                "outcome": outcome,

                "reward": reward,
            }
        )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if reward > 0:

            self.state.successful_actions.append(
                action
            )

            self._reinforce_strategy(
                action=action,
                strength=0.15,
            )

        # ----------------------------------------------------
        # FAILURE
        # ----------------------------------------------------

        elif reward < 0:

            self.state.failed_actions.append(
                action
            )

            self._reinforce_strategy(
                action=action,
                strength=-0.10,
            )

        self._normalize_beliefs()

    # ========================================================
    # ACTION → STRATEGY
    # ========================================================

    def _action_to_strategy(
        self,
        action: str,
    ) -> str | None:

        mapping = {

            "SUBMIT_REQUEST":
                "DIRECT_REQUEST",

            "SUBMIT_EVIDENCE":
                "EVIDENCE_FIRST",

            "ADJUST_AMOUNT":
                "LOWER_AMOUNT",

            "RESPOND_TO_FOLLOWUP":
                "FOLLOWUP_RESPONSE",
        }

        return mapping.get(action)

    # ========================================================
    # REINFORCE STRATEGY
    # ========================================================

    def _reinforce_strategy(
        self,
        action: str,
        strength: float,
    ) -> None:

        strategy = (
            self._action_to_strategy(
                action
            )
        )

        if strategy is None:

            return

        current = (
            self.state.strategy_beliefs[
                strategy
            ]
        )

        current += strength

        self.state.strategy_beliefs[
            strategy
        ] = max(
            0.01,
            current,
        )

    # ========================================================
    # NORMALIZE STRATEGY BELIEFS
    # ========================================================

    def _normalize_beliefs(
        self,
    ) -> None:

        total = sum(
            self.state.strategy_beliefs.values()
        )

        if total <= 0:

            return

        for strategy in (
            self.state.strategy_beliefs
        ):

            self.state.strategy_beliefs[
                strategy
            ] /= total

    # ========================================================
    # SELECT NEXT ACTION
    # ========================================================

    def select_action(
        self,
        available_actions: list[str],
        exploration_rate: float = 0.20,
    ) -> str:

        """
        Select the next action using:

            exploration
                OR
            exploitation
        """

        available_actions = [

            action

            for action in available_actions

            if action in ACTIONS

        ]

        if not available_actions:

            return "ABANDON_CASE"

        # ====================================================
        # EXPLORATION
        # ====================================================

        if (
            self.rng.random()
            < exploration_rate
        ):

            return self.rng.choice(
                available_actions
            )

        # ====================================================
        # EXPLOITATION
        # ====================================================

        scored_actions = []

        for action in available_actions:

            strategy = (
                self._action_to_strategy(
                    action
                )
            )

            if strategy is None:

                score = 0.01

            else:

                score = (
                    self.state.strategy_beliefs[
                        strategy
                    ]
                )

            scored_actions.append(
                (
                    score,
                    action,
                )
            )

        scored_actions.sort(
            reverse=True
        )

        return scored_actions[0][1]

    # ========================================================
    # GET STRATEGY BELIEFS
    # ========================================================

    def get_belief_summary(
        self,
    ) -> dict[str, float]:

        return dict(
            self.state.strategy_beliefs
        )

    # ========================================================
    # GET INFERRED AGENT A POLICY
    # ========================================================

    def get_inferred_policy(
        self,
    ) -> dict[str, float]:

        return dict(
            self.state.inferred_policy
        )

    # ========================================================
    # GET OBSERVATIONS FOR CLAIM
    # ========================================================

    def get_claim_observations(
        self,
        claim_type: str,
    ) -> list[dict]:

        return list(
            self.state.observations_by_claim.get(
                claim_type,
                [],
            )
        )