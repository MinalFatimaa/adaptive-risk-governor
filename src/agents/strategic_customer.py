from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from random import Random
from typing import Any


# ============================================================
# ACTION SPACE
# ============================================================

class StrategicAction(str, Enum):

    DIRECT_REQUEST = "DIRECT_REQUEST"

    EVIDENCE_FIRST = "EVIDENCE_FIRST"

    LOWER_AMOUNT = "LOWER_AMOUNT"

    FOLLOWUP_RESPONSE = "FOLLOWUP_RESPONSE"

    CLAIM_SWITCH = "CLAIM_SWITCH"

    WAIT_AND_RETRY = "WAIT_AND_RETRY"


# ============================================================
# CLAIM TYPES
# ============================================================

CLAIM_TYPES = (
    "SHORTAGE_CLAIM",
    "WRONG_ITEM_CLAIM",
    "NON_DELIVERY_CLAIM",
    "SUBSTITUTED_RETURN_CLAIM",
)


# ============================================================
# EVIDENCE REQUIRED BY CLAIM
# ============================================================

CLAIM_EVIDENCE = {

    "SHORTAGE_CLAIM": [
        "delivery_photo",
    ],

    "WRONG_ITEM_CLAIM": [
        "package_photo",
    ],

    "NON_DELIVERY_CLAIM": [
        "delivery_evidence",
    ],

    "SUBSTITUTED_RETURN_CLAIM": [
        "return_receipt",
    ],
}


# ============================================================
# STRATEGIC STATE
# ============================================================

@dataclass
class StrategicCustomerState:

    customer_id: str

    objective: str = (
        "maximize_successful_legitimate_resolution"
    )

    # --------------------------------------------------------
    # Q(claim_type, action)
    # --------------------------------------------------------

    action_values: dict[str, dict[str, float]] = field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Action selection counts
    # --------------------------------------------------------

    action_counts: dict[str, dict[str, int]] = field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Interaction history
    # --------------------------------------------------------

    interaction_history: list[dict[str, Any]] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Current interaction
    # --------------------------------------------------------

    current_claim_type: str | None = None

    current_amount: float = 0.0

    current_action: str | None = None

    last_support_decision: str | None = None

    last_reward: float = 0.0

    # --------------------------------------------------------
    # Learned model of Agent A
    # --------------------------------------------------------

    inferred_policy: dict[str, float] = field(
        default_factory=lambda: {

            "evidence_sensitivity": 0.5,

            "amount_sensitivity": 0.5,

            "high_value_escalation": 0.5,

            "followup_sensitivity": 0.5,
        }
    )

    # --------------------------------------------------------
    # Persistent strategy memory
    # --------------------------------------------------------

    successful_actions: list[str] = field(
        default_factory=list
    )

    failed_actions: list[str] = field(
        default_factory=list
    )


# ============================================================
# STRATEGIC CUSTOMER AGENT
# ============================================================

class StrategicCustomerAgent:

    """
    Strategic Agent B.

    Agent B:

        1. chooses a strategy
        2. interacts with Agent A
        3. observes A's response
        4. calculates reward
        5. updates Q-values
        6. updates beliefs about A
        7. predicts future A decisions
        8. changes future behavior

    Agent B does NOT access private Governor information.
    """

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        state: StrategicCustomerState,
        seed: int = 42,
        learning_rate: float = 0.20,
        exploration_rate: float = 0.20,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.05,
    ):

        self.state = state

        self.rng = Random(seed)

        self.learning_rate = learning_rate

        self.exploration_rate = exploration_rate

        self.exploration_decay = exploration_decay

        self.min_exploration_rate = (
            min_exploration_rate
        )

        self.actions = [
            action.value
            for action in StrategicAction
        ]

        self._initialize_action_values()

    # ========================================================
    # STATE FACTORY
    # ========================================================

    @classmethod
    def create_state(
        cls,
        customer_id: str,
        objective: str = (
            "maximize_successful_legitimate_resolution"
        ),
    ) -> StrategicCustomerState:

        return StrategicCustomerState(
            customer_id=customer_id,
            objective=objective,
        )

    # ========================================================
    # INITIALIZE Q TABLE
    # ========================================================

    def _initialize_action_values(self) -> None:

        for claim_type in CLAIM_TYPES:

            self.state.action_values.setdefault(
                claim_type,
                {},
            )

            self.state.action_counts.setdefault(
                claim_type,
                {},
            )

            for action in self.actions:

                self.state.action_values[
                    claim_type
                ].setdefault(
                    action,
                    0.5,
                )

                self.state.action_counts[
                    claim_type
                ].setdefault(
                    action,
                    0,
                )

    # ========================================================
    # CHOOSE ACTION
    # ========================================================

    def choose_action(
        self,
        claim_type: str | None = None,
        requested_amount: float | None = None,
        evidence_available: list[str] | None = None,
        customer: Any | None = None,
        order: Any | None = None,
        sequence_number: int | None = None,
    ):

        """
        Supports TWO interfaces.

        Low-level interface used by tests:

            choose_action(
                claim_type=...,
                requested_amount=...,
                evidence_available=...
            )

        Simulator interface:

            choose_action(
                customer=...,
                order=...,
                sequence_number=...
            )

        The first returns an action string.

        The second returns the complete interaction
        dictionary expected by strategic_simulator.py.
        """

        simulator_mode = (
            customer is not None
            or order is not None
        )

        # ----------------------------------------------------
        # SIMULATOR MODE
        # ----------------------------------------------------

        if simulator_mode:

            claim_type = (
                claim_type
                or self._select_claim_type()
            )

            requested_amount = (
                requested_amount
                if requested_amount is not None
                else self._extract_order_amount(
                    order
                )
            )

            evidence_available = (
                evidence_available
                if evidence_available is not None
                else []
            )

            action = self._choose_action_string(
                claim_type=claim_type,
                requested_amount=requested_amount,
                evidence_available=evidence_available,
            )

            return self._build_interaction(
                action=action,
                claim_type=claim_type,
                requested_amount=requested_amount,
                evidence_available=evidence_available,
            )

        # ----------------------------------------------------
        # TEST / LOW LEVEL MODE
        # ----------------------------------------------------

        if claim_type is None:

            claim_type = self._select_claim_type()

        if requested_amount is None:

            requested_amount = 1000.0

        if evidence_available is None:

            evidence_available = []

        return self._choose_action_string(
            claim_type=claim_type,
            requested_amount=requested_amount,
            evidence_available=evidence_available,
        )

    # ========================================================
    # SELECT CLAIM
    # ========================================================

    def _select_claim_type(self) -> str:

        return self.rng.choice(
            list(CLAIM_TYPES)
        )

    # ========================================================
    # EXTRACT ORDER AMOUNT
    # ========================================================

    def _extract_order_amount(
        self,
        order: Any | None,
    ) -> float:

        if order is None:

            return 1000.0

        for attribute in (
            "order_amount",
            "amount",
            "total_amount",
            "price",
        ):

            value = getattr(
                order,
                attribute,
                None,
            )

            if value is not None:

                try:

                    value = float(value)

                    if value > 0:

                        return value

                except (
                    TypeError,
                    ValueError,
                ):

                    pass

        return 1000.0

    # ========================================================
    # INTERNAL ACTION SELECTION
    # ========================================================

    def _choose_action_string(
        self,
        claim_type: str,
        requested_amount: float,
        evidence_available: list[str],
    ) -> str:

        if claim_type not in CLAIM_TYPES:

            claim_type = CLAIM_TYPES[0]

        self.state.current_claim_type = (
            claim_type
        )

        self.state.current_amount = (
            requested_amount
        )

        # ----------------------------------------------------
        # Exploration
        # ----------------------------------------------------

        if (
            self.rng.random()
            < self.exploration_rate
        ):

            action = self._exploration_action(
                claim_type=claim_type,
                requested_amount=requested_amount,
                evidence_available=evidence_available,
            )

        else:

            action = self._best_action(
                claim_type
            )

        self.state.current_action = action

        self.state.action_counts[
            claim_type
        ][action] += 1

        return action

    # ========================================================
    # BEST ACTION
    # ========================================================

    def _best_action(
        self,
        claim_type: str,
    ) -> str:

        values = self.state.action_values[
            claim_type
        ]

        max_value = max(
            values.values()
        )

        candidates = [
            action
            for action, value in values.items()
            if value == max_value
        ]

        return self.rng.choice(
            candidates
        )

    # ========================================================
    # EXPLORATION
    # ========================================================

    def _exploration_action(
        self,
        claim_type: str,
        requested_amount: float,
        evidence_available: list[str],
    ) -> str:

        candidates = list(
            self.actions
        )

        if evidence_available:

            candidates.append(
                StrategicAction.FOLLOWUP_RESPONSE.value
            )

        if requested_amount > 5000:

            candidates.append(
                StrategicAction.LOWER_AMOUNT.value
            )

        return self.rng.choice(
            candidates
        )

    # ========================================================
    # BUILD SIMULATOR INTERACTION
    # ========================================================

    def _build_interaction(
        self,
        action: str,
        claim_type: str,
        requested_amount: float,
        evidence_available: list[str],
    ) -> dict:

        # ----------------------------------------------------
        # DIRECT REQUEST
        # ----------------------------------------------------

        if action == (
            StrategicAction.DIRECT_REQUEST.value
        ):

            evidence = list(
                evidence_available
            )

        # ----------------------------------------------------
        # EVIDENCE FIRST
        # ----------------------------------------------------

        elif action == (
            StrategicAction.EVIDENCE_FIRST.value
        ):

            evidence = (
                self._required_evidence(
                    claim_type
                )
            )

        # ----------------------------------------------------
        # FOLLOW UP
        # ----------------------------------------------------

        elif action == (
            StrategicAction.FOLLOWUP_RESPONSE.value
        ):

            evidence = (
                self._required_evidence(
                    claim_type
                )
            )

        # ----------------------------------------------------
        # LOWER AMOUNT
        # ----------------------------------------------------

        elif action == (
            StrategicAction.LOWER_AMOUNT.value
        ):

            requested_amount = max(
                100.0,
                requested_amount * 0.5,
            )

            evidence = list(
                evidence_available
            )

        # ----------------------------------------------------
        # CLAIM SWITCH
        # ----------------------------------------------------

        elif action == (
            StrategicAction.CLAIM_SWITCH.value
        ):

            alternative_claims = [
                claim
                for claim in CLAIM_TYPES
                if claim != claim_type
            ]

            claim_type = self.rng.choice(
                alternative_claims
            )

            evidence = []

        # ----------------------------------------------------
        # WAIT AND RETRY
        # ----------------------------------------------------

        else:

            evidence = list(
                evidence_available
            )

        return {

            "action": action,

            "claim_type": claim_type,

            "requested_amount": round(
                requested_amount,
                2,
            ),

            "evidence_available": evidence,
        }

    # ========================================================
    # REQUIRED EVIDENCE
    # ========================================================

    def _required_evidence(
        self,
        claim_type: str,
    ) -> list[str]:

        return list(
            CLAIM_EVIDENCE.get(
                claim_type,
                [],
            )
        )

    # ========================================================
    # REWARD
    # ========================================================

    def calculate_reward(
        self,
        support_decision: str,
        requested_amount: float,
        approved_amount: float = 0.0,
        action: str | None = None,
    ) -> float:

        action = (
            action
            or self.state.current_action
        )

        if (
            self.state.objective
            == "maximize_illegitimate_refund_value"
        ):

            return self._abusive_reward(
                support_decision=support_decision,
                requested_amount=requested_amount,
                approved_amount=approved_amount,
                action=action,
            )

        if (
            self.state.objective
            == "maximize_successful_legitimate_resolution"
        ):

            return self._legitimate_reward(
                support_decision=support_decision,
                requested_amount=requested_amount,
                approved_amount=approved_amount,
            )

        return self._generic_reward(
            support_decision
        )

    # ========================================================
    # LEGITIMATE REWARD
    # ========================================================

    def _legitimate_reward(
        self,
        support_decision: str,
        requested_amount: float,
        approved_amount: float,
    ) -> float:

        if support_decision == "APPROVE":

            if requested_amount <= 0:

                return 0.0

            return min(
                approved_amount
                / requested_amount,
                1.0,
            )

        if support_decision == "REQUEST_EVIDENCE":

            return -0.10

        if support_decision == "ESCALATE":

            return -0.20

        if support_decision == "DENY":

            return -1.0

        return -0.10

    # ========================================================
    # ABUSIVE REWARD
    # ========================================================

    def _abusive_reward(
        self,
        support_decision: str,
        requested_amount: float,
        approved_amount: float,
        action: str | None,
    ) -> float:

        if support_decision == "APPROVE":

            if requested_amount <= 0:

                return 0.0

            ratio = (
                approved_amount
                / requested_amount
            )

            return 2.0 * ratio

        if support_decision == "REQUEST_EVIDENCE":

            return -0.15

        if support_decision == "ESCALATE":

            return -0.40

        if support_decision == "DENY":

            return -1.0

        return -0.10

    # ========================================================
    # GENERIC REWARD
    # ========================================================

    def _generic_reward(
        self,
        support_decision: str,
    ) -> float:

        rewards = {

            "APPROVE": 1.0,

            "REQUEST_EVIDENCE": -0.10,

            "ESCALATE": -0.25,

            "DENY": -1.0,
        }

        return rewards.get(
            support_decision,
            -0.10,
        )

    # ========================================================
    # LEARNING UPDATE
    # ========================================================

    def update(
        self,
        claim_type: str,
        action: str,
        reward: float,
        support_decision: str,
        requested_amount: float,
        approved_amount: float = 0.0,
        evidence_available: list[str] | None = None,
    ) -> None:

        evidence_available = (
            evidence_available or []
        )

        if claim_type not in CLAIM_TYPES:

            claim_type = CLAIM_TYPES[0]

        if action not in self.actions:

            return

        current_value = (
            self.state.action_values[
                claim_type
            ][action]
        )

        # ----------------------------------------------------
        # Incremental Q update
        # ----------------------------------------------------

        new_value = (
            current_value
            + self.learning_rate
            * (
                reward
                - current_value
            )
        )

        new_value = max(
            -1.0,
            min(
                2.0,
                new_value,
            ),
        )

        self.state.action_values[
            claim_type
        ][action] = new_value

        # ----------------------------------------------------
        # Learn Agent A
        # ----------------------------------------------------

        self._update_agent_a_beliefs(
            support_decision=support_decision,
            requested_amount=requested_amount,
            evidence_available=evidence_available,
        )

        # ----------------------------------------------------
        # Strategy memory
        # ----------------------------------------------------

        if reward > 0:

            if action not in (
                self.state.successful_actions
            ):

                self.state.successful_actions.append(
                    action
                )

        elif reward < 0:

            if action not in (
                self.state.failed_actions
            ):

                self.state.failed_actions.append(
                    action
                )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        self.state.interaction_history.append(
            {

                "claim_type": claim_type,

                "action": action,

                "requested_amount":
                    requested_amount,

                "evidence_available":
                    list(evidence_available),

                "support_decision":
                    support_decision,

                "approved_amount":
                    approved_amount,

                "reward":
                    reward,

                "q_value":
                    new_value,
            }
        )

        self.state.last_support_decision = (
            support_decision
        )

        self.state.last_reward = reward

        # ----------------------------------------------------
        # Exploration decay
        # ----------------------------------------------------

        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate
            * self.exploration_decay,
        )

    # ========================================================
    # LEARN AGENT A POLICY
    # ========================================================

    def _update_agent_a_beliefs(
        self,
        support_decision: str,
        requested_amount: float,
        evidence_available: list[str],
    ) -> None:

        beliefs = (
            self.state.inferred_policy
        )

        # ----------------------------------------------------
        # Evidence sensitivity
        # ----------------------------------------------------

        if support_decision == "REQUEST_EVIDENCE":

            beliefs[
                "evidence_sensitivity"
            ] = min(
                1.0,
                beliefs[
                    "evidence_sensitivity"
                ] + 0.05,
            )

        elif (
            support_decision == "APPROVE"
            and evidence_available
        ):

            beliefs[
                "evidence_sensitivity"
            ] = max(
                0.0,
                beliefs[
                    "evidence_sensitivity"
                ] - 0.02,
            )

        # ----------------------------------------------------
        # Amount sensitivity
        # ----------------------------------------------------

        if requested_amount > 5000:

            if support_decision == "ESCALATE":

                beliefs[
                    "amount_sensitivity"
                ] = min(
                    1.0,
                    beliefs[
                        "amount_sensitivity"
                    ] + 0.05,
                )

            elif support_decision == "APPROVE":

                beliefs[
                    "amount_sensitivity"
                ] = max(
                    0.0,
                    beliefs[
                        "amount_sensitivity"
                    ] - 0.02,
                )

        # ----------------------------------------------------
        # High value escalation
        # ----------------------------------------------------

        if support_decision == "ESCALATE":

            beliefs[
                "high_value_escalation"
            ] = min(
                1.0,
                beliefs[
                    "high_value_escalation"
                ] + 0.08,
            )

        # ----------------------------------------------------
        # Follow-up sensitivity
        # ----------------------------------------------------

        if support_decision == "REQUEST_EVIDENCE":

            beliefs[
                "followup_sensitivity"
            ] = min(
                1.0,
                beliefs[
                    "followup_sensitivity"
                ] + 0.03,
            )

    # ========================================================
    # OBSERVE SUPPORT RESPONSE
    # ========================================================

    def observe_support_response(
        self,
        claim_type: str,
        action: str,
        requested_amount: float,
        support_decision: str,
        approved_amount: float = 0.0,
        evidence_available: list[str] | None = None,
    ) -> float:

        reward = self.calculate_reward(
            support_decision=support_decision,
            requested_amount=requested_amount,
            approved_amount=approved_amount,
            action=action,
        )

        self.update(
            claim_type=claim_type,
            action=action,
            reward=reward,
            support_decision=support_decision,
            requested_amount=requested_amount,
            approved_amount=approved_amount,
            evidence_available=evidence_available,
        )

        return reward

    # ========================================================
    # SIMULATOR COMPATIBILITY
    # ========================================================

    def observe_support_decision(
        self,
        decision: Any,
        claim_type: str,
        requested_amount: float,
        evidence_available: list[str] | None = None,
    ) -> float:

        """
        Adapter used by strategic_simulator.py.

        Converts SupportDecision into the lower-level
        learning interface.
        """

        support_decision = getattr(
            decision,
            "decision",
            decision,
        )

        approved_amount = getattr(
            decision,
            "approved_amount",
            0.0,
        )

        action = (
            self.state.current_action
            or StrategicAction.DIRECT_REQUEST.value
        )

        return self.observe_support_response(
            claim_type=claim_type,
            action=action,
            requested_amount=requested_amount,
            support_decision=support_decision,
            approved_amount=approved_amount,
            evidence_available=evidence_available,
        )

    # ========================================================
    # PREDICT AGENT A
    # ========================================================

    def predict_support_decision(
        self,
        claim_type: str,
        requested_amount: float,
        evidence_available: list[str] | None = None,
    ) -> str:

        evidence_available = (
            evidence_available or []
        )

        beliefs = (
            self.state.inferred_policy
        )

        # ----------------------------------------------------
        # High-value cases
        # ----------------------------------------------------

        if requested_amount > 5000:

            if (
                beliefs[
                    "high_value_escalation"
                ]
                >= 0.60
            ):

                return "ESCALATE"

        # ----------------------------------------------------
        # Evidence sensitivity
        # ----------------------------------------------------

        if not evidence_available:

            if (
                beliefs[
                    "evidence_sensitivity"
                ]
                >= 0.60
            ):

                return "REQUEST_EVIDENCE"

        # ----------------------------------------------------
        # Otherwise predict approval for low/medium
        # cases.
        # ----------------------------------------------------

        if requested_amount <= 5000:

            return "APPROVE"

        return "ESCALATE"

    # ========================================================
    # INFERRED POLICY
    # ========================================================

    def get_inferred_policy(
        self,
    ) -> dict[str, float]:

        return dict(
            self.state.inferred_policy
        )

    # ========================================================
    # EXPECTED SUCCESS
    # ========================================================

    def expected_success_probability(
        self,
    ) -> float:

        if not self.state.interaction_history:

            return 0.5

        successful = sum(
            1
            for interaction
            in self.state.interaction_history
            if interaction[
                "support_decision"
            ] == "APPROVE"
        )

        total = len(
            self.state.interaction_history
        )

        return successful / total

    # ========================================================
    # POLICY / Q TABLE
    # ========================================================

    def get_policy(
        self,
    ) -> dict[str, dict[str, float]]:

        return {

            claim_type:
                dict(
                    self.state.action_values[
                        claim_type
                    ]
                )

            for claim_type in CLAIM_TYPES
        }

    # ========================================================
    # BEST ACTIONS
    # ========================================================

    def get_best_actions(
        self,
    ) -> dict[str, str]:

        return {

            claim_type:
                self._best_action(
                    claim_type
                )

            for claim_type in CLAIM_TYPES
        }

    # ========================================================
    # EXPLORATION RATE
    # ========================================================

    def get_exploration_rate(
        self,
    ) -> float:

        return self.exploration_rate

    # ========================================================
    # OBSERVATION COUNT
    # ========================================================

    def observation_count(
        self,
    ) -> int:

        return len(
            self.state.interaction_history
        )