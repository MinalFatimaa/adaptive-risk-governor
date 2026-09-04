from datetime import datetime

from ..schemas.customer import CustomerObservableState
from ..schemas.merchant import MerchantState
from ..schemas.order import OrderState
from ..schemas.support import (
    RefundRequest,
    SupportDecision,
)


class SupportAgent:

    def __init__(
        self,
        merchant: MerchantState,
    ):
        self.merchant = merchant

    # ========================================================
    # PUBLIC DECISION METHOD
    # ========================================================

    def decide(
        self,
        customer: CustomerObservableState,
        order: OrderState,
        request: RefundRequest,
    ) -> SupportDecision:

        now = request.submitted_at

        # ----------------------------------------------------
        # 1. BASIC ELIGIBILITY
        # ----------------------------------------------------

        eligibility_result = self._check_basic_eligibility(
            customer=customer,
            order=order,
            request=request,
            now=now,
        )

        if eligibility_result is not None:
            return eligibility_result

        # ----------------------------------------------------
        # 2. CLAIM-SPECIFIC VALIDATION
        # ----------------------------------------------------

        claim_result = self._validate_claim(
            customer=customer,
            order=order,
            request=request,
        )

        if claim_result is not None:
            return claim_result

        # ----------------------------------------------------
        # 3. AMOUNT / POLICY DECISION
        # ----------------------------------------------------

        return self._evaluate_amount(
            customer=customer,
            order=order,
            request=request,
        )

    # ========================================================
    # BASIC ELIGIBILITY
    # ========================================================

    def _check_basic_eligibility(
        self,
        customer: CustomerObservableState,
        order: OrderState,
        request: RefundRequest,
        now: datetime,
    ) -> SupportDecision | None:

        policy = self.merchant.refund_policy

        # Customer mismatch
        if request.customer_id != customer.customer_id:

            return self._decision(
                request=request,
                decision="DENY",
                reason_code="POLICY_VIOLATION",
                message="The refund request does not match the customer account.",
            )

        # Order mismatch
        if request.order_id != order.order_id:

            return self._decision(
                request=request,
                decision="DENY",
                reason_code="ORDER_CONTRADICTION",
                message="The refund request does not match the order.",
            )

        # Invalid amount
        if request.requested_amount <= 0:

            return self._decision(
                request=request,
                decision="DENY",
                reason_code="POLICY_VIOLATION",
                message="The requested refund amount must be greater than zero.",
            )

        # Cannot refund more than order value
        if request.requested_amount > order.order_amount:

            return self._decision(
                request=request,
                decision="DENY",
                reason_code="POLICY_VIOLATION",
                message="The requested amount exceeds the order value.",
            )

        # Refund window
        if order.delivery_timestamp is not None:

            days_since_delivery = (
                now - order.delivery_timestamp
            ).days

            if days_since_delivery > policy.refund_window_days:

                return self._decision(
                    request=request,
                    decision="DENY",
                    reason_code="POLICY_VIOLATION",
                    message="The refund request is outside the merchant refund window.",
                )

        return None

    # ========================================================
    # CLAIM VALIDATION
    # ========================================================

    def _validate_claim(
        self,
        customer: CustomerObservableState,
        order: OrderState,
        request: RefundRequest,
    ) -> SupportDecision | None:

        claim_type = request.claim_type

        # ----------------------------------------------------
        # SHORTAGE
        # ----------------------------------------------------

        if claim_type == "SHORTAGE_CLAIM":

            if order.quantity <= 1:

                return self._decision(
                    request=request,
                    decision="DENY",
                    reason_code="ORDER_CONTRADICTION",
                    message="The order quantity does not support a shortage claim.",
                )

            if "delivery_photo" not in request.evidence_available:

                return self._decision(
                    request=request,
                    decision="REQUEST_EVIDENCE",
                    reason_code="INSUFFICIENT_EVIDENCE",
                    message="Please provide evidence supporting the shortage claim.",
                    requested_evidence=[
                        "delivery_photo"
                    ],
                    requires_followup=True,
                )

        # ----------------------------------------------------
        # WRONG ITEM
        # ----------------------------------------------------

        elif claim_type == "WRONG_ITEM_CLAIM":

            if "package_photo" not in request.evidence_available:

                return self._decision(
                    request=request,
                    decision="REQUEST_EVIDENCE",
                    reason_code="INSUFFICIENT_EVIDENCE",
                    message="Please provide a photo of the received item.",
                    requested_evidence=[
                        "package_photo"
                    ],
                    requires_followup=True,
                )

        # ----------------------------------------------------
        # NON DELIVERY
        # ----------------------------------------------------

        elif claim_type == "NON_DELIVERY_CLAIM":

            if order.delivery_status == "DELIVERED":

                if "delivery_evidence" not in request.evidence_available:

                    return self._decision(
                        request=request,
                        decision="REQUEST_EVIDENCE",
                        reason_code="DELIVERY_CONTRADICTION",
                        message=(
                            "Our records indicate that the order "
                            "was delivered. Please provide additional "
                            "delivery-related evidence."
                        ),
                        requested_evidence=[
                            "delivery_evidence"
                        ],
                        requires_followup=True,
                    )

            else:

                # Undelivered orders are directly eligible
                return None

        # ----------------------------------------------------
        # SUBSTITUTED RETURN
        # ----------------------------------------------------

        elif claim_type == "SUBSTITUTED_RETURN_CLAIM":

            if "return_receipt" not in request.evidence_available:

                return self._decision(
                    request=request,
                    decision="REQUEST_EVIDENCE",
                    reason_code="INSUFFICIENT_EVIDENCE",
                    message=(
                        "Please provide return documentation "
                        "supporting the substituted-return claim."
                    ),
                    requested_evidence=[
                        "return_receipt"
                    ],
                    requires_followup=True,
                )

        # ----------------------------------------------------
        # UNKNOWN CLAIM
        # ----------------------------------------------------

        else:

            return self._decision(
                request=request,
                decision="DENY",
                reason_code="POLICY_VIOLATION",
                message="The submitted claim type is not supported.",
            )

        return None

    # ========================================================
    # AMOUNT / MERCHANT POLICY
    # ========================================================

    def _evaluate_amount(
        self,
        customer: CustomerObservableState,
        order: OrderState,
        request: RefundRequest,
    ) -> SupportDecision:

        policy = self.merchant.refund_policy

        amount = request.requested_amount

        # ----------------------------------------------------
        # HIGH VALUE
        # ----------------------------------------------------

        if amount > policy.human_review_threshold:

            return self._decision(
                request=request,
                decision="ESCALATE",
                reason_code="HIGH_VALUE_CASE",
                message=(
                    "This refund request exceeds the automatic "
                    "refund limit and requires human review."
                ),
                requires_followup=True,
            )

        # ----------------------------------------------------
        # EVIDENCE THRESHOLD
        # ----------------------------------------------------

        if amount > policy.evidence_required_above:

            if not request.evidence_available:

                return self._decision(
                    request=request,
                    decision="REQUEST_EVIDENCE",
                    reason_code="AMOUNT_REQUIRES_REVIEW",
                    message=(
                        "Additional evidence is required "
                        "for this refund amount."
                    ),
                    requested_evidence=[
                        "supporting_documentation"
                    ],
                    requires_followup=True,
                )

        # ----------------------------------------------------
        # AUTO REFUND
        # ----------------------------------------------------

        if amount <= policy.max_auto_refund_amount:

            return self._decision(
                request=request,
                decision="APPROVE",
                reason_code="ELIGIBLE",
                message=(
                    "The refund request satisfies the "
                    "current merchant refund policy."
                ),
                approved_amount=amount,
            )

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        return self._decision(
            request=request,
            decision="ESCALATE",
            reason_code="AMOUNT_REQUIRES_REVIEW",
            message=(
                "The refund request requires additional "
                "merchant review."
            ),
            requires_followup=True,
        )

    # ========================================================
    # DECISION BUILDER
    # ========================================================

    def _decision(
        self,
        request: RefundRequest,
        decision: str,
        reason_code: str,
        message: str,
        requested_evidence: list[str] | None = None,
        approved_amount: float = 0.0,
        requires_followup: bool = False,
    ) -> SupportDecision:

        return SupportDecision(
            decision=decision,
            reason_code=reason_code,
            message=message,
            requested_evidence=(
                requested_evidence or []
            ),
            approved_amount=approved_amount,
            requires_followup=requires_followup,
            request_id=request.request_id,
            customer_id=request.customer_id,
            order_id=request.order_id,
            timestamp=request.submitted_at,
        )