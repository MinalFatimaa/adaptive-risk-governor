from pydantic import BaseModel, Field


class MerchantPolicy(BaseModel):
    refund_window_days: int = 30

    max_auto_refund_amount: float = 5000

    evidence_required_above: float = 2000

    human_review_threshold: float = 5000

    eligible_reasons: list[str] = [
        "DAMAGED",
        "WRONG_ITEM",
        "NOT_RECEIVED",
        "MISSING_PART",
        "QUALITY_ISSUE",
    ]


class MerchantEconomicState(BaseModel):
    total_refund_amount: float = 0.0

    legitimate_refund_amount: float = 0.0

    abusive_refund_amount: float = 0.0

    prevented_loss: float = 0.0

    review_count: int = 0

    review_cost: float = 0.0


class MerchantState(BaseModel):
    merchant_id: str

    name: str = "Synthetic Merchant"

    currency: str = "INR"

    refund_policy: MerchantPolicy

    economic_state: MerchantEconomicState