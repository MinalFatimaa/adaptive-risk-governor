from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CustomerObservableState(BaseModel):
    customer_id: str
    account_id: str

    account_age_days: int = Field(ge=0)

    order_count: int = Field(ge=0)
    completed_order_count: int = Field(ge=0)

    refund_count: int = Field(ge=0)
    refund_amount_total: float = Field(ge=0)

    chargeback_count: int = Field(ge=0)

    current_order_ids: list[str] = []
    device_ids: list[str] = []
    address_ids: list[str] = []
    payment_ids: list[str] = []

    created_at: datetime


class CustomerPrivateState(BaseModel):
    customer_id: str

    objective: str

    current_strategy: str

    strategy_beliefs: dict[str, float] = {}

    successful_strategies: list[str] = []
    failed_strategies: list[str] = []

    observed_support_responses: list[dict] = []
    observed_outcomes: list[dict] = []

    private_memory: list[dict] = []


class CustomerGroundTruth(BaseModel):
    customer_id: str

    population: str
    behavior_type: str

    abuse_family: Optional[str] = None

    is_abusive: bool

    # Important experimental variable.
    counterparty_type: str