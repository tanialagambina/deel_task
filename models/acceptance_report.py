from pydantic import BaseModel
from typing import Dict
from datetime import datetime


class AcceptanceReport(BaseModel):
    """Acceptance Report Model"""
    external_ref: str
    status: bool
    source: str
    ref: str
    date_time: datetime
    state: str
    cvv_provided: bool
    amount: float
    country: str
    currency: str
    rates: Dict[str, float]