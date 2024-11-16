from pydantic import BaseModel

class ChargebackReport(BaseModel):
    """Acceptance Report Model"""
    external_ref: str
    status: bool
    source: str
    chargeback: bool